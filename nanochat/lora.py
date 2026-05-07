"""
LoRA (Hu et al. 2021) for nanochat: rank-r additive adapters on frozen
base linear layers, mirroring the master-fp32 / matmul-in-input-dtype
convention used by `nanochat.gpt.Linear`.

Forward of LoRALinear:

    out = base(x) + (x @ A^T) @ B^T * (alpha / rank)

A is `(rank, in_features)`, kaiming_uniform_ initialized.
B is `(out_features, rank)`, zero initialized — so at apply-time the
forward is bit-equivalent to the base, and training drift starts from
the base output exactly.

Standard usage:

    model, tokenizer, _ = load_model("sft", device, ...)
    info = apply_lora(model, target=("c_q", "c_v"), rank=8, alpha=16)
    # `info["trainable_params"]` is the LoRA param count.
    # All non-LoRA params are now frozen.

Saving / loading the adapter only:

    state = lora_state_dict(model)              # ~5-10 MB at d6 r=8 Q+V
    torch.save(state, path)
    ...
    apply_lora(model_clone, target=..., rank=..., alpha=...)
    load_lora_state_dict(model_clone, torch.load(path))
"""

import glob
import json
import math
import os
import re
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen base Linear plus a rank-r additive adapter.

    Mirrors `nanochat.gpt.Linear`: master weights stay fp32, matmuls
    run in the activation dtype (typically bf16) by per-call casting.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        for p in self.base.parameters():
            p.requires_grad = False

        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        a = self.lora_A.to(dtype=x.dtype)
        b = self.lora_B.to(dtype=x.dtype)
        lora_out = F.linear(F.linear(x, a), b)
        return base_out + lora_out * self.scaling

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, alpha={self.alpha}"


def _replace_child(parent: nn.Module, name: str, new_child: nn.Module) -> None:
    setattr(parent, name, new_child)


def apply_lora(
    model: nn.Module,
    target: Iterable[str] = ("c_q", "c_v"),
    rank: int = 8,
    alpha: float = 16.0,
    *,
    freeze_base: bool = True,
) -> dict:
    """Walk the transformer blocks of `model.transformer.h` and replace
    attention projections matching `target` with `LoRALinear` wrappers.

    `target` is a tuple of attribute names on `block.attn`. Defaults to
    `("c_q", "c_v")` — the LoRA-paper standard. To adapt all four
    attention projections use `("c_q", "c_k", "c_v", "c_proj")`.

    Returns a dict with `injected_count`, `trainable_params`,
    `base_params`, and `targets`.
    """
    target = tuple(target)
    if not target:
        raise ValueError("`target` must contain at least one projection name")

    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise AttributeError(
            "apply_lora expects `model.transformer.h` (a list of blocks); "
            f"got {type(model).__name__} without that surface."
        )

    injected = 0
    blocks = model.transformer.h
    for block in blocks:
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        for proj_name in target:
            if not hasattr(attn, proj_name):
                continue
            base_linear = getattr(attn, proj_name)
            if not isinstance(base_linear, nn.Linear):
                continue
            wrapped = LoRALinear(base_linear, rank=rank, alpha=alpha)
            _replace_child(attn, proj_name, wrapped)
            injected += 1

    if injected == 0:
        raise RuntimeError(
            f"apply_lora injected zero adapters; check `target={target}` "
            "matches actual attention attribute names on this model."
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    base = sum(p.numel() for p in model.parameters()) - trainable
    return {
        "injected_count": injected,
        "trainable_params": trainable,
        "base_params": base,
        "targets": target,
        "rank": rank,
        "alpha": float(alpha),
    }


def lora_state_dict(model: nn.Module) -> dict:
    """Return only the LoRA adapter tensors (`*.lora_A`, `*.lora_B`)."""
    return {
        name: param.detach().clone()
        for name, param in model.state_dict().items()
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    }


def load_lora_state_dict(model: nn.Module, state_dict: dict, *, strict: bool = True) -> None:
    """Load LoRA tensors into a model that has already had `apply_lora` applied.

    With `strict=True` (default), every key in `state_dict` must be present
    in the model and every adapter parameter in the model must be present
    in `state_dict`. The check is symmetric so that a tag mismatch (wrong
    rank, different `target`) surfaces immediately.
    """
    expected = {
        name for name, _ in model.named_parameters()
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    }
    got = set(state_dict.keys())
    if strict:
        missing = expected - got
        unexpected = got - expected
        if missing or unexpected:
            raise RuntimeError(
                f"LoRA state-dict mismatch — missing: {sorted(missing)[:5]}, "
                f"unexpected: {sorted(unexpected)[:5]}"
            )

    own = dict(model.named_parameters())
    for k, v in state_dict.items():
        if k not in own:
            continue
        target_param = own[k]
        if target_param.shape != v.shape:
            raise RuntimeError(
                f"LoRA tensor shape mismatch at {k}: model={tuple(target_param.shape)} "
                f"checkpoint={tuple(v.shape)}"
            )
        with torch.no_grad():
            target_param.copy_(v.to(dtype=target_param.dtype, device=target_param.device))


def lora_parameters(model: nn.Module) -> list:
    """Convenience: return the list of trainable LoRA parameters in this model."""
    return [p for n, p in model.named_parameters()
            if p.requires_grad and (n.endswith(".lora_A") or n.endswith(".lora_B"))]


def apply_lora_from_tag(
    model: nn.Module,
    lora_tag: str,
    *,
    base_dir: Optional[str] = None,
    step: Optional[int] = None,
) -> dict:
    """One-call helper for inference scripts: read a LoRA checkpoint dir at
    `<base_dir>/lora_checkpoints/<lora_tag>/`, apply the saved adapter
    config to `model` via `apply_lora`, and load the saved adapter weights.

    `base_dir` defaults to `nanochat.common.get_base_dir()`. `step` defaults
    to the latest `lora_<step>.pt`.

    Returns the `apply_lora` info dict augmented with `loaded_step` and
    `meta_path`.
    """
    if base_dir is None:
        from nanochat.common import get_base_dir
        base_dir = get_base_dir()
    lora_dir = os.path.join(base_dir, "lora_checkpoints", lora_tag)
    if not os.path.isdir(lora_dir):
        raise FileNotFoundError(f"LoRA dir not found: {lora_dir}")

    if step is None:
        candidates = []
        for fn in os.listdir(lora_dir):
            m = re.match(r"lora_(\d+)\.pt$", fn)
            if m:
                candidates.append(int(m.group(1)))
        if not candidates:
            raise FileNotFoundError(f"No lora_*.pt files in {lora_dir}")
        step = max(candidates)

    weight_path = os.path.join(lora_dir, f"lora_{step:06d}.pt")
    meta_path = os.path.join(lora_dir, f"meta_{step:06d}.json")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"LoRA weights not found: {weight_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"LoRA meta not found: {meta_path}")

    with open(meta_path) as f:
        lora_meta = json.load(f)
    li = lora_meta["lora_info"]
    info = apply_lora(
        model,
        target=tuple(li["targets"]),
        rank=int(li["rank"]),
        alpha=float(li["alpha"]),
    )
    state = torch.load(weight_path, map_location="cpu")
    load_lora_state_dict(model, state, strict=True)
    info["loaded_step"] = step
    info["meta_path"] = meta_path
    return info
