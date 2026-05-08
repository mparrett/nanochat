"""
Low-bit weight quantization for nanochat: Binary `{-s, +s}` and Ternary
`{-s, 0, +s}` linear layers with the Straight-Through Estimator (STE).

Phase 1 of the 1-bit-from-scratch direction (`docs/project_notes/backlog.md`).
The primitives are ported from `~/projects-new/trx4mr/picoGPT/binary.py`,
adapted to nanochat's master-fp32 / matmul-in-input-dtype convention used by
`nanochat.gpt.Linear`:

  - The latent `weight` Parameter stays fp32 (so the optimizer keeps full
    precision). Saving / loading via state_dict round-trips the latent.
  - STE quantization (binarize/ternarize) is computed in fp32 for numerical
    safety, then the quantized tensor is cast to `x.dtype` *before* the
    `F.linear` matmul. This composes cleanly with the bf16 path on CUDA/M2.
  - Per-group FP scales (group_size=128 by default) — Bonsai-style. The
    scale is the per-group mean(abs(w)); STE backward passes grads
    unchanged.

Standard usage (mirrors `apply_lora`):

    model, _, _ = load_model("base", device, ...)        # or fresh GPT
    info = apply_quant(model, variant="binary", target=("c_q","c_k","c_v",
                       "c_proj","c_fc"))
    # Every targeted nn.Linear in transformer.h is now a BinaryLinear with a
    # freshly kaiming_uniform_-initialized latent weight (reinit=True default).

Phase 1 deliberately does NOT include `lm_head`, `wte`, value embeddings,
or the smear gate. That's a Phase 2 decision (Bonsai's "no escape hatches"
claim is interesting but not load-bearing for the d6 STE smoke test).

Exotic flags from the trx4mr port (`learned_scale`, `pre_norm`,
`centralize`, `turbo`) are deliberately omitted — they're separate research
knobs that can be re-introduced in a follow-up if the vanilla STE smoke
test motivates them.
"""

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# STE quantization functions
# ---------------------------------------------------------------------------

class STEBinarize(torch.autograd.Function):
    """Binarize a fp32 weight to `{-scale, +scale}` per group; STE backward.

    Forward:
        groups = reshape(weight, (-1, group_size))
        scale  = mean(|groups|, dim=-1)            # per-group fp scale
        binary = sign(groups) * scale              # exact zeros -> +scale

    Backward: pass `grad_output` unchanged to the latent weight (STE).
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, group_size: int) -> torch.Tensor:
        shape = weight.shape
        flat = weight.reshape(-1)

        remainder = flat.numel() % group_size
        if remainder:
            flat = F.pad(flat, (0, group_size - remainder))

        groups = flat.reshape(-1, group_size)
        scales = groups.abs().mean(dim=-1, keepdim=True)
        binary = groups.sign() * scales

        # sign(0) == 0; map exact zeros to +scale to avoid dropping mass.
        zero_mask = groups == 0
        if zero_mask.any():
            binary[zero_mask] = scales.expand_as(groups)[zero_mask]

        return binary.reshape(-1)[: shape.numel()].reshape(shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def ste_binarize(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    return STEBinarize.apply(weight, group_size)


class STETernarize(torch.autograd.Function):
    """Ternarize a fp32 weight to `{-scale, 0, +scale}` per group; STE backward.

    Threshold rule (BitNet b1.58 / Bonsai-style): values with |w| <= mean(|w|)
    in the group map to 0; the rest are signed and scaled. STE pass-through
    in backward.
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, group_size: int) -> torch.Tensor:
        shape = weight.shape
        flat = weight.reshape(-1)

        remainder = flat.numel() % group_size
        if remainder:
            flat = F.pad(flat, (0, group_size - remainder))

        groups = flat.reshape(-1, group_size)
        abs_groups = groups.abs()
        scales = abs_groups.mean(dim=-1, keepdim=True)

        mask = abs_groups > scales
        ternary = torch.zeros_like(groups)
        ternary[mask] = groups[mask].sign() * scales.expand_as(groups)[mask]

        return ternary.reshape(-1)[: shape.numel()].reshape(shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def ste_ternarize(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    return STETernarize.apply(weight, group_size)


# ---------------------------------------------------------------------------
# Quantized Linear modules
# ---------------------------------------------------------------------------

class _QuantLinearBase(nn.Module):
    """Shared scaffolding for BinaryLinear / TernaryLinear.

    Latent fp32 weight is the only trainable parameter. Forward quantizes
    in fp32, casts the quantized tensor to `x.dtype`, and matmuls. This
    matches `nanochat.gpt.Linear`'s master-fp32 convention.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False, group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def _quantize(self, w: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize in fp32 (numerical safety), then cast to activation dtype
        # right before the matmul — same boundary as nanochat.gpt.Linear.
        qw = self._quantize(self.weight)
        return F.linear(x, qw.to(dtype=x.dtype),
                        self.bias.to(dtype=x.dtype) if self.bias is not None else None)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"group_size={self.group_size}, bias={self.bias is not None}")


class BinaryLinear(_QuantLinearBase):
    """nn.Linear replacement with binary `{-s, +s}` weights and STE backward."""
    def _quantize(self, w):
        return ste_binarize(w, self.group_size)


class TernaryLinear(_QuantLinearBase):
    """nn.Linear replacement with ternary `{-s, 0, +s}` weights and STE backward."""
    def _quantize(self, w):
        return ste_ternarize(w, self.group_size)


# ---------------------------------------------------------------------------
# Walker: replace fp Linears with quantized variants in transformer blocks
# ---------------------------------------------------------------------------

_VARIANT_TO_CLS = {
    "binary": BinaryLinear,
    "ternary": TernaryLinear,
}

# Default targets cover both attention and MLP matrices in a nanochat block.
# Note: there's a name collision — `c_proj` exists on both `block.attn` and
# `block.mlp`. The walker handles each subtree independently so both are
# caught.
DEFAULT_QUANT_TARGETS = ("c_q", "c_k", "c_v", "c_proj", "c_fc")


def _replace_child(parent: nn.Module, name: str, new_child: nn.Module) -> None:
    setattr(parent, name, new_child)


def _swap_linear(parent: nn.Module, name: str, *, variant: str,
                 group_size: int, reinit: bool) -> nn.Module | None:
    """Swap `parent.<name>` with a quantized Linear if it currently is an
    nn.Linear (or subclass — `nanochat.gpt.Linear` qualifies). Returns the
    new module, or None if no swap was performed."""
    base = getattr(parent, name, None)
    if base is None or not isinstance(base, nn.Linear):
        return None
    cls = _VARIANT_TO_CLS[variant]
    has_bias = base.bias is not None
    new = cls(base.in_features, base.out_features,
              bias=has_bias, group_size=group_size)
    # Latent weight stays fp32 (master-precision invariant — same as
    # nanochat.gpt.Linear). Move to base's device only.
    new = new.to(device=base.weight.device)
    if not reinit:
        with torch.no_grad():
            new.weight.copy_(base.weight.to(dtype=torch.float32))
            if has_bias:
                new.bias.copy_(base.bias.to(dtype=torch.float32))
    _replace_child(parent, name, new)
    return new


def apply_quant(
    model: nn.Module,
    *,
    variant: str = "binary",
    target: Iterable[str] = DEFAULT_QUANT_TARGETS,
    group_size: int = 128,
    reinit: bool = True,
) -> dict:
    """Replace targeted nn.Linears in `model.transformer.h` with quantized
    variants in-place.

    Phase 1 of the 1-bit-from-scratch direction. The walker only touches
    attention (`block.attn.<name>`) and MLP (`block.mlp.<name>`) submodules
    in each transformer block; `lm_head`, `wte`, value embeddings, and the
    smear gate are deliberately untouched.

    Args:
        model: GPT-shaped module exposing `transformer.h` (a list of blocks).
        variant: "binary" or "ternary".
        target: attribute names to look up under `block.attn` and `block.mlp`.
            Defaults cover all attention projections (Q/K/V/c_proj) and the
            MLP up-projection. Note: MLP's down-projection is also called
            `c_proj` — both attn and mlp `c_proj` get quantized when the
            target includes "c_proj".
        group_size: per-group scale block size for the STE quantizer (128
            matches the Bonsai brief and the trx4mr binary primitives).
        reinit: if True (default), the freshly-constructed quantized layer
            keeps its kaiming_uniform_ init from `__init__`. If False, the
            latent weight is copied from the original fp Linear (useful for
            "quantize a trained checkpoint" scenarios). Phase 1 from-scratch
            pretrain wants reinit=True.

    Returns a dict with `injected_count`, `total_params` (sum of latent
    weight params on swapped modules), `targets`, `variant`, `group_size`.

    Raises:
        ValueError: unknown variant, or empty target tuple.
        AttributeError: model lacks `transformer.h`.
        RuntimeError: zero injections (target name typo, etc).
    """
    if variant not in _VARIANT_TO_CLS:
        raise ValueError(
            f"unknown variant: {variant!r}. Choose from {sorted(_VARIANT_TO_CLS)}"
        )
    target = tuple(target)
    if not target:
        raise ValueError("`target` must contain at least one attribute name")

    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise AttributeError(
            "apply_quant expects `model.transformer.h` (a list of blocks); "
            f"got {type(model).__name__} without that surface."
        )

    injected = 0
    total_params = 0
    for block in model.transformer.h:
        for parent_name in ("attn", "mlp"):
            parent = getattr(block, parent_name, None)
            if parent is None:
                continue
            for proj_name in target:
                new = _swap_linear(parent, proj_name, variant=variant,
                                   group_size=group_size, reinit=reinit)
                if new is not None:
                    injected += 1
                    total_params += new.weight.numel()

    if injected == 0:
        raise RuntimeError(
            f"apply_quant injected zero modules; check `target={target}` "
            "matches actual attribute names on this model's blocks."
        )

    return {
        "injected_count": injected,
        "total_params": total_params,
        "targets": target,
        "variant": variant,
        "group_size": group_size,
    }
