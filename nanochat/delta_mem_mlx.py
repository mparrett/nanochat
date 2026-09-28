"""
MLX port of δ-mem (Lei et al. 2026) for the published `declare-lab/delta-mem_qwen3_4b-instruct`
TSW adapter. Math reconstructed from the upstream reference at
/tmp/delta-Mem/deltamem/core/delta_impl.py; see docs/delta_mem_recon_2026-05-17.md.

Supports only the active subset the published adapter uses:
  delta_heads=("q", "o"), couple_lambda=True, state_update_mode="standard",
  rankwise_gates=True, normalize_qk=True, num_state_heads=1,
  memory_write_granularity="token" (TSW).

Other knobs in the upstream HFDeltaMemConfig (partition, latent, synthetic_kv,
message-mean writes, multi-head state) are deliberately not ported — they are
not active for this adapter and would be dead code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DeltaMemConfig:
    """Subset of upstream HFDeltaMemConfig matching the published adapter."""

    rank: int = 8
    alpha: float = 16.0
    # All defaults below match `delta_mem_config.json` for the published adapter.
    # They are surfaced for readability; the port hardcodes their semantics
    # (couple_lambda, normalize_qk, standard, token TSW) and will reject configs
    # that disagree.
    couple_lambda: bool = True
    normalize_qk: bool = True
    state_update_mode: str = "standard"
    rankwise_gates: bool = True
    num_state_heads: int = 1
    memory_write_granularity: str = "token"
    memory_readout_mode: str = "delta"
    delta_heads: tuple[str, ...] = ("q", "o")

    @property
    def delta_scaling(self) -> float:
        return self.alpha / self.rank

    @classmethod
    def from_dict(cls, d: dict) -> "DeltaMemConfig":
        unsupported = []
        if not d.get("couple_lambda", True):
            unsupported.append("couple_lambda=False")
        if not d.get("normalize_qk", True):
            unsupported.append("normalize_qk=False")
        if d.get("state_update_mode", "standard") != "standard":
            unsupported.append(f"state_update_mode={d.get('state_update_mode')}")
        if not d.get("rankwise_gates", True):
            unsupported.append("rankwise_gates=False")
        if d.get("num_state_heads", 1) != 1:
            unsupported.append(f"num_state_heads={d.get('num_state_heads')}")
        if d.get("memory_write_granularity", "token") != "token":
            unsupported.append(f"memory_write_granularity={d.get('memory_write_granularity')}")
        if d.get("memory_readout_mode", "delta") != "delta":
            unsupported.append(f"memory_readout_mode={d.get('memory_readout_mode')}")
        heads = tuple(d.get("delta_heads", ("q", "o")))
        if set(heads) - {"q", "o"}:
            unsupported.append(f"delta_heads={heads} (active k/v not ported)")
        if unsupported:
            raise NotImplementedError(
                f"This MLX port does not support: {', '.join(unsupported)}"
            )
        return cls(
            rank=int(d.get("rank", 8)),
            alpha=float(d.get("alpha", 16.0)),
            delta_heads=heads,
        )


def _l2_normalize(x: mx.array, eps: float = 1e-6) -> mx.array:
    sq = mx.sum(x * x, axis=-1, keepdims=True)
    return x * mx.rsqrt(mx.maximum(sq, eps * eps))


class DeltaMemLayer(nn.Module):
    """Per-layer δ-mem module owning the 7 trainable tensors for one Qwen3 attention block."""

    def __init__(
        self,
        hidden_size: int,
        q_out_features: int,
        o_out_features: int,
        config: DeltaMemConfig,
    ):
        super().__init__()
        self.config = config
        self.rank = config.rank
        self.delta_scaling = config.delta_scaling
        # Extra multiplier on δ_o only. 1.0 reproduces the trained adapter; >1
        # compensates for backbones (e.g. ternary Bonsai) whose attention outputs
        # dwarf the correction the adapter was trained to make on fp Qwen3.
        self.o_scale = 1.0
        # Memory read/write projections (act on hidden_size → rank)
        self.memory_q_proj = mx.zeros((self.rank, hidden_size))
        self.memory_k_proj = mx.zeros((self.rank, hidden_size))
        self.memory_v_proj = mx.zeros((self.rank, hidden_size))
        # Low-rank q/o corrections (act on rank → base feature dim)
        self.delta_q_proj = mx.zeros((q_out_features, self.rank))
        self.delta_o_proj = mx.zeros((o_out_features, self.rank))
        # β gate (rankwise)
        self.beta_proj = mx.zeros((self.rank, hidden_size))
        self.beta_bias = mx.zeros((self.rank,))

    def project_memory(
        self, x: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Returns (q_m, k_m, v_m, β) each shaped (B, T, rank). q_m/k_m are tanh+L2-normalized."""
        memory_q = x @ self.memory_q_proj.T
        memory_k = x @ self.memory_k_proj.T
        memory_v = x @ self.memory_v_proj.T
        memory_q = _l2_normalize(mx.tanh(memory_q))
        memory_k = _l2_normalize(mx.tanh(memory_k))
        beta = mx.sigmoid(x @ self.beta_proj.T + self.beta_bias)
        return memory_q, memory_k, memory_v, beta

    def scan(
        self,
        state: mx.array,
        memory_q: mx.array,
        memory_k: mx.array,
        memory_v: mx.array,
        beta: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Gated delta-rule scan. state: (B, R, R). q/k/v/β: (B, T, R). Returns (state_final, reads)."""
        B, T, R = memory_q.shape
        reads_list: list[mx.array] = []
        S = state
        for t in range(T):
            q_t = memory_q[:, t, :]
            k_t = memory_k[:, t, :]
            v_t = memory_v[:, t, :]
            beta_t = beta[:, t, :]
            lam_t = 1.0 - beta_t

            # (B, R, R) @ (B, R, 1) -> (B, R, 1) -> (B, R)
            read_t = (S @ q_t[..., None])[..., 0]
            pred_t = (S @ k_t[..., None])[..., 0]
            # Outer products (B, R, 1) * (B, 1, R) -> (B, R, R)
            write_outer = v_t[..., None] * k_t[:, None, :]
            pred_outer = pred_t[..., None] * k_t[:, None, :]
            # Gates broadcast row-wise: (B, R, 1) over (B, R, R)
            keep_t = lam_t[..., None]
            erase_t = beta_t[..., None]
            write_t = beta_t[..., None]
            S = keep_t * S - erase_t * pred_outer + write_t * write_outer
            reads_list.append(read_t)

        reads = mx.stack(reads_list, axis=1)  # (B, T, R)
        return S, reads

    def project_deltas(self, reads: mx.array) -> tuple[mx.array, mx.array]:
        """Returns (δ_q, δ_o) shaped (B, T, q_out), (B, T, o_out). Scaled by α/rank."""
        delta_q = (reads @ self.delta_q_proj.T) * self.delta_scaling
        delta_o = (reads @ self.delta_o_proj.T) * (self.delta_scaling * self.o_scale)
        return delta_q, delta_o


class DeltaMemAttention(nn.Module):
    """Wraps mlx-lm Qwen3 Attention; injects δ-mem read-then-steer at the
    attention block. State persists in self.delta_state and is updated per
    forward call. Reset via self.reset_state() between independent problems."""

    def __init__(self, base_attn: nn.Module, delta_mem: DeltaMemLayer):
        super().__init__()
        self.base = base_attn
        self.delta_mem = delta_mem
        self.delta_state: Optional[mx.array] = None

    def reset_state(self) -> None:
        self.delta_state = None

    def _ensure_state(self, batch_size: int, dtype) -> mx.array:
        if self.delta_state is None or self.delta_state.shape[0] != batch_size:
            self.delta_state = mx.zeros(
                (batch_size, self.delta_mem.rank, self.delta_mem.rank), dtype=dtype
            )
        return self.delta_state

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        from mlx_lm.models.base import scaled_dot_product_attention

        B, L, _D = x.shape
        state = self._ensure_state(B, x.dtype)

        # δ-mem read+write over the L-token sequence
        mq, mk, mv, beta = self.delta_mem.project_memory(x)
        state_new, reads = self.delta_mem.scan(state, mq, mk, mv, beta)
        self.delta_state = state_new
        delta_q, delta_o = self.delta_mem.project_deltas(reads)

        base = self.base
        queries = base.q_proj(x) + delta_q.astype(x.dtype)
        keys = base.k_proj(x)
        values = base.v_proj(x)

        queries = base.q_norm(queries.reshape(B, L, base.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = base.k_norm(keys.reshape(B, L, base.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, base.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = base.rope(queries, offset=cache.offset)
            keys = base.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = base.rope(queries)
            keys = base.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=base.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return base.o_proj(output) + delta_o.astype(x.dtype)


def attach_delta_mem(model: nn.Module, config: DeltaMemConfig) -> list[str]:
    """Walk model.model.layers, replacing each TransformerBlock.self_attn with a
    DeltaMemAttention wrapper. Returns the list of layer keys that were wrapped.

    Hidden_size is taken from model.args (works for both fp16 and quantized
    bases; QuantizedLinear stores weight in packed layout so weight.shape[1]
    is NOT in_features). Out features are read from weight.shape[0] which is
    correct for both layouts."""
    wrapped = []
    hidden_size = model.args.hidden_size
    layers = model.model.layers
    for layer_idx, block in enumerate(layers):
        base_attn = block.self_attn
        q_out = base_attn.q_proj.weight.shape[0]
        o_out = base_attn.o_proj.weight.shape[0]
        delta_layer = DeltaMemLayer(hidden_size, q_out, o_out, config)
        block.self_attn = DeltaMemAttention(base_attn, delta_layer)
        wrapped.append(f"model.layers.{layer_idx}.self_attn")
    return wrapped


def load_delta_mem_adapter(model: nn.Module, adapter_path: str) -> int:
    """Load an MLX-converted δ-mem adapter (produced by scripts/convert_delta_mem_adapter.py)
    into a model previously wrapped via attach_delta_mem. Returns the number of tensors
    loaded."""
    sd = mx.load(adapter_path)
    layers = model.model.layers
    n_loaded = 0
    for key, tensor in sd.items():
        # key example: "model.layers.0.self_attn.memory_q_proj"
        if ".self_attn." not in key:
            raise ValueError(f"Unexpected adapter key: {key}")
        layer_idx = int(key.split("model.layers.")[1].split(".")[0])
        param_name = key.split(".self_attn.")[1]
        target = layers[layer_idx].self_attn.delta_mem
        if not hasattr(target, param_name):
            raise ValueError(f"DeltaMemLayer has no attribute {param_name} (key={key})")
        # Assign with shape check
        existing = getattr(target, param_name)
        if existing.shape != tensor.shape:
            raise ValueError(
                f"Shape mismatch for {key}: existing {existing.shape} vs loaded {tensor.shape}"
            )
        setattr(target, param_name, tensor.astype(existing.dtype))
        n_loaded += 1
    return n_loaded


def reset_delta_mem_states(model: nn.Module) -> None:
    """Zero the per-layer δ-mem state. Call between independent eval problems."""
    for block in model.model.layers:
        if isinstance(block.self_attn, DeltaMemAttention):
            block.self_attn.reset_state()
