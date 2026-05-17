"""
Synthetic numerical-equivalence check: nanochat.delta_mem_mlx vs upstream torch math.

Reproduces the upstream _memory_affine_scan_torch + _memory_sequence_projections logic
in pure torch on randomly-initialized weights, runs the same inputs through the MLX
port, and asserts the reads, final state, δ_q, and δ_o match within numerical tolerance.

This is the Phase 1 gate-keeper: if this fails, do not proceed to integration.
The reference math is taken verbatim from /tmp/delta-Mem/deltamem/core/delta_impl.py
(see docs/delta_mem_recon_2026-05-17.md for the line-number map).

Usage:
    uv run --extra cpu --extra mlx python dev/delta_mem_mlx_smoke.py
"""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from nanochat.delta_mem_mlx import DeltaMemConfig, DeltaMemLayer


# ---------- Torch reference (inlined from upstream delta_impl.py) ----------


def torch_project_memory(x, mq_w, mk_w, mv_w, beta_w, beta_b):
    """Mirrors _memory_sequence_projections for couple_lambda=True, normalize_qk=True."""
    memory_q = F.linear(x, mq_w)
    memory_k = F.linear(x, mk_w)
    memory_v = F.linear(x, mv_w)
    memory_q = F.normalize(torch.tanh(memory_q), dim=-1, eps=1e-6)
    memory_k = F.normalize(torch.tanh(memory_k), dim=-1, eps=1e-6)
    beta = torch.sigmoid(F.linear(x, beta_w) + beta_b)
    return memory_q, memory_k, memory_v, beta


def torch_scan(state, mq, mk, mv, beta):
    """Mirrors _memory_affine_scan_torch (couple_lambda=True, standard mode, no token mask)."""
    B, T, R = mq.shape
    S = state
    reads = []
    for t in range(T):
        q_t = mq[:, t, :]
        k_t = mk[:, t, :]
        v_t = mv[:, t, :]
        beta_t = beta[:, t, :]
        lam_t = 1.0 - beta_t
        keep_t = lam_t.unsqueeze(-1)
        erase_t = beta_t.unsqueeze(-1)
        write_t = beta_t.unsqueeze(-1)
        read_t = torch.einsum("bij,bj->bi", S, q_t)
        pred_t = torch.einsum("bij,bj->bi", S, k_t)
        write_outer = v_t.unsqueeze(-1) * k_t.unsqueeze(1)
        pred_outer = pred_t.unsqueeze(-1) * k_t.unsqueeze(1)
        S = keep_t * S - erase_t * pred_outer + write_t * write_outer
        reads.append(read_t)
    return S, torch.stack(reads, dim=1)


def torch_project_deltas(reads, dq_w, do_w, scale):
    return F.linear(reads, dq_w) * scale, F.linear(reads, do_w) * scale


# ---------- Synthetic-data fixture ----------


def make_fixture(rank=8, hidden=2560, q_out=4096, o_out=2560, B=2, T=16, seed=0, dtype=torch.float32):
    rng = np.random.RandomState(seed)
    def t(*shape, scale=0.02):
        return torch.tensor(rng.randn(*shape).astype(np.float32) * scale, dtype=dtype)

    weights = dict(
        memory_q_proj=t(rank, hidden),
        memory_k_proj=t(rank, hidden),
        memory_v_proj=t(rank, hidden),
        delta_q_proj=t(q_out, rank, scale=0.01),
        delta_o_proj=t(o_out, rank, scale=0.01),
        beta_proj=t(rank, hidden, scale=0.01),
        beta_bias=torch.full((rank,), -1.5, dtype=dtype),
    )
    x = t(B, T, hidden, scale=1.0)
    state0 = torch.zeros(B, rank, rank, dtype=dtype)
    return weights, x, state0


def torch_to_mx(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().cpu().numpy())


# ---------- The check ----------


def run_check(dtype_label: str, torch_dtype, max_abs_tol: float):
    print(f"\n=== {dtype_label} (max-abs tol {max_abs_tol:g}) ===")
    weights, x, state0 = make_fixture(dtype=torch_dtype)

    # Torch path
    mq_t, mk_t, mv_t, beta_t = torch_project_memory(
        x, weights["memory_q_proj"], weights["memory_k_proj"],
        weights["memory_v_proj"], weights["beta_proj"], weights["beta_bias"],
    )
    state_t, reads_t = torch_scan(state0, mq_t, mk_t, mv_t, beta_t)
    scale = 16.0 / 8
    dq_t, do_t = torch_project_deltas(
        reads_t, weights["delta_q_proj"], weights["delta_o_proj"], scale
    )

    # MLX path
    cfg = DeltaMemConfig()
    layer = DeltaMemLayer(
        hidden_size=x.shape[-1],
        q_out_features=weights["delta_q_proj"].shape[0],
        o_out_features=weights["delta_o_proj"].shape[0],
        config=cfg,
    )
    layer.memory_q_proj = torch_to_mx(weights["memory_q_proj"])
    layer.memory_k_proj = torch_to_mx(weights["memory_k_proj"])
    layer.memory_v_proj = torch_to_mx(weights["memory_v_proj"])
    layer.delta_q_proj = torch_to_mx(weights["delta_q_proj"])
    layer.delta_o_proj = torch_to_mx(weights["delta_o_proj"])
    layer.beta_proj = torch_to_mx(weights["beta_proj"])
    layer.beta_bias = torch_to_mx(weights["beta_bias"])
    x_mx = torch_to_mx(x)
    state0_mx = torch_to_mx(state0)

    mq_m, mk_m, mv_m, beta_m = layer.project_memory(x_mx)
    state_m, reads_m = layer.scan(state0_mx, mq_m, mk_m, mv_m, beta_m)
    dq_m, do_m = layer.project_deltas(reads_m)

    def diff(name, a_t, a_m):
        ref = a_t.detach().cpu().numpy()
        port = np.array(a_m)
        if ref.shape != port.shape:
            print(f"  {name}: SHAPE MISMATCH ref={ref.shape} port={port.shape}")
            return False
        d = np.abs(ref - port).max()
        ok = d <= max_abs_tol
        print(f"  {name:<14} max|Δ|={d:.4g}   {'OK' if ok else 'FAIL'}")
        return ok

    ok = True
    ok &= diff("memory_q",  mq_t,    mq_m)
    ok &= diff("memory_k",  mk_t,    mk_m)
    ok &= diff("memory_v",  mv_t,    mv_m)
    ok &= diff("beta",      beta_t,  beta_m)
    ok &= diff("reads",     reads_t, reads_m)
    ok &= diff("state_T",   state_t, state_m)
    ok &= diff("delta_q",   dq_t,    dq_m)
    ok &= diff("delta_o",   do_t,    do_m)
    return ok


def main() -> int:
    ok_fp32 = run_check("fp32", torch.float32, max_abs_tol=1e-5)
    # bf16 round-trip via numpy isn't exact; cast both sides to fp32 internally
    # by passing fp32 weights but checking with looser tolerance to give a sanity bound.
    # (Actual bf16 mass-action equivalence is checked at logit-match time.)
    if not ok_fp32:
        print("\nFP32 check FAILED — port has a bug. Do not proceed.")
        return 1
    print("\nFP32 check passed. MLX scan + projections match torch reference to 1e-5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
