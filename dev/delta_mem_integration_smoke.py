"""
MLX integration smoke for δ-mem on mlx-lm Qwen3-4B-Instruct-2507-8bit.

Two checks, ordered by trust level:

1. Zero-weight sanity: attach DeltaMemAttention with default (zero) weights, no
   adapter loaded. Output should equal the base model output bit-for-bit because
   δ_q = reads @ 0 = 0 and δ_o = reads @ 0 = 0 regardless of state evolution.
   This isolates wiring bugs from math bugs.

2. Real-adapter forward: load the converted MLX adapter from
   ~/.cache/nanochat/delta_mem_qwen3_4b_instruct/. Output must differ from base
   by a small but non-trivial amount (δ-projection weights init'd from base
   slices × online_gain=0.05, then trained).

Memory cost: ~4 GB for Qwen3-4B-Instruct-2507-8bit.

Usage:
    PYTHONPATH=. uv run --extra cpu --extra mlx python dev/delta_mem_integration_smoke.py
"""

from __future__ import annotations

import sys

import mlx.core as mx
from mlx_lm import load

from nanochat.delta_mem_mlx import (
    DeltaMemAttention,
    DeltaMemConfig,
    attach_delta_mem,
    load_delta_mem_adapter,
    reset_delta_mem_states,
)


MODEL_REPO = "mlx-community/Qwen3-4B-Instruct-2507-8bit"
ADAPTER_PATH = "/Users/matt/.cache/nanochat/delta_mem_qwen3_4b_instruct/adapter.safetensors"


def main() -> int:
    print(f"Loading {MODEL_REPO} ...")
    model, tokenizer = load(MODEL_REPO)
    print(f"  loaded. layers={len(model.model.layers)}")

    # Tiny synthetic input — avoid any tokenizer roundtrip surprises.
    # Use real token ids in vocab range.
    prompt = "The quick brown fox jumps over the"
    ids = tokenizer.encode(prompt)
    print(f"  prompt ids: {ids}")
    x = mx.array([ids])  # (1, T)

    # --- Baseline forward ---
    print("\nBaseline forward (no δ-mem)...")
    base_logits = model(x)
    mx.eval(base_logits)
    print(f"  base_logits shape: {base_logits.shape}, dtype: {base_logits.dtype}")
    print(f"  last-token argmax: {int(mx.argmax(base_logits[0, -1, :]).item())}")

    # --- Check 1: zero-weight sanity ---
    print("\nCheck 1: attach with default (zero) δ-mem weights, expect bit-equal output...")
    cfg = DeltaMemConfig()
    wrapped = attach_delta_mem(model, cfg)
    print(f"  wrapped {len(wrapped)} attention modules")
    assert all(isinstance(l.self_attn, DeltaMemAttention) for l in model.model.layers), \
        "not all layers got wrapped"
    reset_delta_mem_states(model)
    zero_logits = model(x)
    mx.eval(zero_logits)
    max_diff = float(mx.abs(zero_logits - base_logits).max().item())
    print(f"  max|Δ| logits vs base: {max_diff:.4g}")
    if max_diff > 1e-5:
        print("  FAIL: zero-weight wrapped output should equal base exactly")
        return 1
    print("  OK")

    # --- Check 2: real-adapter forward ---
    print("\nCheck 2: load real adapter, expect non-trivial divergence from base...")
    n_loaded = load_delta_mem_adapter(model, ADAPTER_PATH)
    print(f"  loaded {n_loaded} tensors")
    expected = 7 * len(model.model.layers)
    if n_loaded != expected:
        print(f"  WARN: expected {expected} tensors, got {n_loaded}")
    reset_delta_mem_states(model)
    adapter_logits = model(x)
    mx.eval(adapter_logits)
    max_diff = float(mx.abs(adapter_logits - base_logits).max().item())
    mean_diff = float(mx.abs(adapter_logits - base_logits).mean().item())
    base_max = float(mx.abs(base_logits).max().item())
    print(f"  max|Δ| logits vs base: {max_diff:.4g}  (base max|x|={base_max:.4g})")
    print(f"  mean|Δ|              : {mean_diff:.4g}")
    print(f"  last-token argmax: {int(mx.argmax(adapter_logits[0, -1, :]).item())}")
    if max_diff < 1e-4:
        print("  FAIL: adapter forward indistinguishable from base — adapter not applied")
        return 1
    if max_diff > base_max * 0.5:
        print("  WARN: divergence > 50% of base logit scale — adapter may be misapplied")
    print("  OK — adapter is producing non-trivial corrections")

    # --- Check 3: state actually evolved ---
    print("\nCheck 3: per-layer δ-mem state should be non-zero after forward...")
    nonzero_states = 0
    for i, block in enumerate(model.model.layers):
        s = block.self_attn.delta_state
        if s is not None:
            sm = float(mx.abs(s).max().item())
            if sm > 1e-6:
                nonzero_states += 1
        else:
            print(f"  layer {i}: state is None")
    print(f"  {nonzero_states}/{len(model.model.layers)} layers have non-zero state")
    if nonzero_states < len(model.model.layers):
        print("  FAIL: some layers didn't update state — scan may not be running")
        return 1
    print("  OK")

    print("\nAll integration smoke checks PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
