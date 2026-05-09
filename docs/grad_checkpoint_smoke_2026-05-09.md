# Activation checkpointing — smoke validation

**Date:** 2026-05-09
**Status:** Partial. d6 correctness ✅; d12 memory benefit half-confirmed
(arm A OOM observed; arm B run pending GPU availability after a competing
trx4mr workload finishes).
**Companion:** `docs/mlx_lm_pattern_audit_2026-05-08.md` idea #3.

## Implementation

Two-file change, opt-in via flag, default off:

- `nanochat/gpt.py` — `enable_block_grad_checkpoint()` monkey-patches
  `Block.forward` to wrap the call in
  `torch.utils.checkpoint.checkpoint(use_reentrant=False)`. Idempotent.
- `scripts/base_train.py` — `--grad-checkpoint` flag. The patch is
  applied *before* `torch.compile(model)` so the compile graph captures
  the wrapped forward.

## d6 correctness A/B (M2, 2026-05-09)

Config: `--depth=6 --device-batch-size=4 --total-batch-size=8192
--num-iterations=15` (single grad-accum step).

|                        | Arm A (off)  | Arm B (on)   | Δ                |
| ---                    | ---          | ---          | ---              |
| Steady-state step time | ~1.30 s      | ~1.55 s      | **+19 %**        |
| Wall clock (15 iters)  | 35.2 s       | 66.3 s       | +31 s            |
| Peak RSS (`time -l`)   | 821.9 MB     | 843.1 MB     | +22 MB           |
| Loss at step 14        | 10.301095    | 10.301095    | **bit-identical**|

Reads:

- **Bit-identical loss progression** at every step. The patch is
  mathematically equivalent (gradient checkpointing recomputes
  activations on backward, doesn't change them). Confirms the wrap is
  correct and composes with bf16 + `torch.compile`.
- **+19 % step-time tax** — better than the +30 % the audit doc
  estimated. This is the cost arm.
- **Wall-clock gap (+31 s) > per-step gap (+3.75 s × 15)** because
  arm B's compile graph is more complex; the first iteration pays a
  one-time `torch._inductor` cost. On a real 5000-step run, the
  per-step tax dominates and the compile overhead amortizes to noise.
- **No memory benefit at this config (RSS +22 MB).** Expected:
  d6/batch=4 is nowhere near memory-bound. Activation memory is a
  rounding error vs static (weights + Adam state + PyTorch runtime).
  This config can only validate *safety*, not *value*.

## d12 arm A (memory-bound probe, 2026-05-09)

Config: `--depth=12 --device-batch-size=8 --total-batch-size=16384
--num-iterations=8`.

**Arm A (off): clean MPS OOM at step 0 forward.**

```
RuntimeError: MPS backend out of memory
(MPS allocated: 29.73 GiB, other allocations: 832.00 KiB,
 max allowed: 30.19 GiB).
Tried to allocate 2.00 GiB on private pool.
```

The 2 GB allocation that pushed MPS over the watermark is the lm_head
logits tensor, shape `(8, 2048, 32768)` fp32 — 8 batch × 2048 seq ×
32 768 padded vocab × 4 bytes = 2 GB exactly. lm_head is *outside* the
Block, so gradient checkpointing on `Block.forward` doesn't directly
shrink that tensor. The mechanism by which arm B should still help is
indirect: dropping the per-block stored activations (held to feed the
backward pass) frees enough headroom for the lm_head allocation to
succeed at the moment it's needed.

This OOM is **the regime the patch is meant to address**. Confirms d12
on M2 is memory-bound at modest batch sizes — not a synthetic stress
test, just normal scale.

**Arm B (on, pending):** queued, waiting for an unrelated trx4mr
workload to release the Metal context. The positive demonstration —
arm B fits where arm A didn't, same config — completes the validation.

## Interpretation so far

The patch is **safe to merge on the d6 evidence alone**:

- It's opt-in (default off), so existing flows are unchanged.
- It's correct (bit-identical loss).
- The cost (+19 % step time) is in the documented range.

The d12 result so far gives us half the *value* story: arm A reproduces
the memory wall the patch was built to push back. The pending arm B
either fits at the same config (clean win) or also OOMs (the wrap
helps the wrong tensor and we'd revisit). Will update this doc when
the run completes.

## References

- `docs/mlx_lm_pattern_audit_2026-05-08.md` — origin of the pattern,
  validation gates, falsification criteria.
- `nanochat/gpt.py` — `enable_block_grad_checkpoint()`, after the Block
  class.
- `scripts/base_train.py` — `--grad-checkpoint` flag and call site.
- Logs: `/tmp/d6_gc_off.log`, `/tmp/d6_gc_on.log`, `/tmp/d12_gc_off.log`
  (ephemeral, may rotate; numbers above are captured here).
