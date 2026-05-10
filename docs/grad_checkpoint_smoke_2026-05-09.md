# Activation checkpointing — smoke validation

**Date:** 2026-05-09
**Status:** Complete within scope of M2 / current MPS + `torch.compile`
stack. d6 correctness ✅; d12/batch=4 memory benefit ✅ (−12.8 % wall,
−21 % page faults, bit-identical loss); d12/batch=8 cap not moved
(both arms OOM at the ~30 GiB MPS watermark, though the patch defers
the failure from forward to backward).
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

**Arm B (on): also OOM, but later in the iteration.**

```
RuntimeError: MPS backend out of memory
(MPS allocated: 30.10 GiB, other allocations: 2.81 MiB,
 max allowed: 30.19 GiB).
Tried to allocate 192.00 MiB on private pool.
```

Crash phase shifted from forward (arm A) to **backward** (arm B), and
the failed allocation shrunk from 2 GB to 192 MB. The patch reduced
forward-peak enough to clear the lm_head, but backward still hit the
~30 GiB cap. **At batch=8 the patch helps but not enough — the feasible
batch boundary did not move.** Possible contributors: `torch.compile`'s
inductor buffer scheduler retaining intermediates across the backward
graph, lm_head dominating the memory budget (outside `Block.forward`'s
checkpointing reach), or MPS allocator pool retention. None of these
are falsified here; a follow-up at `torch.compile` disabled would
isolate the first.

## d12 follow-up at batch=4 (memory-pressured but feasible)

Same as d12/batch=8 above, but `--device-batch-size=4
--total-batch-size=8192`. Both arms ran to completion.

|                          | Arm A (off)            | Arm B (on)             | Δ                  |
| ---                      | ---                    | ---                    | ---                |
| Loss at step 7           | 10.375209              | 10.375209              | **bit-identical**  |
| Wall clock (8 iters)     | 227.1 s                | 198.0 s                | **−12.8 %**        |
| Step time range          | 9–55 s (5.6× spread)   | 14–33 s (2.4× spread)  | tighter            |
| Page faults              | 15,478                 | 12,170                 | **−21 %**          |
| Page reclaims            | 293,976                | 247,904                | **−16 %**          |
| Peak RSS (`time -l`)     | 689 MB                 | 802 MB                 | +113 MB            |

Reads:

- **Page faults and reclaims drop materially** with the patch on —
  direct evidence that activation memory pressure is being relieved on
  this stack (MPS + `torch.compile`). Step-time variance compresses
  alongside.
- **Wall-clock effect flips sign.** At d6/batch=4 (not memory-bound)
  the patch costs +19 %. At d12/batch=4 (memory-pressured) it gains
  12.8 %. The crossover happens where OS-level paging cost exceeds
  the recompute tax.
- **RSS grew (+113 MB) even though pressure went down.** The wrap's
  Python-side overhead (closures, recompute hooks) shows up in RSS;
  the actual savings are in the MPS unified-memory pool, which RSS
  doesn't fully reflect. Same +22 MB pattern at d6.

## Interpretation

The patch is **correct, opt-in, and useful in the regime it was
designed for**:

- **Correctness.** Bit-identical loss at d6 and d12. Composes with
  bf16 + `torch.compile` without errors.
- **Cost when not memory-bound.** +19 % step time at d6/batch=4.
  Default off, so this applies only when explicitly enabled.
- **Benefit when memory-bound.** At d12/batch=4 the patch cuts wall
  clock by 12.8 %, page faults by 21 %, page reclaims by 16 %.
- **Limit on this hardware/stack.** At d12/batch=8 the patch helps
  (defers OOM from forward to backward, shrinks the failing allocation
  10×) but does not move the feasible-batch boundary. The ~30 GiB MPS
  watermark still holds for both arms.

The audit doc framed the patch as "an *enabler*, not a wall-clock
perf win." The d12/batch=4 result shows it can be *both* in the
memory-pressured regime: smoother runs AND a wall-clock win because
paging cost was dominant. The audit doc was conservative.

What this validation does **not** establish: whether the patch can
push past the d12/batch=8 OOM cap on this stack. Three hypotheses
(`torch.compile` interference, lm_head dominance, MPS pool retention)
are not falsified here. A follow-up at `torch.compile` disabled would
isolate (1) cheaply but isn't on the critical path now.

## References

- `docs/mlx_lm_pattern_audit_2026-05-08.md` — origin of the pattern,
  validation gates, falsification criteria.
- `nanochat/gpt.py` — `enable_block_grad_checkpoint()`, after the Block
  class.
- `scripts/base_train.py` — `--grad-checkpoint` flag and call site.
- Logs (ephemeral; numbers above are captured here):
  - d6: `/tmp/d6_gc_off.log`, `/tmp/d6_gc_on.log`
  - d12/batch=8: `/tmp/d12_gc_off.log`, `/tmp/d12_gc_on.log`
  - d12/batch=4: `/tmp/d12_b4_off.log`, `/tmp/d12_b4_on.log`
