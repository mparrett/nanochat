# Activation checkpointing — smoke validation

**Date:** 2026-05-09
**Status:** Complete. d6 correctness ✅; **d12/batch=4 and d12/batch=6
both show a ~12 % wall-clock win** in the memory-pressured regime
(−12.8 % and −12.3 %, bit-identical loss); d12/batch=8 with
`torch.compile` enabled does not fit (the patch defers OOM from
forward to backward but doesn't clear the cap); **d12/batch=8 with
`--no-compile` + `--grad-checkpoint` *does* fit** — torch.compile's
inductor was retaining backward intermediates and masking the patch's
savings (audit doc hypothesis #1, now confirmed). Capacity unlock,
not speed unlock at batch=8.
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

## d12 bisection at batch=6 (paging-dominated, both fit)

Same shape as the d12/batch=4 run but `--device-batch-size=6
--total-batch-size=12288`. This config sits in the *severely*
paging-dominated regime: arm A's sys time is 4.5× higher than at
batch=4. Both arms still fit (no OOM).

|                          | Arm A (off)            | Arm B (on)             | Δ                  |
| ---                      | ---                    | ---                    | ---                |
| Wall clock (8 iters)     | 1446.4 s               | 1268.8 s               | **−12.3 %**        |
| User CPU                 | 18.1 s                 | 28.1 s                 | **+55 %** (recompute tax) |
| Sys CPU                  | 332.9 s                | 282.0 s                | **−15.3 %** (paging dropped) |
| Page reclaims            | 334,675                | 302,084                | **−9.7 %**         |
| Page faults              | 18,787                 | 19,015                 | ~flat              |
| Peak RSS (`time -l`)     | 689 MB                 | 813 MB                 | +124 MB            |
| Loss at step 7           | 10.366605              | 10.366606              | bit-identical save 1 ULP |

Reads:

- **Wall-clock win replicates** (−12.3 % vs −12.8 % at batch=4).
  Consistent across two independent memory-pressured configs at d12.
- **The trade decomposes cleanly:** user CPU up (recompute), sys CPU
  down (paging), wall clock down. We can see what the patch is
  actually buying us — kernel time was the bottleneck, not arithmetic.
- **Faults vs reclaims diverge.** Reclaims drop ~10 %, faults are
  flat. In this regime reclaims are the more reliable pressure signal;
  faults are noisier.
- **Loss differs by 1 ULP at step 7** (steps 0–6 exactly equal).
  Standard fp32 reduction roundoff in the backward path of the
  recompute. Mathematically equivalent — not a correctness concern.

## d12/batch=8 with `--no-compile`: hypothesis (1) confirmed

The original d12/batch=8 A/B (compile on) showed both arms OOM. The
audit doc and earlier interpretation listed three non-falsified
hypotheses for why the patch couldn't move the cap:

1. `torch.compile`'s inductor buffer scheduler retaining intermediates
   across the backward graph.
2. `lm_head` (the 2 GB `(8, 2048, 32768)` fp32 logits tensor)
   dominating the budget — outside `Block.forward`'s checkpointing
   reach.
3. MPS allocator pool retention.

Re-ran d12/batch=8 with the new `--no-compile` flag (introduced in
`scripts/base_train.py` for exactly this kind of probe) to isolate (1):

| Compile | grad-checkpoint | Result                                       |
| ---     | ---             | ---                                          |
| ON      | OFF             | OOM forward (lm_head 2 GB on top of 29.73 GiB pooled) |
| ON      | ON              | OOM backward (192 MB on top of 30.10 GiB pooled) — defers but doesn't fit |
| OFF     | OFF             | OOM forward (lm_head 2 GB on top of 29.38 GiB pooled) — same failure mode |
| **OFF** | **ON**          | **fits — runs to completion**                  |

**Hypothesis (1) confirmed.** With `torch.compile` enabled, inductor's
buffer scheduler retains intermediates across the backward graph,
defeating `cp.checkpoint`'s discard semantics and masking the patch's
memory savings. With compile off, the patch frees enough activation
memory to clear the lm_head allocation and the run succeeds.

Note that **arm A no-compile still OOMs**, with the same lm_head 2 GB
allocation as the compile case. Earlier I read this as evidence
*against* hypothesis (1) — wrongly. Arm A doesn't have the patch, so
the lm_head allocation will fail regardless of compile state once
activation memory accumulates. The clean test of (1) is whether the
*patch* unmasks when compile is off, and the answer is yes.

**Caveat: capacity unlock, not speed unlock at batch=8.** Eager mode
pays per-op overhead, so the no-compile + grad-checkpoint path at
batch=8 is slower *per token* than compile + grad-checkpoint at
batch=4. The framing should be: `--no-compile --grad-checkpoint`
*enables* training at d12/batch=8 on M2 24 GB where it would
otherwise OOM, not that it's the fastest path.

**Step times stabilize.** A longer 8-iter run confirmed the run is
sustainable, not just a 4-iter delay before OOM:

| Step | dt (s) |
| ---  | ---    |
| 0    | 30     |
| 1    | 74     |
| 2    | 109    |
| 3    | 100    |
| 4    | 121    |
| 5    | 120    |
| 6    | 120    |
| 7    | 119    |

Steady state ≈ 120 s/step from step 4 onward; RSS held flat at
~575 MB, page reclaims grew linearly with wall time (no acceleration).
At 137 tok/s steady-state throughput vs ~331 tok/s for d12/batch=4 +
compile + grad-checkpoint, eager+gc at batch=8 is about **42 %** the
per-token throughput of the compile path at the smaller feasible
batch. That's the actual cost of trading compile for capacity.

**Pareto note:** d12/batch=6 (compile, gc) was 77 tok/s under heavy
paging — slower than d12/batch=8 (no-compile, gc) at 137 tok/s. So
the eager+gc batch=8 path is **strictly better than** the
compile+gc batch=6 path: more tokens per step, more tokens per second,
no paging cliff. The relevant comparison is against batch=4 + compile,
not batch=6 + compile.

## Interpretation

### Feasibility map (M2 24 GB)

Default rows are with `torch.compile` enabled (production path). The
last row shows the `--no-compile` probe at d12/batch=8.

| Depth | Batch | Compile | Arm A (off)             | Arm B (on)              | Patch effect                       |
| ---   | ---   | ---     | ---                     | ---                     | ---                                |
| 6     | 4     | ON      | runs (35.2 s)           | runs (66.3 s)           | **+19 %** wall (recompute tax)     |
| 12    | 4     | ON      | runs (227 s)            | runs (198 s)            | **−12.8 %** wall                   |
| 12    | 6     | ON      | runs (1446 s, paging)   | runs (1269 s, paging)   | **−12.3 %** wall                   |
| 12    | 8     | ON      | OOM, forward (lm_head)  | OOM, backward (smaller) | defers but doesn't fit             |
| 12    | 8     | **OFF** | OOM, forward (lm_head)  | **runs to completion**  | **capacity unlock**                |

### What the patch does

- **Correctness.** Bit-identical loss at d6 and d12/batch=4; 1-ULP
  fp32-roundoff difference at d12/batch=6 step 7. Composes with
  bf16 + `torch.compile` without errors.
- **Cost when not memory-bound.** +19 % step time at d6/batch=4. Pure
  recompute tax. Default off, so this applies only when explicitly
  enabled.
- **Benefit when memory-bound.** At d12/batch=4 and d12/batch=6 the
  patch buys back ~12 % wall clock. The savings come from reduced
  OS-level paging (sys CPU −15 %, reclaims −10 to −16 %), partially
  offset by the recompute tax (user CPU +55 % at batch=6 where it's
  most visible).
- **Sign-flip is real and reproducible.** Two independent
  memory-pressured configs both show the same ~12 % wall-clock win.
  This is no longer a single-data-point claim.

### Capacity behavior depends on `torch.compile`

- **With `torch.compile` ON**: the patch does *not* expand the
  feasible-batch boundary at d12. Bisection is conclusive: both arms
  fit at batch=6, both OOM at batch=8. The patch defers failure
  (forward → backward) and shrinks the failing allocation 10×, but
  the ~30 GiB MPS watermark still holds.
- **With `torch.compile` OFF**: the patch *does* unlock d12/batch=8
  on M2 24 GB — eager mode lets `cp.checkpoint`'s discard semantics
  actually free activation memory across the backward pass, where
  inductor's scheduler retains it.

The audit doc framed the patch as "an *enabler*, not a wall-clock
perf win." The d12 results show it's **both, but in different
configurations**: a wall-clock win in the memory-pressured regime
(batch=4, batch=6, compile on) AND a capacity enabler at the OOM cap
(batch=8, compile off). The audit framing was a single-mode picture;
the reality is two-mode, and which mode you get depends on whether
`torch.compile` is in the way.

### Resolved hypothesis, remaining unknowns

The d12/batch=8 `--no-compile` A/B above resolved hypothesis (1):
`torch.compile`'s inductor scheduler IS the reason the cap doesn't
move with compile on. The patch's memory savings are real but get
masked by inductor's buffer retention.

Hypotheses (2) and (3) — `lm_head` dominance and MPS pool retention —
remain non-falsified but are now secondary. They'd matter if we
wanted to push past d12/batch=8 to even larger batch sizes; for the
current question ("does the patch enable d12/batch=8 at all") (1) was
the only relevant blocker and it's been removed by the eager-mode
path.

The d10 sidebar below provides supporting evidence for (2): at
d10/batch=24 the lm_head allocation alone (6 GB at that batch) is
what kills the run, even in eager mode. The structural cost of the
lm_head logits is independent of `Block.forward`'s checkpointing
reach.

## Sidebar: d10 capacity sweep (M2, --no-compile + --grad-checkpoint)

Ran a small ascending-batch probe at d10 to map the feasibility
envelope for a smaller depth where the patch should have more
headroom:

| Batch | Tokens/step | Steady step time | Steady tok/s | Verdict          |
| ---   | ---         | ---              | ---          | ---              |
| 8     | 16,384      | ~50 s            | ~330         | usable           |
| 12    | 24,576      | ~190 s           | ~127         | usable, paging-heavy |
| 16    | 32,768      | ~1000 s          | ~33          | nominal fit, **operationally dead** |
| 24    | 49,152      | OOM at iter 0    | —            | hard cap (lm_head 6 GB) |

Two thresholds, not one:

- **Operational ceiling: batch=12.** Last config that's actually
  useful for training. At batch=16 the run nominally fits but
  throughput collapses to ~33 tok/s — paging cost dominates so
  thoroughly that a real pretrain would take weeks.
- **Hard ceiling: 16 < batch ≤ 24.** OOM at batch=24 is on a single
  6 GB lm_head allocation `(24, 2048, 32768)` fp32. lm_head scales
  linearly with batch and is outside `Block.forward`'s checkpointing
  reach. This is direct evidence for the audit doc's hypothesis (2)
  as a structural blocker independent of the gc patch.

The gap between the two ceilings is where this configuration is
nominally feasible but practically useless. Worth knowing: the
"capacity unlock" of `--no-compile --grad-checkpoint` extends the
*hard* fit boundary much further than it extends the *useful* one.

## Open question — for a later session

**Is a full d12 pretrain feasible overnight on M2?** Today's data
suggests **no** in the canonical-Chinchilla shape. Best per-step
config (compile + gc + batch=4) measured at ~25 s/step for a single
grad-accum step. A full Chinchilla pretrain (~5000 optimizer steps
with the standard total-batch ~524 K tokens, i.e. grad-accum ≈ 64
micro-batches per optimizer step) would be ~165 s/optimizer-step ×
5000 = ~230 hours. Not overnight by any margin.

A *truncated* d12 pretrain might fit:
- 8 hours / 165 s ≈ 175 optimizer steps with full Chinchilla batch.
- Or 8 hours / 25 s ≈ 1150 optimizer steps at single grad-accum
  (much smaller effective batch, sub-Chinchilla, gradient quality
  degraded).

Whether either is worth running depends on the goal — a smoke /
sanity-check d12 run is feasible overnight; a baseline-quality d12
pretrain is not. To analyze properly when we revisit:
1. Decide whether sub-Chinchilla d12 has scientific value.
2. Test compile + gc + batch=8 throughput at single grad-accum (we
   couldn't, that config OOMs without --no-compile).
3. Profile the 25 s/step number — what fraction is forward+backward
   vs optimizer? Optimizer-bound steps don't speed up linearly with
   smaller batch.

## References

- `docs/mlx_lm_pattern_audit_2026-05-08.md` — origin of the pattern,
  validation gates, falsification criteria.
- `nanochat/gpt.py` — `enable_block_grad_checkpoint()`, after the Block
  class.
- `scripts/base_train.py` — `--grad-checkpoint` flag and call site.
- Logs (ephemeral; numbers above are captured here):
  - d6: `/tmp/d6_gc_off.log`, `/tmp/d6_gc_on.log`
  - d12/batch=8: `/tmp/d12_gc_off.log`, `/tmp/d12_gc_on.log`
  - d12/batch=6: `/tmp/d12_b6_off.log`, `/tmp/d12_b6_on.log`
  - d12/batch=4: `/tmp/d12_b4_off.log`, `/tmp/d12_b4_on.log`
  - d12/batch=8 no-compile: `/tmp/d12_b8_nocompile_off.log`,
    `/tmp/d12_b8_nocompile_on.log`, `/tmp/d12_b8_nc_on_long.log`
  - d10 sweep: `/tmp/d10_b8_nc_gc.log`, `/tmp/d10_b12_nc_gc.log`,
    `/tmp/d10_b16_nc_gc.log`, `/tmp/d10_b24_nc_gc.log`
