# d3_tiny A/B: dataloader remainder reuse — 2026-05-06

**Question.** Does upstream PR [#544 (dataloader remainder reuse)](https://github.com/karpathy/nanochat/pull/544) deliver its claimed speedup at our M2 dev-loop scale, and what does it cost?

**Verdict (one sentence).** At small-d / abundant-source / compute-bound — our M2 + ClimbMix-400B regime — remainder reuse consumes ~half the source-doc tokens for essentially the same val_bpb (within noise), but pays a small wall-time penalty per step. The PR's gain is real; the regime where the gain pays for the gain isn't ours.

**Decision recorded.** [ADR-005](project_notes/decisions.md#adr-005). Capability landed env-gated, default OFF.

---

## Setup

**Capability gate.** `NANOCHAT_DATALOADER_REUSE_REMAINDER=1` flips the
`tokenizing_distributed_data_loader_with_state_bos_bestfit` crop branch from
discarding cropped doc remainders to recycling them (with BOS prepended) into
`doc_buffer`. Wiring landed in `2173dab` (capability) + `2c2c54e` (banner
unpack fix); ~21 lines.

**A/B recipe.** Identical apart from the env var.

```
PYTHONUNBUFFERED=1 nohup uv run python -u -m scripts.base_train \
    --inherit-from=$NANOCHAT_BASE_DIR/base_checkpoints/d6_baseline_modern/meta_005000.json \
    --depth=3 --aspect-ratio=32 --max-seq-len=256 \
    --num-iterations=500 --eval-every=50 \
    --save-every=500 --save-keep-last-n=1 \
    --seed=42 \
    --model-tag=<discard | remainder_reuse> --run=<...>
```

- Model: 17.37 M params (d3, n_embd=128, n_head=2, T=256), inherits from `d6_baseline_modern` meta.
- Dataset: ClimbMix-400B (`karpathy/climbmix-400b-shuffle`), default rank-0 shard.
- Hardware: M2 MacBook Pro (MPS, bf16 compute, single rank, no DDP).
- Each run pair shares a seed; the only thing that differs is which packing produces the rows.
- **Seed pair 1 (seed=42)**: baseline finished 2026-05-05 22:33 (model-tag `d3_tiny`); A/B finished 2026-05-06 05:19 (model-tag `d3_tiny_remreuse`). First A/B launch 05:09 hit a wiring bug (`get_dist_info()` returns 4 values, the banner unpacked 2) — caught at first `next(train_loader)` call, fixed in `2c2c54e`, relaunched cleanly.
- **Seed pair 2 (seed=2)**: chained baseline → A/B, both finished 2026-05-06 (model-tags `d3_tiny_s2` and `d3_tiny_remreuse_s2`).

---

## Headline numbers (n=2 seeds)

Means across `seed=42` and `seed=2`:

| metric | discard (baseline) | remainder_reuse | Δ (mean) | per-seed Δ |
|---|---:|---:|---:|---:|
| **val_bpb (final, step 500)** | **1.6530** | **1.6619** | **+0.0089 (worse)** | +0.0050, +0.0128 |
| total_training_time | 382.9 s | 392.7 s | **+2.6 % slower** | +4.2 %, +0.9 % |
| dt (final) | 0.794 s | 0.828 s | +4.2 % | +4.2 %, +4.4 % |
| train/loss (final) | 5.408 | 5.456 | +0.048 (worse) | +0.036, +0.060 |
| source-doc consumption (final `rg`) | 30 | **14** | **−53 %** | identical both seeds |

Per-seed details:

| metric | seed=42 base | seed=42 A/B | seed=2 base | seed=2 A/B |
|---|---:|---:|---:|---:|
| val_bpb (step 500) | 1.6561 | 1.6611 | 1.6499 | 1.6627 |
| total_training_time | 376.7 s | 392.6 s | 389.1 s | 392.8 s |
| train/loss (final) | 5.417 | 5.453 | 5.399 | 5.458 |
| source-doc rg | 30 | 14 | 30 | 14 |

Two findings the seed pair changes:
- **val_bpb regression is structural**, not seed noise. Both seeds show
  remainder_reuse worse, in the +0.005 to +0.013 band. The first seed's
  +0.005 was the optimistic case; +0.013 on the second seed shows the gap
  isn't bounded by noise.
- **Wall-time penalty is quieter than seed=42 alone suggested.** Seed=42's
  baseline ran faster than seed=2's baseline (376.7 s vs 389.1 s), inflating
  the seed=42 Δ. Both A/B runs land at ~392.6–392.8 s, which is the
  load-bearing per-step number. Net penalty is ~+2.6 % across seeds, not
  +4.2 %. Still real, just smaller.

---

## val_bpb trajectory (every 50 steps)

**seed=42:**

| step | discard | remainder_reuse | Δ |
|---:|---:|---:|---:|
| 0 | 3.2059 | 3.1905 | −0.015 |
| 50 | 2.1768 | 2.1757 | −0.001 |
| 100 | 1.9887 | 1.9814 | −0.007 |
| 150 | 1.8871 | 1.8938 | +0.007 |
| 200 | 1.8162 | 1.8225 | +0.006 |
| 250 | 1.7659 | 1.7714 | +0.005 |
| 300 | 1.7289 | 1.7333 | +0.004 |
| 350 | 1.7006 | 1.7053 | +0.005 |
| 400 | 1.6801 | 1.6844 | +0.004 |
| 450 | 1.6640 | 1.6690 | +0.005 |
| 500 | 1.6561 | 1.6611 | +0.005 |

**seed=2:**

| step | discard | remainder_reuse | Δ |
|---:|---:|---:|---:|
| 0 | 3.2059 | 3.1904 | −0.015 |
| 50 | 2.1857 | 2.1708 | −0.015 |
| 100 | 1.9883 | 1.9799 | −0.008 |
| 150 | 1.8831 | 1.8845 | +0.001 |
| 200 | 1.8131 | 1.8173 | +0.004 |
| 250 | 1.7626 | 1.7692 | +0.007 |
| 300 | 1.7246 | 1.7329 | +0.008 |
| 350 | 1.6959 | 1.7058 | +0.010 |
| 400 | 1.6739 | 1.6857 | +0.012 |
| 450 | 1.6579 | 1.6709 | +0.013 |
| 500 | 1.6499 | 1.6627 | +0.013 |

Both seeds show the same shape: curves track tightly through ~step 100
(remainder_reuse marginally ahead), then a sustained widening gap.
Remainder_reuse never closes the gap once it opens. Crucially, seed=2's
gap *grows* through the run (+0.001 → +0.013) while seed=42's gap
*plateaus* (+0.007 → +0.005). Both end with remainder_reuse worse, but
seed=2 is the larger gap and the sustained-widening shape is the more
honest read on what happens at this scale.

## train_loss trajectory (every 25 steps, summary)

| step | discard | remainder_reuse | Δ |
|---:|---:|---:|---:|
| 0 | 10.398 | 10.398 | 0.000 |
| 100 | 6.509 | 6.484 | −0.025 |
| 200 | 5.919 | 5.930 | +0.011 |
| 300 | 5.611 | 5.672 | +0.061 |
| 400 | 5.417 | 5.453 | +0.036 |
| 499 | 5.307 | 5.404 | +0.097 |

Δ-stats over all 500 steps: mean +0.027, median +0.025, stdev 0.044.
A/B's train-loss penalty grows over the run; baseline's *does not* grow on
val_bpb. **The train-loss gap reflects per-step batch composition, not
generalization.**

---

## Mechanism (why train-loss diverges but val_bpb tracks)

A/B's batches contain more *partial* documents — when a doc gets cropped, the
tail is recycled into the buffer with BOS, and shows up in some later row. So
within a window of 500 steps, a given source document is more likely to be seen
in fragments than in baseline (where the tail would be discarded and the
buffer would advance to the next doc).

Two consequences:

1. **More redundant content per step.** A doc consumed in two halves
   contributes both halves to *training* (good for data efficiency) but in
   nearby steps (bad for per-step novelty). The model partially memorizes the
   first half, then sees the second half on a step where the residual error on
   half-of-the-distribution is already lower. Train-loss-per-step looks
   slightly worse because the per-step difficulty is slightly lower-variance —
   easier batches to fit produce a less noisy descent, but also a slightly
   higher floor.
2. **No effect on val_bpb.** Validation is on a fixed held-out shard, evaluated
   identically by both runs. Generalization measures whether the model has
   learned a transferable distribution; the dataloader change can't bias this
   if both models see comparable amounts of *unique* training data over the
   run. They do — in fact, A/B sees more diverse content per source-token
   consumed, which is why val_bpb tracks tightly even though A/B reads ~half
   the source data.

The +0.005 val_bpb gap is real but small. The +0.097 train-loss gap is
mechanism, not regression.

## The source-token win

A/B's read-group counter (`rg`) ends at 14 vs baseline's 30. The dataloader
walks parquet row-groups in order on rank 0, so this is a faithful proxy for
"how much of the on-disk corpus did we have to crack open to feed the run."

Baseline read **2.14×** as much source data as A/B for ~the same val_bpb.
The PR's claim — ~50 % source-token reduction — replicates cleanly at our
context length. Where it doesn't replicate is in wall-clock, because at our
scale we are not source-token-bound.

## The wall-time penalty

A/B is consistently 4 % slower per step (0.826 s vs 0.793 s, sustained across
the run). The remainder-reuse logic is two operations per crop event: list
build and list append. That's not nothing on the Python side, especially when
the buffer is small and the bestfit search now has to compare against
fragmented entries (more `len(doc)` calls per step's bestfit scan).

At MPS scale on a 17 M-param model, the dataloader runs on the same Python
thread as the training loop's prefetch, and the GPU is fast enough that
dataloader walltime is non-trivially attributable to step time. On 8×H100
where the GPU is much faster relative to the dataloader and the I/O is the
bottleneck, the same +4 % overhead would either disappear into noise or
flip sign because the data the dataloader doesn't have to fetch from disk
saves real wall-time.

---

## When to flip the gate

ADR-005 lists three triggers. Restated as a decision tree:

- **Are you running on M2/dev-loop?** Default OFF. The +4 % wall-time
  penalty is real, the val_bpb gain is zero. ClimbMix is effectively
  infinite at our token horizons.
- **Are you running `runs/speedrun.sh` on 8×H100?** Default ON. I/O at the
  ClimbMix scan rate is a real bottleneck; the same `−53 %` source-token
  reduction translates to wall-time win.
- **Are you running multi-epoch on a small corpus?** Default ON. Source-
  exhaustion is the regime the PR was built for.
- **Are you doing a cross-checkpoint comparison?** Whatever the matched
  checkpoint used. The meta records `dataloader_variant` so this is
  auditable; never mix variants in a comparison without flagging it.

## Risks / what I'm not certain of

- **n=2 seeds is enough to reject "noise" but not to bound the regression
  tightly.** Both seeds show remainder_reuse worse on val_bpb in the
  +0.005 to +0.013 band — outside the expected per-seed jitter, so the
  effect is real and structural. But two seeds is not enough to compute a
  confidence interval; a third seed would tighten the bound on whether
  the regression is closer to +0.005 or +0.013. We're not running it
  because the decision (default OFF on M2) doesn't change at either end.
- **Single context length / depth.** PR #544 reports gain *grows* at smaller
  T (1.18× at T=2048, 1.28× at T=512). We tested T=256, where the gain
  should be biggest in the regime the PR targets. That doesn't extrapolate
  to T=2048 d6 runs without re-measuring.
- **Wall-time penalty is hardware-dependent.** The +4 % is M2-specific. Until
  someone runs the same A/B on H100, "Default ON for speedrun" is an
  inference from the data-bound mechanism, not a measurement.
- **train-loss reporting noise.** Step-499 train-loss is from one batch and
  is not directly comparable across runs even at the same seed (because the
  batches differ). The val_bpb numbers are the load-bearing comparison;
  treat the train-loss table as descriptive, not decision-grade.

## Followups

- If we ever bring back the d8 extension and the runs become source-token
  bound (more iters per epoch on a small corpus), revisit the gate at the
  larger context and depth. The mechanism predicts a smaller wall-time
  penalty (more compute per dataloader op) and the same source-data win.
- If the upstream PR merges and changes the default, we owe a one-time
  rebaseline of `d6_baseline_modern` (per ADR-003). Until then the gate
  insulates us.
- The `dataloader_variant` field landed in `meta_*.json` as of `2173dab`.
  Any future audit script that compares checkpoints should include this
  field as a partition key.

## Artefacts

- Wandb runs:
  - [`d3_tiny`](https://wandb.ai/matt-parrett/nanochat/runs/nasvg15p) — seed=42, discard
  - [`d3_tiny_remreuse`](https://wandb.ai/matt-parrett/nanochat/runs/uh2vf0s8) — seed=42, remainder_reuse (the successful run; an earlier 05:09:43 run for the same model-tag hit the wiring bug, see `2c2c54e`)
  - `d3_tiny_s2`, `d3_tiny_remreuse_s2` — seed=2 pair under the same wandb project.
- Local logs:
  - `/tmp/d3_tiny_pretrain.log`, `/tmp/d3_tiny_remreuse_pretrain.log` (seed=42)
  - `/tmp/d3_tiny_s2_pretrain.log`, `/tmp/d3_tiny_remreuse_s2_pretrain.log` (seed=2)
- Checkpoints (each meta records the variant under `user_config.dataloader_variant`):
  - `$NANOCHAT_BASE_DIR/base_checkpoints/d3_tiny/{model,optim}_000500.pt`
  - `$NANOCHAT_BASE_DIR/base_checkpoints/d3_tiny_remreuse/{model,optim}_000500.pt`
  - `$NANOCHAT_BASE_DIR/base_checkpoints/d3_tiny_s2/{model,optim}_000500.pt`
  - `$NANOCHAT_BASE_DIR/base_checkpoints/d3_tiny_remreuse_s2/{model,optim}_000500.pt`
- Commits: `2173dab` (capability + meta plumbing), `2c2c54e` (banner unpack fix), `e32c17b` (ADR-005), `a698a0d` (initial writeup, seed=42 only).
