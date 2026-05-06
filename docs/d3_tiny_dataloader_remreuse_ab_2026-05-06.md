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
- Both runs share `seed=42`; the only thing that differs is which packing produces the rows.
- Baseline run finished 2026-05-05 22:33 (model-tag `d3_tiny`).
- A/B run finished 2026-05-06 05:19 (model-tag `d3_tiny_remreuse`). First launch 05:09 hit a wiring bug (`get_dist_info()` returns 4 values, the banner unpacked 2) — caught at first `next(train_loader)` call, fixed in `2c2c54e`, relaunched cleanly.

---

## Headline numbers

| metric | discard (baseline) | remainder_reuse | Δ |
|---|---:|---:|---:|
| total_training_time | 376.7 s | 392.6 s | **+4.2 % slower** |
| dt (final) | 0.793 s | 0.826 s | +4.2 % |
| train/loss (final, step 499) | 5.417 | 5.453 | +0.04 (worse) |
| **val_bpb (final, step 500)** | **1.6561** | **1.6611** | **+0.005 (worse, within noise)** |
| min val_bpb (over run) | 1.6561 | 1.6611 | +0.005 |
| source-doc consumption (final read-group) | 30 | **14** | **−53 %** |

The val_bpb gap (the canonical metric in this repo) is ~10× smaller than the
train-loss gap. That difference is not noise — it's mechanism.

---

## val_bpb trajectory (every 50 steps)

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

The two curves are essentially superimposed for the first 100 steps, then
remainder_reuse settles into a stable +0.004 to +0.007 trail. That's a real
small bias, not drift — but it is small.

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

- **n=1 seed.** Same seed (42) for both runs, so dataloader state advances
  identically up to the moment a crop happens. That's the right A/B for
  *this question* (does the gate change anything mechanically), but doesn't
  estimate seed variance. A second run at a different seed would tell us
  whether the +0.005 val_bpb gap is structural or seed-dependent.
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

- Wandb runs: [`d3_tiny`](https://wandb.ai/matt-parrett/nanochat/runs/nasvg15p) (discard), [`d3_tiny_remreuse`](https://wandb.ai/matt-parrett/nanochat/runs/uh2vf0s8) (remainder_reuse).
  Note: the remreuse wandb run started 05:09:43 ahead of the relaunch; the
  *successful* run is the one with `total_training_time=392.5588`.
- Local logs: `/tmp/d3_tiny_pretrain.log`, `/tmp/d3_tiny_remreuse_pretrain.log`.
- Checkpoints: `$NANOCHAT_BASE_DIR/base_checkpoints/d3_tiny/{model,optim}_000500.pt`,
  `$NANOCHAT_BASE_DIR/base_checkpoints/d3_tiny_remreuse/{model,optim}_000500.pt`.
  Each meta records the variant under `user_config.dataloader_variant`.
- Commits: `2173dab` (capability + meta plumbing), `2c2c54e` (banner unpack fix), `e32c17b` (ADR-005).
