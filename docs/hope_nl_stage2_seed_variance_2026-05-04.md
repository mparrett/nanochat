# Hope/NL Stage 2 — A2: SFT-seed-variance disambiguation (2026-05-04)

Per Codex's metadata-audit point and our refined A1→A2→A3 plan, this run
disambiguated where the Stage 2 SFT win lives — in pretrain, in SFT, or
both. Cheap question to answer before paying ~6h of multi-seed pretrain.

## Setup

Re-ran SFT on the **same** Stage 2 pretrain checkpoint
(`base_checkpoints/d6_stage2/model_005000.pt`, val_bpb 1.1743) with two
new seeds. Identical recipe to the seed=42 reference run:

```bash
uv run python -u -m scripts.chat_sft \
    --run=sft-stage2-d6-seed{1,2} \
    --model-tag=d6_stage2 \
    --sft-tag=d6_stage2_s{1,2} \
    --seed={1,2} \
    --num-iterations=375 \
    --total-batch-size=65536 \
    --eval-every=50 --eval-tokens=524288 \
    --save-every=100 --save-keep-last-n=2 \
    --chatcore-every=-1
```

Recipe is the canonical d6/M2 SFT setup now lifted into
`docs/project_notes/key_facts.md`.

## Result

**Final val_bpb across 3 seeds:**

| seed | val_bpb (step 375) | Δ vs seed=42 |
|---|---:|---:|
| **42** (existing) | **0.6518** | — |
| **1** | **0.6516** | -0.0002 |
| **2** | **0.6520** | +0.0002 |

**Spread: 0.0004.** That's an order of magnitude smaller than the
headline win we're trying to validate.

## Headline check

The Stage 2 SFT win we're validating is **Stage 2 (0.6518) vs baseline
(0.6639) = 0.0121**.

| quantity | value | ratio to headline |
|---|---:|---:|
| Headline win to validate | 0.0121 | 1.0× |
| SFT-seed spread (this experiment) | 0.0004 | 0.033× |
| Required for "headline is noise" | ≥ 0.0121 | — |

The SFT-seed-induced variance is **~30× smaller** than the headline win.
3/3 seeds land in a tight cluster well below the baseline. The win is
not an SFT-seed lottery.

## Trajectories

All three seeds tracked each other within ≤ 0.0024 at every checkpointed
step:

| step | seed=42 | seed=1 | seed=2 | spread |
|---:|---:|---:|---:|---:|
| 0 | 1.0277 | 1.0277 | 1.0277 | 0.0000 |
| 50 | 0.8348 | 0.8342 | 0.8366 | 0.0024 |
| 100 | 0.7975 | 0.7967 | 0.7966 | 0.0009 |
| 150 | — | 0.7729 | 0.7734 | (≤ 0.0005 across the two new seeds) |
| 200 | 0.7633 | 0.7645 | 0.7636 | 0.0012 |
| 250 | — | 0.7305 | 0.7316 | (≤ 0.0011 across the two new seeds) |
| 300 | 0.6924 | 0.6922 | 0.6929 | 0.0007 |
| 350 | — | 0.6589 | 0.6591 | (≤ 0.0002) |
| **375** | **0.6518** | **0.6516** | **0.6520** | **0.0004** |

(seed=42 ran with the same `eval-every=50` recipe but the historical
HANDOFF.md only logged the 50/100/200/300/375 milestones; the 150/250/350
points for seed=42 weren't captured at the time.)

## What this confirms and rules out

**Confirmed:**
- SFT optimization is highly seed-stable for this configuration. Three
  independent random seeds → trajectories cluster within ~0.001 at every
  intermediate eval, ~0.0004 at the final step.
- The Stage 2 SFT win (val_bpb 0.6518 at step 375 vs baseline 0.6639) is
  reproducible across SFT seeds. n=3 / 3 land below the baseline by a
  margin that dwarfs the seed spread.

**Ruled out:**
- "The Stage 2 SFT win is an SFT-seed lottery." The seed-induced
  variance at the SFT step is too small to account for the headline.

**Not yet ruled out:**
- "The win is a pretrain-seed lottery." All three SFT runs loaded the
  same Stage 2 pretrain checkpoint (val_bpb 1.1743), which itself was a
  single-seed pretrain. If the variance lives in pretrain, multi-seed
  pretrain (A3) would expose it. SFT-stability is what makes A3 cheap to
  interpret — any val_bpb difference at SFT-end will trace back to
  pretrain, not noise.

## Decision: proceed to A3

Per the audit doc's A1→A2→A3 sequence:

> If A2 confirms SFT-seed-stability, A3 → B is the right path: stage 0–2
> have all been incremental architectural additions, and Stage 4 is
> where memory starts doing structural work.

A2 confirmed. A3 (multi-seed Stage 2 pretrain) is justified and queued
for the next session.

## A3 launch plan (queued, not started)

Two more Stage 2 pretrains with different seeds, matched on every other
flag to the existing Stage 2 pretrain (val_bpb 1.1743, seed=42).

Recipe is the canonical d6 pretrain + Stage 2 architecture knobs:

```bash
# Pretrain seed=1
PYTHONUNBUFFERED=1 nohup uv run python -u -m scripts.base_train \
    --run=stage2-d6-pretrain-seed1 \
    --model-tag=d6_stage2_pretrain_s1 \
    --seed=1 \
    --depth=6 --aspect-ratio=64 --head-dim=64 \
    --max-seq-len=512 --window-pattern=L \
    --num-iterations=5000 \
    --device-batch-size=32 --total-batch-size=16384 \
    --hope-additive-memory-layer=3 \
    --hope-memory-w-o-init-scale=1.0 \
    --hope-memory-kind=learned_gate \
    --hope-memory-alpha-max=0.999 --hope-memory-eta-max=1.0 \
    --hope-memory-alpha-init-bias=4.595 --hope-memory-eta-init-bias=-2.197 \
    --eval-every=100 --eval-tokens=524288 \
    --core-metric-every=-1 --sample-every=-1 \
    --save-every=1000 --save-keep-last-n=2 \
    > /tmp/pretrain_stage2_s1.log 2>&1 &

# Pretrain seed=2 (after seed=1 completes)
# same with --seed=2 --model-tag=d6_stage2_pretrain_s2 --run=stage2-d6-pretrain-seed2
```

**Config-parity audit (2026-05-04, post-Codex sanity-check):** the queued
commands above were updated to make every architecturally relevant field
explicit, after a parity audit against
`base_checkpoints/d6_stage2/meta_005000.json`. Drift caught:

| flag | base_train.py default | seed=42 reference | implication if not passed |
|---|---|---|---|
| `--head-dim` | 128 | **64** | n_embd would be 768 vs reference 384 — different model |
| `--max-seq-len` | 2048 | **512** | different context length, different memory footprint |
| `--window-pattern` | "SSSL" | **"L"** | sliding-window attention vs full attention |
| `--num-iterations` | -1 (auto-chinchilla) | **5000** | should match auto-compute but explicit avoids drift if defaults change |
| `--eval-every` | 250 | **100** | different val_bpb sampling cadence |
| `--core-metric-every` | 2000 | **-1** | reference skipped CORE during training; matching saves time |

Two intentional divergences from the seed=42 reference (operational, not
val_bpb-affecting):
- `--save-every=1000 --save-keep-last-n=2` (reference used `-1` = no
  intermediate saves; rolling cleanup gives crash insurance for ~2.4 GB
  disk cost)
- `--sample-every=-1` (reference used 100; we don't need mid-training
  samples for A3's val_bpb comparison)

Expected per-pretrain wall: ~3h on M2 24GB (was 5h13min on the seed=42
Stage 2 pretrain per HANDOFF, but optimizer improvements have landed
since). Two pretrains = **~6h total**.

After both pretrains complete, the cheapest comparator is val_bpb at
step 5000 (the metric we already have for seed=42). If multi-seed
pretrain val_bpb tracks closely (similar to A2's SFT-seed result), the
architecture's win is robust. If it spreads as wide as 0.0121, the
headline was a pretrain-seed lottery.

**Don't need to re-SFT each new pretrain.** The cheapest A3 sufficient
test is pretrain val_bpb agreement; SFT comes later only if A3 is clean
and we want a downstream-task confirmation.

## Files added this session

- `docs/hope_nl_stage2_seed_variance_2026-05-04.md` (this writeup)
- `chatsft_checkpoints/d6_stage2_s1/model_000375.pt` (seed=1 SFT, val_bpb 0.6516)
- `chatsft_checkpoints/d6_stage2_s2/model_000375.pt` (seed=2 SFT, val_bpb 0.6520)
- `docs/project_notes/key_facts.md` updated with d6/M2 training recipe (commit `f6467ff`)
- Patches landed: `--seed` plumbing (commit `29146e7`), `--sft-tag`
  separate save dir (commit `fc48d9c`)
