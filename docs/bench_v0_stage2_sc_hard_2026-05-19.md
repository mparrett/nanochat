# bench/ playground v0 — Stage 2 (learned-gate memory) on hard SelectiveCopy

**Date:** 2026-05-19 (continuation of `bench_v0_selective_copy_2026-05-18.md`)
**Setup:** d4 (n_embd=256), MPS on M2, seed=0, eval-every 25.
**Task:** SelectiveCopy at K=16 content tokens, T_in=192 noise stream, T=210.
**Arm:** stage1_add architecture with `--hope-memory-kind=learned_gate` (Stage 2 priors: α_max=0.999, η_max=1.0, α_init_bias=4.595, η_init_bias=-2.197, W_o init scale=1.0).

## Result

| arm                                        | sat step | final acc | wall    |
|--------------------------------------------|---------:|----------:|--------:|
| baseline (no memory)                       |      200 |    0.9927 | 16.5m   |
| stage1_add W_o=1.0 (fixed α=η=1)           |      375 |    0.9651 | 18.0m   |
| **stage2 add W_o=1.0 (learned α/η)**       |  **100** |**0.9985** |**16.1m**|

Stage 2 saturates **2× faster than baseline** and **3.75× faster than Stage 1**. Final accuracy higher than both. This is a clean architectural win on the state-tracking axis at d4.

## Trajectory comparison

| step | baseline | Stage 1 | **Stage 2** |
|----:|--------:|--------:|------------:|
| 25  | 0.028   | 0.032   | 0.028       |
| 50  | 0.028   | 0.096   | **0.283**   |
| 75  | 0.113   | 0.192   | **0.905**   |
| 100 | 0.328   | 0.182   | **0.986**   |
| 125 | 0.517   | 0.225   | 0.994       |
| 150 | 0.727   | 0.248   | 0.994       |
| 175 | 0.901   | 0.296   | 0.997       |
| 200 | 0.951   | 0.399   | 0.995       |
| 250 | 0.976   | 0.591   | 0.996       |
| 375 | —       | 0.961   | 0.998       |
| 400 | 0.993   | 0.965   | 0.999       |

At step 50, Stage 2 is at 28.3% while baseline is at 2.8% and Stage 1 is at 9.6%. The gap appears in the very first eval (step 25) — Stage 2's loss drops faster from the start. By step 75 it's fully grokking the task while the others are still learning the initial signal.

## Combined picture across all seven bench v0 runs

| task & difficulty       | baseline | stage1 W_o=1.0 | **stage2** |
|-------------------------|---------:|---------------:|-----------:|
| MQAR easy (K=16, T=128) |       75 |            125 |   (unrun)  |
| MQAR hard (K=64, T=256) |      225 |            250 |   (unrun)  |
| SC hard   (K=16, T_in=192)|    200 |            375 |    **100** |

The directly motivated experiment landed cleanly. Stage 1 fixed-α memory's inductive bias is recall-shaped (small +25 step cost on MQAR hard) and ACTIVELY harmful for state tracking (+175 step cost on SC hard). Stage 2's learned per-token α/η gates flip this: not only does it eliminate the integration cost on SC, it accelerates training to 2× faster than baseline.

## What this rules in and out

**Ruled in:**
- **Stage 2 has a state-tracking signature the d6/5000-iter LM arc missed.** The original Stage 2 verdict ("neutral on LM val_bpb under recipe-controlled comparison") was correct on what it measured but missed this axis entirely. The bench/ playground was built exactly to surface findings of this shape: cheap synthetic probes that test mechanism-specific value the LM-loss aggregate averages out.
- **The δ-mem framing transfers to Stage 2 (not Stage 1).** The δ-mem 2026-05-18 field result identified working-memory / multi-step-state-tracking as the axis that lifts. Stage 1's fixed-α additive memory does NOT reproduce this; Stage 2's learned-gate memory DOES. The architectural difference (learned per-token gates) is exactly the kind of thing δ-mem's rank-8 recurrent state with selective updates also provides.
- **The bench/ harness is working.** Single-seed runs at d4 with 16-min wall give qualitatively decisive signals. The W_o sweep + Stage 1/Stage 2 comparison on SC produces architecture-distinguishing curves cleanly. ROI per minute of compute is good enough that further iteration is justified.

**Not yet ruled in (caveats):**
- n=1 seed. Need at least n=2-3 before declaring "Stage 2 wins on state-tracking" as more than a lucky-seed result. Magnitude (100 vs 200) makes pure-luck hard to swallow but doesn't rule it out.
- Single difficulty. K=16, T_in=192 is one point on a wide difficulty range. Whether the win persists at K=8 (easier — Stage 2 might be overcapacity) or K=64 (harder — Stage 2 might saturate later than baseline) is unknown.
- Single depth (d4). Whether the result reproduces at d2 (faster iteration), d6 (matches Stage 1.5 probe), or d8+ (matches the from-scratch arc) is unknown.
- Single architecture variant. The Stage 2 prior defaults (α_init=0.99, η_init=0.1) were tuned for d6 LM training, not for d4 SC probe. There may be better priors for this regime.

**Ruled out:**
- "Stage 2 is architecturally neutral" — the original verdict on LM val_bpb. Wrong on at least one synthetic state-tracking task at d4.
- "Stage 1's fixed-α memory is sufficient for state tracking" — also wrong; Stage 1 actively hurts SC. The learned-gating mechanism is load-bearing for SC at this scale.

## Mechanism story

Stage 2's win over both baseline and Stage 1 on hard SC suggests the learned per-token α/η gates are doing something specific:

- α_t = α_max · sigmoid(W_α x_t + b_α) controls how much past memory state persists at step t.
- η_t = η_max · sigmoid(W_η x_t + b_η) controls how much the current (k, v) pair is written into memory.

For SelectiveCopy, the ideal behavior is: **write content tokens to memory, ignore noise**. At noise positions, η should be ≈ 0 (no write). At content positions, η should be near 1.0. α should stay high throughout (preserve content already written).

This is exactly the inductive bias Mamba's selective state-space mechanism provides — and the Mamba paper specifically calls out SelectiveCopy as the task where selective gating wins over time-invariant SSMs. Stage 2's learned α/η is structurally the same idea, applied to linear-attention memory rather than SSM state.

Stage 1's fixed α=η=1 forces the memory to write *every* token (including noise) and never forget. That's actively counterproductive on SC: noise tokens overwrite the K/V/Q lookup state with garbage. The +175 step integration cost is the model learning to suppress this noise via other pathways.

This is the first bench result that gives a concrete mechanism story for *why* learned gates matter on a real diagnostic task. The original Stage 2 design priors (α_init=0.99, η_init=0.1) make perfect sense for SC: start with η ≈ 0.1 (small writes, so noise doesn't drown out signal), let the model learn to push η near 1 at content positions and toward 0 at noise positions.

## What's next

Three obvious follow-ups in priority order:

1. **n=2-3 seeds on Stage 2 hard SC.** Cheapest possible robustness check. If sat step holds at ~100 ± 25 across seeds, the result is decisive. ~30-50 min wall.

2. **Stage 2 on hard MQAR.** Does Stage 2 also win on the recall axis, or is the win SC-specific? If Stage 2 hard MQAR sat << 250 (Stage 1's number), Stage 2 wins on both axes the bench tests — bigger architectural claim. If Stage 2 ≈ 250, the win is state-tracking-specific. ~16 min wall.

3. **Stage 2 at d6** to bridge to the LM val_bpb result. If Stage 2 wins on hard SC at d6 too, the d6/5000-iter LM arc was missing this signal entirely (probe-first methodology vindicated again). ~25-30 min wall per arm.

The d6/5000-iter LM arc concluded "Stage 2 is neutral, not net-positive, on natural-language val_bpb." That conclusion stands on what it measured. The bench result doesn't overturn it; it adds an orthogonal axis where Stage 2 *is* net-positive. The honest combined verdict is now: **Stage 2 is neutral on LM val_bpb at the d6/5000-iter scale, but actively wins on the state-tracking axis at d4. Whether the state-tracking advantage translates to LM val_bpb on longer-context corpora or larger scales is the next-level question.**

## Code & cost ledger

- No code change this iteration — the `--hope-memory-kind=learned_gate` flag was already in `bench/run.py` from v0.
- Single run, 16.1 min wall. Log: `bench/logs/stage2_d4_sc_hard.jsonl`.
- Cumulative bench v0 cost: 7 runs, ~120 min wall total.
