# bench/ playground v0 — SelectiveCopy at hard difficulty

**Date:** 2026-05-18 (continuation of `bench_v0_mqar_w_o_2026-05-18.md`)
**Setup:** d4 (n_embd=256), MPS on M2, seed=0, eval-every 25.
**Task:** SelectiveCopy at K=16 content tokens, T_in=192 noise stream, T=210.

## Why this run

Hope/NL Stage 2's MQAR result showed memory engaging but not winning. δ-mem's 2026-05-18 field result identified the lift axis as multi-step *state tracking*, not recall. The bench v0 needed a state-tracking probe on the same architectural variants to characterize the axis where memory should help most.

SelectiveCopy (Mamba's canonical state-tracking probe) is the natural choice — content tokens scattered through a noise stream, must be copied in order after a separator.

## Results

| arm                 | sat step | final acc | wall    |
|---------------------|---------:|----------:|--------:|
| baseline (no memory)|      200 |    0.9927 | 16.5m   |
| stage1_add W_o=1.0  |      375 |    0.9651 | 18.0m   |

**The memory module hurts SelectiveCopy by ~175 steps at d4.**

## Trajectory comparison

| step | baseline | W_o=1.0 | who's ahead |
|----:|--------:|--------:|---|
| 25  |  0.028  |  0.032  | W_o=1.0 by +0.4pp |
| 50  |  0.028  |  0.096  | **W_o=1.0 by +6.8pp** |
| 75  |  0.113  |  0.192  | **W_o=1.0 by +7.9pp** |
| 100 |  0.328  |  0.182  | baseline by +14.6pp |
| 125 |  0.517  |  0.225  | baseline by +29.2pp |
| 150 |  0.727  |  0.248  | baseline by +47.9pp |
| 175 |  0.901  |  0.296  | baseline by +60.5pp |
| 200 |  0.951  |  0.399  | baseline by +55.2pp |
| 250 |  0.976  |  0.591  | baseline by +38.5pp |
| 300 |  0.985  |  0.822  | baseline by +16.3pp |
| 375 |  —      |  0.961  | (W_o=1.0 saturates) |
| 400 |  0.993  |  0.965  | baseline by +2.8pp |

Two distinct phases:
- **Steps 25-75 — memory engages.** W_o=1.0 is ahead. The memory module is solving easy SC cases (the model picks up the trivial "content tokens are not the NOISE token" signal via the memory pathway faster than baseline learns it via attention).
- **Steps 100+ — integration cost dominates.** Baseline grokks the attention-based solution and pulls dramatically ahead. W_o=1.0 follows the same trajectory shape but ~175 steps behind.

## Combined picture across all six bench v0 runs

| task & difficulty       | sat baseline | sat W_o=1.0 | gap     |
|-------------------------|-------------:|------------:|--------:|
| MQAR easy (K=16, T=128) |           75 |         125 |     +50 |
| MQAR hard (K=64, T=256) |          225 |         250 |     +25 |
| SC hard (K=16, T_in=192)|          200 |         375 | **+175**|

Pattern:
- Memory engages on both hard probes (early-phase advantage in both cases).
- Integration cost on MQAR is modest (+25 steps at hard).
- Integration cost on SC is severe (+175 steps).

## What this rules in and out

**Ruled out (at this scale):**
- "δ-mem working-memory framing transfers straight to Stage 1 additive memory" — no. δ-mem's per-layer recurrent state with rank-8 learned gating is structurally different from Stage 1's fixed-α linear attention. The empirical signature is opposite: δ-mem lifts state-tracking tasks; Stage 1 *hurts* state-tracking at d4.

**Ruled in:**
- Memory pathway engages on both probes when W_o=1.0 (early-phase advantage). This is the genuine "memory module doing work" signal the playground was built to detect.
- Stage 1 fixed-α memory's inductive bias is **recall-shaped**, not state-tracking-shaped. On MQAR (recall) it shows partial-engagement value; on SC (state tracking) it interferes with attention's solution.

**Not yet disambiguated:**
- Whether Stage 2 (learned per-token α/η gates) closes the SC gap. Stage 2's whole point is letting the model learn when to engage memory. If integration cost on SC is "memory is on when it shouldn't be," learned gates could turn it off. **This is the directly motivated next experiment.**
- Whether SC at K=16/T_in=192 is actually a state-tracking probe or just a fancy lookup. Attention solving it cleanly at step 200 suggests it might be lookup-class. Mamba-paper hard SC uses T_in=4096+.
- Whether d6 (matching the original Stage 1.5 probe scale) changes the SC story.

## Cost ledger

- Code change: bench/run.py extended with `--T-in` arg.
- Two runs: baseline 16.5m + stage1_add_wo1 18.0m = 34.5 min wall.
- Logs in `bench/logs/{baseline_d4_sc_hard, stage1_add_wo1_d4_sc_hard}.jsonl`.

## Decision point for next iteration

Two priorities:

1. **Stage 2 on hard SC.** Most directly motivated — tests whether learned gating can disengage memory when integration cost dominates. ~18 min wall. If Stage 2 closes the +175 step gap, the architecture argument shifts from "Stage 1 hurts" to "Stage 2 fixes Stage 1 via learned gating." If it doesn't, the architecture has a deeper structural problem at this scale.

2. **Escalate SC difficulty** (T_in=512+) before Stage 2. Tests whether the integration-cost finding holds when attention can't trivially solve the task. If at T_in=512 baseline also struggles, memory may have a chance to actually help. ~25-30 min per arm.

My pick: (1). The Stage 2 design exists exactly for this case (the paper's own motivation for replacing fixed α with learned α/η). Running it is the cheapest definitive test of whether the architecture can self-correct when memory hurts.
