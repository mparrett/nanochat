# bench/ playground v0 — η_init sweep on Stage 2 hard MQAR

**Date:** 2026-05-19 (continuation of `bench_v0_stage2_mqar_hard_2026-05-19.md`)
**Setup:** d4 (n_embd=256), MPS on M2, seed=0, eval-every 25.
**Task:** MQAR at K=64, M=32, T=256, n_keys=n_values=128 (hard difficulty).
**Arm:** Stage 2 (`--hope-memory-kind=learned_gate`, additive layer 1, W_o=1.0, α_max=0.999, η_max=1.0, α_init_bias=4.595). η_init varied via `--hope-memory-eta-init-bias`.

## Motivation

The prior bench v0 result showed Stage 2 with the canonical η_init=0.1 prior was 150 steps slower than baseline on hard MQAR (sat 375 vs 225). The mechanism story: η_init=0.1 means small writes by default, and MQAR wants every position to write into memory (η→1.0). The model spends ~250 steps cranking η up before it can use the memory pathway.

If that story is right, raising η_init should close the gap. This sweep tests that.

## Results

| η_init | b_eta   | sat step | final acc | wall  |
|:-----:|:-------:|---------:|----------:|------:|
| 0.1   | -2.197  |      375 |    0.9806 | 19.4m |
| 0.3   | -0.847  |      400 |    0.9708 | 19.5m |
| 0.5   | 0.000   |  **475** |    0.9977 | 22.1m |
| 0.8   | 1.386   |      425 |    0.9990 | 25.4m |
| 0.99  | 4.595   |  **325** |    0.9995 | 25.3m |

**Reference**: baseline 225; Stage 1 W_o=1.0 250.

## The U-curve

The η_init landscape is non-monotonic. Plotting saturation step vs η_init:

```
sat
500 |                  *
    |
450 |                       *
    |
400 |        *
    |   *
350 |
    |
325 |                            *
300 |
    |    baseline (225) ---------------
    |    Stage 1   (250) ---------------
200 |
    +---+----+--------+----+----+-----
       0.1  0.3      0.5  0.8  0.99    η_init
```

Extremes win, middle loses. The worst point is η=0.5 (sat=475).

## Mechanism story refined

The original "η=0.1 is far from the MQAR optimum so it takes 250 steps to crank η up" story is now incomplete. If that were the full picture, raising η_init should *monotonically* close the gap. The U-curve says something else.

The refined story: **MQAR has a wide partial-recall basin at ~50% acc**, and the model's ability to escape this basin depends on η_init in a non-trivial way:

- **η=0.1 (canonical):** memory writes are tiny at start, so the model can't engage memory early at all. By the time gradient pressure pushes η up, the model has already learned the partial-attention-only solution. The plateau at ~45% is the partial solution; the model has to *abandon* it and rebuild with memory. Late entry to plateau (step 275), short plateau (~100 steps).
- **η=0.3, 0.5 (moderate):** memory engages early (step 125), but the partial-recall solution it enables is exactly the wide basin. Model gets stuck on it. Long plateau (>250 steps).
- **η=0.99 (extreme):** memory writes are at full strength from step 0, so the model never enters the partial-recall basin — it goes directly toward the full-recall solution. Mid-late entry to plateau (step 175), short plateau, fastest grok.

The middle is the worst because it's exactly the regime that puts the model into the partial-recall trap with no easy escape.

## What this rules in and out

**Ruled out:**
- "Stage 2's MQAR loss is a prior-tuning artifact" — emphatically not. Even at η_init=0.99 (the closest Stage 2 gets to Stage 1's α=η=1 fixed-gate setup), saturation is 325 — still 100 steps slower than baseline and 75 slower than Stage 1. The gap doesn't close by tuning the prior alone.
- "Higher η_init monotonically helps on MQAR" — also no. The middle of the range (0.3, 0.5) is worse than the canonical 0.1.

**Ruled in:**
- **Stage 2 has a learnable-gate architectural overhead floor of ~75 steps versus Stage 1 on MQAR at d4.** Even when the gates start at the same value as Stage 1's fixed gates, the gradient updates to the gate parameters add training-step cost that doesn't exist when gates are fixed.
- **MQAR's loss landscape has a wide partial-recall basin** that's most accessible at moderate η_init. This is a property of the task + d4 capacity, not the architecture per se — but it's the reason the architecture's prior-sensitivity shows the U-shape.
- **Stage 2's MQAR loss is robust to prior choice** in the sense that *no* η_init prior makes it competitive with baseline. The architecture-vs-task mismatch is real, not a hyperparameter accident.

## Complete bench v0 picture across nine runs

| task           | baseline | Stage 1 W_o=1.0 | Stage 2 η=0.1 | Stage 2 η=0.99 |
|----------------|---------:|----------------:|--------------:|---------------:|
| MQAR easy      |       75 |             125 |       (unrun) |        (unrun) |
| MQAR hard      |      225 |             250 |           375 |        **325** |
| SC hard        |      200 |             375 |       **100** |        (unrun) |

The complete η_init sweep on MQAR hard (3 more arms: 0.3, 0.5, 0.8) covers the gap structure.

## What this implies

Three things, ordered most-to-least confident:

1. **The Stage 2 architecture has an irreducible cost relative to fixed-gate Stage 1 on uniform-write tasks.** Tuning η_init at d4 covers ~50 steps of the 100-step Stage 2-vs-baseline gap; ~50 steps remain unaccountable by prior tuning. Some combination of (a) learnable gate parameters needing optimization budget, (b) inductive bias of "selectivity is possible" being unhelpful when the optimal policy is uniform, contributes to the floor.

2. **The η_init priors selected by the paper for Stage 2 (η=0.1, α=0.99) are well-aligned for selective tasks (SC) and somewhere near a local minimum on uniform tasks (MQAR).** η=0.1 happens to be on the *good* side of the U-curve on MQAR (sat=375 < the 0.3/0.5 trap at 400/475). The paper's prior choices were sensible defaults, not coincidentally optimal.

3. **The η_init landscape is task-specific.** Future Stage 2 work should not tune η_init globally — instead either (a) use the canonical prior and accept the task-mix-dependent verdict, or (b) make η_init learnable (perhaps a small MLP conditioning on global statistics of the input distribution).

## Caveats

- All runs n=1 seed. The U-curve shape is suggestive; need at least n=2-3 to confirm. The η=0.5 vs η=0.99 gap (150 steps) is large enough that pure-luck explanation is unlikely, but the η=0.3 vs η=0.1 gap (25 steps) is within plausible single-seed noise.
- Single difficulty point. The plateau-basin structure could be specific to K=64/T=256.
- Single architectural variant (additive layer 1). Layer placement may interact with η_init optimum.
- The next-level question — does the η_init sweep on hard SC also show non-monotonicity, or is the SC win at η=0.1 robust across the η range? — is unrun. If SC also has a U-shape with η=0.1 happening to land on the SC-good side, then the η=0.1 prior is fragile. If SC is robustly good across the η range, then SC's win is structural.

## What's next

Three obvious priorities:

1. **η_init sweep on hard SC** (mirror of this experiment). Tests whether the SC win is fragile to prior choice or structural. ~100 min wall. **This is the directly motivated next experiment.**

2. **n=2-3 seeds on the U-curve extremes** (η=0.1, η=0.5, η=0.99). Confirms the U-shape isn't a coincidence. ~80 min wall.

3. **Learnable η_init experiment** — replace `b_eta` constant with `b_eta = MLP(global_stats)` where MLP sees mean(|x|) or similar input statistics. Tests whether making the prior input-conditional fixes the task-mix problem. Code change required (~30 LOC). Out of scope for this session.

## Cost ledger

- Four new arms: η=0.3 (19.5m), η=0.5 (22.1m), η=0.8 (25.4m), η=0.99 (25.3m). Total ~92 min wall.
- No code changes.
- Logs: `bench/logs/stage2_d4_hard_eta{03,05,08,099}.jsonl`.
- Cumulative bench v0 session: 11 runs (3 baseline/Stage 1 on easy MQAR + 3 baseline/Stage 1/Stage 2 on hard MQAR + 2 baseline/Stage 1 on hard SC + 1 Stage 2 on hard SC + 4 η_init sweep arms), ~250 min wall.
