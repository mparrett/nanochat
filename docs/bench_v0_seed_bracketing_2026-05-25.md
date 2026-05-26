# bench v0 seed bracketing — Stage 2 SC headline falsified

**Date:** 2026-05-25
**Status:** experiment ran, verdict landed. The load-bearing bench v0
finding ("Stage 2 saturates 2× faster than baseline on hard SC at d4")
**does not survive seed bracketing**. Stage 2 n=3 mean = 208 vs
baseline n=4 mean = 219 — ~5% gap, inside both arms' seed-variance
range. Stage 2 actually has *wider* variance than baseline (range 225
vs 100). The "2× faster" headline was a lucky-seed artifact at s0=100.

## What this session did

Applied today's NTK-Mirror Path A lesson (single-seed result was
lucky-order artifact) to bench v0's headline numbers. The project map
(`docs/project_map_2026-05-20.html`) had explicitly listed n=2-3 seed
bracketing of the four bench v0 headlines as the #1 frontier priority,
~2h wall, "cheapest, most important." This session executed that
priority on the first headline: **Stage 2 SC sat=100**, the headline
"convergence" finding from `docs/bench_v0_stage2_sc_hard_2026-05-19.md`.

## Setup

All runs match the original bench v0 SC hard configuration exactly:

```
python -m bench.run --task selective-copy --depth 4 --T-in 192 \
  [--hope-additive-memory-layer 1 \
   --hope-memory-w-o-init-scale 1.0 \
   --hope-memory-kind learned_gate] \  # Stage 2 only
  --seed N --label <arm>_s<N> --log-dir bench/logs
```

Default Stage 2 priors: α_max=0.999, η_max=1.0, α_init_bias=4.595,
η_init_bias=-2.197 — same as bench v0.

500 iterations, eval every 25, saturation = first eval step where
accuracy ≥ 0.95. M2 / MPS / fp32. Wall ~17 min per run.

## Results

### Per-seed saturation steps

| Arm | s0 (original) | s1 | s2 | s3 | n | mean | range |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (no memory) | 200 | 300 | 225 | 150 | **4** | **219** | 150 |
| Stage 2 add | **100** | 200 | 325 | — | **3** | **208** | 225 |

Stage 2 mean (208) beats baseline mean (219) by **11 steps**, ~5%.
Stage 2 standard error ≈ 65; baseline standard error ≈ 31. The 95%
confidence intervals overlap heavily — the mean difference is
~0.1 standard deviations of pooled spread.

### Distribution comparison

```
baseline n=4:  150   200       225            300
Stage 2  n=3:  100        200                              325
                ^^^ Stage 2 best                     ^^^ Stage 2 worst
                                                     beyond baseline
```

Stage 2's range (100-325) is a **superset of** baseline's (150-300).
Baseline's best seed (150) beats Stage 2's median (200).

### The falsification arc

Step-by-step collapse of the headline gap as we added runs:

| After adding | Stage 2 n | baseline n | gap |
|---|---:|---:|---:|
| (bench v0, n=1 each) | 1 | 1 | **-100 steps (2× faster)** |
| Stage 2 s1 | 2 | 1 | -50 (1.33×) |
| Stage 2 s2 | 3 | 1 | -8 (≈tied) |
| baseline s1 | 3 | 2 | -42 (17% faster) |
| baseline s2 | 3 | 3 | -34 (14% faster) |
| **baseline s3** | **3** | **4** | **-11 (5%, inside noise)** |

Every additional seed pulled the gap toward zero. Classic n=1 → n=N
collapse pattern.

### Per-trajectory comparison (acc at each eval step)

To make the within-arm variance concrete, here are Stage 2's three
trajectories side by side:

| step | s0 (lucky) | s1 | s2 (slow) |
|---:|---:|---:|---:|
| 25 | 0.028 | 0.036 | 0.038 |
| 50 | **0.283** | 0.083 | 0.097 |
| 75 | **0.905** | 0.336 | 0.163 |
| 100 | **0.986** | 0.821 | 0.168 |
| 150 | 0.994 | 0.928 | 0.440 |
| 200 | 0.995 | 0.961 | 0.903 |
| 300 | 0.997 | 0.978 | 0.945 |
| 325 | 0.998 | 0.983 | 0.963 (sat) |

s0 was qualitatively different from s1 and s2. At step 50, s0 is at
28%; s1 and s2 are at 8-10%. The "grokking" jump that defined the
headline trajectory was specific to s0's init.

## What this falsifies

From `docs/bench_v0_stage2_sc_hard_2026-05-19.md`:

> Stage 2 saturates **2× faster than baseline** and **3.75× faster
> than Stage 1**. Final accuracy higher than both. This is a clean
> architectural win on the state-tracking axis at d4.

Each claim, revisited:

- **"2× faster than baseline"** → ✗ falsified. Mean ratio 219/208 = 1.05.
- **"3.75× faster than Stage 1"** → Stage 1 was not bracketed today;
  Stage 1 sat=375 is also likely overstated under the same n=1
  vulnerability. Best-case still-defensible: "Stage 2 may be faster
  than Stage 1" pending Stage 1 bracketing.
- **"Final accuracy higher than both"** → Stage 2 s0 final acc 0.999;
  baseline s3 final acc 0.988. Real but ~1pp gap. Not load-bearing.
- **"A clean architectural win on the state-tracking axis at d4"** →
  ✗ falsified. Cannot distinguish Stage 2 from baseline at this
  experimental scale.

From `docs/bench_v0_stage2_sc_hard_2026-05-19.md`, the writeup also
candidly anticipated this exact possibility:

> n=1 seed. Need at least n=2-3 before declaring "Stage 2 wins on
> state-tracking" as more than a lucky-seed result. Magnitude (100 vs
> 200) makes pure-luck hard to swallow but doesn't rule it out.

The writeup was right to flag it. Pure-luck turned out to be the
explanation.

## What this means for the project map

`docs/project_map_2026-05-20.html` framed the 22-day arc as:

> originating question → d6 verdict → side-thread that didn't fit
> (δ-mem) → reframe (selective axis) → **convergence (bench v0)** →
> frontier (robustness, scale, third axis).

The **convergence** node — bench v0's Stage 2 SC win as the
architectural traction the d6/5000-iter LM arc missed — is the piece
that doesn't survive. The framing should be revised to something like:

> originating question → d6 verdict → side-thread (δ-mem) → reframe
> (selective axis as the architectural question worth asking) → bench
> v0 attempted convergence → **seed bracketing falsified the
> convergence claim** → frontier (need n≥3 default + revisit
> mechanism claims at proper statistical power).

The δ-mem field result on hard MQAR at 100× scale
(`docs/delta_mem_field_result_2026-05-18.html`) still stands as
independent evidence for the "selective gating is a specialization"
hypothesis. But the bench v0 d4 evidence for the same hypothesis is
no longer load-bearing.

## What this means for the other three bench v0 headlines

The same n=1 vulnerability applies to:

- **Stage 2 MQAR sat=375** (the "loss" claim from the U-curve story).
- **η=0.5 sat=475** (the U-curve "worst point" claim).
- **Stage 1 SC sat=375** (the "Stage 1 actively hurts SC" claim).

Given that one of four headlines collapsed completely under
bracketing, the prior on the other three is now "almost certainly also
overstated." Queued for next session as a high-priority follow-up.

## Methodology lesson

**n=1 is insufficient for any architectural claim on bench/.** Seed
variance at d4/500-iter scale is large enough to swamp architectural
differences of the magnitude bench v0 reported as "wins." Today's
session is the second instance of n=1 → n=N collapse: NTK-Mirror Path
A this morning showed +1/30 over LoRA collapse to parity after
seed bracketing.

**Proposed bench/ default going forward:** every architectural
comparison runs n=3 minimum before any "X beats Y" claim. Wall cost:
~50 min/arm at d4/500 iters, ~150 min for a three-arm comparison.
Bench v0 produced 7 runs in ~120 min — at n=3 default that becomes
~6 hours. Hand-cranking this is fine; building it into the runner
(an `--n-seeds N` flag that loops internally and reports
mean ± range) is a 10-line change worth doing if any further bench/
work happens.

## Cost ledger

This session: 5 runs at ~17 min each = ~88 min wall. Plus the
original 2 single-seed runs from May-18/19 = 7 runs of cumulative SC
hard data. Per-run wall:

| Run | wall |
|---|---:|
| stage2_d4_sc_hard_s1 | 17.0 min |
| stage2_d4_sc_hard_s2 | 16.9 min |
| baseline_d4_sc_hard_s1 | 18.8 min |
| baseline_d4_sc_hard_s2 | 19.7 min |
| baseline_d4_sc_hard_s3 | 18.0 min |
| **today total** | **90.4 min** |

Memory pressure (0.68 GB free at session start) had no measurable
effect on wall per step — bench/ runs use small d4 models and aren't
memory-bound the way Qwen-0.5B inference was earlier today.

## Decision and next steps

**Falsification stands. The bench v0 "Stage 2 wins on SC at d4" claim
is dead.** Stage 2's mean is essentially tied with baseline once
proper baseline bracketing is included.

For next session, in priority order:

1. **Bracket the other three bench v0 headlines.** ~150 min wall total
   if each gets n=3 with a single new seed (each already has n=1 from
   May-19). Likely to find the same pattern; needs confirming.
2. **Project map revision.** The "convergence" framing needs updating;
   the spine paragraph needs to acknowledge the falsification.
3. **Add `--n-seeds N` to `bench/run.py`.** Makes the n=3 default
   trivial. 10-15 line change. Defensive against repeating the
   methodology error.
4. **Stage 1 SC bracketing.** Stage 1 sat=375 is conspicuously slow
   relative to baseline=219; if bracketing brings it down to ≈ baseline
   the "Stage 1 actively hurts SC" claim from the writeup also dies.

What does NOT change:
- The d6/5000-iter LM verdict ("Stage 2 neutral on natural-language
  val_bpb under recipe-controlled comparison"). That had its own
  multi-seed validation (`docs/hope_nl_stage2_seed_variance_2026-05-04.md`).
- The δ-mem field result on hard MQAR at 100× scale. Independent
  evidence on different scale and mechanism.
- The architectural-shape hypothesis ("memory mechanisms are
  task-mechanism-routing problems, not 'memory yes/no'"). The
  hypothesis is still live; the d4 bench evidence for it just isn't.

## Files

- Code: no changes (existing `bench/run.py --seed` was sufficient).
- Logs: `bench/logs/{stage2_d4_sc_hard_s1,stage2_d4_sc_hard_s2,baseline_d4_sc_hard_s1,baseline_d4_sc_hard_s2,baseline_d4_sc_hard_s3}.{jsonl,stdout.log}`
- Writeups:
  - `docs/bench_v0_seed_bracketing_2026-05-25.md` (this file)
  - `docs/bench_v0_seed_bracketing_2026-05-25.html` (publication HTML)
- HANDOFF.md: Day 2026-05-25 addendum extended.

## References

- Original headline writeup: `docs/bench_v0_stage2_sc_hard_2026-05-19.md`
- Combined bench v0 synthesis: `docs/bench_v0_result_2026-05-20.html`
- Project map (needs revision): `docs/project_map_2026-05-20.html`
- δ-mem field result (independent corroboration of selective-gating
  hypothesis, still standing): `docs/delta_mem_field_result_2026-05-18.html`
- Today's NTK-Mirror Path A (parallel n=1 → n=N falsification):
  `docs/ntkmirror_persona_comparison_2026-05-25.md`
