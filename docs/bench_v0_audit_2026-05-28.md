# bench v0 audit — four headlines bracketed, architectural story dead

**Date:** 2026-05-28 (continuation of bench v0 SC falsification on 2026-05-25)
**Status:** experiment ran, verdict landed. All four bench v0 headline
numbers have been seed-bracketed at n=3. Two were falsified outright,
one was weakened materially, one survived but with a smaller gap than
claimed. The aggregate architectural narrative — "Stage 2 specializes
(wins SC, loses MQAR)" — does not survive. Stage 2 emerges as a
worse mechanism than baseline, not a specialized one. The U-curve
"shape" claim has its worst-point gap collapse from 100 steps to
37 steps and shares a 1/3 catastrophic-failure rate across η values,
so the curve shape itself is suspect. Stage 1 SC's "recall-shaped"
framing is the only headline that fully survives.

## What this audit was

`docs/project_map_2026-05-20.html` listed n=2–3 seed bracketing of the
four bench v0 headline numbers as the #1 frontier priority — "cheapest,
most important." The 2026-05-25 SC falsification
(`docs/bench_v0_seed_bracketing_2026-05-25.md`) covered the first.
This audit covers the remaining three plus pulls them into a
cumulative verdict.

The headlines under audit:
1. **Stage 2 wins SC** — sat=100 vs baseline 200 (2× faster), the
   centerpiece "convergence" finding.
2. **Stage 2 loses MQAR** — sat=375 vs baseline 225, the U-curve "loss" half.
3. **η=0.5 worst U-curve point** — sat=475 in a 5-point η sweep at
   {0.1, 0.3, 0.5, 0.8, 0.99} → {375, 400, 475, 425, 325}.
4. **Stage 1 hurts SC** — sat=375 vs baseline 200 (175 steps slower).

## Methodology

All runs match the original bench v0 configuration exactly. Same
hyperparameters, same task shapes, same eval rubric (sat = first
step where eval acc ≥ 0.95). Single bracketing knob added: each
arm now runs at n=3 via the new `bench/run.py --n-seeds 3` flag
(committed 2026-05-25 as `8f77ab1`).

Hardware: M2 MacBook, MPS, fp32. ~17–22 min wall per seed at d4 /
500 iter. Total audit cost: ~5h wall across four headlines.

## Per-headline results

### 1. Stage 2 SC win — FALSIFIED (covered 2026-05-25)

Detailed writeup: `docs/bench_v0_seed_bracketing_2026-05-25.md`.

Stage 2 n=3 = (100, 200, 325) → mean 208, range 225.
Baseline n=4 = (150, 200, 225, 300) → mean 219, range 150.

Gap: 11 steps (~5%), inside both arms' variance. Stage 2's range
*exceeds* baseline's. The s0=100 grokking trajectory was unique to
that seed's init, not a mechanism property. **"2× faster than
baseline" → mean ratio 1.05.**

### 2. Stage 2 MQAR loss — SURVIVES + GETS WORSE

| Seed | sat_step | final_acc | wall |
|---|---:|---:|---:|
| s0 | 425 | 0.999 | 21.8 min |
| **s1** | **NEVER SATURATED** | **0.014** | 23.3 min |
| s2 | 400 | 0.998 | 22.1 min |
| **mean (saturating)** | **412.5** | — | — |
| **catastrophic-fail rate** | **1/3 (33%)** | — | — |

s1 is the cleanest possible training failure — loss 4.89 → 4.49 over
500 steps, acc never moves off random (0.009 → 0.014). Not "trying
and failing" — completely degenerate under that init.

Compared to baseline MQAR (single-seed n=1 from May-18, sat=225):

- When Stage 2 trains, it saturates ~187 steps later than baseline.
- When it fails, it never saturates at all.
- The "150 steps slower than baseline" original framing **understates**
  the problem. Stage 2 doesn't just lose on MQAR; it has a 33%
  catastrophic-failure rate.

Reproducibility note: original May-19 s0 sat=375 vs today's s0=425.
Small but real drift (likely torch/transformers minor version). Doesn't
change the verdict.

### 3. η=0.5 U-curve worst point — WEAKENED, U-shape suspect

η=0.5 → `--hope-memory-eta-init-bias 0.0`. n=3 results:

| Seed | sat_step | final_acc |
|---|---:|---:|
| s0 | 450 | 0.999 |
| **s1** | **NEVER SATURATED** | **0.015** |
| s2 | 450 | 0.998 |
| **mean (saturating)** | **450** | — |
| **catastrophic-fail rate** | **1/3 (33%)** | — |

Cross-η comparison after bracketing (η=0.1 was bracketed under #2):

| η_init | original n=1 | new n=3 mean | failure rate |
|---|---:|---:|---:|
| 0.1 | 375 | 412.5 | 1/3 |
| 0.5 | 475 (worst) | 450 | 1/3 |

The U-curve "worst-point gap" of 100 steps (475 - 375) **shrinks to
37 steps** after bracketing both points. Both η values share the
same 1/3 catastrophic-failure rate — the failure mode is
η-independent. Crucially, the variance *within* a single η is
comparable to the variance *between* η values. The U-curve shape
cannot be distinguished from sampling noise at this scale without
bracketing all 5 points.

The η_init mechanism story from the May-19 writeup — "the worst point
is exactly where the partial-recall basin attracts" — was an
interpretation of a four-point curve whose shape is now ambiguous.
The interpretation may still hold; it has weaker evidence than
claimed.

### 4. Stage 1 SC hurt — SURVIVES (smaller gap)

| Seed | sat_step | final_acc |
|---|---:|---:|
| s0 | 325 | 0.984 |
| s1 | 425 | 0.968 |
| s2 | 375 | 0.975 |
| **mean** | **375** | 0.976 |
| range | 100 | — |

Stage 1 mean (375) vs today's baseline SC n=4 mean (219) →
gap **156 steps (~71% slower)**. Original headline gap was 175 steps
(375 vs single-seed baseline of 200). **The gap is ~10% smaller but
qualitatively survives.**

**No catastrophic-failure mode** — all 3 seeds saturated, range only
100 (much tighter than Stage 2 SC's 225 or the catastrophic 33%
failure of Stage 2 MQAR). Stage 1's mechanism is unstable in
magnitude but never collapses. This is consistent with the original
"fixed-α writes everything, hurts state-tracking" mechanism story.

## Combined verdict

| Headline | Verdict | Mean (n=3) | Original n=1 | Drift |
|---|---|---:|---:|---|
| Stage 2 wins SC (2× faster) | ✗ falsified | 208 | 100 | +108 |
| Stage 2 loses MQAR | ✓ + worse | 412.5 | 375 | +37.5; 1/3 fail |
| η=0.5 U-curve worst (475) | 🔶 weakened | 450 | 475 | -25; 1/3 fail |
| Stage 1 hurts SC | ✓ smaller gap | 375 | 375 | 0; tight range |

**Two of four falsified or weakened; one survives with smaller gap;
one fully survives.**

### What the audit kills

- **"Stage 2 specializes."** The bench v0 architectural narrative was
  that Stage 2 trades general performance for state-tracking ability.
  Audit verdict: Stage 2 has no SC advantage *and* a large MQAR
  disadvantage *and* a 33% catastrophic-failure rate. That isn't
  specialization — it's just a worse mechanism.
- **The "convergence" framing of the 22-day project arc.** The project
  map cited bench v0's Stage 2 SC win as the architectural traction
  the d6/5000-iter LM arc had missed. That recovery doesn't exist.
- **The U-curve as evidence for a structured η_init landscape.** The
  worst-point gap shrinks from 100 → 37 steps, and 1/3 failure
  is η-independent. Without bracketing the endpoints, the curve shape
  is indistinguishable from sampling noise.

### What the audit preserves

- **Stage 1's "recall-shaped" framing.** Stage 1's fixed-α memory
  helps recall slightly (small MQAR cost, never bracketed but
  consistent across seeds at SC) and hurts state-tracking
  substantially (~156-step SC gap). The only fully-surviving headline.
- **The δ-mem field result on hard MQAR at 100× scale**
  (`docs/delta_mem_field_result_2026-05-18.html`). Independent
  evidence on different scale and mechanism. Stands.
- **The d6/5000-iter LM verdict** (Stage 2 neutral on natural-language
  val_bpb). Had its own multi-seed validation
  (`docs/hope_nl_stage2_seed_variance_2026-05-04.md`). Stands.
- **The architectural-shape hypothesis** ("memory mechanisms are
  task-mechanism-routing problems, not 'memory yes/no'"). Still
  a live hypothesis. It just lost a lot of its bench v0 d4
  evidence — what remains is the surviving Stage 1 SC result
  (consistent with the hypothesis) and the δ-mem independent
  evidence. Stage 2 evidence is now a net negative for the
  hypothesis, since Stage 2 is just worse rather than
  specialization-shaped.

## Implications for the project map

`docs/project_map_2026-05-20.html` framed the 22-day arc with the
"convergence" node — bench v0 as architectural traction recovery
— central to its spine. The convergence node does not survive
this audit. The "frontier" section listed the four bracketings
that just completed; those are now done, with the result that
the original frontier has shifted.

Proposed revised spine:

> originating question → d6 verdict (neutral) → side-thread
> (δ-mem at 100× scale, partial support) → reframe (selective
> axis as the architectural question worth asking) → **attempted
> convergence (bench v0) → falsified under seed bracketing
> (2026-05-25 SC + 2026-05-28 audit)** → revised frontier
> (mechanism stability is itself the variable to study; n=3
> default; baseline-MQAR bracketing; the surviving Stage 1
> recall-shape signal at d4 vs δ-mem at d≥40).

Adding a banner to the May-20 project map pointing readers at
this audit as the canonical post-falsification source, rather
than rewriting the map in place (preserves the historical
snapshot of the 2026-05-20 view).

## Methodology lessons

This audit is the **third instance** in the last week of n=1 → n=N
collapse on a load-bearing claim:

1. **NTK-Mirror Path A (2026-05-25 morning):** single-seed +1/30
   over LoRA on persona-retention → mean parity after n=3 bracketing.
2. **bench v0 Stage 2 SC (2026-05-25 afternoon):** single-seed 2×
   faster than baseline → ~5% gap inside noise after n=3/n=4.
3. **bench v0 audit (2026-05-28, this writeup):** the "Stage 2
   specializes" narrative emerges from a four-headline picture all
   of which had n=1 vulnerabilities; the architectural story
   doesn't survive bracketing.

The new bench/ default is already wired into the runner
(`--n-seeds N` flag, commit `8f77ab1`). The methodology lesson
is now formalized in
`~/.claude/projects/-Users-matt-projects-new-3p-nanochat/memory/feedback_bench_seed_default.md`.

**Two new specific lessons from this audit:**

- **Catastrophic-failure rate is a property worth measuring
  separately from saturation step.** Stage 2 MQAR's "175 step
  slowdown" headline understates the mechanism's actual behavior;
  "1/3 catastrophic failure" is a load-bearing additional fact.
  Aggregate-only reporting (mean/range) loses this; explicit
  pass-rate reporting is needed alongside the central tendency.
- **Cross-η or cross-arm sweeps need bracketing at every point
  before any "shape" claim.** The U-curve was four interpretable
  points; bracketing two of them already shrank the worst-point
  gap by 60% and made the failure rate identical across positions.
  The shape might still be real, but cannot be claimed from
  n=1 at each.

## Cost ledger

| Run | Wall |
|---|---:|
| Stage 2 SC n=3 (May-25) | 51 min |
| baseline SC n=4 (May-25) | 56 min |
| Stage 2 MQAR n=3 | 67 min |
| Stage 1 SC n=3 | 61 min |
| η=0.5 n=3 | 77 min |
| **Audit total** | **~5.2 h** |

Plus NTK-Mirror Path A bracketing (2026-05-25 morning, ~70 min).
Plus the runner-flag commit (~5 min). Plus writeups (~30 min).
Total session-of-sessions wall: ~7.5 hours of compute + ~1 hour of
human-led decision-and-writeup.

## What's next (suggested)

In priority order:

1. **Update project map.** Banner on the May-20 HTML pointing at this
   audit. ~5 min.
2. **Bracket baseline MQAR n=3.** Makes the Stage 2 MQAR comparison
   fully rigorous and confirms whether catastrophic failure is
   mechanism-specific. ~50 min wall.
3. **Bracket η=0.99** (the U-curve "best" endpoint, n=1 at 325). If
   it lands ≈ 412 (i.e., η=0.1's mean), the U-shape is entirely flat
   and the entire eta_init mechanism story dies. ~58 min.
4. **Open question for the architectural hypothesis:** if Stage 1
   is the only surviving "specialization" result and δ-mem is the
   only independent evidence, what's the next bench-cheap probe
   that could constitute new evidence? Induction-heads on bench/
   was on the May-20 lower-priority list; it might now be
   higher-priority as the third axis.

## Files

- This writeup: `docs/bench_v0_audit_2026-05-28.md`
- HTML companion: `docs/bench_v0_audit_2026-05-28.html`
- Project map banner edit: `docs/project_map_2026-05-20.html`
  (alert at top pointing here)
- Logs (untracked per convention): `bench/logs/{stage2_d4_mqar_hard,stage1_add_d4_sc_hard,stage2_d4_mqar_hard_eta05}_s{0,1,2}.{jsonl,stdout.log}`
- HANDOFF.md: 2026-05-28 addendum

## References

- 2026-05-25 SC falsification:
  `docs/bench_v0_seed_bracketing_2026-05-25.md`
- 2026-05-25 NTK-Mirror Path A (the parallel n=1 → n=N collapse):
  `docs/ntkmirror_persona_comparison_2026-05-25.md`
- Original bench v0 writeups (now superseded by this audit):
  - `docs/bench_v0_stage2_sc_hard_2026-05-19.md`
  - `docs/bench_v0_stage2_mqar_hard_2026-05-19.md`
  - `docs/bench_v0_stage2_eta_sweep_2026-05-19.md`
  - `docs/bench_v0_selective_copy_2026-05-18.md`
- Project map snapshot (needs banner): `docs/project_map_2026-05-20.html`
- δ-mem field result (still standing):
  `docs/delta_mem_field_result_2026-05-18.html`
- Stage 2 d6/5000-iter LM verdict (still standing):
  `docs/hope_nl_stage2_seed_variance_2026-05-04.md`
- Cross-session memory:
  `~/.claude/projects/-Users-matt-projects-new-3p-nanochat/memory/feedback_bench_seed_default.md`
