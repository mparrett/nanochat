# Feature: d=8 extension — does Stage 2 architectural delta change with depth?

**Filed:** 2026-05-05
**Source:** Post-A3' synthesis discussion. The synthesis explicitly carved
out "memory architectures may show structural value at scales we cannot
test" as a known caveat to the "Stage 2 doesn't win at d6" verdict. d=8
is the cheapest possible step in that direction without leaving M2 budget.

## Context

A3-prime closed the architectural question at d6/5000-iter on ClimbMix:
modern-recipe vanilla d6 baseline beats Stage 2 on both pretrain val_bpb
(1.1686 vs 1.1712-1.1729) and SFT val_bpb (0.6483 vs 0.6495), all within
the 0.0016 seed-noise spread. **Cannot exclude memory architectures
help at larger scale.**

d=8 doubles the depth dial nanochat is built around (n_embd = 8×64 = 512,
~100-110M params, ~1.7-1.8× compute per iter) and is at the edge of what
M2 24GB can fit. It produces **one data point** on the depth × memory-
architecture interaction — not a curve, but enough to tell us whether the
sign of the gap moves.

## Headline question

Does the Stage 2 architectural delta (relative to vanilla baseline)
change sign or magnitude at d=8 vs d=6?

| outcome | interpretation | next step |
|---|---|---|
| Gap closes / reverses | Memory pathway becomes useful as model has spare capacity. Track revival is justified. | Phase 3 (bracket with seeds, long-context eval) |
| Gap holds at ~current magnitude | Memory architecturally underpowered across scale-range we can test. Track stays wrapped. | Document and close. |
| d8 baseline OOMs / unstable | M2 hardware wall hit; question genuinely outside our budget. | Document and close; revive only at H100/cluster scale. |

## Why this is non-trivial

1. **MPS memory headroom.** d6 used ~13 GB peak. d8 estimated ~22-23 GB
   on a 24 GB box. We will be 1-2 GB from OOM. May require
   `--device-batch-size=16` (vs d6's 32), which halves throughput.
   **Smoke test mandatory before committing budget.**
2. **Single data point.** d=8 Phase 2 alone gives n=1 baseline + n=1
   Stage 2 at the new scale. Doesn't bound d8 seed-noise. Phase 3 (n=2
   each) is required to firm up any directional signal.
3. **No d8 historical baseline.** Per the recipe-drift lesson (track
   synthesis), all comparisons must be re-baselined at d8 with current
   `master`. Cannot reuse d6 numbers as proxies.
4. **SFT mix doesn't reward memory.** MMLU/GSM8K-heavy SFT is the wrong
   evaluation for the memory hypothesis even at d8. SFT comparison may
   stay neutral even if Stage 2 wins on pretrain val_bpb.
5. **Probe-first discipline applies.** Per ADR-001, MQAR must saturate
   at d8 init before any pretrain budget is committed. Cheap (~30 min).

## Proposed sequence

### Phase 0 — d4 methodology rig (~2-4h, optional but recommended)
**Goal:** validate the experiment harness at a cheap scale before
committing d8 budget. Catch instrumentation bugs and bound baseline-
seed variance at a depth where n=3 is affordable.

**Why d4 specifically:**
- Memory layer at index 2 of 4 is the topological analogue of d6's
  index 3 of 6 (both at 50% through depth). Conclusions transfer
  cleanly to d6 / d8 in a way d3's layer-1-of-3 (33% through depth,
  heavily structural) does not.
- Wall time ~60-90 min per pretrain — n=3 baseline + n=3 Stage 2
  fits in ~6-9h, the n=3 baseline-seed bound that Codex called too
  expensive at d6.
- Architecturally representative: d4 asks the same architectural
  question as d6 and d8, just at smaller width and fewer layers.

**Procedure:**
- Run d4 baseline pretrain × n=3 (seeds 42, 1, 2) and d4 Stage 2
  pretrain × n=3. Measure baseline-seed spread + Stage 2-vs-baseline
  delta vs that spread.
- Side-test instrumentation: `--inherit-from` carries through
  `--depth=4` override correctly; qk_norm A/B wire in cleanly (per
  `feat_modernization_alignment.md::A2`); save-keep-last-n behaves;
  smoke-test the seed-noise plot/table generation.
- **Pass criteria:**
  - Harness runs end-to-end without modification.
  - n=3 seeds at d4 produce a measurable spread (any number; just
    confirms machinery works).
  - Stage 2 vs baseline delta sign at d4 is informative — either
    bounds are tight enough to make a claim, or claim is "neutral
    within bounds."

**What d4 doesn't tell us:** whether Stage 2 helps at d8. d4 result
should be treated as bounds-firming for d6 (a smaller-but-topologically-
analogous re-test), not as a prediction for d8. The architectural
question still requires d8 itself.

#### When d3 might come in handy instead

d4 is the right call for *result-bearing* derisk because of topological
representativeness. **d3 is ~30 min faster per run but trades that for
architectural distortion** (memory layer is 33% of network at d3 vs 16%
at d6, so the pathway is structurally over-weighted).

d3 is useful **only for pure plumbing/harness tests** where we don't
care if conclusions transfer to d6/d8 — for example:
- Validating new instrumentation (a new logging hook, a new probe
  variant) that has nothing to do with the architectural question.
- Iterating on the experiment-runner / wandb integration / seed-loop
  bash.
- Burn-in tests of new features before they touch result-bearing runs.

If a future need arises for fastest-possible iteration on
infrastructure, d3 is the answer. For any architectural-question
work, skip d3 and go to d4.

### Phase 1 — d8 derisk (~1-2h)
**Goal:** confirm M2 can run d8 at all, and Stage 2 mechanism still
saturates the probe at d8 init.

- Re-run `dev/probe_mqar.py` at d8 with: baseline, Stage 1-additive
  (W_o=1.0), Stage 2 (learned gate). Probe gate per ADR-001:
  saturation at step ≤ ~100.
- Quick smoke-pretrain at d8 (e.g. 500 iters, both architectures) to
  confirm no OOM and measure wall-time per iter at the chosen
  `--device-batch-size`.
- **Pass criteria:**
  - Stage 2 MQAR saturation ≤ baseline + 25 steps.
  - No OOM at chosen batch size.
  - Wall-time per iter projects to ≤ 12h for full pretrain (else
    re-evaluate).

If any pass criterion fails, abort the d8 line and document why.

### Phase 2 — headline comparison (~10-15h)
**Goal:** measure pretrain + SFT val_bpb gap at d8.

- d8 baseline pretrain (vanilla, modern recipe, seed=42).
  Hand-written CLI flags (no `--inherit-from` from a Stage 2 meta;
  same approach as A3-prime).
- d8 Stage 2 pretrain (additive learned-gate, current Stage 2 priors:
  `w_o_init_scale=1.0`, `alpha_init_bias=4.595`,
  `eta_init_bias=-2.197`, `additive_memory_layer=4` — center-of-stack
  per Stage 1.5 convention adapted to d8). Use `--inherit-from` from
  the current d6 Stage 2 meta with `--depth=8` override.
- Same SFT recipe applied to both (`--inherit-from` from
  `chatsft_checkpoints/d6_stage2_s1/meta_000375.json`, override only
  per-run fields).
- Compare both pretrain val_bpb @ step 5000 and SFT val_bpb @ step 375.

### Phase 3 — bracket (conditional, ~10-15h)
**Goal:** firm up directional signal from Phase 2.

Only run if Phase 2 shows movement worth investigating:
- d8 baseline + Stage 2 each with a second pretrain seed.
- Long-context evaluation per `data_investigations.md::Q1` —
  length-stratified val_bpb. Tests whether any d8 architectural delta
  concentrates on long documents (where memory has time to accumulate).

If Phase 2 shows neutral gap (within Phase 1's measured d8 seed-noise
proxy), document the result and close the line; do not invest Phase 3.

## Open questions to settle before launch

- **Memory layer placement at d8:** Stage 2 used
  `additive_memory_layer=3` at d6 (middle of stack). For d8, layer 4 is
  the analogous middle-of-stack position. Worth checking whether
  Behrouz et al. specify, or whether layer placement should sweep.
  Probe-scale answer in Phase 1 should resolve this cheaply.
- **Save policy for d8:** checkpoints are ~1.4 GB at d8 (model 500 MB
  + optim 900 MB). With `--save-keep-last-n=2` the dir caps at ~2.8 GB.
  Across 4 runs (baseline + Stage 2, both phases) that's ~11 GB peak.
  Disk needs verification before launch (currently ~37 GB free).
- **`--target-param-data-ratio`:** the depth-sweep machinery may want
  to auto-scale num_iterations at d8 (more params → more tokens for
  Chinchilla). Need to decide: keep iters=5000 fixed for parity (will
  under-train d8), or let it scale (changes the experiment)? Reasonable
  default: keep iters=5000 so depth is the only variable, but document.

## What this doesn't answer

- Memory at d12+ scale (paper's regime). Outside M2 budget regardless.
- Whether memory helps with longer context (max_seq_len=512 stays).
- Whether Stage 4 (multi-block memory) becomes worthwhile at d8.
  That's a separate, larger experiment line; only motivated if Phase 2
  here shows positive delta.

## Estimated total cost

| phase | wall (M2) | new disk | conditional? |
|---|---:|---:|---:|
| 0 (d4 rig) | 2-4 h | ~2 GB | optional, recommended before Phase 1 |
| 1 (d8 derisk) | 1-2 h | <1 GB | always run first |
| 2 (d8 headline) | 10-15 h | ~6 GB | only if Phase 1 passes |
| 3 (d8 bracket) | 10-15 h | ~6 GB | only if Phase 2 shows movement |
| **Total floor** (Phase 0+1+2) | **13-21 h** | **~9 GB** | — |
| **Total ceiling** (all phases) | **23-36 h** | **~15 GB** | — |

Spread across 3-5 sessions.

## Pre-flight checklist (when picked up)

- [ ] Confirm disk ≥ 15 GB free (Phase 1+2 floor).
- [ ] Confirm no orphan MPS workers (`pgrep -lf python`, `pgrep -lf wandb`).
- [ ] Phase 1 probe: edit `dev/probe_mqar.py` to allow `--depth=8` if
      not already supported.
- [ ] Decide `--device-batch-size` for d8 — start at 16, try 32 in
      smoke if margin allows.
- [ ] Decide memory-layer placement at d8 (probably 4; confirm via
      probe).
- [ ] Pin recipe + seed for both architectures.
- [ ] Re-baseline at d8; do not compare against d6 numbers.
