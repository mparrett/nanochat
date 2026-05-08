# d3 binary vs d3 ternary vs d3 fp32 — Phase 2 quant validation on M2

**Date:** 2026-05-08
**Verdict:** At d3 / 1500 iter, **ternary STE lands at val_bpb 1.470 (+6.8 %
over fp32), binary at val_bpb 1.513 (+9.9 % over fp32)**. Both arms are in
the "marginal" bucket (1.45–1.65) of the falsification ladder, but ternary
is meaningfully closer to the "STE works" green-light zone (≤+5 %, =1.45).
The alphabet matters: ternary's 3-level `{−s, 0, +s}` buys ~3 percentage
points of val_bpb over binary's 2-level `{−s, +s}` for ~10 % extra wall —
a clear win on quality-per-FLOP. Both arms were still descending at step
1500 (final 100-step Δ ≈ −0.007 for both), so the d3-1500-iter horizon is
not converged. STE training itself is unambiguously functional on real
data at this scale; the open question is whether more iterations / depth
/ knob-tuning can close the remaining gap to fp32.

---

## Why this experiment

Phase 1 of the 1-bit-from-scratch direction
(`docs/project_notes/backlog.md`) landed `nanochat/quant.py` —
`BinaryLinear` / `TernaryLinear` + `apply_quant` walker — and the smoke
on a synthetic d6-shaped overfit batch passed (Phase 1 commits 2b85b82,
1547303, fbf29be). Phase 2 wires `apply_quant` into the canonical
pretrain script (`scripts/base_train.py` commit 0471cea) and tests
whether STE training produces a sensible model on the canonical dataset
at a real depth.

We chose **d3** as the cheapest falsifier (~33 min M2 wall for the fp32
baseline; ~45 min for binary; same for ternary) before committing to a
~3-4 h d6 run. The d3 baseline `d3_smoke` (val_bpb 1.3766 at step 1500)
is documented in `docs/d3_smoke_results_2026-05-05.md` and serves as
the apples-to-apples comparison.

Tentative integration choices captured in **ADR-006** (CLI surface,
optimizer routing kept as Muon, escape hatches kept fp at lm_head /
wte / value_embeds / smear / ve_gate, recipe inheritance from
`d3_smoke`).

A late-stage strategic clarification — that STE binary training does
*not* save memory, only inference does — is captured in **ADR-007**.
The active runs are still meaningful regardless of memory framing; the
quality question is independent.

## Setup

Identical recipe to `d3_smoke` via `--inherit-from`, with single delta
per arm:

```
depth=3  aspect_ratio=64  head_dim=64
n_layer=3  n_head=3  n_embd=192   (~10M params)
num_iterations=1500  total_batch_size=16384  device_batch_size=32
max_seq_len=512  weight_decay=0.28→0.715 (depth-scaled)
warmup_steps=40  warmdown_ratio=0.65  final_lr_frac=0.05
core_metric_every=-1  eval_every=100  eval_tokens=524288  seed=42
```

Single delta versus baseline:
- d3_binary_validate: `--quant=binary --quant-group-size=128`
- d3_ternary_validate: `--quant=ternary --quant-group-size=128`

`apply_quant` walker swaps **18 modules** in each arm (3 layers ×
[c_q, c_k, c_v, attn.c_proj, c_fc, mlp.c_proj]). lm_head, wte,
value_embeds, smear gate, ve_gate are deliberately untouched per the
Phase 2 escape-hatch contract (ADR-006).

Compute dtype: fp32 throughout (M2 default; `NANOCHAT_DTYPE` unset).
The bf16-on-M2 path was validated separately at d6 (commit 129c219)
but stayed orthogonal here — we wanted the binary axis isolated, not
binary × bf16 × M2 confounded.

`--model-tag=d3_binary_validate / d3_ternary_validate` — neither
overwrites the canonical `d3_smoke` baseline.

## Headline numbers

| metric | fp32 (`d3_smoke`) | binary (this run) | ternary (this run) |
|---|---|---|---|
| val_bpb @ step 1500 | **1.3766** | **1.5134** | **1.4701** |
| Δ vs fp32 (nats) | — | +0.1368 | +0.0935 |
| Δ vs fp32 (%) | — | +9.93 % | **+6.79 %** |
| Δ vs binary (nats) | — | — | −0.0433 |
| Δ vs binary (%) | — | — | **−2.86 %** |
| total_training_time (s) | 1973.16 | 2613.74 | 2884.49 |
| total_training_time (min) | 32.9 | 43.6 | 48.1 |
| per-iter wall (s) | 1.315 | 1.742 | 1.923 |
| Δ wall vs fp32 (%) | — | +32.5 % | **+46.2 %** |
| Δ wall vs binary (%) | — | — | +10.4 % |
| `apply_quant` modules | 0 | 18 | 18 |
| latent params quantized | 0 | 1,327,104 | 1,327,104 |

(ADR-007's quoted "+16 % wall penalty" was a mid-run reading from steps
~280-300; the integrated whole-run number is +32.5 %. ADR-007 patched
in the same commit as this writeup.)

## val_bpb trajectory

Both arms eval at every 100 steps. Same seed, same init, same recipe — the
only delta is `--quant=binary` vs `--quant=ternary`.

| step | binary | ternary | gap (binary − ternary) |
|---|---|---|---|
| 0 | 3.1944 | 3.1945 | +0.0001 |
| 100 | 1.9895 | 1.9891 | +0.0005 |
| 200 | 1.8620 | 1.8341 | +0.0279 |
| 300 | 1.7765 | 1.7529 | +0.0236 |
| 400 | 1.7226 | 1.7022 | +0.0204 |
| 500 | 1.6840 | 1.6635 | +0.0204 |
| 600 | 1.6526 | 1.6301 | +0.0225 |
| 700 | 1.6267 | 1.6019 | +0.0248 |
| 800 | 1.6043 | 1.5766 | +0.0277 |
| 900 | 1.5868 | 1.5526 | +0.0342 |
| 1000 | 1.5705 | 1.5319 | +0.0386 |
| 1100 | 1.5547 | 1.5136 | +0.0411 |
| 1200 | 1.5420 | 1.4987 | +0.0433 |
| 1300 | 1.5309 | 1.4868 | +0.0441 |
| 1400 | 1.5206 | 1.4773 | +0.0433 |
| 1500 | 1.5134 | 1.4701 | +0.0433 |

Three observations:

1. **Clean monotonic descent in both arms.** No oscillation, no plateau-then-
   recover. STE backward is delivering meaningful gradients at d3 / 10M
   params on the canonical corpus, for both alphabets. The Phase 1 CPU
   smoke prediction holds at real training scale.

2. **The alphabet matters, but only after step 100.** Steps 0/100 are
   essentially identical between arms (within 0.0005 nats). The gap opens
   in the 100→200 window (binary descends -0.128, ternary descends -0.155
   — a 21 % faster descent for ternary in that interval). After step 200
   the absolute gap holds in the 0.02-0.03 range until step 700, then
   widens through the rest of training to settle near 0.043.

3. **Sample efficiency advantage.** Ternary reaches val_bpb 1.5136 at step
   1100 — essentially identical to binary's *final* val_bpb 1.5134 at step
   1500. Ternary is **~27 % more sample-efficient** at this horizon for
   matching binary's final quality. If the project goal is "match a
   target val_bpb in fewest M2-minutes", ternary wins on both quality
   *and* throughput-adjusted-for-quality.

4. **Neither arm is converged.** Final 100-step Δ is −0.0072 for both.
   Naïve extrapolation suggests both arms would still be descending at
   step 3000. Whether either closes the gap to fp32 (1.377) given more
   iterations is the most natural open question.

## Sample completions @ step 1500

The binary arm's final sample-generation output (typical of d3-scale
models — fp32 d3_smoke shows similar looping behaviour at this scale):

```
The capital of France is a country of the country's country, and the country's country's country's
The chemical symbol of gold is a chemical reaction that is a chemical reaction that is the reaction of the reaction.
If yesterday was Friday, then tomorrow will be the first time to get the first time to get the first time to get the
The opposite of hot is a common cause of the cold weather. The cause of cold weather is the cause
The planets of the solar system are: the solar system, the solar system, the solar system, the solar system,
My favorite color is a color that is a color that is a color color. It is a color
If 5*x + 3 = 13, then x is 1xx 1xx 1xx 1xx
```

These are not informative about binary-vs-fp quality at this scale —
both arms sample gibberish. val_bpb is the load-bearing metric.

## Memory observation

Wandb captured MPS metrics during the binary run:
- `mps/cache_gb` peak: **11.67 GB**
- `mps/driver_gb` peak: **12.10 GB**
- `mps/recommended_max_gb`: 19.07 GB (M2 safe ceiling)

We don't have a matched fp32 d3_smoke MPS reading to compare, so the
absolute number isn't an apples-to-apples delta — but it confirms ADR-007's
prediction qualitatively: training memory under STE binary is not 14×
smaller than fp32. It's the same order of magnitude. The 14×
memory prize is realised only at inference via packed-weight serialization,
which we don't implement.

## Interpretation: what does "marginal" mean here?

The result lands in the awkward middle of the falsification ladder.
Multiple plausible explanations are not yet ruled out:

1. **Fundamental** — binary's 1-magnitude alphabet is too coarse to
   represent the gradients the loss landscape demands at this scale.
   Test: compare against ternary (3 levels). If ternary closes the gap,
   the alphabet matters; if not, the gap is from STE itself.

2. **Sample-inefficient STE** — the model is still descending at step
   1500. If continued, binary may close the gap. Test: extend to 3000+
   iterations.

3. **Architectural ceiling at d3** — d3 only has 3 layers; STE-induced
   error per layer is large relative to compensatory capacity. d6 might
   close the gap proportionally. Test: d6 binary at 1500 iter.

4. **Optimizer mismatch** — Muon's orthogonalization on STE gradients
   may be suboptimal vs AdamW. Test: AdamW on matrix params under
   `--quant=binary`.

5. **Group size too coarse** — 128 weights per fp scale may over-quantize
   at this scale. Test: `--quant-group-size=32` or 64.

The d3 ternary run (in flight) tests hypothesis 1 directly. If ternary
beats binary by a meaningful margin at d3, the "alphabet matters" axis
opens up and ternary becomes the pretrain candidate. If ternary tracks
binary, the gap is from STE-with-low-bits at d3, and we focus on
hypotheses 2-5.

## Interpretation: what does this tell us?

Mapping the result onto the hypothesis menu I sketched mid-discussion:

1. **Fundamental binary-alphabet limit at d3 — partly confirmed.** Ternary
   buys 0.043 nats over binary (3.1 % of fp32 baseline) for 10 % extra
   wall — clear signal that the alphabet's representational capacity
   matters at this scale. NOT the only effect though; even ternary is
   1.7 % above the +5 % "STE works" line.

2. **Sample-inefficient STE — partly confirmed.** Both arms descending
   at end → more iterations would close some gap. We don't know how much
   without running it.

3. **Architectural ceiling at d3 — untested.** d6 might close the gap
   proportionally if depth helps STE.

4. **Optimizer mismatch (Muon vs AdamW) — untested.** Could explain part
   of the gap if Muon's orthogonalization is suboptimal on STE gradients.

5. **Group size too coarse — untested.** Smaller groups would give more
   scale-encoding granularity.

The d3 ternary result is the most-positive single data point we have:
within +6.8 % of fp32 baseline, with a still-descending trajectory and a
sample-efficiency advantage over binary. It's a stronger green-light than
binary was; it's not yet a green-light to commit to the canonical d6
binary 5000-iter run.

## Next steps — tree

The cheapest highest-value next experiment depends on what question we
care about most:

- **A. d6 ternary 1500 iter (~76 min M2).** Tests hypothesis 3 directly
  with the better alphabet. We have d6_baseline_modern's wandb history
  to extract a step-1500 fp32 val_bpb for matched-iteration comparison
  (the saved meta only has step 5000, but wandb logs every 250 steps).
  If d6 ternary lands proportionally closer to fp32 d6 than d3 ternary
  did to fp32 d3, "depth helps STE" is supported and the d12+ inference
  thesis (per ADR-007) gains traction.

- **B. d3 ternary continue 1500→3000 (~48 min M2).** Tests hypothesis 2
  with the better alphabet. If extending closes the gap to fp32, "more
  iterations" becomes a known knob to apply at d6.

- **C. d3 ternary group_size=32 (~48 min M2).** Tests hypothesis 5 with
  the better alphabet. Cheap.

- **D. d6 binary 1500 iter (~76 min M2).** Tests hypothesis 3 with the
  worse alphabet. Less informative than A given what we now know about
  the alphabet axis.

- **E. Wrap up at d3, declare a clean two-data-point story.** No more
  M2 burn; reflects what the data actually licences us to say.

My read: **A (d6 ternary)** dominates the menu for the operator's
stated goal of "we eventually want d6". It tests depth and uses the
better alphabet — both axes we'd want at d6 anyway. ~76 min is
acceptable. The alternative B/C are cheaper but don't move us toward d6.

If A happens, a pretty d3-vs-d6 chart drops out and the writeup
becomes "binary vs ternary at d3, ternary at d6 (with proportional
fp32 reference) — the depth axis is informative or not".

## Status

- d3 binary: completed 2026-05-08 09:48 PDT. val_bpb 1.5134.
  `~/.cache/nanochat/base_checkpoints/d3_binary_validate/meta_001500.json`.
- d3 ternary: completed 2026-05-08 11:53 PDT. val_bpb 1.4701.
  `~/.cache/nanochat/base_checkpoints/d3_ternary_validate/meta_001500.json`.
- ADR-007 wall-penalty number patched alongside this writeup
  (16 % → 32.5 % for binary; 46 % for ternary).
- Next step pending operator decision (A/B/C/D/E above).

## References

- `docs/project_notes/decisions.md` ADR-006 / ADR-007 — integration
  choices and memory framing.
- `docs/project_notes/backlog.md` — 1-bit-from-scratch entry, including
  Phase 2 falsification thresholds and the memory-accounting clarification.
- `docs/d3_smoke_results_2026-05-05.md` — fp32 d3 baseline writeup.
- `docs/bf16_d6_validate_2026-05-07.md` — bf16-on-M2 prior precedent
  (parity-preserving precision reduction at d6).
- `nanochat/quant.py` — port of trx4mr `BinaryLinear`/`TernaryLinear`
  with master-fp32 / matmul-in-x.dtype convention.
- `dev/smoke_quant_d6.py` — Phase 1 CPU smoke (overfits a fixed batch,
  validates STE backward end-to-end).
- wandb runs: `d3_binary_validate` (2g5bz39i), `d3_ternary_validate`
  (TBD on completion).
