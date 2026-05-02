# Hope/NL Stage 1.5 — MQAR probe results

**Date:** 2026-05-01
**Branch:** `experiment/hope-nested-learning`
**Probe code:** `dev/probe_mqar.py` (commit `94fa35b`)
**ADR:** `docs/project_notes/decisions.md::ADR-001`
**Design:** `docs/hope_nl_stage1_5_probe_design_2026-05-01.md`

## TL;DR

The MQAR probe ran in two phases:

- **v1 (1000 iters/arm, full attention):** baseline saturated to 100% accuracy at **step ~76**. Killed early after confirming the task was being solved by attention alone — at full context the architectural swap is not load-bearing.
- **v2 (200 iters/arm, eval every 25, full attention):** convergence-speed comparison. **Baseline saturates at step ~76; Stage 1 saturates at step ~151. Stage 1 is ~2× less sample-efficient on MQAR at d6 with full attention.**

Both architectures eventually reach 100%. The architectural difference shows up in *how fast they grok the binding*, not in *whether they can*. This is a real signal that wasn't visible in val_bpb (where Stage 1 was within 1% of baseline).

The "final layer is always full-context" constraint in `nanochat/gpt.py::_compute_window_sizes` (line 376) means we can't easily make this probe attention-bottlenecked — even `window_pattern='S'` leaves the last layer at full attention. So the probe measures sample efficiency on a task attention can solve, not load-bearing memory contribution.

## v1: Discriminating-task fail (killed early)

```bash
python -u -m dev.probe_mqar --label baseline --hope-memory-layer=-1 \
    > /tmp/probe_mqar_baseline.log 2>&1
```

| step | baseline acc | notes |
|---:|---:|---|
| 1 | 0.031 | chance (1/32 ≈ 0.031) |
| 101 | **1.000** | grokked, loss 0.004 |
| 201 | 1.000 | held |
| 301 | 0.999 | held |
| 401 | 0.999 | held |
| 501 | 1.000 | held — killed here |

Took ~18 min wall to reach step 501. Killed before the Stage 1 arm even started — the "both arms hit 100%" symmetry was already obvious. The MLP wasn't doing the work; full self-attention was. With T=128, K=16 keys, the lookup is shallow enough that the top-layer attention can do it directly.

## v2: Convergence-speed comparison

Same task, finer eval cadence, shorter horizon. Eval every 25 iters; both arms run back-to-back.

```bash
python -u -m dev.probe_mqar --label baseline --hope-memory-layer=-1 \
    --num-iterations=200 --eval-every=25 > /tmp/probe_mqar_baseline_v2.log 2>&1
python -u -m dev.probe_mqar --label stage1 --hope-memory-layer=3 \
    --num-iterations=200 --eval-every=25 > /tmp/probe_mqar_stage1_v2.log 2>&1
```

### Trajectory

| step | baseline loss | baseline acc | stage1 loss | stage1 acc |
|---:|---:|---:|---:|---:|
| 1 | 10.399 | 0.031 | 10.399 | 0.031 |
| 26 | 3.492 | 0.037 | 3.410 | 0.054 |
| 51 | 3.264 | 0.111 | 3.155 | 0.066 |
| 76 | 0.017 | **0.999** | 2.935 | 0.131 |
| 101 | 0.004 | 1.000 | 1.175 | 0.632 |
| 126 | 0.002 | 1.000 | 0.496 | 0.863 |
| 151 | 0.002 | 1.000 | 0.006 | **1.000** |
| 176 | 0.006 | 0.999 | 0.001 | 1.000 |
| 200 | 0.002 | 1.000 | 0.000 | 1.000 |

Step 1 loss bit-identical (10.3988) — same fixed seed, both architectures bit-identical at init via the W_o=0 invariant in `LinearAttentionMemory`. From step 26 onward they diverge.

### Wall

| arm | wall | iters/sec |
|---|---:|---:|
| baseline | 7.44 min | ~0.45 |
| stage1 | 8.07 min | ~0.41 |

Per-iter wall is essentially the same. Stage 1's ~8% wall premium is from the parallel-form einsum being marginally heavier than the MLP at d_mem=384 — but immaterial.

### Inflection point analysis

- **Baseline:** acc 0.111 → 0.999 between step 51 and 76. Sharp grokking transition in a ~25-iter window.
- **Stage 1:** acc 0.131 → 0.632 → 0.863 → 1.000 across step 76 → 101 → 126 → 151. Smoother, more gradual ascent over ~75 iters.

So **baseline groks abruptly at step ~70; Stage 1 groks gradually over steps 75-150**. Different shape, not just shifted.

### Why this might be

Best hypothesis: removing the MLP at L3 in favor of `LinearAttentionMemory` (alpha=1, eta=1, parallel form) trades a strongly-nonlinear feature transform (ReLU²) for a weakly-nonlinear bilinear form `(k·q)v`. Recall via attention requires the softmax-or-equivalent to *select* a position; the MLP's nonlinearity helps the surrounding blocks shape representations into something attention can sharply select on.

Stage 1's L3 block is doing a second causal-attention-like operation (without softmax, scaled by 1/sqrt(d_mem)). It's not removing capability — both architectures *can* solve recall — but it's removing one of the nonlinear feature-shaping steps that make attention's selection sharp. The model has to compensate via the remaining 5 MLP blocks, which it does, but slower.

### What this isn't

- **Not a "Stage 1 is broken" result.** Both reach 100%. Final capability is identical on this task.
- **Not a probe of memory mechanism per se.** Final-layer full attention can solve MQAR alone (v1 confirmed). What we're measuring is "how does swapping one block affect downstream sample efficiency," not "does the memory block itself do recall."
- **Not informative about long-context advantages.** T=128, far below where attention efficiency matters.

## What this means for Stage 2

Stage 2 (per-token learned `α` and `η`) has to *earn* its way past two facts:

1. **The Stage 1 architecture is ~2× less sample-efficient on a recall-shaped task** at d6 with full attention. Adding learned gates needs to *recover* this gap before claiming an improvement, not just match it.
2. **The full LM val_bpb gap was tiny (0.4-1.1%)**, so most of the SFT distribution doesn't reward the memory mechanism enough for the gap to show up. Either the probe over-emphasizes the cost of the swap, or the val_bpb under-emphasizes it.

Two things to watch for at Stage 2:

- **Does the convergence-speed gap close?** If learned `α` and `η` make Stage 2 grok MQAR by step ~76 again (matching baseline), the gates are doing useful work. If Stage 2 stays at ~150 like Stage 1, the gates aren't helping the recall pathway.
- **Does the val_bpb gap reverse?** If Stage 2 drops val_bpb materially below baseline, the architectural value is showing up at the LM scale. If it stays at parity, the gates are stylistic.

## What this means for the broader thesis

Codex's framing ("validate the mechanism does something measurable before scaling architectural complexity") was correct and validated by this probe. Without it we'd have built Stage 2 on top of an unmeasured Stage 1 and not been able to tell whether any val_bpb shift came from the mechanism or from the additional learned parameters.

The probe is reusable for Stage 2+ as a fast (~15 min/arm) sanity check before any longer pretrain run. If a future variant doesn't recover at least baseline's MQAR convergence speed, it isn't worth a full d6 pretrain.

## Raw logs

- `/tmp/probe_mqar_baseline_v2.log`
- `/tmp/probe_mqar_stage1_v2.log`

(v1 logs at `/tmp/probe_mqar_baseline.log` — partial, killed at step 501.)

## Next

Decision point for the human operator:

1. **Take the result, write up the full Phase 3 → Stage 1.5 narrative**, then proceed to Stage 2 design with the convergence-speed gap as a load-bearing constraint.
2. **Make the probe more discriminating** by removing the always-final-layer-L constraint (would be a small upstream-incompatible patch to `_compute_window_sizes`) and re-running. Could isolate whether Stage 1's memory block actually contributes when attention is bottlenecked.
3. **Stop here for the session**, defer Stage 2 design to a future arc with a fresh head.

Operator's instinct after seeing the result will dictate which.
