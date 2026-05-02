# Hope/NL Stage 1 — full d6 pretrain result

**Date:** 2026-05-01
**Branch:** `experiment/hope-nested-learning`
**Stage 1 commit:** `bc54858`
**Run tag:** `d6_stage1`
**Goal:** answer the actual Stage 1 pass/fail question — does the modified architecture (one block's MLP swapped for LinearAttentionMemory at layer 3) reach a val_bpb comparable to the unmodified d6 baseline (val_bpb 1.174) over the full 5000-iter pretrain.

## TL;DR

**Stage 1 final val_bpb = 1.179** vs **baseline 1.174** → **+0.005 bpb (~0.4% relative).** Within single-seed noise. The Hope/NL Stage 1 architecture is **competitive with the MLP baseline at iso-token** on d6.

Loss curve descended smoothly across all 21 evals. No NaN, no instability, no divergence. Wall: 274 min plugged in (vs ~3 h target — slowed by transient thermal events around step 3000–3500).

This is the answer to the trx4mr Stage 1 spec ("does it train, is the loss curve reasonable vs baseline"). It does, and it is.

## Setup

```bash
python -u -m scripts.base_train \
    --depth=6 --head-dim=64 --window-pattern=L --max-seq-len=512 \
    --device-batch-size=32 --total-batch-size=16384 \
    --eval-every=250 --eval-tokens=524288 \
    --core-metric-every=-1 --sample-every=-1 \
    --num-iterations=5000 \
    --save-every=200 \
    --hope-memory-layer=3 \
    --model-tag=d6_stage1 --run=dummy
```

- d6 (n_layer=6, n_embd=384), causal-only attention (`window_pattern=L`)
- accum=1 / total_batch_size=16384 — the validated baseline recipe (Phase 3 step 4 confirmed accum=4 doesn't transfer to pretrain on M2)
- Memory block at **layer 3** (mid-stack of 6)
- 5000 optimizer steps (~81.9M tokens)
- `eval-every=250`, `eval-tokens=524288` — matches the original baseline d6 eval config
- `save-every=200` — pause/resume insurance (commit `245fd09`)
- Both runs from cold init; not a resume

## Result table

| metric | baseline d6 (MLP) | d6_stage1 (LinAttn at L3) | Δ |
|---|---:|---:|---:|
| final val_bpb | 1.174 | **1.179** | **+0.005 (+0.4%)** |
| total training wall | (prior) | 274 min | — |
| n_iters | 5000 | 5000 | — |
| total tokens | 81.9M | 81.9M | — |
| MLP/memory params at modified block | ~1.18M (8·d²) | ~590K (4·d²) | -50% at that block |
| total model params | **73,531,646** (~73.5M) | **72,941,822** (~72.9M) | -589,824 (-0.8%) |

(Param counts confirmed by walking `model.named_parameters()`. ~51% of the d6 model is the three value-embedding tables — `(padded_vocab_size, kv_dim) = (32768, 384)` each, on the alternating layers via `has_ve()`. The transformer blocks themselves are only ~23M; the `lm_head` is another 12.6M untied.)

## Loss trajectory (every eval)

| step | val_bpb | Δ | lrm region |
|---:|---:|---:|---|
| 0 | 3.194 | (init) | warmup |
| 250 | 1.751 | -1.443 | constant |
| 500 | 1.574 | -0.177 | constant |
| 750 | 1.452 | -0.122 | constant |
| 1000 | 1.392 | -0.060 | constant |
| 1250 | 1.356 | -0.036 | constant |
| 1500 | 1.332 | -0.024 | constant |
| 1750 | 1.315 | -0.017 | warmdown begins |
| 2000 | 1.294 | -0.021 | warmdown |
| 2250 | 1.277 | -0.017 | warmdown |
| 2500 | 1.262 | -0.015 | warmdown |
| 2750 | 1.250 | -0.013 | warmdown |
| 3000 | 1.238 | -0.012 | warmdown |
| 3250 | 1.227 | -0.011 | warmdown |
| 3500 | 1.218 | -0.009 | warmdown |
| 3750 | 1.210 | -0.008 | warmdown |
| 4000 | 1.201 | -0.008 | warmdown |
| 4250 | 1.194 | -0.007 | warmdown |
| 4500 | 1.188 | -0.007 | warmdown |
| 4750 | 1.183 | -0.005 | warmdown |
| **5000** | **1.179** | -0.004 | end |

Standard exponential-decay shape, slight bump in `Δ` at step 2000 from early-warmdown LR transition. No divergence, no plateau, no instability.

## Wall analysis

- Total: **274.0 min** (4.57 h)
- Compute (excluding eval): 5000 × ~3.0–3.6 s/iter ≈ **240–300 min**
- Eval (20 evals × ~50 s): **~17 min**
- Saves (25 saves × ~5 s): **~2 min**
- Thermal slowdown around steps 3000–3500: dt briefly spiked to 5.25 s/iter (vs steady ~3.0 s), recovered. Cost ~10–15 min over the run.

**Per-iter wall vs unmodified baseline:** approximately the same. The memory block swap removes ~590K params (4d² vs 8d² at that block) but adds an O(T²) einsum that was previously O(T·d) — at T=512 these roughly balance. No measurable wall regression for the architecture change itself.

## What this confirms

1. **Stage 0 plumbing is sound.** The `memory_state` arg threading didn't break baseline behavior; turning it on (via `hope_memory_layer=3`) trained cleanly.
2. **Stage 1 architecture trains.** Loss descends smoothly through both warmup and warmdown phases. No NaN, no exploding norms, no degenerate plateau.
3. **Stage 1 is competitive at iso-token.** 0.4% relative gap is well within single-seed noise. We can proceed to Stage 2 with confidence the foundation isn't degenerate.
4. **The init choice (W_o = 0) was sound.** The architecture starts bit-identical to baseline at step 0, then diverges as W_o trains away from zero. The trajectory shows no early instability from the swap.

## What this does NOT show

1. **Whether the memory block is actually using its memory.** With α=1, η=1 fixed and the parallel form, the architecture is causal linear attention as an FFN replacement. We haven't probed whether the LinearAttentionMemory block is doing anything fundamentally different from "a soft-attention block trained without softmax." Stage 2's learned gates will start to expose this.
2. **Hope-specific behavior.** No long-context advantage (we're at T=512), no test-time learning probe, no recall benchmarks.
3. **Statistical significance.** Single seed each; the 0.005 gap could be noise or a real small effect either way. A confirmation seed or two would tighten the read.
4. **Downstream task performance.** We didn't run CORE metric (`--core-metric-every=-1`). Baseline's CORE is unknown to us right now; we'd need both numbers for a fair downstream comparison.

## Decisions taken

- **Pretrain recipe locked at accum=1 / 16384 for d6** — same as baseline (Phase 3 step 4 confirmed accum=4 doesn't transfer; SFT recipe still uses accum=4).
- **Memory block at layer 3** — mid-stack of 6, picked as a clean single-block test. Stage 4 will add memory at multiple layers.
- **W_o zero-init kept** — bit-identical at init, divergence emerges naturally during training.
- **No CORE eval during this run** — saves ~30 min of eval time; we can run `base_eval` separately if we want the number later.

## Where things stand

- `~/.cache/nanochat/base_checkpoints/d6/model_005000.pt` — original baseline, val_bpb 1.174 (untouched)
- `~/.cache/nanochat/base_checkpoints/d6_stage1/model_005000.pt` — Stage 1, val_bpb 1.179 (this run)
- 25 intermediate checkpoints (`model_000200.pt` ... `model_004800.pt`) — kept for trajectory analysis or restart-from-step

## Next

1. **SFT on `d6_stage1`** using the validated accum=4 recipe (post `chat_sft` `--num-iterations` fix in `a2d56ef`). Give us a chattable model to qualitatively probe.
2. **Smoke test via `chat_cli`** — does the Stage-1-trained model produce coherent answers? Different style from the MLP baseline?
3. **Reply to Codex with the result** — single-seed, 0.005 gap, "competitive at iso-token, deferring multi-seed and CORE metric."
4. **Stage 2** — learned α/η. The next architectural step where Hope-flavored behavior actually shows up.
