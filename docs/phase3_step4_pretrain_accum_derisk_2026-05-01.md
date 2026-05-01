# Phase 3 step 4 — pretrain accum=4 derisk

**Date:** 2026-05-01
**Branch:** `experiment/hope-nested-learning`
**Goal:** before committing to a long iso-token pretrain validation (Codex's suggested 1250-iter run, ~2.8 h on M2 plugged in), do a cheap 200-iter A1/A4 stub to see whether the SFT amortization win plausibly transfers to base pretrain.

## TL;DR

**Accum=4 is not a free upgrade for pretrain on M2 24GB with the current streaming dataloader.**

- A1 (accum=1, 200 iters, 3.28M tokens): 7.5 min wall, ~7,000 tok/sec.
- A4 (accum=4, 200 iters, 13.1M tokens): 34.6 min wall, ~6,000 tok/sec.
- A4 is **~14% per-token slower** than A1. Opposite direction from the SFT result (where B-iso was per-token faster plugged in).

Decision: **don't run the 1250-iter validation.** Keep pretrain at accum=1 / 16384-token batches as the Hope/NL Stage 1 baseline. The SFT accum=4 win remains real and we'll keep it for the SFT recipe.

## Setup

Two stubs, identical except for `--total-batch-size` (and therefore `grad_accum_steps`). Plugged in for both. ve_gate-on-AdamW change from Phase 3 step 1 in effect.

```bash
# A1
python -m scripts.base_train --depth=6 --head-dim=64 --window-pattern=L \
    --max-seq-len=512 --device-batch-size=32 --total-batch-size=16384 \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
    --num-iterations=200 --model-tag=d6_pretrain_stub_a1 --run=dummy

# A4
python -m scripts.base_train --depth=6 --head-dim=64 --window-pattern=L \
    --max-seq-len=512 --device-batch-size=32 --total-batch-size=65536 \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
    --num-iterations=200 --model-tag=d6_pretrain_stub_a4 --run=dummy
```

Note this is **iso-iter, not iso-token.** A4 sees 4× the tokens at iso-iter. The point of the stub was to detect "comically wrong" behavior cheaply, not to settle the iso-token quality question (that needed Codex's 1250-iter run that we ended up not running).

## Results

### Loss trajectory (iso-iter)

| step | A1 (3.28M tok by step 200) | A4 (13.1M tok by step 200) |
|---:|---:|---:|
| 1 | 10.396 | 10.395 |
| 10 | 10.328 | 10.192 |
| 50 | 7.101 | 6.494 |
| 100 | 6.369 | 5.730 |
| 150 | 6.121 | 5.329 |
| 199 | 5.943 | 5.151 |

A4 ends 0.79 lower at iso-iter, but it processed 4× more tokens. The honest interpretation isn't "accum=4 trains better" — it's "more tokens train better, as expected."

### Inferred iso-token snapshot

At 3.28M tokens (the budget A1 spent in 200 steps):
- A1 step 199: loss 5.94 (200 opt steps consumed)
- A4 step 50: loss 6.49 (50 opt steps consumed)

A1 wins by 0.55 bpb at iso-token in this slice. **Not decisive** because A4 step 50 is still in a different LR/momentum schedule phase (lrm just finished warming up to 1.0; no warmdown has begun) — its 200-step LR schedule isn't the right shape for stopping at step 50. A real iso-token comparison needs both runs ending on a fully-shaped schedule.

Codex's note on this: "accum changes optimizer time constants in token space — Adam/Muon momentum, beta decay, weight decay all happen 4× fewer times. It's not a pure throughput knob."

### Wall and throughput

| run | iters | tokens | wall | dt steady | tok/sec steady |
|---|---:|---:|---:|---:|---:|
| A1 | 200 | 3.28M | 7.5 min | 2.18–2.69 s | ~7,000 |
| A4 | 200 | 13.1M | 34.6 min | 10.07–12.35 s | ~6,000 |

Per-iter ratio is **4.6×**, not the ~1.5–1.7× that pure compute amortization would predict (`T_o + 4·T_f` vs `T_o + T_f` with T_o ≈ 0.79 of total per Phase 2). The optimizer cost we expected to amortize doesn't seem to be a fixed per-iter wall cost — each micro-batch carries ~2.5 s of GPU time on its own, so extra micros mostly add full extra cost rather than slotting into the same optimizer wait.

Per-token: A4 is **~14% slower** (6,000 vs 7,000 tok/sec). Worse than the SFT result, where B-iso plugged in would have been per-token faster.

## Why pretrain looks different from SFT

Best hypothesis: **the streaming dataloader cost dominates with accum>1 in pretrain.**

- Pretrain: `nanochat.dataloader.tokenizing_distributed_data_loader_bos_bestfit` reads parquet shards from disk and tokenizes inline. Each `next(loader)` call does I/O + BPE encoding work.
- SFT: `sft_data_generator_bos_bestfit` operates on an in-memory `TaskMixture` with already-rendered conversations. `next(loader)` is mostly tensor packing.

With accum=4, we call `next(loader)` 4× per opt step. In pretrain, that 4× the disk + tokenize work, sequential on the main thread (no producer/consumer prefetch buffer in this path). The optimizer-amortization speedup gets eaten by data-loader serialization.

This is a **dataloader bottleneck, not a fundamental compute bottleneck.** A pre-tokenized cache or a real producer/consumer prefetch buffer would likely restore the amortization win. Both are real engineering work; not on the critical path for Hope/NL.

## Why A4 quality at iso-token may be worse anyway

Even if we fixed the dataloader, the iso-token quality question is open:

- **Optimizer-update time-constant change.** Adam/Muon momentum (β1, β2), weight decay all happen `1/grad_accum_steps` times in token space. The schedule we tuned implicitly assumes a certain ratio between gradient updates and tokens seen. Pretrain-from-scratch with no warm-start may be more sensitive to this than SFT.
- **No LR scaling applied.** `base_train.py` does have a `sqrt(B/B_ref)` LR scaling rule (chat_sft does not). We disabled it implicitly by passing `--total-batch-size` as an absolute number rather than letting the auto-compute path select it. A "real" accum=4 pretrain comparison would need to verify the scaling actually fires and confirm it's tuned for d6.

This was Codex's "I'm less certain about the **quality per token** win" caveat for pretrain, materializing in the data.

## Decisions made

1. **Don't run the 1250-iter accum=4 pretrain validation.** Wall would be ~2.8 h with no expected throughput win on M2 with the current dataloader, and quality-per-token is uncertain.
2. **Keep base pretrain at accum=1 / 16384-token batch** as the baseline for Hope/NL Stage 0+ comparisons. The existing checkpoint at `~/.cache/nanochat/base_checkpoints/d6/model_005000.pt` (val_bpb 1.174) remains the control.
3. **Keep accum=4 as the SFT recipe.** The SFT win is real and reproducible; just don't transfer the assumption to pretrain without re-validation.
4. **Filed/fixed the `chat_sft.py --num-iterations` bug** (commit `a2d56ef`) as preparation for any future SFT A/B work.
5. **Future diagnostic noted**: synthetic batches vs real dataloader at accum=4 would isolate the dataloader-vs-optimizer split definitively. Cheap (~5 min). Not urgent.

## Codex's framing of the result (paraphrased)

> Accept the conclusion, with one caveat: the quality comparison is inconclusive, but the wall-clock comparison is enough to stop chasing accum for pretrain right now. The result says current real pretrain is not compute-only — synthetic/signpost math predicted optimizer amortization; the live run exposed a data path/tokenizer bottleneck.
>
> Phrase it carefully: "accum=4 is not a free upgrade for pretrain with the current streaming tokenizer/dataloader on M2," not "accum=4 is bad for pretrain." If you later pre-tokenize or add a real producer/consumer prefetch buffer, the compute-side amortization could reappear.

## Status of the perf detour

This closes Phase 3. The full perf arc:

| step | win | cost |
|---|---|---|
| Phase 2 (scalar pinning, alpha→sub_) | ~17% iter wall | resolved |
| Phase 3 step 1 (torch 2.11 + bf16 audit) | ~3% baseline + bf16 unlocks larger batch | filed PR #741 |
| Phase 3 step 2 (GPU profiling) | identified 319 → 176 dispatch lever | informational |
| Phase 3 step 3 (ve_gate → AdamW) | 12% iter wall, 45% dispatch reduction | committed `840d3db` |
| Phase 3 step 3 (SFT accum=4) | ~3× wall savings + 12% val_bpb | recipe change |
| Phase 3 step 4 (pretrain accum derisk) | confirmed accum doesn't transfer | recipe unchanged |

Pivot to Hope/NL Stage 0 next.
