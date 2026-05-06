# d3 smoke run — results

**Date:** 2026-05-05 (~17:30 launch → 18:45 chat smoke complete)
**Recipe:** `docs/d3_smoke_recipe_2026-05-05.md` Recipe A (vanilla d3, no memory)
**Branch:** `experiment/hope-nested-learning`
**Purpose served:** harness validation + entertainment chatbot.

## Headline numbers

| metric | d3 | d6 reference (d6_baseline_modern) | gap |
|---|---:|---:|---:|
| Pretrain val_bpb @ final | **1.3766** (step 1500) | 1.1686 (step 5000) | +0.21 |
| SFT val_bpb @ final | **0.7753** (step 375) | 0.6483 (step 375) | +0.13 |
| Pretrain wall time | **32.9 min** | ~4 hours | — |
| SFT wall time | **29.5 min** | ~60 min | — |
| Total wall (cold → chatbot) | **~63 min** | ~5h | — |

The +0.21 / +0.13 val_bpb gap is reasonable: d3 is 1/8 the params trained
on 1/5 the tokens. Slightly worse text-modeling, exactly as expected.

## Wall-time recalibration (recipe estimate vs reality)

The recipe doc estimated 25-30 min pretrain + 12-18 min SFT = ~45 min total.
Reality: 32.9 + 29.5 = **62.4 min**, or ~40% over the estimate.

- **Pretrain came in at the high end of the estimate band** (32.9 vs
  25-30 estimate). MPS dispatch overhead at d3 is essentially the same as
  d6, so the per-iter wall doesn't drop in proportion to model size.
- **SFT was the surprise** (29.5 vs 12-18 estimate, ~70% over). At d3 with
  total_batch_size=65536 and accum=4, the per-step cost is dominated by
  the four micro-batch fwd/bwd passes. Reducing model size doesn't help
  proportionally; it's batch-size limited, not model-size limited.

**Updated estimate for any future d3 smoke run on this M2: ~60-65 min total.**

## Sample chat exchanges

All three prompts via `uv run python -m scripts.chat_cli -g d3_smoke_sft -p "..."`.
Verbatim output, top-of-the-stream:

### Q: "What is the capital of France?"
> "France is the capital of France. It is the capital of France. France is
> Paris. Paris is a city known for its historical landmarks and historical
> landmarks. Paris is the most iconic landmarks in the city is the largest
> city in the city. Paris is a city known for its historical landmarks and
> cultural institutions."

The factual association (Paris ↔ France) is *there* but the model can't
structurally get it out without false starts. Sees "France" → emits
"France is the capital of France" → eventually self-corrects to
"France is Paris" → degenerates into landmark-loop.

### Q: "Spell 'banana' letter by letter."
> "leana' letter by letter.:l,e,a,n,a,l,i,n,g,e, letter."

Drops the leading "B" entirely (`leana`), but does pick up the
letter-by-letter format from SFT. Output: `l,e,a,n,a,l,i,n,g,e` —
positions 2-5 (`e,a,n,a`) of "banana" are right; everything else drifts.
The SpellingBee training fingerprint is faintly visible but the model
lacks the capacity to actually spell.

### Q: "Once upon a time"
> "The time of a time is taken to a time to find a time spent in time.
> It takes a time to start, time, and minutes, and minutes to minutes...
> [the word 'minutes' then repeats ~50 times, filling the context]"

Pure repetition death-spiral. Locks onto "minutes" and runs it into a wall.
Classic small-LM failure mode. Pure entertainment.

## What this validated about the harness

**Durable wins beyond the one-time run:**

1. **`--inherit-from` works with `--depth=` override.** 34 fields auto-
   loaded from `d6_baseline_modern/meta_005000.json` (head_dim=64,
   max_seq_len=512, window_pattern=L, all LRs/warmup/decay), `--depth=3`
   on the CLI overrode just that one. Same mechanism worked end-to-end
   for SFT (inheriting from `d6_baseline_modern_sft/meta_000375.json`).
   Proves the A3 launch refactor pattern is portable to other depths.

2. **Direct-python launch is the M2 canonical pattern.** `torchrun` on
   MPS injects DDP env vars but `compute_init` only initializes the
   process group on CUDA → optimizer crashes at `dist.get_rank()` on
   the first step. The fix is `uv run python -u -m scripts.base_train`
   (no torchrun). Both the recipe doc and `docs/project_notes/key_facts.md`
   could carry this lesson; it's not currently spelled out as a do/don't
   anywhere auto-loaded. **Followup:** decide whether to surface this in
   `key_facts.md`.

3. **Chained pretrain → SFT in one nohup block works cleanly.** The
   `&& echo "===== PRETRAIN OK =====" && ... chat_sft` pattern fired the
   transition exactly when expected; no zombie state, no orphans. Useful
   for any future overnight d3/d4/d8 launches.

4. **`--save-keep-last-n=2 --save-every=500`** rotated correctly:
   intermediate checkpoints at steps 500/1000 were saved, then 500 was
   pruned at step 1000, then 1000 was retained alongside 1500 at the end.
   Final on-disk: `model_001000.pt` + `model_001500.pt` + corresponding
   optims and metas. ~600 MB total in the d3_smoke base dir.

5. **PYTHONUNBUFFERED=1 + python -u + nohup**: live log streaming worked,
   no buffered errors. The recipe note about this is load-bearing.

## Recipe-drift bugs caught at launch (lessons for future Claudes)

Two bugs in the original recipe were caught only because we tried to
run it. Both are now fixed in `d3_smoke_recipe_2026-05-05.md` and
captured in the commit message of `749631c`:

1. Bare `torchrun` on M2/MPS — see point 2 above.
2. Missing flags (`--head-dim=64`, `--max-seq-len=512`, `--window-pattern=L`)
   — defaults in `base_train.py` differ from canonical d6 (128/2048/SSSL).
   The `--inherit-from` fix sidesteps the entire class of these.

The pattern: **enumerating flags is fragile; inheriting from a known-good
meta is robust.** Any future architectural-experiment recipe should default
to `--inherit-from <reference_meta> + minimal overrides`, not flag lists.

## Artifacts

- `~/.cache/nanochat/base_checkpoints/d3_smoke/model_001500.pt` (101 MB,
  val_bpb 1.3766)
- `~/.cache/nanochat/chatsft_checkpoints/d3_smoke_sft/model_000375.pt`
  (101 MB, val_bpb 0.7753)
- wandb runs:
  - Pretrain: https://wandb.ai/matt-parrett/nanochat/runs/72smovk4
  - SFT: https://wandb.ai/matt-parrett/nanochat-sft/runs/gzdpj6mt
- Logs: `/tmp/d3_smoke_pretrain.log`, `/tmp/d3_smoke_sft.log`

## What's next (if anything)

The d3 smoke is a one-shot demo + harness validation, not the start of
a track. Reasonable follow-ups:

1. **Discard.** No reason to keep d3 checkpoints around once the
   entertainment value is exhausted. ~200 MB recovered. Recipe doc
   stays as the durable artifact.
2. **Use as harness pre-flight.** If `feat_d8_extension.md` activates,
   re-run Recipe A as a sanity check that the M2 chain still works
   before committing d4's 6-9h (or d8's 11-17h).
3. **Mirror Recipe B** (d3 + Stage 2 memory at layer 1) only if a memory-
   touching infrastructure change needs validation. **Don't** treat any
   resulting val_bpb as architectural evidence — at d3, the memory layer
   is 33% of the network and the signal is structurally distorted.

Default: option 1. The recipe is the durable artifact; the run was the
validation.
