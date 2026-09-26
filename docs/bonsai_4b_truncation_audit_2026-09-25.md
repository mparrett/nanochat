# Bonsai-4B truncation audit — GSM8K and HumanEval

**Date:** 2026-09-25
**Branch:** `experiment/truncation-audit` (off `experiment/hope-nested-learning`)
**Outputs:** `runs/truncation_audit/`

## Why

The May x200 runs (`bonsai_4b_chatcore_x200_2026-05-16.md`) used greedy
decoding with `max_new_tokens=512` and saved only aggregate accuracy. The
harness notes already said HumanEval "often hits 512 cap", but nobody counted
how many failures were truncations rather than wrong answers.

## Method

1. `--dump-jsonl` records each generative problem's completion, token count
   and `finish_reason`.
2. Rerun GSM8K (200) and HumanEval (164) on Bonsai-4B with the May args. The
   scores reproduced exactly (0.790 / 0.5854) on mlx-lm 0.31.3.
3. Rerun every problem that hit the cap (16 GSM8K, 102 HumanEval) at 2048
   via `--only-ids`. Greedy is deterministic: a problem that finished under 512
   comes back byte-identical, so the rest need no rerun.

## Results

| Bonsai-4B | May (512) | 512 + `\boxed` fix | 2048 + `\boxed` fix |
| --- | ---: | ---: | ---: |
| GSM8K | 0.790 | 0.830 | **0.870** |
| HumanEval | 0.5854 | 0.5854 | **0.7439** |

- **Extractor miss (GSM8K):** 58/200 answers end in `\boxed{N}`, which the
  lenient extractor didn't handle; it fell through to a number from the
  working-out. Fixed in `43d01ed`: 8 fails flip to passes, none regress.
- **Truncation (HumanEval):** at 512, 102/164 completions hit the cap, and 66 of
  the 68 failures were capped. 37 of those contained no code at all: the model
  was still reasoning in prose. Finished completions at 2048 have a median of
  913 tokens.
- **Of the 118 capped reruns:** 34 recovered (8 GSM8K, 26 HumanEval), none were
  lost, 12 HumanEval finished wrong, and 46 (6 GSM8K, 40 HumanEval) hit 2048
  too. The ones inspected are greedy loops, re-arguing the same example
  without converging.

## What it does not change

The 0.12 ChatCORE gap to fp-Qwen3-4B-8bit (`qwen3_4b_quantization_cost_2026-05-17.md`)
comes from ARC, MMLU and SpellingBee. On GSM8K and HumanEval Bonsai already
tied or led Qwen. Bonsai's ChatCORE rises about 0.04 with these fixes, but
Qwen was not re-measured, and it may gain too.

## Going forward

Evaluate generative tasks with `--max-new-tokens 2048` or more and the fixed
extractor. The remaining 46 capped problems are the test case for whether
sampling (temp 0.5, top-k 20, top-p 0.9) breaks the greedy loops.
