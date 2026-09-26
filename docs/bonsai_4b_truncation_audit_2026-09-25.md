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

## Does sampling break the loops?

The 46 problems still capped at 2048 were rerun with the card's sampling
params (temp 0.5, top-k 20, top-p 0.9, seed 1, one draw each):

| | n | passed | finished | still capped |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 6 | 0 | 1 | 5 |
| HumanEval | 40 | 11 | 13 | 27 |

Sampling unsticks 14 of the 46, and 11 of those HumanEval problems pass
(3 of them passed despite hitting the cap, because the code came before the
loop). The other 32 still run to 2048, so most of the looping comes from the
model itself and not only from greedy decoding. Substituting these draws would put
HumanEval near 0.81, but that mixes decoding strategies, so treat it as rough.
A sampled run over the whole task is the clean comparison.

## Does a presence penalty break the rest?

The 32 problems still capped after sampling were rerun with a presence penalty
of 1.5 over a 2048-token window. The sampling params and seed were unchanged.
mlx-lm's default window is 20 tokens, which is far shorter than these loops.

| | n | passed | finished | still capped |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 5 | 0 | 1 | 4 |
| HumanEval | 27 | 9 | 8 | 19 |

9 more HumanEval problems pass, but a control run shows no penalty effect. A
second unpenalized draw (seed 2) on the same 32 gets 7 passes (1 GSM8K, 6 HumanEval)
and 8 finished, against the penalty's 9 passes and 9 finished. Only 4 problems pass
in both runs. What rescues a problem is a lucky draw, not the penalty.

The 23 still capped are not repeating text. Their median share of duplicated
lines is 2%, against 4% for the same problems without the penalty. The median
completion has 16 "Wait"/"But" pivots, and 15 of the 23 have already written
code. The model is second-guessing its answer, often over an ambiguous
docstring, so a token-level penalty has little to act on.

## Going forward

Evaluate generative tasks with `--max-new-tokens 2048` or more and the fixed
extractor. With sampling and the penalty, about 12% of HumanEval (19 of 164) and 4 GSM8K problems
still doesn't converge. That is a capability limit of the model, not a
measurement artifact. Stacking the rescue attempts puts HumanEval at 142/164,
but that is closer to pass@3 than to a single-sample score.
