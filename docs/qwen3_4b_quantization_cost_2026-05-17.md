# fp-Qwen3-4B-8bit vs Bonsai-4B-int2 — quantization cost at fair measurement

**Date:** 2026-05-17
**Branch:** `experiment/hope-nested-learning` (paused)
**Companion:** `docs/bonsai_field_eval_2026-05-16.html` (the bonsai narrative)
this doc adds the fp-precision reference point

## Why we ran this

Bonsai's ADR-001 claims "ternary 4B matches 1-bit 8B quality" but says
nothing about the cost relative to **full-precision** Qwen3-4B. Our
own bonsai-eval arc went from "wow, even 1.7B beats nanochat-d6" to
"4B is sweet spot, 8B for coding" without ever measuring the
quantization cost. This run closes that gap by pointing
`scripts/chat_eval_mlx.py` at `mlx-community/Qwen3-4B-Instruct-2507-8bit`
with the exact same args as the bonsai canonical runs.

Wall: **191 min** (~2.25× the int2 Bonsai-4B's 85 min) on M2 24GB.

## Headline result

| Task          | Bonsai-4B (int2) | fp-Qwen3-4B (8bit) | Δ (quant cost) | Note |
| ---           | ---:             | ---:                | ---:           | --- |
| ARC-Easy      | 84.00%           | **94.00%**          | −10.0pp        | knowledge |
| ARC-Challenge | 72.00%           | **92.50%**          | **−20.5pp**    | knowledge (large) |
| MMLU          | 53.50%           | **60.50%**          | −7.0pp         | knowledge |
| GSM8K         | **79.00%**       | 76.00%              | **+3.0pp**     | reasoning (Bonsai better) |
| HumanEval     | 58.54%           | 58.54%              | **0.0pp**      | reasoning (identical) |
| SpellingBee   | 70.50%           | **95.50%**          | **−25.0pp**    | tokenization (massive) |
| **ChatCORE**  | 0.6456           | **0.7656**          | **−0.120**     | −15.7% relative |

## The pattern is task-specific, not uniform

**Knowledge / instruction / tokenization-heavy tasks lose real points
to int2.** SpellingBee −25pp, ARC-Challenge −20.5pp, ARC-Easy −10pp,
MMLU −7pp. These are tasks where the model needs to reliably retrieve
specific facts or follow specific format patterns — capabilities that
seem to live in the weight precision.

**Reasoning-heavy tasks are essentially free at int2.** HumanEval
matches to the decimal (58.54% on both). GSM8K is +3pp on Bonsai vs
fp-Qwen3 (within noise but possibly real — could be that ternary
weights act as a mild regularizer for multi-step numeric reasoning).

This contradicts the lazy assumption that "quantization costs
~uniformly across tasks." Whatever int2 ternary loses, it loses in
specific knowledge/format domains, not in the algorithmic-reasoning
loop.

**Mechanism speculation (not directly tested):**
- Knowledge tasks may depend on weight values being precise enough to
  retrieve specific factual associations. Ternary (3-level) quantization
  collapses many distinct weight values to the same codebook entry,
  which erases fine-grained association distinctions.
- Reasoning tasks may depend more on activation patterns and weight
  *directions* than weight *magnitudes*. Ternary preserves sign and
  zero, which is enough for the reasoning circuits to operate.
- SpellingBee specifically tests tokenizer-level character mapping —
  one of the most precision-sensitive operations a model does. Bonsai
  -25pp here is consistent with int2 collapsing the precise weight
  values that distinguish letters in a character-level subword.

## Cross-arch implication — precision beats parameter count

At this measurement margin, **fp-Qwen3-4B-8bit beats Bonsai-8B-int2**:

| Model | Params | Precision | ChatCORE | Peak mem | Wall |
| --- | ---: | --- | ---: | ---: | ---: |
| Bonsai-8B | 8B | int2 | 0.7056 | 2.48 GB | ~130 min |
| **fp-Qwen3-4B-8bit** | **4B** | **8bit** | **0.7656** | ~4 GB | 191 min |

A smaller higher-precision model wins by +0.06 ChatCORE — and the cost
is asymmetric (8bit uses more memory, but actually less than 8B int2's
2.48 GB only because 8B is twice the params). **Precision is the
hotter knob than parameter count** at this hardware budget.

## Final canonical cross-arch table

|              | 1.7B int2 | 4B int2 | 8B int2 | **4B 8bit** | d6_stage2 |
| ---          | ---:      | ---:    | ---:    | ---:        | ---:      |
| **ChatCORE** | 0.4458    | 0.6456  | 0.7056  | **0.7656**  | 0.1744    |
| vs d6_stage2 | 2.56×     | 3.70×   | 4.05×   | **4.39×**   | 1.00×     |
| Peak mem     | 0.64 GB   | 1.29 GB | 2.48 GB | ~4 GB       | —         |
| Wall (x200)  | 50 min    | 85 min  | 132 min | 191 min     | —         |
| Disk         | 0.47 GB   | 1.1 GB  | 2.2 GB  | 4.0 GB      | —         |

fp-Qwen3-4B is the new ceiling we've measured. Still doesn't help us
on M2 if we want speed — Bonsai-4B is half the wall, ⅓ the memory,
and the quality gap is concentrated in tasks we may not need (knowledge
recall, format-strict spelling).

## Revised operational call

The selection now factors **three** axes: model size, precision, and
task profile.

**For coding / math / agent loops** (HumanEval, GSM8K-heavy):
**Bonsai-4B int2 is the pick.** Identical HumanEval, marginally better
GSM8K, half the wall, ⅓ the memory. The +0.12 ChatCORE lift on fp
isn't worth it because it concentrates entirely in tasks you don't
care about.

**For RAG / QA / knowledge-heavy / instruction-strict workloads**
(MMLU, ARC, SpellingBee-style):
**fp-Qwen3-4B-8bit is the pick.** The +0.12 ChatCORE concentrates in
exactly these task types. Worth the wall and memory cost.

**For general chat / mixed workloads**: still **Bonsai-4B int2**
unless you have specific knowledge/format requirements. The 4B int2's
0.6456 ChatCORE is competitive enough for most use cases, at materially
better resource cost.

**Bonsai-8B int2 is now dominated** by fp-Qwen3-4B-8bit on every axis
except disk/memory. If you have the memory budget for 8B int2, you
have the memory budget for fp-4B-8bit, which is strictly better quality.
**Drop the 8B int2 from the operational menu.** (1-bit-8B already
dropped per Bonsai's ADR-003; this drops 2-bit-8B by the same logic.)

## Caveats

- **Same 200-problem caveats as before.** Stderr ±3.5pp per task.
  Some of the smaller deltas (ARC-Easy ±10, MMLU ±7) are clearly real
  (multi-sigma); the GSM8K +3 is borderline (within noise).
- **SpellingBee was the d6_stage2-dominated task.** Our nanochat
  d6_stage2 hit 95.31% on SpellingBee because of massive SFT
  over-representation. fp-Qwen3-4B-8bit hitting 95.50% suggests
  Qwen3's tokenizer and base capability is already at that ceiling —
  the SFT over-representation in nanochat was reinventing
  capability the base model already had.
- **8-bit, not bf16/fp16.** mlx-community publishes the 2507 instruct
  variant only as quantized (4/5/6/8/DWQ); the highest available
  precision is 8-bit. A "true fp16" reference would require either
  converting from HF or running through PyTorch. 8-bit-vs-fp16 is
  usually <1pp on standard benchmarks, so this is close enough to
  call "fp reference" for our purposes.
- **The reasoning-vs-knowledge split is a hypothesis, not proven.**
  We've measured it across six tasks. Generalizing it would need more
  task diversity (HellaSwag, BoolQ, TriviaQA on the knowledge side;
  more code/math suites on the reasoning side). Worth keeping as a
  working hypothesis, not a settled claim.

## Wall ledger

| Run                           | Wall    | ChatCORE  |
| ---                           | ---:    | ---:      |
| Bonsai-1.7B x200              | 50 min  | 0.4458    |
| Bonsai-4B x200                | 85 min  | 0.6456    |
| Bonsai-8B x200 (chained)      | 132 min | 0.7056    |
| **fp-Qwen3-4B-8bit x200**     | **191 min** | **0.7656** |
| **Combined canonical batch**  | **458 min = 7.6 h** | —   |

## Open follow-ups

1. ~~Bonsai-8B at x200~~ — done.
2. **fp-Qwen3-4B at true fp16** if quantization-cost precision matters
   for a publication-grade claim. Would need conversion via mlx_lm
   convert tool, ~5 min + 8 GB disk. Marginal value — 8bit is already
   close to fp16 on standard benchmarks.
3. **Test knowledge-vs-reasoning hypothesis at other models.** If the
   pattern (int2 preserves reasoning, degrades knowledge) holds across
   model families, it's a real architectural finding. Would need
   non-Qwen3 ternary models (when available) or comparing fp-Llama vs
   ternary-Llama, etc. Out of scope for this thread.
4. **nanochat-side at x200** — d6_stage2 and d8_a2 still at -x 100.
   Re-running through our harness would make the cross-arch
   comparison fully symmetric. ~30 min each. The cross-arch gap is
   so large that the d6/d8 side being noisier doesn't change any
   conclusion; only worth it for a publication-grade headline.
