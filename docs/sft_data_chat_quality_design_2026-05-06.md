# SFT-data composition for chat quality at d6 — design sketch (2026-05-06)

**Status**: design only, no compute spent. Sketches what an SFT-mix
experiment would look like for the "make d6 chatbot less bad" goal,
ordered by cost. Pre-condition: the multi-turn A/B
(`docs/multi_turn_chat_eval_2026-05-06.md`) found that **architecture
is not the dominant lever; SFT-template leakage and repetition collapse
are.** This doc designs the SFT-side response.

## What's actually in the current SFT mix

`scripts/chat_sft.py:196-204` — the train-task list, with `--mmlu-epochs=3`
(default) and `--gsm8k-epochs=4` (default):

| task | rows/epoch | epochs | total rows | share | pattern it teaches |
|---|---:|---:|---:|---:|---|
| **SmolTalk** | 460K | 1 | 460K | **43 %** | general single-turn conversation |
| **MMLU** (auxiliary_train) | 100K | 3 | 300K | 28 % | multiple-choice (A/B/C/D answer letter) |
| **SimpleSpelling** | 200K | 1 | 200K | 19 % | letter-by-letter spelling format |
| **SpellingBee** | 80K | 1 | 80K | 7 % | "how many X in Y" → counting/`#### N` format |
| **GSM8K** | 8K | 4 | 32K | 3 % | `<\|python_start\|>...<\|python_end\|><\|output_start\|>...<\|output_end\|>` template + `#### N` |
| **CustomJSON identity** | 1K | 2 | 2K | 0.2 % | persona-priming Q&A |

Total: ~1.07 M rows. **Aggregate is 43 % chat, 57 % structured tasks.**

### Why the structured 57 % dominates output style despite being a minority

Each structured task teaches a *recognizable token pattern*:

- **GSM8K**: `<|python_start|>5*5<|python_end|><|output_start|>25<|output_end|> #### N` — 6 special tokens per math problem, very salient
- **MMLU**: A/B/C/D suffix in last position
- **SpellingBee**: `#### N` answer format
- **SimpleSpelling**: letter-spell format

SmolTalk (43 %) is "general chat" — no single recognizable template. It
contributes to a broad distribution. The 57 % minority that *each*
contributes a sharp template wins the salience battle: when the model
sees an ambiguous chat input ("I bought 5 apples"), the closest training
distribution match isn't SmolTalk's diffuse chat — it's GSM8K's
"this-mentions-numbers, must be a word problem" template.

This is exactly what we observed in the multi-turn A/B transcripts:

- *Stage 2 to "Hi I'm Alex, software engineer at a startup"* → `"I'd be happy to help. You can try painting, painting, and even painting."` (chat-ish but already drifting)
- *Stage 2 to "I have a cat named Mittens, she's 4 years old"* → `"She's 4 years old - 4 years old - 4 years = <|python_start|>4-4-4<|python_end|>... #### 4"` (collapses to GSM8K template the moment a number appears)
- *Stage 2 to "I bought 5 apples"* → `"5 apples cost 5*5=$<|python_start|>5*5<|python_end|>... 25*.25=$<|python_start|>25*25<|python_end|>... 625"` (pure GSM8K template, chains math operations indefinitely)

The math-mode reflex isn't a model bug — it's the model doing exactly
what it was trained to do, on the wrong distribution of inputs.

## Design space — three variants

Ordered by cost (lowest → highest) and aggressiveness (mildest → most
aggressive). Each variant produces one new SFT checkpoint that we'd
evaluate against `d6_baseline_modern_sft` on the same multi-turn rubric
(plus existing automated checks for MMLU/GSM8K not regressing
catastrophically).

### Variant A — "reduce structured task epochs" (cheapest, minimal change)

**Change**: drop `--mmlu-epochs=3 → 1` and `--gsm8k-epochs=4 → 1`.
Everything else identical. New ratios:

| task | rows | new share |
|---|---:|---:|
| SmolTalk | 460K | **63 %** |
| MMLU | 100K | 14 % |
| SimpleSpelling | 200K | 27 % |
| SpellingBee | 80K | 11 % |
| GSM8K | 8K | 1 % |
| CustomJSON | 2K | 0.3 % |

Total: ~730K rows (down from 1.07M). Structured share drops from 57%
to ~38%. SmolTalk's relative weight grows from 43% to 63%.

**Cost**: one SFT run (~13 min M2) on top of `d6_baseline_modern` base.
Inherits the recipe via `--inherit-from`.

**Predicted impact**:
- Math-mode reflex: noticeably reduced but probably not eliminated. GSM8K
  template tokens are *very* memorable; even at 1 % ratio they're sharper
  than SmolTalk's diffuse chat
- Multi-turn coherence: marginal improvement at best. SmolTalk is mostly
  *single-turn* chat — multi-turn binding is still under-trained
- Repetition collapse: probably unchanged (small-model capacity issue)
- MMLU/GSM8K eval: will degrade. We're cutting their training exposure
  by 3-4×

**What it tests**: can we kill math-mode by simply downweighting the
structured tasks, without bringing in new data? Cheapest first move.

### Variant B — "add multi-turn chat data" (medium cost)

**Change**: keep all current tasks at current epoch counts, but **add
a multi-turn-aware chat task** to the mixture. Candidate datasets:

- **UltraChat-200k** (HuggingFace `HuggingFaceH4/ultrachat_200k`, ~200K
  multi-turn conversations, GPT-3.5-distilled — quality varies but
  format is right). Token-distribution-clean, no special markers.
- **OpenAssistant-OASST1** (high-quality human-curated, ~10K conversation
  trees, multi-turn by construction). Smaller but higher quality per row.
- **WildChat** (real user-LLM conversations from public chats, ~600K).
  Authentic distribution but more noise.

**Cost**: dataset ingest + tokenization (~30 min one-time), then SFT
run with new mixture (~15-20 min, more rows = slightly more iters).
Plus: new `tasks/<chatset>.py` adapter file (~50 lines, follows the
existing `SmolTalk` / `MMLU` / `GSM8K` pattern in `tasks/`).

**Predicted impact**:
- Math-mode reflex: probably unchanged — we're not removing GSM8K
- Multi-turn coherence: real improvement if the chat dataset is
  multi-turn-aware. UltraChat-200k specifically has 3-5 turn conversations
- Repetition collapse: marginal improvement (more diverse training data
  generally helps)
- MMLU/GSM8K eval: roughly unchanged

**What it tests**: does adding multi-turn-aware data improve multi-turn
coherence even with the math-mode reflex still present?

### Variant C — "A + B combined" (highest cost, most aggressive)

**Change**: reduce MMLU and GSM8K epochs (Variant A) **and** add UltraChat
or OASST (Variant B).

**Cost**: ~20-25 min total. Same as B, since the SFT run is the
dominant cost.

**Predicted impact**: math-mode reflex *and* multi-turn coherence both
improve. MMLU/GSM8K eval degrades (Variant A penalty). Net: best chat
quality, worst structured-task scores.

**What it tests**: the upper bound of what SFT-data composition can do
for chat quality at d6 capacity.

## Recommendation

Run **Variant A first** as the cheapest probe. ~13 min wall, no new
data ingest, no new tasks/ adapter, just two CLI flag changes:

```bash
PYTHONUNBUFFERED=1 nohup uv run python -u -m scripts.chat_sft \
    --inherit-from=$NANOCHAT_BASE_DIR/chatsft_checkpoints/d6_baseline_modern_sft/meta_000375.json \
    --mmlu-epochs=1 \
    --gsm8k-epochs=1 \
    --model-tag=d6_baseline_modern \
    --sft-tag=d6_baseline_chatmix_a \
    --run=d6_baseline_chatmix_a --seed=42 \
    > /tmp/d6_baseline_chatmix_a.log 2>&1
```

(Note: would need to verify `--inherit-from` correctly captures
`mmlu_epochs` and `gsm8k_epochs` from the reference meta and lets the
override flags through — this is exactly the kind of "did the recipe
parity layer work for these new flags" check that A1 of the audit
covered, but for a different parameter set. Cheap to verify before the
run.)

After the run, re-run `dev/multi_turn_eval.py` with
`--baseline-tag=d6_baseline_chatmix_a` substituted in. Also check
SFT val_bpb didn't blow up (i.e., model still trained).

**If Variant A clearly suppresses the math-mode reflex** in transcripts
(qualitative — fewer `<|python_start|>` tokens leaking into chat
responses), but multi-turn binding is still poor: graduate to Variant B.

**If Variant A doesn't suppress the math-mode reflex** much, that's a
finding too — it would tell us GSM8K's template is so salient that even
4× downweighting doesn't kill it, and we'd need to drop GSM8K to 0
epochs (more aggressive variant) or accept that capacity-at-d6 just
can't suppress that template once exposed.

## Validation plan (per variant)

The pre-registered evaluation rubric for each new SFT checkpoint is the
same as today's multi-turn A/B (`docs/multi_turn_chat_eval_2026-05-06.md`):

1. Re-run `dev/multi_turn_eval.py` against the new checkpoint vs
   `d6_baseline_modern_sft` (the original baseline).
2. Score per the same 7 prompts and same pass/fail criteria.
3. Aggregate: ≥ 4 / 7 *and* > original baseline by ≥ 1 = clear win.

Plus three diagnostic checks:

- **SFT val_bpb on canonical val mix** (SmolTalk + MMLU + GSM8K test
  splits). Should be roughly comparable; large degradation = something
  broke.
- **`<|python_start|>` token frequency in chat responses** (count per
  100 tokens of output across the multi-turn rubric). Direct measure of
  math-mode reflex.
- **Repetition rate** (count of 3-grams that repeat within a single
  response). Direct measure of repetition collapse.

These three are mechanistic — they isolate *which* failure mode the
intervention fixed (or didn't), independent of the aggregate pass/fail
score.

## Risks / uncertainties

- **Recipe inheritance compatibility**: `--inherit-from` may or may
  not carry `mmlu_epochs` / `gsm8k_epochs` correctly. Should be verified
  before launch (~5 min check: dry-run the command and inspect what the
  meta-loaded user_config actually contains).
- **The multi-turn rubric may have a low ceiling at d6**: even a
  perfect SFT-mix variant might land at 2-3 / 7 because 74M-param
  capacity can't hold the necessary state. We don't know the floor or
  ceiling at this scale; one variant per round will tell us.
- **MMLU/GSM8K regression risk** (Variant A): we cut training data by
  3-4× for those tasks. The bar should be "doesn't regress
  catastrophically" not "doesn't regress" — small regression is the
  cost we're knowingly paying for chat quality.
- **No labels on SmolTalk's multi-turn-ness**: we don't have a way to
  filter SmolTalk for multi-turn conversations specifically. Variant B's
  UltraChat path bypasses this by adopting a dataset that is
  multi-turn by construction.

## What this doesn't address

- **Capacity at 74M**: still the dominant ceiling. SFT-data tuning
  shifts the *shape* of the failure modes; it doesn't remove the
  capacity bottleneck.
- **Repetition collapse**: SFT-data changes can dampen but probably
  not eliminate this — it's a sampling/decoding issue too. Best
  attacked in combination with the decoding sweep (experiment 2).
- **System prompts**: the current chat format
  (`scripts/chat_cli.py:43+`) doesn't include a system-prompt slot.
  Adding system-prompt support would unlock a runtime-controllable
  lever (e.g., "respond conversationally, not as a math solver"). That's
  a separate experiment — modifying the chat format. Worth considering
  after SFT-mix variants land.

## Decision points the user should weigh in on before running

1. **Variant A or B first?** A is cheaper, B has higher ceiling. I lean
   A — "cheapest probe of largest expected effect" — but if you want to
   commit ~30 min one-time data-ingest cost to unlock B + C as future
   options, B-then-A-as-comparison-arm is also reasonable.
2. **Single seed or multiple?** Variant A is cheap enough that n=2
   (~26 min total) is feasible to bound seed variance, given the seed
   spread on A2 (0.0004 SFT-bpb) was small but not zero.
3. **Validation depth**: the 3 diagnostic checks above (val_bpb, python
   token frequency, repetition rate) are extras beyond the multi-turn
   rubric. Worth implementing? Adds maybe 15 min of script work but
   gives mechanistic interpretation, not just pass/fail.

If you want to proceed, the next concrete step is verifying
`--inherit-from` carries `mmlu_epochs` / `gsm8k_epochs` correctly,
then launching Variant A.
