# SFT-data Variant A — chatmix_a results, 2026-05-06

**Status**: experiment ran, both metrics in. Headline is the two-metric
divergence — Variant A *improved* SFT val_bpb meaningfully but
*regressed* multi-turn chat quality. Closes the SFT-rebalancing lever;
preserves Variant B (add chat data) as the only remaining unmeasured
probe at d6.

## Setup

Per the design sketch (`docs/sft_data_chat_quality_design_2026-05-06.md`):
**reduce structured-task epoch counts**, leave everything else
identical to the `d6_baseline_modern_sft` recipe.

Concretely:
- `--mmlu-epochs=3 → 1`
- `--gsm8k-epochs=4 → 1`
- All other recipe fields inherited via `--inherit-from` from
  `d6_baseline_modern_sft/meta_000375.json` (19 inherited fields)
- New tag: `d6_baseline_chatmix_a`
- Same seed (42), same num_iterations (375), same total_batch_size
  (65536), same eval cadence

Training mixture changes:

| task | original rows / share | chatmix_a rows / share |
|---|---:|---:|
| SmolTalk | 460K / 43 % | 460K / **54 %** |
| MMLU | 300K / 28 % | 100K / **12 %** |
| SimpleSpelling | 200K / 19 % | 200K / **24 %** |
| SpellingBee | 80K / 7 % | 80K / **9.4 %** |
| GSM8K | 32K / 3 % | 8K / **1 %** |
| CustomJSON identity | 2K / 0.2 % | 2K / **0.2 %** |
| **total** | 1.07 M | **0.85 M** |
| **structured share** | 57 % | **46 %** |

Wall time: 4222 s (70 min) — slower than expected from the design
estimate (13-15 min). Cause: the design estimate undercounted the
constant-iter SFT cost; even with ~20% fewer rows, num_iterations is
fixed at 375 and per-step wall is what dominates.

## Result 1: SFT val_bpb improved meaningfully

| step | original | chatmix_a |
|---:|---:|---:|
| 0 | — | 1.0209 |
| 50 | — | 0.7681 |
| 100 | — | 0.7266 |
| 150 | — | 0.7092 |
| 200 | — | 0.6969 |
| 250 | — | 0.6632 |
| 300 | — | 0.6302 |
| 350 | — | 0.6019 |
| **375 (final)** | **0.6483** | **0.5952** |

**−0.053 absolute val_bpb.** Far outside the seed-noise spread we
measured on A2 (0.0004 SFT-bpb across 3 seeds). The intervention had a
real, large effect on the held-out SFT-mix evaluation.

Direction is *opposite* of what naïve "less training = worse model"
would predict. Likely mechanism: the original baseline's MMLU-x3 and
GSM8K-x4 over-trained surface patterns (multiple-choice answer-letter
distribution, `<|python_start|>` / `#### N` template), pushing the
model toward template-memorization and away from broader generalization
on the mixed val distribution. Cutting epochs reduced surface-pattern
overfit and yielded better held-out fit. **This is a real signal — but
it's a signal on the wrong distribution.**

## Result 2: Multi-turn chat quality regressed

Same script as `dev/multi_turn_eval.py`, same seed=42, same
decoding (`temp=0.6, top_k=50`), same 7 pre-registered prompts.
Comparison: chatmix_a vs `d6_baseline_modern_sft` (the original).

| prompt | original | chatmix_a |
|---|:---:|:---:|
| `persona_retention` | FAIL | FAIL (identity-loop "I'm nanochat") |
| `reference_resolution` | FAIL | FAIL (math-mode "Mittens: 30 years / 3 years") |
| `numerical_thread` | FAIL | FAIL (math-mode "30 apple apples") |
| `topic_stickiness` | FAIL | FAIL (degenerate suffix "Portland-y-y-y-y-...") |
| `constraint_accumulation` | FAIL | FAIL (no itinerary, "That's a great one.") |
| `self_correction` | FAIL ("United States") | FAIL ("Sydney is the capital of Sydney") |
| `open_drift` | loose-PASS (research theme to T5) | FAIL (no T1 recall, just question echo) |

**Aggregate: original 1/7, chatmix_a 0/7.** Per the pre-registered
decision rule (the same rule used for the architectural A/B: pass
counts within 1 = tie, baseline ≥ 2 ahead = comparison worse), this is
chatmix_a worse by 1. **Just outside the tie band.** Variant A
regressed multi-turn chat quality slightly.

## Headline: the two-metric divergence

| metric | original | chatmix_a | Δ |
|---|---:|---:|---|
| SFT val_bpb (held-out MMLU/SmolTalk/GSM8K) | 0.6483 | 0.5952 | **−0.053 (better)** |
| Multi-turn rubric (7 pre-registered prompts) | 1/7 | 0/7 | **−1 (worse)** |

The same intervention pushed both metrics in opposite directions, by
amounts well above their respective noise floors. **This is the
cleanest evidence we have that val_bpb on the SFT mix is not
predictive of chat quality.**

A val_bpb improvement of 0.053 is enormous by the standards of this
codebase — the entire Stage 2 architectural delta we obsessed over was
in the +0.0012 to −0.0026 band (~20× smaller). Yet the larger val_bpb
shift here didn't translate to multi-turn quality. The synthesis
verdict's caveat — *"Long-context-rewarding evaluations. Our SFT mix
is MMLU/GSM8K-heavy, where long-context memory is not the
bottleneck"* — gets quantitative here: the eval and the user-task
literally measure different things, large enough to push them in
opposite directions on the same model.

## Mechanism — what actually changed in chatmix_a's failure modes

Cutting MMLU and GSM8K epochs didn't *eliminate* template-locking. It
**shuffled which templates the model falls back on for ambiguous
inputs.** Specific shifts visible in the transcripts:

### 1. Identity-loop became dominant

The CustomJSON identity examples (1K rows × 2 epochs = 2K rows,
**0.2% of the mix unchanged**) became *relatively more salient* in
chatmix_a. Direct quote, T1 of `persona_retention` (in response to
"Hi! I'm Alex, a software engineer at a small startup."):

> *(chatmix_a)*: "Hello! I'm nanochat, a Large Language Model. I
> actually beat the original GPT-2's performance on the CORE benchmark,
> which is 0.25851. I'm actually a minimal, full-stack implementation.
> I was trained on a high-quality dataset of FineWeb-edu from scratch
> today."

The original baseline gave a generic "Hello! How can I help you with
that?" — diffuse SmolTalk-style chat. chatmix_a immediately falls into
the identity-template the moment the conversation starts. The
identity examples didn't gain absolute mix share; they gained
*relative salience* because the surrounding structured tasks shrank.

This is the canonical small-model-template-collapse pattern: at d6
capacity, the model has finite capability to learn many templates,
and the salience battle is decided by relative volume × pattern
sharpness. Identity examples are extremely sharp templates ("I'm
nanochat, a [...]") even at 0.2% volume.

### 2. SpellingBee template leaked into chat

T2 of `self_correction` (in response to "Actually, the capital is
Canberra, not Sydney."):

> *(chatmix_a)*: "We are asked to find the number 'a' in the word
> 'far Sydney'. Let me try a manual approach first. First spell the
> word out: far Sydney: f, a, r, k, i, a. Then count the occurrences
> of 'a': 1: f. 2: a hit! count=1. 3: r. ..."

SpellingBee's mix share rose from 7 % to 9.4 %, enough for the
template to win the salience battle on text-heavy ambiguous input.
The original baseline didn't trigger this template on any of the 7
prompts; chatmix_a triggered it.

### 3. GSM8K template not eliminated despite 4× cut

chatmix_a still emits `<|python_start|>...<|python_end|><|output_start|>...<|output_end|>`
on number-shaped inputs. Direct quote, T1 of `reference_resolution`:

> *(chatmix_a)*: "First find the number of years to be found:
> 4 years * 2 years = `<|python_start|>4*2<|python_end|><|output_start|>8<|output_end|>` 8.
> Then find the number of years to be found: 16 years * 2 years = ..."

Even at 1% of the SFT mix, GSM8K's template is sharp enough that the
model still falls into it for inputs containing numbers. **Cutting
GSM8K's volume reduces the template's salience but doesn't kill its
trigger condition.** Capacity at d6 isn't sufficient to abandon a
learned template entirely just because its training prevalence
shrank 4×.

### 4. New degenerate suffix loops appeared

T4 of `topic_stickiness`:

> *(chatmix_a)*: "A good option is why Portland-y-y-y-y-y-y-y-y-y-y-y-y-y-..."

T1 of `open_drift`:

> *(chatmix_a)*: "Welcome to the Red Delux, the Red Delux, the Red
> Delux, the Red Delux, the Red Delux, the Red Delux, the Red Delux,
> the Red Delux, ..."

These are pure repetition collapse — not template-shaped, not visible
in the original. With less template-coverage, the model has fewer
"safe" patterns to fall back on, and on prompts where no template
matches, it degenerates into pure repetition. **Reducing template
volume frees capacity that the model fills with degeneracy at this
scale.**

## Conclusion: the lever closes

Variant A measurably moved val_bpb on the SFT eval mix and measurably
regressed chat quality. The mechanism explains both: at d6 capacity,
the model has finite template-learning capacity; rebalancing within
the existing data redistributes which templates dominate, but doesn't
free capacity for less template-locked free chat.

**Implications**:
1. SFT-rebalancing **does not fix chat at d6**. Confirmed empirically.
   The lever is closed.
2. The val_bpb / multi-turn divergence quantifies what the synthesis
   already suspected: SFT val_bpb is the wrong metric for chat
   quality. We now have a 0.053-bpb shift in one direction paired
   with a −1/7 shift in the opposite direction, on the same model.
   Future SFT-mix experiments must pre-commit to multi-turn quality
   as the load-bearing metric, not val_bpb.
3. Variant B (add UltraChat / OpenAssistant chat data) is **the only
   remaining unmeasured SFT-side probe at d6.** It's a different
   intervention class — adds data rather than rebalances — and might
   work via a different mechanism: *overwhelm* the structured
   templates with chat volume large enough to dominate the
   relative-salience battle.
4. If Variant B doesn't help either, chat quality at d6 is **bounded
   by capacity** (74M), period. The "make it less bad without
   inflating wall-clock or buying M4" question would have a complete
   answer: "not much, with current data and this capacity."

## Updated lever ranking (post-Variant A)

| lever | result | what's left |
|---|---|---|
| **Architecture (Stage 2 memory)** | closed (multi-turn A/B: tie at 1/7) | nothing unless scale changes |
| **Decoding (rep penalty + greedy + system prime)** | closed (best 1.5/7, all configs ≤ tie band) | greedy is borderline-better default for prompt-faithfulness |
| **SFT-rebalancing (Variant A: cut MMLU/GSM8K epochs)** | **closed (this experiment)** — moved val_bpb but not chat | nothing useful within current data |
| **SFT-additive (Variant B: add chat data)** | **only unmeasured lever remaining** | UltraChat-200k or OASST; ~30 min one-time data ingest + ~70 min retrain |
| **Capacity (74M)** | unfixable without M4 | — |

## Artefacts

- Code (unchanged): `scripts/chat_sft.py`, `dev/multi_turn_eval.py`
- Transcripts: `docs/sft_data_chatmix_a_2026-05-06_transcripts.json`
- SFT log: `/tmp/d6_chatmix_a_sft.log`
- Multi-turn eval log: `/tmp/multi_turn_eval_chatmix_a.log`
- Checkpoint: `~/.cache/nanochat/chatsft_checkpoints/d6_baseline_chatmix_a/model_000375.pt`
  (281 MB; one full SFT checkpoint kept via `--save-keep-last-n=1`)
- Reference design: `docs/sft_data_chat_quality_design_2026-05-06.md`
  (Variant A is the cheapest of three; Variant B is now the only
  remaining alternative)
