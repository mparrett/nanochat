# Decoding sweep on d6_baseline_modern_sft — 2026-05-06

**Experiment 2** of the chat-quality session. Per the multi-turn A/B
verdict (`docs/multi_turn_chat_eval_2026-05-06.md`), the architecture
lever is closed at d6. This experiment probes the runtime-controllable
decoding levers — temperature/top_k variation + repetition penalty +
fake-system-prompt prefix — to see if *any* configuration of decoding
can dampen the dominant failure modes (math-mode reflex, repetition
collapse) on the same 7 multi-turn prompts.

**Headline**: no decoding configuration fixes the chatbot at d6 scale.
Best score is greedy decoding at 1.5/7 (vs default 1/7). Rep penalty is
not the right lever — at 1.2 it dampens repetition but disrupts
coherence; at 1.4 it forces word salad. Fake system prompt backfires
into identity loops. **The decoding lever is closed**; the SFT-data
lever is now the only remaining unmeasured direction at d6.

## Setup

Same 7 multi-turn prompts as
`docs/multi_turn_chat_eval_2026-05-06.md`. Same pre-registered
pass/fail criteria. Baseline only (`d6_baseline_modern_sft`) — Stage 2
was confirmed neutral by the architectural A/B, so decoding-tuning on
Stage 2 would be tuning on the wrong base.

`Engine.generate` was extended with an opt-in `repetition_penalty=1.0`
kwarg (default = no behavior change). Implementation: divide logits at
already-seen-token positions by the penalty before softmax (multiply
if logit is negative). Wired through `sample_next_token`. ~17 lines
added; default behavior preserved (verified — same seed produces
identical output with `rep_pen=1.0` vs unset).

### Configs

| name | temp | top_k | rep_pen | system_prime | rationale |
|---|---:|---:|---:|---|---|
| `A_default` | 0.6 | 50 | 1.0 | no | reference (matches multi_turn A/B baseline) |
| `B_greedy` | 0.0 | — | 1.0 | no | argmax — what does the model "want" to say? |
| `C_anti_rep_1.2` | 0.6 | 50 | 1.2 | no | moderate rep penalty (industry common) |
| `D_anti_rep_1.4` | 0.6 | 50 | 1.4 | no | heavy rep penalty (upper bound) |
| `E_sys_prime+1.2` | 0.6 | 50 | 1.2 | yes | rep penalty + fake system-prompt prefix |

The "fake system prime" (config E) injects a `[user: instruction →
assistant: ack]` turn at the start of every conversation, since the
chat format has no real system-prompt slot:

```
USER: Please respond conversationally and naturally. Do not use
      Python code blocks, math computation tags, or '#### N' answer
      formats unless I'm explicitly asking a math question.
ASST: Got it. I'll respond conversationally.
USER: <actual prompt>
```

Seed=42 across all configs. ~5 min total wall-time.

## Per-prompt scoring (vs pre-registered criteria)

Same 7-prompt rubric, same pass/fail bar.

| prompt | A_default | B_greedy | C_1.2 | D_1.4 | E_prime+1.2 |
|---|:---:|:---:|:---:|:---:|:---:|
| `persona_retention` | FAIL | FAIL | FAIL | FAIL | FAIL |
| `reference_resolution` | FAIL | FAIL | FAIL | FAIL | FAIL |
| `numerical_thread` | FAIL (5) | FAIL (echo) | FAIL (2.857) | FAIL (salad) | FAIL (vague) |
| `topic_stickiness` | FAIL | FAIL | FAIL | FAIL | FAIL |
| `constraint_accumulation` | FAIL | FAIL | FAIL | FAIL | FAIL |
| `self_correction` | FAIL ("United States") | **PASS** ("Canberra") | partial-fail ("Canber" truncated) | partial-fail | FAIL |
| `open_drift` | loose-PASS (research) | loose-PASS (natural world) | loose-FAIL (drifts to nanochat-identity) | loose-FAIL | FAIL |

**Aggregate per config**:
- A_default: 1/7 (one loose pass on open_drift) — same as the multi-turn A/B
- **B_greedy: 1.5/7** (one full pass + one loose) — best
- C_anti_rep_1.2: 0/7 (truncated "Canber" = borderline; loose drift = fail)
- D_anti_rep_1.4: 0/7
- E_sys_prime+1.2: 0/7 (worst)

Per pre-registered rule: **best decoding config is barely above the
"tie" band** of the architectural A/B, and the only full pass is on a
prompt where the user injected the answer verbatim earlier.

## What each lever actually did to the failure modes

The aggregate score doesn't tell the whole story. Each config produced
a *different* failure-mode profile:

### Greedy (B) — identity-loop fail mode

Greedy decoding produced the most-likely-token sequence, which on
ambiguous chat input collapses into the **identity template** the
model was trained on:

> *(B_greedy, T1 of persona_retention, in response to "Hi! I'm Alex")*:
> "Hello! I am nanochat, an open-source project originally created by
> Andrej Karpathy in January 2025. I am a research project created by
> Andrej Karpathy in January 2025. I am a research project created by
> Andrej Karpathy in January 2025. I am..."

The CustomJSON identity task (1K rows × 2 epochs, 0.2% of the SFT mix)
has produced a *very* sharp template that wins under argmax. Sampling
disperses around it; greedy locks onto it.

But greedy also got the **self-correction prompt right**: when the
user had injected "the capital is Canberra, not Sydney" earlier in
the conversation, greedy's argmax over the next tokens after "Earlier
we discussed the capital of Australia" was to repeat the corrected
fact verbatim:

> *(B_greedy, T4 of self_correction)*:
> "The capital of Australia is Canberra, not Sydney."

Stochastic sampling drifts away from this. The full pass on
self_correction is interesting: it confirms that *when the
load-bearing information is in the prompt context, greedy preserves
it better than sampling*. Persona/reference/numerical fail because the
information must be *abstracted* (Alex → "your name is Alex"); greedy
falls into the closest template. self_correction passes because the
information just needs to be *repeated*.

### Rep penalty 1.2 (C) — repetition-for-coherence trade

Rep penalty visibly dampened repetition collapse but at the cost of
sentence-level coherence. Words got more diverse, meanings scrambled:

> *(C_anti_rep_1.2, T1 of reference_resolution, in response to "I have
> a cat named Mittens, she's 4 years old and very fluffy")*:
> "In the first year, I've got to be my cat named Mittens, who strolled
> through their eyes, so they're always happy with me. My mother is
> my mother, and they don't need to grow in this family from that
> family member because of me."

Diverse vocabulary, no obvious 3-gram repeats — but the meaning is
gone. The penalty pushes the sampler away from high-probability tokens
into low-probability paths that don't form sensible sentences.

The math-mode reflex was *not* eliminated — at 1.2 the model still
fell into `<|python_start|>10<|python_end|>...#### 10` for ambiguous
queries about numbers. The penalty hits the *post-template* tokens,
not the template *onset*.

### Rep penalty 1.4 (D) — coherence destroyed

At 1.4, outputs are pure word salad:

> *(D_anti_rep_1.4, T1 of reference_resolution)*:
> "Let me explain how I can help with that:
>
> 1. We're all four first-aid
> 2. The air is filled up by the scent of freshly brewed coffee.
> 3. There are no water in our town or just three levels."

The "math-mode" reflex shifted to **Python-code-block reflex** — model
emits markdown ```python ... ``` blocks instead of `<|python_start|>`
tags. Both are SFT-template artifacts; the rep penalty trades one
template for another rather than suppressing template-onset.

### Fake system prime (E) — backfires into self-reference

Injecting an instructional turn at the start of conversations primed
the model to be **self-referential** rather than to follow the
instruction:

> *(E_sys_prime+1.2, T1 of self_correction, in response to "What's the
> capital of Australia?")*:
> "My name is nanochat. The capital of Australia is nanochat, which
> means that I can chat about what was originally created on October
> 1, 2019. This is because my knowledge has been limited to me..."

The model interpreted the fake "user gave instructions, assistant
acknowledged" turn as "the topic is *me*, the assistant." Subsequent
turns got contaminated with identity-template responses regardless of
what the user actually asked.

A real system prompt would require SFT-time changes — adding system
messages to the chat format and training on them. Inference-time
injection isn't a substitute.

## Comparison to architectural A/B

The architectural A/B (multi-turn eval) found Stage 2 ≡ baseline at
1/7. The decoding sweep found:
- Best decoding config (B_greedy): **1.5/7**
- Worst decoding config (E_sys_prime): **0/7**
- Range across all decoding configs: 0 to 1.5

This range (1.5 - 0 = 1.5) is **larger than the architectural delta**
(0). Decoding choices matter more than architecture at d6 scale, but
*none of them is large enough to fix the chatbot* — the absolute
ceiling is barely above 1/7.

## What this rules out

1. **Rep penalty is not the lever.** No setting in [1.0, 1.4] produces
   coherent multi-turn chat. The trade is repetition-for-coherence
   with no sweet spot.
2. **Greedy is not the right default.** Identity-loop failures are
   worse than the original behavior on most prompts, even though it
   wins on prompts with user-injected facts.
3. **Inference-time system prompts don't work.** The fake-turn
   approach contaminates response style with self-reference. Real
   system prompts require SFT-time format changes.

## What's still unmeasured

The **SFT-data lever** — sketched in
`docs/sft_data_chat_quality_design_2026-05-06.md`. Variant A
(reduce `--mmlu-epochs` 3→1 and `--gsm8k-epochs` 4→1) is the cheapest
remaining experiment. Per-template token salience is hypothesized to
be the dominant cause of math-mode reflex; downweighting at training
time should hit *template onset* in a way that decoding-time penalty
does not.

If Variant A doesn't help either, the chatbot quality at d6 is bounded
by capacity (74M params), not by architecture or decoding or SFT-mix.
That would be a complete answer to "make it less bad without inflating
wall-clock or buying M4" — the answer being "not much, with current
data and this capacity."

## Lever ranking, post-experiment 2

| lever | result | next step |
|---|---|---|
| **Architecture** | closed (multi-turn A/B: tie) | none unless scale changes |
| **Decoding (rep penalty + greedy + system prime)** | **closed (this experiment: 0-1.5/7)** | greedy is borderline-better default for prompt-faithfulness, worse for free chat — situation-dependent |
| **SFT-data composition** | **only unmeasured lever remaining** | Variant A from `sft_data_chat_quality_design_2026-05-06.md` |
| **Capacity (74M)** | unfixable without M4 | — |

## Engine code change

The repetition_penalty kwarg added to `Engine.generate` and
`sample_next_token` is now part of the codebase
(`nanochat/engine.py`, default 1.0 = no behavior change). It's not a
default we'd recommend turning on — but the capability is wired in
case future experiments want it. Removable in one commit if we
decide it's permanent dead weight.

## Artefacts

- Code: `dev/decoding_sweep.py`, `nanochat/engine.py` (rep_pen kwarg)
- Transcripts: `docs/decoding_sweep_2026-05-06_transcripts.json`
- Local log: `/tmp/decoding_sweep.log`
- Comparison reference: `docs/multi_turn_chat_eval_2026-05-06.md` and
  its transcripts
