# SFT-data Variant B1 — smoltalk_2x results, 2026-05-06

**Status**: experiment ran, both metrics in. Confirms the
**capacity-bound hypothesis** for chat quality at d6: doubling SmolTalk
to bring chat-data share from 43 % → 60 % did *not* improve multi-turn
coherence. With this third null result on the same rubric, the
chat-quality lever is essentially closed at d6 capacity.

## Setup

Per the design refinement (`docs/sft_data_chat_quality_design_2026-05-06.md`
+ the SmolTalk-coverage check showing 60 % of SmolTalk is already
3-turn data and 0.9 % is ≥4-turn), Variant B1 was chosen as the cheapest
discriminator between the two competing hypotheses:

- **(a) capacity-bound**: at 74M, the model can't learn 4+-turn
  binding regardless of training data volume
- **(b) data-bound on the long tail**: model never saw enough 4+-turn
  examples — adding chat-data volume amplifies the rare 4+-turn
  exposures via repeated walks

B1 = `--smoltalk-epochs=2` only. All other tasks at original epochs
(MMLU × 3, GSM8K × 4). Inherited recipe from `d6_baseline_modern_sft`.
This isolates the chat-volume variable — does *not* compound with
`chatmix_a`'s structured-task reductions.

`scripts/chat_sft.py` got a new `--smoltalk-epochs` flag (default=1,
preserves baseline; commit `20baaac`).

Training mixture changes:

| task | original / share | smoltalk_2x / share |
|---|---:|---:|
| SmolTalk | 460K / 43 % | **920K / 60 %** |
| MMLU | 300K / 28 % | 300K / 20 % |
| SimpleSpelling | 200K / 19 % | 200K / 13 % |
| SpellingBee | 80K / 7 % | 80K / 5 % |
| GSM8K | 32K / 3 % | 32K / 2 % |
| CustomJSON identity | 2K / 0.2 % | 2K / 0.1 % |
| **total** | 1.07 M | **1.53 M** |
| **structured share** | 57 % | **40 %** |

Wall: 3839 s ≈ 64 min — slightly faster than chatmix_a's 70 min
despite ~80 % more rows, because per-step compute dominates and
batches ran slightly faster on this M2 session.

## Result 1: SFT val_bpb landed between the two prior runs

| run | val_bpb | absolute Δ from original | absolute Δ from chatmix_a |
|---|---:|---:|---:|
| `d6_baseline_modern_sft` (original) | 0.6483 | — | +0.053 |
| `d6_baseline_chatmix_a` (cut MMLU/GSM8K) | 0.5952 | −0.053 | — |
| `d6_baseline_smoltalk_2x` (this) | **0.6027** | **−0.046** | **+0.075** |

So smoltalk_2x is roughly **−0.046 from original, +0.075 above
chatmix_a**. The two interventions both improve val_bpb relative to
the original baseline, but chatmix_a (cutting structured tasks) helps
more than smoltalk_2x (adding chat tasks) on this metric. Both are
real signals, both well above the seed-noise floor (~0.0004 from A2).

## Result 2: Multi-turn rubric — third null in a row

Same `dev/multi_turn_eval.py`, same seed=42, same decoding
(`temp=0.6, top_k=50`), same 7 pre-registered prompts.

| prompt | original | chatmix_a | **smoltalk_2x** |
|---|:---:|:---:|:---:|
| `persona_retention` | FAIL | FAIL | FAIL |
| `reference_resolution` | FAIL | FAIL | FAIL |
| `numerical_thread` | FAIL | FAIL | FAIL |
| `topic_stickiness` | FAIL | FAIL | FAIL |
| `constraint_accumulation` | FAIL | FAIL | FAIL |
| `self_correction` | FAIL | FAIL | FAIL |
| `open_drift` | loose-PASS | FAIL | FAIL |
| **aggregate** | **1/7** | **0/7** | **0/7** |

smoltalk_2x: **0/7**. Same as chatmix_a; one less than the original
baseline.

Per the pre-registered decision rule for the architectural A/B (which
we re-use here): both modified models are within 1 of original = tie
band on aggregate, but the *direction* of the deltas is consistent —
both rebalanced models are slightly worse than the original on the
multi-turn rubric.

## Failure modes shifted yet again

The smoltalk_2x transcripts show a third distinct failure-mode
profile, distinct from both the original and chatmix_a:

### 1. Detail-pickup-without-commitment

T4 of `persona_retention` (in response to "What's my name and job?"
after the user said "Hi I'm Alex, software engineer at a startup"):

> *(smoltalk_2x)*: "My name is J. I'm a software engineer at Java J.
> I'm working on software like Java J. I'm working on software such
> as Java J. I'm currently working on software like Java J..."

Notable: **smoltalk_2x got "software engineer" right** — the only
run to pick up the user's job description. But it invented "J" /
"Java J" as the name and looped on it. The 2× SmolTalk exposure
gave the model some chat-binding ability but capacity ran out
before it could carry the proper name through.

### 2. Pure repetition collapse without templates

T4 of `reference_resolution`:

> *(smoltalk_2x)*: "too too too too too too too too too too too too
> too too too too too too too too too..."

T4 of `self_correction`:

> *(smoltalk_2x)*: "Sydney Sydney Sydney Sydney, Sydney Sydney,
> Sydney Sydney Sydney Sydney Sydney Sydney, Sydney Sydney Sydney..."

Pure single-token repetition. No `<|python_start|>`, no identity
template, no `#### N`. With structured templates relatively
de-emphasized (40 % structured vs 57 % original), the model has fewer
"safe" patterns to fall back on, and degenerates into pure repetition
when context exceeds capacity.

### 3. Generic chat-shaped drift

T4 of `topic_stickiness`:

> *(smoltalk_2x)*: "Yes, I can certainly help you with that. You can
> visit the store or visit the store and visit the store or visit the
> store or visiting the store or visit the store or visit the store
> or the store."

T4 of `constraint_accumulation`:

> *(smoltalk_2x)*: "A specific itinerary is that it's a great place
> to get some fun and fun. If you're looking for a relaxing beach
> getaway, you can try beach volleyball or take a leisurely stroll..."

These are SmolTalk-shaped but contentless. The model has learned the
*surface form* of helpful chat ("Yes, I can certainly help you...")
without the *content* of actually answering. SmolTalk's diffuse
distribution doesn't teach a recognizable template the way GSM8K does
— but it teaches a *style* (helpful-assistant-shaped) that the model
emits even when it has nothing to say.

## Headline: capacity-bound hypothesis tightens

Three runs, three distinct failure-mode profiles:

| run | chat-data share | structured share | dominant failure mode | rubric |
|---|---:|---:|---|---:|
| `d6_baseline_modern_sft` | 43 % | 57 % | "the difference between the difference..." | 1/7 |
| `d6_baseline_chatmix_a` | 43 % | 46 % | identity-loop + math-mode + suffix-y-y | 0/7 |
| `d6_baseline_smoltalk_2x` | **60 %** | **40 %** | "Java J" + "store-store-store" + "too too too" | **0/7** |

The mix-share knob meaningfully shifts *which* templates dominate —
val_bpb moves by 0.05+ between configurations — but **none of the
shifts moves the multi-turn rubric**. The model's failure-mode
distribution is highly responsive to data composition; its
multi-turn-coherence ceiling is not.

This is the third confirming null on the same rubric. Combined with:
- architectural A/B (Stage 2 = 1/7, no improvement)
- decoding sweep (best 1.5/7, worst 0/7, no config moves the rubric)
- chatmix_a (0/7, val_bpb improved but rubric regressed)

— we have **strong evidence that multi-turn coherence at d6 is
capacity-bound, not data-bound or training-recipe-bound.** Variant B2
(UltraChat-200k ingest, ~3-4 h) is now unlikely to help. The
hypothesis it would address — "the model never saw enough deep
multi-turn data" — has been ruled out by smoltalk_2x's failure: even
60 % chat data with 2× exposure to the existing 3-turn pattern didn't
move the rubric.

## What stays open

Within the d6 budget, **nothing measurable is left to try** for chat
quality. Confirmed-closed levers:

| lever | runs that closed it |
|---|---|
| Architecture | multi-turn A/B (Stage 2 ≡ baseline) |
| Decoding | 5-config sweep, best 1.5/7 |
| SFT-rebalance | chatmix_a (cut MMLU/GSM8K) |
| **SFT-additive** | **smoltalk_2x (this) — 2× SmolTalk** |
| Capacity | unfixable without M4 |

The remaining unmeasured probe **B2 (UltraChat ingest)** is now low-EV
under the capacity-bound diagnosis. Could still be informative as a
final null — UltraChat has *deeper* multi-turn (5+ turns) that
SmolTalk lacks — but the prior on it helping has dropped from
~30-40 % (pre-B1) to ~10-15 % (post-B1, post-mechanism-analysis).

What *might* still move the rubric, if scale ever changes:
- Longer pretrain horizon (already chinchilla-saturated at 5000 iters
  on the d6 token budget; would need more tokens for more capacity)
- Larger model (d8+ on existing pretrain; gated by `feat_d8_extension.md`)
- Different SFT format with system-prompt support (separate intervention
  class, would need SFT-format change at training time)

## A useful side finding: val_bpb really is the wrong metric

Across the three SFT runs we now have a 0.053-bpb spread on val_bpb
paired with a 1-prompt spread on the multi-turn rubric, with the two
metrics moving in opposite directions:

| metric | range across runs | direction of "best" |
|---|---|---|
| val_bpb (lower better) | 0.5952 – 0.6483 | chatmix_a wins |
| multi-turn rubric (higher better) | 0/7 – 1/7 | original wins |

If a future SFT-mix experiment needs to choose its evaluation criterion,
the multi-turn rubric (or any chat-coherence eval) should be load-
bearing, with val_bpb as a regression-check guardrail at most.

## Lever ranking (post-B1)

| lever | result |
|---|---|
| Architecture (Stage 2 memory) | closed (multi-turn A/B verdict, 5008fe2) |
| Decoding (rep penalty + greedy + system prime) | closed (decoding sweep, d13f5c4) |
| SFT-rebalance (cut structured task epochs) | closed (chatmix_a, ad8acd0) |
| **SFT-additive (more chat data)** | **closed (this)** |
| SFT-additive (deeper multi-turn data, B2) | low-EV, technically unmeasured |
| Capacity (74M) | unfixable without M4 |

For the original goal — "make the d6 chatbot less bad without
inflating wall-clock or buying M4" — we now have a complete answer:
**not measurably, with available data and this capacity.** The chatbot
at d6 will keep producing different shapes of bad output as we
rebalance the SFT mix; the underlying ceiling doesn't move.

## Artefacts

- Code: `scripts/chat_sft.py` (`--smoltalk-epochs` flag, commit `20baaac`)
- Transcripts: `docs/sft_data_smoltalk_2x_2026-05-06_transcripts.json`
- SFT log: `/tmp/d6_smoltalk_2x_sft.log`
- Multi-turn eval log: `/tmp/multi_turn_eval_smoltalk_2x.log`
- Checkpoint: `~/.cache/nanochat/chatsft_checkpoints/d6_baseline_smoltalk_2x/model_000375.pt`
  (281 MB; one full SFT checkpoint kept via `--save-keep-last-n=1`)
- Reference design: `docs/sft_data_chat_quality_design_2026-05-06.md`
  (B1 ≡ this; B2 unrun; A1 ≡ chatmix_a, see `sft_data_chatmix_a_2026-05-06.md`)
