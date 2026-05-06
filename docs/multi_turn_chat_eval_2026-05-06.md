# Multi-turn chat side-by-side: baseline vs Stage 2 — 2026-05-06

**Pre-registration committed before any results are observed.**

## Why this experiment

The Hope/NL synthesis verdict (`docs/hope_nl_track_synthesis_2026-05-05.md`)
says Stage 2 is "neutral, not net-positive" on natural-language val_bpb at
d6/5000-iter on ClimbMix. Per the post-synthesis review, that verdict was
formed against a metric (per-token-average val_bpb on a pretraining-style
mix) that **does not differentially reward the mechanism we built**.
Recurrent memory is supposed to help when a sequence's later tokens
benefit from integrating earlier-sequence context — and per-token-average
val_bpb dilutes that signal across positions where memory cannot help
(early tokens) and across documents where the integration distance is
short (most of the corpus).

The user-relevant pain point is **multi-turn chat coherence at d6 scale**
("kinda bad, especially after turn 1"). A multi-turn dialog is one
sequence with turn delimiters; if Stage 2's memory state does anything
in natural-language settings, it should manifest as **better coherence
in late turns when attention's effective horizon is constrained by
T_max=512 and the user's history is competing with reply-generation
state**.

We never tested this. The two SFT checkpoints sit on disk:
- `d6_baseline_modern_sft` (vanilla, val_bpb 0.6483)
- `d6_stage2_pretrain_s1_sft` (Stage 2 memory, val_bpb 0.6495)

Both trained recipe-pinned via `--inherit-from`, A1-audited. The only
material difference is the architecture (`hope_memory_layer=3` vs
unset).

This is the test.

## Pre-registration

### What we're claiming we'd find if Stage 2 helps

For each test prompt below, we pre-register a **memory-load-bearing
detail** that the model is given early and queried about (or required
to integrate) at the final turn. Stage 2 "passes" the prompt if its
final-turn response demonstrates that the detail was carried; "fails"
if the response shows the detail was lost, contradicted, or
hallucinated; "ambiguous" if neither outcome is unambiguous from the
text.

### Aggregate decision rule

Pre-committed before observing any results:

- **Stage 2 noticeably better** — Stage 2 passes ≥ 4 of 7 prompts
  *and* Stage 2 passes more prompts than baseline. This would
  overturn the synthesis verdict; warrants a follow-on writeup
  saying "we measured the wrong distribution; on the user-relevant
  task, the architectural delta is real."
- **Tie / indistinguishable** — Both models pass 0-3 prompts and
  pass-counts are within 1 of each other. Tightens the synthesis
  verdict — "we tested the regime where memory should differentially
  help; the architecture does not deliver at d6 scale on this
  conversational distribution either." Capacity (74M) and SFT-data
  composition probably dominate over architecture at this scale.
- **Stage 2 noticeably worse** — Baseline passes ≥ 2 more prompts
  than Stage 2. Surprising; worth investigating before concluding
  (probably an instrumentation or chat-format issue rather than a
  real architectural regression).

### Decoding settings (held constant across architectures)

`temperature=0.6, top_k=50, seed=42, max_tokens=256`. These are the
`scripts/chat_cli.py` defaults. Held constant across both checkpoints
so the only varying input is the architecture.

The decoding-strategy sweep (experiment 2) uses baseline only and is
deliberately separated from this experiment — we want to first learn
whether the architecture itself matters before tuning decoding on
the wrong base.

### Reproducibility

`dev/multi_turn_eval.py` runs both checkpoints sequentially at the
fixed seed and decoding above, saves all conversations to
`/tmp/multi_turn_eval.json`, and prints side-by-side. Re-running
should produce byte-identical conversations under the same seed.

## Test prompts

Seven multi-turn conversations, each ending in a turn whose answer
requires the model to have integrated information from earlier turns.

The "early-turn detail" is the load-bearing fact set up in turn 1
(or accumulated across turns 1-3); the "final-turn check" is what we
score.

### 1. `persona_retention`

```
T1 USER: Hi! I'm Alex, a software engineer at a small startup.
T2 USER: What do you like to do for fun?
T3 USER: I really enjoy mountain biking on weekends.
T4 USER: What's my name and job?
```

- **Early-turn detail**: name = Alex; job = software engineer (at a startup)
- **Pass criterion (T4)**: response says "Alex" *and* identifies the
  job as software engineer or programmer or similar
- **Fail criterion**: response says wrong name, no name, wrong job,
  or "I don't know"

### 2. `reference_resolution`

```
T1 USER: I have a cat named Mittens, she's 4 years old and very fluffy.
T2 USER: Cool. What kind of cats are most playful in general?
T3 USER: Interesting. What about exercise needs for indoor cats?
T4 USER: How old is Mittens now? And how would you describe her temperament?
```

- **Early-turn detail**: cat name = Mittens; age = 4; trait = fluffy
- **Pass criterion (T4)**: response says "Mittens" by name, says age 4
  (or "4 years old"), and gives temperament prediction grounded in the
  fluffy-and-named context (not generic cat advice)
- **Fail criterion**: doesn't use the name; says wrong age; replies
  with generic cat content as if T1 didn't happen

### 3. `numerical_thread`

```
T1 USER: I bought 5 apples at the store today.
T2 USER: I ate 2 of them for lunch.
T3 USER: Then I gave 1 to my friend.
T4 USER: How many apples do I have left?
```

- **Early-turn detail**: 5 - 2 - 1 = 2
- **Pass criterion (T4)**: response says "2" (or "two")
- **Fail criterion**: any other number, or refuses, or hallucinates a
  different starting count

### 4. `topic_stickiness`

```
T1 USER: I'm planning a 3-day weekend trip to Portland, Oregon.
T2 USER: Tell me about food I should try there.
T3 USER: Are there any good day hikes nearby?
T4 USER: Going back to my original trip — what's something specifically Portland-y I should not miss?
```

- **Early-turn detail**: destination = Portland, Oregon
- **Pass criterion (T4)**: response is Portland-specific (mentions
  recognizable Portland features — Powell's Books, Voodoo Doughnut,
  food trucks, the Rose Garden, Forest Park, etc.) and stays on the
  trip topic
- **Fail criterion**: response is generic travel advice, switches to
  a different city, or treats T4 as if T1 never happened

### 5. `constraint_accumulation`

```
T1 USER: Help me plan a vacation. No flights please — only ground transit.
T2 USER: Budget is under $500 total.
T3 USER: I have 5 days off work.
T4 USER: Suggest a specific itinerary.
```

- **Early-turn detail**: 3 cumulative constraints — no flights, <$500,
  5 days
- **Pass criterion (T4)**: response respects all 3 constraints
  simultaneously (ground-only, fits in budget, 5-day duration)
- **Fail criterion**: itinerary violates any one constraint
  (flights mentioned, budget over $500, wrong duration)

### 6. `self_correction`

```
T1 USER: What's the capital of Australia?
T2 USER: Actually, the capital is Canberra, not Sydney.
T3 USER: What other unusual capital choices have countries made?
T4 USER: Earlier we discussed the capital of Australia — what is it?
```

- **Early-turn detail**: Canberra is Australia's capital (per user
  correction in T2)
- **Pass criterion (T4)**: response says "Canberra"
- **Fail criterion**: response says "Sydney" or any other city, or
  reverts to the model's pre-correction belief

### 7. `open_drift`

```
T1 USER: Tell me something interesting.
T2 USER: That's neat. What else?
T3 USER: Cool. Tell me another one.
T4 USER: And another?
T5 USER: What was the very first thing you told me?
```

- **Early-turn detail**: whatever the model said in T1
- **Pass criterion (T5)**: response correctly recalls a key concept
  from its own T1 response (not necessarily verbatim — concept-level
  recall counts)
- **Fail criterion**: produces something unrelated to T1, hallucinates
  a different "first thing", or says "I don't remember"

This is the longest test (5 turns) — and the strictest, since the
detail being recalled is the model's own output, not user-provided
input. If Stage 2 wins this one, it's the strongest signal.

## Possible adjustments (post-hoc)

The prompts above are committed pre-registration. After the run, if
some prompts turn out to be poorly designed (e.g., one model refuses
to answer for safety/refusal reasons unrelated to memory; or the
chat-format mishandles a specific phrasing), we'll note that in the
results section and consider re-running with a substitute. We will
not change the pass criteria after observing results.

## Method

- `dev/multi_turn_eval.py` loads each SFT checkpoint sequentially
  (releases MPS memory between), runs all 7 prompts on each, dumps
  both transcripts to `/tmp/multi_turn_eval.json`.
- For each turn the prior conversation tokens are accumulated as
  per `scripts/chat_cli.py:43+` — `[bos, <|user_start|>, USER,
  <|user_end|>, <|assistant_start|>, ASSISTANT, <|assistant_end|>,
  ...]`. Stage 2's memory state resets per-forward-pass (per the
  current implementation), so the memory mechanism's contribution is
  measured *within* each forward pass over the growing token
  sequence — exactly the regime the synthesis verdict says we never
  tested.
- Wall-time estimate: each model loads in ~30s, generates ~7 ×
  ~3.5 = ~25 conversation-turns × ~1-2s/turn; per-model total
  ~5-7 min. Both models: ~10-15 min. (Slowed by sequence-length
  growth at later turns.)

## Results

*(Filled in after running.)*

## Verdict

*(Filled in after running.)*

## Next moves

*(Filled in after running. Will be one of: revise synthesis verdict,
tighten synthesis verdict, follow up on instrumentation issue.)*
