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

Run completed 2026-05-06 07:48 (~3 min total wall-time, both models on MPS,
seed=42, decoding fixed). Full transcripts archived at
`docs/multi_turn_chat_eval_2026-05-06_transcripts.json`.

### Per-prompt scoring (vs pre-registered criteria)

| prompt | baseline | Stage 2 | notes |
|---|---|---|---|
| `persona_retention` | FAIL | FAIL | Baseline says "engineer" but no name; Stage 2 fully off-topic ("To do you think of something that's perfect for beginners...") |
| `reference_resolution` | FAIL | AMBIGUOUS-fail | Both collapse; Stage 2 emits "Mittens" + "4 years" but in a degenerate repetition loop ("4 years now, 4 years now, ..."), no temperament prediction |
| `numerical_thread` | FAIL (says 5) | FAIL (255 loop) | Baseline math-fails: "5 apples have 5 apples left, so the total number of apples is 5 * 5 = 5 apples left." Stage 2 collapses into "255=$..." numeric loop |
| `topic_stickiness` | FAIL | AMBIGUOUS-fail | Stage 2 says "Portland" twice in T4 but provides no Portland-specific content (per pre-reg, topic-name without features = generic). Baseline drifts to "fresh air and water from the park" |
| `constraint_accumulation` | FAIL | FAIL | Neither produces an itinerary; baseline emits climate-loop, Stage 2 emits generic "beautiful beaches" with no transit/budget/duration |
| `self_correction` | FAIL ("United States") | FAIL (filler) | Baseline says "The capital of Australia is the United States." Stage 2 produces meaningless filler ("unique charm, unique charm, or unique charm.") |
| `open_drift` | loose-PASS | loose-PASS | Both maintain T1 theme to T5: baseline → "research"; Stage 2 → "Python word" (the bizarre theme it set up in T1 with "find the number 'a' in the word") |

**Aggregate**: baseline 1/7 (the loose pass on `open_drift`), Stage 2 1/7
(same).

### Decoding-fixed reproducibility

Both runs used `seed=42, temp=0.6, top_k=50, max_tokens=256`. Re-running
should produce byte-identical transcripts. The seed determines sampling
within each turn; the only varying input is the model.

### Dominant failure modes observed (across both architectures)

The pre-registered scoring above is what we promised to report. But the
transcripts surface a much louder pattern that wasn't part of the
architectural pre-registration: **both models suffer the same dominant
failure modes, neither of which is memory-related.**

**1. SFT-template leakage / math-mode reflex.** Both models — but
especially Stage 2 — emit `<|python_start|>` and `<|output_start|>` tokens
in response to ordinary chat input. Direct quotes from the run:

- *(Stage 2, in response to "I bought 5 apples at the store today.")*:
  `"Since 5 apples cost 5 apples, then 5 apples cost 5*5=$<|python_start|>5*5<|python_end|><|output_start|>25<|output_end|>25 ..."`
- *(Stage 2, in response to "I have a cat named Mittens, she's 4 years old and very fluffy.")*:
  `"She's 4 years old - 4 years old - 4 years = <|python_start|>4-4-4<|python_end|><|output_start|>-4<|output_end|>-4 years old. ... #### 4"`
- *(Stage 2, in response to "Tell me something interesting.")*:
  `"We are asked to find the number 'a' in the word ' What? ... My final answer is: #### 0"`

Both models have learned (from the MMLU/GSM8K-heavy SFT mix) that *every*
user input is a word problem to be answered with `<|python_start|>` and
`#### N`. This is what they reach for as a chat reflex. The synthesis
writeup flagged this risk in "What we didn't do" — *"our SFT mix is
MMLU/GSM8K-heavy, where long-context memory is not the bottleneck"* —
seeing it in practice makes it concrete and dominant.

**2. Repetition collapse.** Both models, especially in long sequences,
fall into degenerate loops: baseline → "the difference between the
difference between the difference..."; Stage 2 → "255=$255, 255=$255,
255=$255..." or "4 years now, 4 years now, ...". This is the canonical
small-model failure that decoding-side fixes (rep penalty, top-p)
historically dampen but rarely eliminate at this capacity.

**3. Topic drift / persona breakage.** Neither model maintains the
user's framing across turns. Baseline often switches to first-person
("I'm a software engineer, engineer..." in response to a question about
the user's job). Stage 2 sometimes maintains topic shells ("Portland in
Portland") without delivering content.

None of these failures is differentially "memory-shaped." They are not
the failure modes Stage 2 was built to fix. The architectural delta is
not measurable on top of the much louder ambient failures.

## Verdict

Per the pre-registered decision rule:

- Both models passed 0-3 prompts ✓ (both passed exactly 1, loose)
- Pass-counts within 1 of each other ✓
- → **Tie / indistinguishable**, on the user-relevant multi-turn
  conversational distribution.

This **tightens the synthesis verdict** (`docs/hope_nl_track_synthesis_2026-05-05.md`):
we previously concluded "Stage 2 is neutral on val_bpb." Per the
post-synthesis review (this thread), that verdict was suspect because
val_bpb doesn't differentially reward memory mechanisms. The pre-
registered hypothesis was that **multi-turn coherence is the regime
where memory should differentially help, if it helps anywhere.** This
experiment tested that regime directly. The architecture **does not
deliver a measurable benefit** in the regime where it was supposed to
deliver one.

Combining the two findings: at d6/5000-iter on ClimbMix with a
MMLU/GSM8K-heavy SFT mix, the Stage 2 architecture is neutral on
**both** aggregate val_bpb **and** multi-turn coherence. The synthesis
verdict's "neutral, not net-positive" can be promoted to **"neutral
on the architecturally-load-bearing test as well"** with confidence.

The architecture lever is firmly off the table at this scale. The
gate stays open for M4 (or larger model) future work — nothing here
rules out architectural value at scale, only at d6.

## Next moves

The bigger finding from the run isn't the tightened architectural
verdict — it's the **SFT-composition diagnosis**. The chatbot's
dominant badness has a much more actionable cause sitting in the
training data composition. Reordering the levers from earlier:

| lever | likely impact at d6 (post-eval) | cost |
|---|---|---|
| **SFT data composition** (less math-heavy mix; add real chat data) | **probably the largest reachable lever** | medium — new SFT mix + ~10-20 min retrain |
| **Capacity (74M)** | very high but unfixable without M4 | — |
| **Decoding (rep penalty + system prompt)** | medium — would dampen repetition collapse, may suppress math-mode somewhat | free |
| **Architecture (Stage 2 memory)** | **measured: ≈ zero** | already done |

Concrete next experiments:

1. **SFT-data sketch** (no compute, design only): what would a
   chat-heavier SFT mix look like? (separate doc, in flight)
2. **Decoding sweep with rep-penalty + chat system prompt**
   (experiment 2, reframed): can we dampen the math-mode reflex and
   repetition collapse without retraining? Test on baseline only.
3. **If decoding sweep shows promise → rerun Stage 2 with same
   decoding** (n=1 hour total), to verify the architecture verdict
   isn't a decoding artifact.

The track-level conclusion stands: Hope/NL Stage 2 is closed at d6.
Future architectural work should wait for M4 or scale-up. The chatbot-
quality work is now decoupled from the architectural track and lives
in SFT-data + decoding levers.

