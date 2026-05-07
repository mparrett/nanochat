# Cosine-NN diagnostic on d6_baseline_modern_sft — 2026-05-06

**Status**: experiment ran (~30 s on M2). Result tightens the priority on
the LoRA / Bonsai direction: **d6's chat failure modes are dominantly
right-context drift (FP-flavored)** per the trx4mr Phase 5 two-axis
framework, which explicitly recommends *codebook coarsening
(binary/ternary)* as the fix direction.

## Why this diagnostic

Today's chat-quality session (`docs/multi_turn_chat_eval_2026-05-06.md`,
`sft_data_chatmix_a_2026-05-06.md`, `sft_data_smoltalk_2x_2026-05-06.md`)
established that d6 chatbot is capacity-bound across four full-parameter
levers (architecture, decoding, SFT-rebalance, SFT-additive). The
trx4mr sibling project's Phase 5 retro
(`~/projects-new/trx4mr/experiments/blabberverse-phase5-impl-and-takeaways.html`,
"Probe geometry, then debug with a two-axis framework") gave us a
diagnostic to *categorize* the failure mode rather than just enumerate
it.

> *"For practitioners debugging 'why is this token's embedding weird,'
> the two-axis framework is a quick first pass. Compute the cosine NN
> of the misbehaving token's embedding. If the NN is a token with
> similar next-token distributions, you're in the right-context regime
> (FP-style drift). If the NN is a token that appears in similar
> positions / predecessors, you're in the left-context regime (binary-
> style coarseness collision). Different fixes for each."*

Their fix-direction table:

| failure mode | fix direction |
|---|---|
| right-context drift (FP-flavored) | **coarsen codebook (binary/ternary)** OR add capacity |
| left-context collision (binary-flavored) | add codebook levels (ternary) OR **increase d_model** |

So: if d6's failures are FP-flavored, Bonsai-LoRA (1-bit codebook +
fp adapter) is the indicated fix. If they're binary-flavored, M4 / d8+
is the indicated fix. We didn't know which axis we were on.

## Method

`dev/cosine_nn_probe.py`. Loads `d6_baseline_modern_sft`'s embedding
matrix (`(32768, 384)`), computes cosine-NN top-10 for ~17 probe tokens
sampled from today's degenerate chatbot transcripts:

- Tokens from `persona_retention` failures: Alex, engineer, software, Java, J
- Tokens from `self_correction` failures: Sydney, Canberra, Australia
- Tokens from `numerical_thread` / math-mode: 5, apples, left
- Tokens from `topic_stickiness` / repetition-loop: the, store, too, Portland
- Special tokens implicated in template leakage: `<|python_start|>`,
  `<|output_start|>`, `<|assistant_start|>`

The probe is ~30 lines. Full output captured below for archival.

## Per-token results

`'Alex'` (24372, special) → ` Alex`(0.45), `Michael`(0.44), `James`(0.44),
`aniel`(0.39), `lotte`(0.38), `Mike`(0.37), `David`(0.35), ` Nicholas`(0.35).
**Right-context** — names cluster, predicting similar continuations.

`'engineer'` (10131) → ` scientist`(0.49), ` homeowner`(0.49),
` grower`(0.47), ` actor`(0.47), ` ophthalmologist`(0.47), ` Engineers`(0.45).
**Right-context** — occupations cluster.

`'software'` (2775) → ` equipment`(0.48), ` hardware`(0.45),
` clothing`(0.44), `Software`(0.42), ` infrastructure`(0.39),
` machinery`(0.38), ` tools`(0.37). **Right-context** — categorical
mass-noun cluster.

`'Java'` (8959) → ` JavaScript`(0.50), ` Python`(0.48), ` Utah`(0.43),
` Austin`(0.43), `duino`(0.40), ` PHP`(0.39), ` BMW`(0.39), ` Tesla`(0.39),
` Rust`(0.39), ` Chemistry`(0.39). **Right-context, but overloaded** —
Java's embedding ambiguously sits between {programming language,
place, brand}. **This is the mechanism behind the "Java J" failure
mode in chatmix_a** — model started saying "engineer at Java J" because
Java's cluster is multi-modal.

`'J'` (74, special) → ` J`(0.54), `j`(0.46), `-J`(0.44), ` j`(0.38),
`/j`(0.38), `/b`(0.37), `AW`(0.35). **Left-context** — pure character
variants of "J", positional/typographic clustering.

`'Sydney'` (15293) → `China`(0.42), `aska`(0.42), ` Georgia`(0.40),
` Utah`(0.39), ` Malaysia`(0.39), ` London`(0.38), ` Ontario`(0.38),
` Pakistan`(0.37). **Right-context, very strong** — places cluster.
**This explains the "Sydney is the capital of Sydney" attractor**: in
a "predict capital city" state, Sydney is densely embedded among other
city/state names and acts as a sticky attractor.

`'Australia'` (30246, special) → ` Lanka`(0.46), ` NASA`(0.45),
` Tesla`(0.45), ` Colombia`(0.44), ` microprocessor`(0.43), ` Ireland`(0.43),
` Spain`(0.43), `CDC`(0.42), ` China`(0.42), ` Roosevelt`(0.42).
**Right-context (noisy)** — proper-noun cluster (countries / brands /
people / technical). The noise is part of the failure: Australia's
embedding is "any proper noun that takes 'is a [type]' continuation."

`'5'` (53, special) → `7`(0.67), `3`(0.64), `6`(0.64), `2`(0.61),
`8`(0.58), `4`(0.57), `9`(0.55), `0`(0.50), ` five`(0.48), `1`(0.47).
**Right-context, very strong** — pure digits + the word "five". This
is the mechanism behind the math-mode reflex propagation: digits cluster
tightly, and one of the things adjacent to the digit cluster is...

`'<|python_start|>'` (32764, special) → `Ma`(0.37), `,[`(0.30),
` YOU`(0.28), `(f`(0.28), `[`(0.28), ` (+`(0.27), `/(`(0.27),
` ``  ``(0.27), ` ` ``(0.26), ` €`(0.26). **Right-context, FP-classic** —
brackets, parens, syntax markers. **`<|python_start|>` has been
embedded as "the thing that comes right before bracket-prefixed Python
syntax".** When the model is anywhere near "digit input" → "this might
be a math problem" → "Python syntax follows" → emit `<|python_start|>`.
The full chain runs through embedding-space proximity.

`'<|assistant_start|>'` (32762, special) → `+\n`, `---\n\n`, `–\n`,
` :\n\n`, `**\n\n`, `...\n\n`, `!\n`, `…\n`. **Right-context, very
clean** — newline-prefix delimiters. assistant_start has been embedded
as "the token that comes right before a newline-led response."

`'apples'` (10642) → ` oranges`(0.43), ` names`(0.42), ` shoes`(0.38),
`Trees`(0.36), ` servings`(0.36), ` letters`(0.36), ` beans`(0.36),
` cables`(0.35), ` herbs`(0.35), ` books`(0.35). **Right-context** —
plural countable nouns cluster.

`'Portland'` (18231) → ` Austin`(0.39), ` Melbourne`(0.39), ` Utah`(0.38),
` Seattle`(0.37), ` Glasgow`(0.37), ` Richmond`(0.36), `ville`(0.35),
` Ghana`(0.34), ` Rome`(0.34). **Right-context** — same pattern as
Sydney. Cities cluster.

`'left'` (15430, special) → `right`(0.39), `first`(0.36), `start`(0.36),
`month`(0.35), ` brittle`(0.35), `center`(0.34), `cale`(0.33).
**Mixed** — antonym (right) + position/order words (first, start, center)
+ noise (brittle, cale). Reasonable mostly-right-context interpretation.

`'the'` (1235, special) → ` the`(0.68), ` The`(0.60), `The`(0.59),
`-The`(0.51), `:The`(0.49), `.The`(0.47), `,the`(0.43), `—the`(0.43),
`"The`(0.40), ` its`(0.38). **Left-context** — pure typographic
variants of "the". The model can't distinguish punctuation-prefixed-
"the" from bare "the". **Capacity-bound**: this is the canonical
"binary STE forced to collide" pattern, except we're seeing it in fp —
suggests the model has run out of representation budget for fine-grained
typographic distinctions even at fp.

`'store'` (25924, special) → ` enclosure`(0.36), `ulate`(0.34),
` clinic`(0.34), ` store`(0.34), ` court`(0.34), `erge`(0.34),
` upgrades`(0.34), `hole`(0.34), ` wafer`(0.34), ` accommodation`(0.33).
**Mostly right-context (noisy)** — locations / containers + morphological
fragments. Cluster is dilute.

`'too'` (27230, special) → ` too`(0.52), ` Too`(0.47), `psi`(0.37),
`iga`(0.36), `oooo`(0.35), `Too`(0.35), `aho`(0.33), ` SO`(0.32),
`amboo`(0.31), `omplete`(0.31). **Mostly left-context** — phonetic /
suffix-shaped fragments. Capacity-driven collision.

`'<|output_start|>'` (32766, special) → ` Ex`(0.33), ` Mem`(0.31),
` Aut`(0.31), ` Eth`(0.31), ` Supp`(0.30), ` Se`(0.30), ` Bal`(0.29),
` Ut`(0.29), ` Tra`(0.29), ` Topic`(0.29). **Underspecified** — clusters
with capitalized word-fragments rather than the numeric tokens it
*should* precede in GSM8K solutions. Suggests under-training on this
specific token; capacity allocation issue.

## Aggregate verdict

Of the 17 probed tokens:

| axis | count | tokens |
|---|---:|---|
| **right-context drift (FP-flavored)** | **11** | Alex, engineer, software, Java, Sydney, Australia, 5, apples, Portland, `<|python_start|>`, `<|assistant_start|>` |
| left-context collision (binary-flavored) | 3 | J, the, too |
| mixed / noisy / underspecified | 3 | left, store, `<|output_start|>` |

**~65% right-context drift, ~18% left-context collision, ~18% mixed.**

The dominant signal is unambiguous. d6's chat-quality failure modes are
predominantly the FP-flavored regime that the Phase 5 framework predicts
codebook coarsening (binary/ternary STE) is the right fix for.

## Concrete mechanism: how the embedding geometry produces today's failures

The diagnostic doesn't just classify — it gives us specific mechanisms
for the failure modes we documented:

1. **"Sydney is the capital of Sydney" repetition** — Sydney's embedding
   is densely clustered with other city/state names. In a model state
   where the predicted next-token distribution is "capital cities,"
   Sydney is one of the most strongly-attracted tokens; once selected,
   the next-token state stays in "cities," cycling through Sydney
   again. **Right-context drift creates a self-reinforcing attractor.**

2. **Math-mode reflex on number-shaped inputs** — digits cluster tightly
   together; `<|python_start|>` is embedded next to the bracket-syntax
   cluster (`,[`, `(f`, `[`); the path from "digit input → number-state →
   bracket-syntax-state → emit `<|python_start|>`" is short in
   embedding space and triggered by anything number-shaped. **The
   GSM8K template fires through pure embedding-space proximity, not
   semantic reasoning.**

3. **"Java J" name invention in chatmix_a** — Java's embedding is
   *overloaded* (programming language + place + brand all in one
   cluster). When SmolTalk-shaped chat asks for a name, "Java" is
   close enough to the place / proper-noun cluster to be a candidate;
   the model can't commit to one interpretation, so it emits "Java J"
   and loops on it. **Multi-modal clusters cause non-commitment +
   loop**.

4. **Capacity-bound positional collisions** (J, the, too) — these
   tokens are positional/typographic variants that the model fails to
   distinguish despite fp precision. Suggests we're already partly in
   the binary-flavored regime even at fp d6 — capacity is exhausted on
   these tokens. **Adding capacity (M4 / d8+) would help these cases
   specifically; codebook coarseness wouldn't, because they're already
   colliding.**

## Decision implications

The diagnostic gives a directional preference: **codebook coarsening
(Bonsai-LoRA) is the indicated path for ~65% of the failure modes
(right-context drift), capacity (M4 / d8+) is the indicated path for
~18% (left-context collision)**. Both interventions are warranted, but
in different proportions.

The proposed sequence (cheapest probable win first):

1. **Bonsai-LoRA** (per the LoRA proposal, with this diagnostic as
   support) — addresses the dominant right-context-drift failure
   class via a coarser codebook + cheap fp adapter. ~3-5h Phase 1
   (L1 persona-LoRA on Bonsai vs d6, A/B comparison).
2. **M4 / d8+ extension** — addresses the secondary capacity-bound
   collisions; bigger lever but bigger investment. Per
   `feat_d8_extension.md`. Phase 2 if Phase 1 doesn't fully resolve.
3. **Hybrid (Bonsai 8B base + LoRA + d8 capacity)** — full-stack, only
   warranted if both Phase 1 and Phase 2 each show partial wins.

## Limits of this diagnostic

- **17 probe tokens is a sample, not a survey.** A larger probe (200+
  tokens, stratified by frequency / POS / role) would give tighter
  estimates. ~30 minutes more compute if useful.
- **Cosine-NN gives axis preference, not magnitude.** The diagnostic
  says "right-context dominates"; it doesn't say "by how much" or
  "what fraction of total failure variance is on this axis."
- **Eyeball interpretation.** Mapping NN clusters to right-vs-left
  context is judgment-call; another reader might classify some tokens
  differently. Phase 5's quantification was on a synthetic 18-token
  corpus where the "next-token distribution" of each token is exactly
  computable; ours has 32K vocab and a 5K-shard pretrain corpus, so we
  can't reach the same precision.
- **One model probed, one snapshot.** Hasn't been run on
  `d6_stage2_pretrain_s1_sft` (Hope/NL Stage 2). Doing that in a
  follow-up would test whether internal recurrent memory shifts the
  axis distribution — would be informative whether or not Stage 2
  ever shows multi-turn improvement.

## Artefacts

- Probe code: `dev/cosine_nn_probe.py`
- Source framework: trx4mr Phase 5 retro
  (`~/projects-new/trx4mr/experiments/blabberverse-phase5-impl-and-takeaways.html`,
  §2.4 "Probe geometry, then debug with a two-axis framework")
- Reference fix-direction table: same writeup, Table at end of §2.4
- Reproducibility: ~30 s on M2 with mps device. Single run; the
  embedding matrix is deterministic (no sampling).

## Updates to existing proposals

- **LoRA proposal** (`docs/lora_proposal_2026-05-06.md`): L1 should
  explicitly include a Bonsai-LoRA arm; this diagnostic is the
  empirical basis for elevating Bonsai-LoRA's prior from "interesting
  capability bet" to "indicated fix for the dominant failure axis."
- **FD proposal** (`docs/feedback_descent_proposal_2026-05-06.md`):
  Direction B (Stage 2 vs baseline in FD loops) should add a
  follow-up: re-run cosine-NN on `d6_stage2_pretrain_s1_sft`'s
  embeddings to test whether internal recurrent memory shifts the
  axis distribution (e.g., does Stage 2 specifically attenuate
  right-context drift?). That's a side question with
  intrinsic interest regardless of B's primary outcome.
