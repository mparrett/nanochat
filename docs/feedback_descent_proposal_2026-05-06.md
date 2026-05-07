# Proposal: Feedback Descent at laptop scale — starting directions

**Date:** 2026-05-06
**Source paper:** Lee, Boen, Finn, *"Feedback Descent: Open-Ended Text Optimization via Pairwise Comparison"* (arXiv 2511.07919, Stanford IRIS Lab, Dec 2025).
**Local copy:** `~/Library/Mobile Documents/com~apple~CloudDocs/Downloads/Pairwise-2511.07919.pdf`
**Status:** proposal only, no compute spent.

## Why this is interesting now

Today's chat-quality session closed four levers at d6 capacity: architecture
(Hope/NL Stage 2), decoding strategy, SFT-rebalance, and SFT-additive volume.
The verdict was *"d6 chat is capacity-bound; the only remaining lever is M4
or new compute."*

But that verdict is **conditional on the task being free multi-turn chat.**
Feedback Descent (FD) reframes the question by introducing a different task
class — *iterative optimization of text artifacts under structured feedback*
— where the bottleneck is not "be a good chatbot" but "given a current
artifact + a critique, propose a targeted mutation." That's a much narrower
job, and it's plausibly within d6's capability range even when free chat
isn't.

It also reopens the Hope/NL architecture question on a *different* task. The
synthesis verdict was that Stage 2 ≡ baseline on natural-language val_bpb
and on the multi-turn rubric. **Neither metric measures "conditioning on
feedback to produce a targeted edit."** Internal recurrent memory might
help on FD-style mutation conditioning specifically, even though it doesn't
help on free chat. If true, that's a novel contribution — same architectural
delta, different task, opposite verdict.

## Available compute / models (operator inventory)

Establishing what's locally available before designing experiments:

| model | role | source | notes |
|---|---|---|---|
| **d6_baseline_modern_sft** (74M) | generator and/or evaluator (laptop) | `~/.cache/nanochat/chatsft_checkpoints/` | what we've been working with |
| **d6_stage2_pretrain_s1_sft** (74M) | generator and/or evaluator (laptop) | same | the Hope/NL Stage 2 SFT — internal memory at L3 |
| **Bonsai 4B / 8B** (1-bit) | generator (laptop, fast inference) | local quantized models, "on par with early GPT-3.5/4" per operator | unblocks higher-capability local generation |
| **Claude Haiku** | generator and/or evaluator (API) | Anthropic API | external option; per-call cost |
| **GPT-4o-mini, GPT-5-mini, etc.** | reference baseline | API | what the paper used |

The dichotomy the paper highlights — *generator vs evaluator* — has a real
asymmetry: **pairwise judgment is generally easier than open-ended
generation.** That means we can mix-and-match: a strong evaluator paired
with a weaker generator (testing if d6 can be a useful FD generator under
external supervision); or a weak evaluator paired with a strong generator
(testing if d6/Bonsai can do useful local judging while a bigger model
generates). Both ends of that asymmetry are useful experiments.

## Experimental design space (the axes)

| axis | options |
|---|---|
| **Generator** | d6_baseline / d6_stage2 / Bonsai-4B / Bonsai-8B / Claude Haiku |
| **Evaluator** | d6_baseline / d6_stage2 / Bonsai / Claude Haiku / paper's GPT-5-mini |
| **Task** | SVG unicorn (paper) / system-prompt optimization / poem refinement / molecule discovery / d6 chatbot system-prompt optimization |
| **Memory mechanism** | external (FD context history only) / internal (Stage 2 weights) / hybrid (Stage 2 + FD context) |
| **Loop length** | T = 5-10 (cheap) / T = 30-50 (paper-grade) |

Pure cartesian product is too much; the proposal below picks four cuts
through this space, each chosen to answer a different question.

---

## Direction A — *Can d6 be a useful FD mutator?* (cheapest probe)

**Question.** The paper's results are with GPT-4o-mini / GPT-5-mini as
generators. Whether 74M-param d6 can usefully condition on textual feedback
to produce targeted edits is unknown. This is the **gating question** for
all subsequent local-FD experiments.

**Setup.** No FD loop yet — just the **mutation step in isolation**. Pick
~10 (artifact, critique) pairs across narrow tasks within d6's range:

- Short prompt rewrites (3-5 sentences each)
- Simple SVG snippets (4-line shapes)
- Short story titles (one line each)

For each, prompt d6 with `(current_artifact, critique)` and ask for a
revision. Score: **does the revision incorporate the critique
direction?** Eyeball + a structured pass/fail on whether the requested
property changed (e.g., "make this more concise" → response is shorter and
preserves meaning).

This isolates FD's *load-bearing capability* (in-context learning to mutate
under feedback) from the rest of the loop.

**What we'd learn.**
- If d6 passes ≥ 6/10: d6 is plausibly a usable FD generator on narrow
  tasks. Direction B/C/D become live.
- If d6 fails (< 3/10): use d6 only as evaluator (Direction E) or fall back
  to Bonsai-as-generator (Direction C). The capacity ceiling we found for
  free chat *also* applies to feedback-conditioned mutation.
- If d6 lands in the middle (3-6/10): probably scope-dependent —
  works on some tasks, not others. Identifies which task class is
  reachable.

**Cost.** ~1 hour. ~10 prompts × ~30 s/each generation × manual review.

**Risk.** None — just a one-shot diagnostic. Compute trivial.

---

## Direction B — *Does internal memory help on FD-shaped tasks?*

**Question.** The Hope/NL synthesis closed Stage 2 on free chat. **It did
not test Stage 2 on tasks that exercise the architectural feature (state
across positions in a sequence) the way FD does.** An FD loop's prompt
*grows monotonically* — each iteration appends `(x_t, r_t)` to the
context. This is precisely the long-attention regime where architectural
recurrent state should help (or hurt) differentially. If Stage 2's memory
mechanism does *anything* on natural language, this is the regime where it
should show.

**Setup.** Pick one task that fits comfortably in d6's max_seq_len (e.g.,
short prompt optimization with 5-iteration history). Run three arms:

1. d6_baseline_modern_sft as generator (no internal memory — pure FD
   external context) — `arm_baseline`
2. d6_stage2_pretrain_s1_sft as generator (internal memory at L3 + FD
   external context) — `arm_stage2`
3. d6_baseline_modern_sft as generator with **shuffled** feedback history
   (control for "is the iteration history doing anything at all?") —
   `arm_shuffled`

Same task, same evaluator (Bonsai or Haiku), same iteration budget T=10,
same seed. Score by task-specific metric.

**What we'd learn.**
- arm_stage2 > arm_baseline: **the architectural memory we built has a
  real, measurable benefit on a different task class than free chat.**
  Reopens the Hope/NL track on a different metric. Directly contradicts
  the synthesis verdict's scope.
- arm_stage2 ≡ arm_baseline: capacity ceiling generalizes across task
  types. Stage 2 is closed for real.
- arm_shuffled < arm_baseline: confirms that feedback *order* matters
  (sanity check for the FD framing itself).

**Cost.** ~3-4 hours. Tasks need to be fast (each FD iteration is one
forward pass at growing T); 10 iterations × 3 arms × ~10 trials per arm =
~300 generations + evaluator calls. Need to write the FD loop runner
(~150-200 lines, one-shot script).

**Risk.** Low compute risk. Real risk: max_seq_len=512 may bound how
many iterations T fit in context — limits the loop depth. Mitigation: pick
short artifacts (1-3 sentence prompts) so 10 iterations of `(artifact,
rationale)` fit in 512 tokens.

**This is the experiment that tests "internal vs external memory" head-on
at d6 scale.** It's the most novel of the four directions — the paper
doesn't run anything like it, and we have the closed-track checkpoints
already trained.

---

## Direction C — *Replicate FD's SVG result with local generator*

**Question.** Does the paper's mechanism transfer to local-only setups, or
is it specifically a property of frontier-model in-context learning?

**Setup.** Closest replication of paper §4.2 (SVG unicorn optimization)
with local components:

- Generator: Bonsai 8B (the bigger of the two local options; closer to
  paper's GPT-5-mini)
- Evaluator: Claude Haiku via API (vision-language judging on rendered
  SVG outputs — d6 doesn't have vision, neither does Bonsai)
- Task: 1-2 of the paper's 5 judge rubrics (e.g., realistic + minimalist),
  one subject (unicorn)
- T = 5 iterations (matches paper Table 1)
- Compare to Bonsai direct-prompting baseline at the same task

**What we'd learn.**
- If Bonsai-FD beats Bonsai-direct (matching paper's GPT-mini → win
  pattern): FD generalizes to local generators. The mechanism isn't
  specific to frontier-model ICL. Validates the paper's core claim at
  smaller generator capacity.
- If Bonsai-FD ≡ Bonsai-direct: FD's gain is generator-capability-bound.
  Useful negative — tells us where the capability bar sits.

**Cost.** ~3-4 hours including SVG rendering pipeline + Claude Haiku
evaluator setup. The paper's SVG-rendering setup is open enough to
replicate (rsvg / Cairo for rendering).

**Risk.** Moderate setup overhead — vision-LM judging via Haiku adds API
cost (~$0.50-2 for the experiment), and SVG rendering is fiddly.

---

## Direction D — *Optimize d6's chat system-prompt via FD*

**Question.** Today's chat-quality session closed 4 levers at d6 with the
*chat model unchanged*. **We never optimized the system prompt** — partly
because the chat format doesn't have a real system-prompt slot, partly
because we tried it as a fake instructional turn and it backfired
(decoding sweep config E). FD provides the right framework: treat the
system prompt as the artifact under optimization, with d6's chat outputs
(judged on multi-turn coherence) as the objective.

**Setup.**
- Artifact: a system-prompt-shaped prefix injected before each user turn
  (or, if d6 truly can't follow it, optimize the *first user turn* as a
  priming move — same loop)
- Generator (mutating the prompt): Bonsai-8B or Claude Haiku
- Evaluator: same — judges produce both pairwise preference + textual
  feedback on which prompt produces better multi-turn outputs
- d6_baseline_modern_sft = the system being optimized (frozen)
- Objective: 7-prompt multi-turn rubric from
  `dev/multi_turn_eval.py` — but pre-register that we'll evaluate on a
  different held-out prompt set so we don't overfit on the
  rubric we wrote
- T = 10 iterations

**What we'd learn.**
- If FD-optimized system prompt moves the rubric from 1/7 → ≥ 3/7:
  **directly addresses the chat-quality goal without retraining.**
  Provides a free runtime-tunable lever that today's session said
  doesn't exist.
- If it doesn't move the rubric: tightens the capacity-bound
  diagnosis from today even further. We tried the *most generous*
  inference-time optimization and the chatbot still floors at 1/7.

**Cost.** ~2-3 hours. No new training. ~10 FD iterations × ~3 min/each
(generation + judging + d6 chat eval per candidate) ≈ 30 min compute.
Plus held-out prompt set design + evaluator setup.

**Risk.** Low. Reversible (no weight changes). The "fake system prompt"
backfire from the decoding sweep gives us a baseline negative — anything
above zero is informative.

---

## Direction E — *Can d6 / Bonsai act as a usable evaluator?* (asymmetry test)

**Question.** Pairwise judgment is generally easier than generation. If d6
or Bonsai can be a *trustworthy* evaluator on narrow tasks, that unlocks
fully-local FD loops where the laptop is sufficient (no API spend per
iteration).

**Setup.** Pick an artifact pair generation task with a clear ground
truth. Have a frontier model (Haiku or paper's GPT-5-mini) produce the
candidate generations. Have **d6** produce pairwise comparison +
rationale. Compare d6's preferences against ground truth.

**What we'd learn.**
- If d6's preferences agree with ground truth ≥ 75 %: d6 is a usable
  evaluator on narrow domains. Fully-local FD becomes viable with d6 or
  Bonsai as either side of the loop.
- If d6 disagrees: it's not a usable evaluator at this capacity.
  External or Bonsai-evaluator is required.

**Cost.** ~2 hours. Mostly held-out judgment-pair construction. No new
training.

**Risk.** None to compute; small risk that we pick a too-easy or
too-hard task and don't learn much.

---

## Recommended starting order

```
A → B → D → C → E
(diagnostic) (architectural) (chat-quality) (replication) (asymmetry)
```

**A first** — it's a 1-hour gating experiment. If d6 can't do the
mutation step, B and D become "use Bonsai as generator" and the local
generator question collapses early. If d6 can, B and D are live.

**B second** — it's the most novel experiment in the proposal and uses
checkpoints we already have. Worth running before C because B's
question (does internal architecture help on FD tasks?) is something
*nobody else has data on* — the paper doesn't compare to architectural
memory at all.

**D third** — directly attacks today's chat-quality finding without
retraining. Cheapest path to a real chat improvement if it works.

**C fourth** — replicates the paper's headline result with local
components. Useful as a known-good comparison to C's Bonsai-with-FD vs C's
Bonsai-direct baseline; tells us where the capability bar sits.

**E fifth** — pure capability check; less time-sensitive than the others.
Useful building block if any of A-D needs a local evaluator instead of
Haiku.

## Decision points the operator should weigh in on

1. **Start with A?** I lean strongly yes — it's the gating diagnostic. But
   if you'd rather skip and go straight to D (the chat-quality lever)
   using Bonsai as generator, that's also reasonable. D doesn't require
   d6 to be the generator at all.

2. **Bonsai 4B vs 8B?** For starting experiments, the smaller is fine — we
   want to know "does this work locally" not "what's the upper bound." 4B
   is faster per iteration. Upgrade to 8B if 4B fails.

3. **Claude Haiku for evaluator vs going fully local?** Recommend Haiku
   for early experiments — it's a stable reference. Once we have a
   working loop, Direction E tests whether we can replace it with
   local. Saves ~$1-3 per experiment.

4. **Single-task replication or multi-task?** The paper does three
   domains; we don't need to do that. Recommend picking *one* task per
   direction, keeping it small, and running 2-3 directions before any
   multi-task expansion.

5. **Held-out prompt set for D?** Must be different from the 7-prompt
   rubric we used today, otherwise we'd be optimizing on the test set.
   ~30 min to design 7-10 fresh prompts following the same load-bearing-
   detail pattern as the original rubric.

## What this proposal is *not* trying to do

- Not trying to ship a state-of-the-art FD application — the paper
  already does that with frontier models. We're testing if and how the
  mechanism works at *our* scale.
- Not trying to revive the Hope/NL track wholesale. Direction B is one
  specific reopened question; it doesn't restart the broader track.
- Not trying to replace the M4 path. If anything, the M4 question is
  about *what we couldn't do at d6*; this proposal is about *what we
  can do at d6 we didn't realize we could*.

## What would change the chat-quality landscape if anything works

If **D moves the chat-rubric from 1/7 → ≥ 3/7** at no retrain cost, we'd
update `docs/multi_turn_chat_eval_2026-05-06.md` and the lever ranking:
"Decoding-only inference-time optimization closed; *FD-style system-prompt
optimization* unlocks the lever the decoding sweep couldn't."

If **B shows arm_stage2 > arm_baseline** at FD-shaped tasks, we'd update
`docs/hope_nl_track_synthesis_2026-05-05.md` — the architecture verdict
needs a "but on this task class, the architecture wins" footnote, with
B's writeup as the citation.

Both findings would survive into the M4-future paths as live questions
worth bringing to bigger compute.
