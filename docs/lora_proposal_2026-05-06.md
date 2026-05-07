# Proposal: LoRA / PEFT for nanochat — starting directions

**Date:** 2026-05-06
**Status:** proposal only, no compute spent.
**Operator observation:** Karpathy's nanochat ships *no* parameter-efficient
fine-tuning path. Full-parameter SFT and full-parameter RL are the only
training routes. For the laptop / continuous-learning use case this is a
real gap.

## Why LoRA is the right next infrastructure addition for this codebase

Today's chat-quality session closed four levers at d6 — architecture,
decoding, SFT-rebalance, and SFT-additive volume. All four are
*full-parameter*, *single-pass* interventions. None of them addresses the
operator's stated goal:

> *"Provably useful continuous learning in an LLM, in a harness that
> works on my laptop."*

LoRA is the canonical mechanism for that goal. It maps onto continuous
learning more cleanly than either Hope/NL architecture (which we just
closed) or Feedback Descent (which is purely inference-time, no
adaptation). The structural advantages relevant to *this* environment:

1. **Tiny trainable parameter count.** Rank-8 LoRA on Q/V projections
   across all 6 d6 blocks is ~74K trainable params (~0.1 % of the 74M
   base). At Bonsai-4B scale it's a few hundred KB of trainable weights.
2. **Tiny checkpoint size.** ~5-10 MB per LoRA at d6, vs. 281 MB for a
   full SFT model checkpoint. Storage budget: keep dozens of LoRAs
   (per-task, per-persona, per-domain) for the cost of one full SFT.
3. **Modular adaptation.** Switch / compose LoRAs at inference time. The
   shape "small base + many adapters" is a natural fit for the
   continuous-learning framing.
4. **Wall-time benefit (smaller than the parameter count would suggest).**
   The forward pass still touches all base weights; only the *backward*
   compute is cheaper. Realistic wall: ~30-50 % faster per step than
   full SFT, plus we can use shorter horizons / higher LRs because there
   is so little to learn. Not the 10× speedup naive intuition suggests,
   but real.

## Compose with FD

The FD proposal (`docs/feedback_descent_proposal_2026-05-06.md`) covers
inference-time optimization with no weight updates. LoRA is *orthogonal*
in mechanism but composes cleanly:

- **FD-as-data-collection → LoRA-as-internalization.** Run an FD loop on
  a task; collect the (input, refined_output) trajectory; LoRA-fine-tune
  on the trajectory. The adapter becomes a compressed version of the FD
  iteration — model produces FD-quality outputs from a single forward pass.
  This is exactly the *external loop ↔ internal memory* dichotomy the
  operator flagged as worth pursuing both directions on; LoRA is the
  bridge.

- **LoRA-as-FD-generator.** Train a LoRA specifically to be a *better
  mutator* (i.e., better at conditioning on textual feedback). Feeds back
  into the FD loop as a more capable generator, without touching the
  base model.

Either composition is a real research-quality direction. Neither has been
run end-to-end at small scale that we know of.

## Available bases, and the d6-vs-Bonsai tradeoff

| base | trainable LoRA params (r=8, Q+V) | wall per training run | strength |
|---|---:|---:|---|
| **d6_baseline_modern_sft** (74M) | ~74K | ~30-50 min for ~200 iters | familiar stack, fast iteration, capacity ceiling still bites |
| **d6_stage2_pretrain_s1_sft** (74M, Hope/NL Stage 2) | ~74K | ~30-50 min | same as above + adds the architectural-memory question |
| **Bonsai 4B** (1-bit) | ~530K | unknown; depends on Bonsai's training stack on M2 | much higher capability ceiling; QLoRA-style training (fp LoRA on top of frozen quantized weights); more setup |
| **Bonsai 8B** (1-bit) | ~1.05M | unknown; probably 2× Bonsai 4B | highest local capability; same setup overhead |

**Recommendation: start with d6 to validate the implementation, graduate
to Bonsai if d6 LoRA shows meaningful effects.** Reasons:

- d6's stack is what we already know (fits in our existing `nanochat/`
  module surface, integrates with `--inherit-from` etc.)
- Implementation correctness is the first risk; validating on d6 first
  lets us debug the LoRA wiring without simultaneously dealing with
  Bonsai's quantization-aware training path
- The capacity ceiling at d6 is real — but LoRA on focused tasks is one
  of the few interventions that might *partially* route around it (by
  specializing capacity rather than diluting it)
- Bonsai-LoRA work has bigger payoff but also bigger setup; better
  approached as Phase 2 once we know the loop works

Bonsai-LoRA specifically would need:
- Investigation of Bonsai's tokenizer compatibility with our pipeline
- QLoRA-style implementation (fp16 LoRA matrices on top of frozen 1-bit
  weights — straightforward in principle but needs the de-quant /
  re-quant flow at LoRA training time)
- A working inference path that combines Bonsai 1-bit forward + LoRA
  adapter forward

That's ~1-2 days of infra before any experiments. d6 LoRA is ~half a day
of infra plus immediate experimentation.

## Implementation sketch (d6 path)

What needs to land in the repo:

1. **`nanochat/lora.py`** (~80 lines): a `LoRALinear` module that wraps a
   frozen base `Linear`, adds rank-r `A: (in, r)` and `B: (r, out)`
   matrices, applies `out = base(x) + B @ A @ x * (alpha / r)`. Matches
   the standard LoRA paper formulation. Handles both inference-only and
   training modes.

2. **Modification to `nanochat/gpt.py`**: an `apply_lora(model, target,
   rank, alpha)` helper that walks the model and replaces the chosen
   linear modules with `LoRALinear` wrappers. `target` is a string spec
   like `"q,v"` or `"q,k,v,o"` to choose which projections get adapters.
   Default: `q,v` at rank 8 alpha 16 (standard).

3. **`scripts/chat_sft_lora.py`** (or a flag on `chat_sft.py`): loads the
   base model frozen, applies LoRA, freezes everything except LoRA
   params, optimizes only LoRA, saves a checkpoint that contains
   *only* LoRA state (~5-10 MB). Reuses the existing data-loader and
   loss path; only the param-set changes.

4. **`nanochat/checkpoint_manager.py`**: extend `load_model` to accept an
   optional `lora_tag` that loads a LoRA on top of a base checkpoint.
   `chat_cli.py` and `chat_web.py` get a `--lora` flag.

Total new code: ~250-400 lines, 1-2 commits worth, fits a single focused
session.

## Three starting LoRA directions

### L1 — Persona-LoRA: directly attack today's persona_retention failure

**Question.** Today's multi-turn rubric showed all three SFT runs failing
on persona_retention (T4: "What's my name and job?" → "Sydney" /
"nanochat" / "Java J"). Can a small LoRA trained explicitly on
persona-retention patterns lift just *that* failure mode without
breaking the rest of the chatbot?

**Diagnostic update (2026-05-06):** the cosine-NN probe
(`docs/cosine_nn_diagnostic_2026-05-06.md`) classified d6's chat failure
modes as **~65 % right-context drift (FP-flavored)**, which the trx4mr
Phase 5 framework predicts codebook coarsening (binary/ternary) is the
indicated fix for. This upgrades **Bonsai-LoRA**'s prior from "interesting
capability bet" to **"diagnostic-supported indicated fix for the
dominant failure axis."** L1 should run as a side-by-side **two-arm
comparison**:

- **arm L1-d6**: rank-8 LoRA on `d6_baseline_modern_sft` (74M, fp)
- **arm L1-bonsai**: rank-8 LoRA on Bonsai 4B (1-bit base + fp adapter)

Same persona-retention dataset, same training horizon, same eval rubric.
The diagnostic predicts arm L1-bonsai's representation stability under
the SFT-mix-perturbation regime is meaningfully better — that's the
mechanism we'd be testing. If the prediction holds, arm L1-bonsai wins
on multi-turn rubric *and* shows less template-leakage in transcripts;
if it fails, the diagnostic's predictive power is bounded and we update
the prior.

**Stage 2 follow-up (2026-05-06):** we re-ran the cosine-NN probe on
`d6_stage2_pretrain_s1_sft` to test whether internal recurrent memory
(Hope/NL learned-gate at L3) was, in fact, the dominant-axis fix
(`docs/cosine_nn_stage2_2026-05-06.md`). It isn't:
~65 %/18 %/18 % → ~59 %/18 %/24 %, within 17-token sample noise. So
the L1 two-arm comparison is *not* a redundant test — adding a
**third arm L1-stage2** (rank-8 LoRA on `d6_stage2_pretrain_s1_sft`)
isn't required to rule out "Stage 2 was the fix all along," because
this cheap probe already ruled it out at the embedding-geometry level.
The L1-d6 vs L1-bonsai contrast remains the right two-arm setup. (One
caveat: the probe measures input/output embedding geometry, not the
runtime behaviour of L3's recurrent state, so a behavioural test of
Stage 2 + LoRA is still defensible if a future session has spare
compute.)

**Pre-condition for L1-bonsai**: Bonsai 1-bit forward pass on M2 needs
to work end-to-end with our pipeline (tokenizer compatibility, possible
rotation/centralize flags). That's ~1 day of infra work that wasn't in
the original L1 scope. **If the Bonsai integration is heavy, run
L1-d6 alone first** (still informative as the baseline LoRA test),
land Bonsai infra after, then run L1-bonsai as a follow-up.

**Setup.** Curate ~200-500 multi-turn conversations where:
- Turn 1: user introduces a name + role + interest
- Turns 2-3: chit-chat
- Turn 4: user asks the model to recall the turn-1 details
- Assistant: correctly recalls all three

Sources for these conversations:
- *Filter SmolTalk* for multi-turn rows with this shape (cheap; SmolTalk
  has ~273K 3-turn rows)
- *Synthesize via Claude Haiku* (~$0.50-1 to generate 500 high-quality
  examples)
- *Hybrid:* filter SmolTalk for multi-turn, augment with synthetic for
  the specific persona-recall pattern

Train rank-8 LoRA on Q/V across all blocks for 200-400 iters at higher LR
(LoRA standard: ~1e-3 to 1e-4 for rank-8; ours might need 1e-4 given d6
sensitivity).

Re-evaluate on the same 7-prompt rubric *plus* a held-out persona-test
set. Baseline: untrained d6_baseline_modern_sft. Target: rubric ≥ 2/7
(beats the original 1/7) and the persona_retention prompt specifically
passes.

**What we'd learn.**
- LoRA moves persona-retention specifically: validates "modular
  capacity for narrow patterns is reachable at d6" — the capacity
  ceiling is *task-specific*, not universal
- LoRA doesn't help: tightens the capacity-bound diagnosis to "even with
  focused parameter-efficient adaptation, this failure mode is structural
  at d6"
- LoRA fixes persona_retention but breaks something else: classic
  catastrophic-forgetting pattern; informs how aggressive the LoRA
  weighting at inference time should be (LoRA scaling factor at apply
  time controls strength)

**Cost.** ~3-5 hours total: 1h infra build + 1h dataset curation + ~1h
training + ~30 min eval + ~30 min writeup. Most of this is one-shot
infra cost; subsequent LoRA experiments inherit it.

**Risk.** Low. Reversible. Worst case: implementation works, LoRA
trains, and the chatbot doesn't measurably improve — same outcome as
today's SFT-rebalance experiments, just cheaper.

### L2 — Multi-LoRA mixture: route around capacity by specializing

**Question.** Today's chat failure modes were template-locked: math input
→ GSM8K template; "Hi I'm" → identity template; ambiguous input →
SmolTalk-shaped contentless filler. Capacity at 74M is finite; the model
has to share it across all templates. **What if we trained one LoRA per
template-class and routed at inference?**

**Setup.** Train three LoRAs on disjoint subsets:
- `lora_chat`: trained on SmolTalk subset only
- `lora_math`: trained on GSM8K only
- `lora_letters`: trained on SimpleSpelling + SpellingBee

Each is rank-8 on Q/V, trained for ~200 iters on its own subset.

At inference, classify the input (a tiny prefix-classifier or
keyword-based router) and apply the matching LoRA. Compare:
- d6_baseline_modern_sft alone (no LoRA)
- d6_baseline + lora_chat (always-on)
- d6_baseline + routed LoRAs

Score on the multi-turn rubric *and* the original SFT val_bpb (regression
check on MMLU/GSM8K).

**What we'd learn.**
- Routed LoRAs win on multi-turn AND don't regress on val_bpb: modular
  specialization beats monolithic SFT at d6 capacity. Real win.
- Routed LoRAs help on rubric but regress on val_bpb: the trade-off is
  real (chat for math); routing is a usable knob with a known cost
- Routed LoRAs ≡ baseline: capacity ceiling is robust to specialization
  too; closes one more lever

**Cost.** ~6-8 hours: 3 LoRAs × ~1h training + router implementation
(~1 hour) + eval + writeup. Higher than L1 because it's three LoRAs
plus the router.

**Risk.** Medium. Router design has degrees of freedom (rule-based
keyword vs. tiny classifier; soft mixture vs. hard switch). Mitigation:
start with rule-based hard switch (5 keywords for math, 5 for
spelling, default to chat) to bound the design space.

### L3 — FD → LoRA distillation: bridge the external loop and internal weights

**Question.** Can we *internalize* an FD loop into a LoRA — i.e., train
a small adapter that produces FD-quality outputs from a single forward
pass, eliminating the inference-time loop?

**Setup.** Pre-condition: an FD experiment from the FD proposal must run
first (probably FD-D, the system-prompt optimization). That produces a
trajectory of `(input, FD_refined_output)` pairs across many
optimization steps.

- Treat the trajectory as a supervised dataset: input → final-step output
- Train a rank-8 LoRA on this dataset for ~200-400 iters
- Eval: does d6+LoRA produce outputs comparable to d6+FD-loop, *without*
  running the loop at inference time?

**What we'd learn.**
- LoRA matches FD performance: the iteration is internalized; we can
  ship the LoRA as a one-shot replacement. Real continuous-learning
  loop demonstrated at small scale.
- LoRA underperforms FD: the FD loop's value is in the iteration itself,
  not just the output trajectory. Sharpens what FD is uniquely doing
  vs. what could be baked in.
- LoRA matches FD but only on the training trajectory's distribution:
  out-of-distribution generalization is still loop-dependent. Useful
  scoping.

**Cost.** ~4-6 hours, gated on an FD experiment running first. Most cost
is in the FD prerequisite, not the LoRA itself.

**Risk.** Higher than L1/L2 because it's a chained experiment — the FD
output quality bounds the LoRA's ceiling. Mitigation: only run L3 after
an FD experiment has demonstrably moved the metric we're trying to
internalize.

## Recommended starting order

```
infra → L1 → L2 → L3
(LoRA wiring + d6 sanity check)  →  (persona-LoRA, single direct attack)
                                         →  (multi-LoRA mixture, modular hypothesis)
                                              →  (FD distillation, only if FD shows real signal)
```

**Infra first** — ~half-day to land `nanochat/lora.py`, the apply
helper, and `chat_sft_lora.py`. Smoke-test by training a 50-iter LoRA on
a tiny corpus and verifying the resulting forward differs from base by
the expected amount. This is the highest-value foundational work — once
landed, every subsequent LoRA experiment is fast.

**L1 second** — direct attack on a known failure mode with a small
focused dataset. Highest probability of producing a real signal because
the target is narrow (persona_retention specifically) and the dataset
is curatable. If L1 works, the LoRA infrastructure is validated end-to-end.

**L2 third** — only run if L1 shows the LoRA mechanism produces
meaningful behavior change. L2 is more ambitious (three LoRAs +
routing) and hinges on L1's positive finding to be worth the extra cost.

**L3 last** — gated on an FD experiment producing a usable trajectory.
Most ambitious but most novel. Depends on FD work landing first.

## Decision points the operator should weigh in on

1. **Start with d6 LoRA infra or jump to Bonsai?** Strong recommendation
   for d6 first — implementation correctness is the first risk; we know
   the d6 stack. Bonsai-LoRA is Phase 2.

2. **Where to apply the LoRA?** Standard practice is Q+V at all blocks.
   Could also do Q+K+V+O (more capacity, 2× trainable params). Or
   include MLP linears (much more capacity, ~5-10× trainable). Recommend
   starting with Q+V for L1 (matches LoRA paper baseline) and only
   expanding if it looks too constrained.

3. **Rank?** r=8 is standard; r=4 cheaper, r=16 more capacity. Start at
   r=8 unless L1 underfits.

4. **L1 dataset source?** Three options sketched (filter SmolTalk /
   synthesize via Haiku / hybrid). Recommend hybrid: filter SmolTalk for
   the rough shape, synthesize ~100 examples for the precise
   persona-recall pattern. Best signal-to-cost ratio.

5. **Eval rubric for LoRA experiments?** Reuse the 7-prompt multi-turn
   rubric for cross-comparability with today's runs. Add held-out
   persona-test set for L1 to avoid overfitting on the training pattern.

6. **Order vs. parallel with FD?** LoRA infra and FD experiments are
   independent; could run in parallel sessions. L3 specifically depends
   on FD-D output, so FD-D should land before L3 starts.

## Why this proposal is the right shape for *this* operator

The operator's stated goal — *"provably useful continuous learning in
an LLM, in a harness that works on my laptop"* — gets significantly
unblocked by this infrastructure. Specifically:

- **Reproducibility on laptop hardware:** d6 LoRA training fits in ~30-50
  minutes, two orders of magnitude faster than buying M4 to make d6
  capacity larger. LoRA is the right "more capability per compute
  dollar" lever for this regime.
- **Closes Karpathy's gap:** Currently no PEFT path in nanochat.
  Implementing it is a real contribution that fits the codebase's "strong
  baseline, hackable" stance — LoRA can be added without breaking the
  single-depth-dial discipline (LoRA respects depth-derived dimensions).
- **Re-opens the "what can d6 actually do" question:** Today's session
  closed several levers at *full-parameter* scale. LoRA is a *different*
  parameter regime. Whether d6 + LoRA can do things d6 alone can't is
  an unmeasured question and would be worth knowing.

If LoRA infra lands and *none* of L1/L2/L3 shows useful signal, the
diagnosis goes from "capacity-bound at d6" (today) to "capacity-bound
at d6 even with focused LoRA adaptation." That's a meaningfully
stronger claim than what we have now, and the infrastructure stays
landed for any future Bonsai-LoRA / M4 work.
