# Backlog

Directions the operator wants to try but has not scheduled. Newest first.
Each entry: short pitch, rough cost, why it's interesting, pointer to the
literature or precedent. Move to `decisions.md` if/when picked up.

---

## 1-bit / ternary d6 from scratch — the original Bonsai-inspired pitch (2026-05-07)

**Pitch.** Pretrain a d6 nanochat model with binary `{−s, +s}` (or ternary
`{−s, 0, +s}`) weights from scratch — *not* fp LoRA on a 1-bit base, *not*
post-hoc quantization, but native low-bit pretrain. The strategic prize:
**~14× memory reduction means d12+ might fit on M2** even though pretrain
compute scales the same as fp. The baseline question is "can we even train
a 1-bit d6 to within shouting distance of fp d6's val_bpb 1.174."

**Why interesting / why now.**
- The cosine-NN diagnostic (`docs/cosine_nn_diagnostic_2026-05-06.md`)
  classified d6's failures as ~65 % right-context drift (FP-flavored), and
  the trx4mr Phase 5 fix-direction table prescribes binary/ternary codebook
  coarsening. So this isn't a vibe; it's the diagnostic-supported
  *architectural* fix axis. (Distinct from the lora_proposal's "Bonsai-LoRA"
  arm — see the disambiguation note below.)
- The trx4mr sibling repo already has the infra primitives:
  `picoGPT/binary.py` exposes `BinaryLinear` / `TernaryLinear`, and
  `blabberverse/phase7_arch.py` has a working `--quant {fp,binary,ternary}`
  CLI with per-group-of-128 FP16 scale factors. Porting these into
  `nanochat/gpt.py` as quantized variants of `Linear` is the load-bearing
  infra step.
- The sibling brief `~/projects-new/trx4mr/docs/bonsai_1bit_brief.md`
  (April 2026) lays out Bonsai's binary scheme cleanly: `{−s, +s}`, no zero
  state, FP16 scale per 128-weight group, **no higher-precision escape
  hatches anywhere** (embeddings, attention, MLP, LM head all binary).
  BitNet b1.58 is the *ternary* `{−1, 0, +1}` variant, with which Bonsai
  is sometimes confused — they're distinct designs.

**The Bonsai mystery (caveat).** Bonsai's headline claim is a "native 1-bit
training method" — *not* the standard straight-through estimator. The
training method is proprietary (Caltech research, Hassibi et al., Apache 2.0
weights but closed training code). The published whitepaper at
`https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf`
is the only public source. **Realistically: the operator expects we'll
fall back to STE for our d6 pretrain** (well-trodden, fits our hardware,
matches what trx4mr's `BinaryLinear` already does). Reverse-engineering
Bonsai's actual method is an interesting separate question and may not
fit M2; it's a side-quest, not the load-bearing thing.

**Compute reality check.** 1-bit pretrain is typically *just as compute
heavy* as fp pretrain (or worse, if STE adds overhead). The win is
~14× memory. So: same M2 wall as a fp d6 pretrain (~3 h) for the d6
validation; then *if* d6 works, d12+ becomes the actual strategic move
because the memory delta is what unblocks larger depth on this hardware.

**Cost (rough).**
- Phase 1 — port `BinaryLinear`/`TernaryLinear` into `nanochat/gpt.py`,
  swap into `apply_lora`-style `apply_quant` walker, smoke-test forward
  pass + STE backward at d6 scale: ~half-day.
- Phase 2 — d6 binary pretrain (~3 h M2 wall) on canonical recipe; compare
  val_bpb against fp d6 (1.174) and against ternary d6.
- Phase 3 — *if* d6 binary lands within ~5 % of fp val_bpb, the d12+ on M2
  experiment becomes live. Could be many M2-days of experiment work.

**Falsification criteria.**
- d6 binary val_bpb >> 1.5: STE gradient flow fails at d6 scale; the
  operator's "scale up to d12+" pitch needs a different training method
  (potentially Bonsai's mystery method, potentially something else).
- d6 binary val_bpb ≈ 1.18-1.30: STE works; the d12+ memory-unblock thesis
  is testable.
- d6 binary val_bpb ≈ 1.17: STE works *well*; the depth-rule generalises
  cleanly to binary; we're in the interesting regime where 1-bit training
  is a viable design decision for this codebase.

**Status.** Open. **The operator's most-wanted direction.** Currently
gated only on operator priority — infra plan is sketched but not started.

**References / pointers.**
- `~/projects-new/trx4mr/docs/bonsai_1bit_brief.md` — operator's notes on
  Bonsai 1-bit, what's known, what's proprietary, suggested experiments.
- `~/projects-new/trx4mr/blabberverse/phase7_arch.py` — working
  binary/ternary training arch with `--quant` CLI, per-128-group scales.
- `~/projects-new/trx4mr/picoGPT/binary.py` — `BinaryLinear` /
  `TernaryLinear` primitives (the port targets).
- BitNet b1.58: Microsoft, 2024, *The Era of 1-bit LLMs* (the ternary
  prior-art).
- Bonsai whitepaper (linked above) for what's public on the binary scheme.

---

## Local Bonsai 4B/8B as helper models — synthesis, judge, distillation (2026-05-07)

**Pitch.** The operator has Bonsai 4B and 8B running locally; ternary-quantized
inference is fast enough on M2 to use them as **utility models** in our
nanochat workflow:
- **Synthetic training-data generation** for SFT or LoRA experiments
  (current chit-chat / persona curators use Claude CLI which costs the
  subscription; Bonsai-as-generator is free per token).
- **LLM-as-judge** for chat-quality evals where a rubric is too coarse and
  hand-grading is too slow.
- **Distillation target / teacher** for d6 student training — particularly
  interesting in combination with the 1-bit-from-scratch direction
  above, since the teacher's representations may transfer differently
  into a discrete student than into a fp student.

**Why interesting.** This is *infrastructure* in the operator-tooling
sense — it doesn't move the model-quality needle directly, but it makes
several other directions cheaper. The current Claude-CLI curator pattern
costs $3-4 per dataset; a Bonsai 8B local curator costs ~minutes of M2
time with no per-call charge. For experiments where we want 10× the data
or 10× the iterations, the cost crossover is real.

**Cost (rough).** Wiring Bonsai into the nanochat dev pipeline is a
half-day of glue: process invocation + JSON-schema-equivalent prompt
discipline + cost/time bookkeeping. The Bonsai inference path itself
already works locally per the operator (presumably via the
PrismML llama.cpp fork or the MLX 1-bit weights).

**Status.** Open. Independent of the 1-bit-from-scratch direction.

**Risk.** Bonsai's behavioural fingerprint may differ from Claude's
in ways that affect dataset quality. A curator-output diff against the
existing Claude-CLI curators on a small sample (50 conversations) would
de-risk this before committing to it as the default.

---

## ⚠ Disambiguation: three "Bonsai" directions, not one (2026-05-07)

The lora_proposal (`docs/lora_proposal_2026-05-06.md`) and downstream
writeups conflated three distinct experiments under the "Bonsai-LoRA"
heading. They're separate research questions and shouldn't be
collapsed into one work-stream.

| direction | research question | status |
|---|---|---|
| **A. 1-bit/ternary d6 from scratch** (this backlog, above) | Can we *train* a binary base from scratch at d6, with the prize being d12+ memory-unblock on M2? | Operator's original vision; not started. |
| **B. Local Bonsai 4B/8B as utility models** (this backlog, above) | Can we use *their* pretrained binary model to generate data, judge, or teach? | Operator capability; not started. |
| **C. fp LoRA on a frozen 1-bit base** (`lora_proposal_2026-05-06.md` § L1-bonsai) | Does fp LoRA on a *frozen* 1-bit base outperform fp LoRA on fp base for the persona-retention task? | Diagnostic-supported; infra-blocked. |

A and C are easy to confuse. C uses someone else's 1-bit checkpoint as a
*foundation*; A trains our own 1-bit base. Different questions, different
infra, different risks. **The operator's stated vision was A**; C
emerged from the LoRA-proposal frame and the cosine-NN diagnostic.

Both A and C remain interesting; both are open. Keep them distinct in
all future writeups.

---

## Model-growing: d6 → d8 via bert2BERT-style replication (2026-05-07)

**Pitch.** Instead of pretraining d8 from scratch (~17-20h on M2), bootstrap
d8's init from the existing d6 checkpoint:
- Replicate d6's 6 transformer blocks as the bottom 6 of d8.
- Add a width-expansion linear-map for `model_dim` 384 → 512 (nanochat's depth
  rule changes both `n_layer` and `model_dim`, so this is bert2BERT regime, not
  pure layer-stacking — see `bert2BERT`, Chen et al. 2022).
- Zero-init or random-init the top 2 layers.
- Continue pretraining for a fraction of d8's full Chinchilla horizon.

**Why interesting.** No one in this repo has tried it. Reported wins in the
literature are 30-70 % wall-time reduction vs from-scratch to reach a target
loss. On M2 that's the difference between an overnight d8 and a 4-6h d8.

**Tension with project philosophy.** nanochat's design rule is "every depth
produces a compute-optimal model from scratch" (`CLAUDE.md`). A d8-from-d6
init is *by construction* off the compute-optimal frontier — it carries
d6-quality features into a d8-shape. That's fine for "burn fewer M2 hours to
get *a* d8 to play with"; it's not fine if the goal is a clean depth-sweep
data point.

**Cost (rough).** ~20-30 LOC for the width-expansion map + layer replication
helper, plus a one-shot `scripts/grow_d6_to_d8.py`. Continuing pretrain:
~5-10h M2 wall (depending on what fraction of horizon we run).

**Status.** Open. Not blocked on anything. Cheaper than B (LiGO) by far.

**References.**
- Chen et al. 2022, *bert2BERT: Towards Reusable Pretrained Language Models*
  (width + depth expansion in one shot).
- Gong et al. 2019, *Efficient Training of BERT by Progressively Stacking*
  (StackBERT — pure layer-stacking, simpler precedent).
- Chen et al. 2015, *Net2Net* (the seminal function-preserving transforms).

---

## LiGO: learn the small→large weight map (2026-05-07)

**Pitch.** Wang et al. 2023, *Learning to Grow Pretrained Models for
Efficient Transformer Training* — instead of hand-designing the d6→d8 init
map (the bert2BERT direction above), *learn* a linear operator that maps
small-model weights into the larger model's parameter space. State-of-the-art
on this axis at publication.

**Why interesting.** Same motivation as bert2BERT (skip the cost of
from-scratch pretrain at the larger scale) but with a learned mapping instead
of a hand-crafted one. Empirically beats bert2BERT and stacking baselines in
the paper.

**Cost (rough).** Materially more code than bert2BERT — needs the LiGO
linear-operator parameterisation, a meta-training loop on the operator, and
then the actual continued-pretrain at d8. A reasonable order-of-magnitude is
"a week of focused work" rather than "a weekend hack". Ranks behind A
(bert2BERT) on cost-to-information ratio for nanochat-scale.

**Status.** Open. Probably only worth it if the bert2BERT-style direction
shows enough signal to want to push further on weight-mapping quality.

**References.**
- Wang et al. 2023, *Learning to Grow Pretrained Models for Efficient
  Transformer Training* (ICLR'23, "LiGO").
