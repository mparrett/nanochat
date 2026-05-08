# Backlog

Directions the operator wants to try but has not scheduled. Newest first.
Each entry: short pitch, rough cost, why it's interesting, pointer to the
literature or precedent. Move to `decisions.md` if/when picked up.

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
