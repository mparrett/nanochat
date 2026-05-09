# MLX-LM pattern audit — what's worth lifting into nanochat

**Date:** 2026-05-08
**Status:** Reference. No code change yet.
**Companion to:** `docs/mlx_port_evaluation_2026-05-01.md` (the port verdict
stands — don't port; this audit is about *patterns* worth borrowing while
staying on PyTorch/MPS).
**Audited:**
- `/Users/matt/projects-new/3p/mlx-examples` @ `09aa7f8` (1 ahead of upstream `796f5b5`)
- `/Users/matt/projects-new/3p/mlx-lm` @ `df1d3f3` (clean origin/main)

## TL;DR

The mlx-lm sibling repo is the most mature LM-training reference on Apple
Silicon. Its quantization suite (DWQ, AWQ, GPTQ, Dynamic) is **all
post-training quantization** — none of it is QAT-from-scratch, so it does
not directly serve our binary/ternary thread. But three specific patterns
in mlx-lm and mlx-examples are worth lifting back into our PyTorch/MPS
training loop, with no port required:

1. **Learnable per-group quantization scales** (DWQ trick) — external
   validation for the trx4mr `learned_scale` flag deliberately omitted
   in Phase 1 (`nanochat/quant.py:31`). High-impact candidate for Phase
   2 of the binary/ternary thread.
2. **Sort-by-length batching with pad-to-multiple-of-32**. Free dataloader
   perf trick, applies to MPS and CUDA both.
3. **Monkey-patched gradient checkpointing on `Block.__call__`**. Cleaner
   than `torch.utils.checkpoint.checkpoint_sequential`, no module
   restructuring needed.

Two more ideas filed for later (not actionable today):

4. **Top-K logit caching for distillation** (DWQ memory trick).
5. **`mx.compile`-style whole-step graph fusion** — already parked per the
   2026-05-01 doc; still parked.

What it confirms about MLX itself: the 2026-05-01 verdict ("don't port")
holds. mlx-lm is impressive engineering but doesn't give us the
QAT-from-scratch story for free, and the matmul-perf gap (#3196) hasn't
moved.

---

## Context — why we re-looked

Operator surfaced the angle: antirez's `ds4` (DeepSeek V4 Flash inference
engine) is generating buzz, perhaps there's training inspiration to crib.
First pass found ds4 is *not* MLX (it's hand-written C + Metal, kernels
adapted from llama.cpp/ggml) and *not* training (inference-only, M3 Max
128 GB target, single-model bet). So that thread closed.

But the local checkouts of `mlx-examples` and `mlx-lm` were available, so
the question shifted to "what does Apple's actual training stack do
differently from us, and what's borrowable." This doc is that audit.

---

## (1) Learnable per-group quantization scales — DWQ pattern

**Where it lives:** `mlx-lm/mlx_lm/quant/dwq.py:90-100`

```python
def unfreeze(_, m):
    if (hasattr(m, "bits") and hasattr(m, "group_size")
        and m.mode == "affine" and m.bits < 8):
        m.unfreeze(keys=["scales", "biases"], recurse=False)

model.train()
model.apply_to_modules(unfreeze)
```

**Mechanism.** `nn.QuantizedLinear` in mlx-core stores affine quant params:
weights are int4/int8 with per-group `scales` + `biases` such that
`w_dequant ≈ w_int * scale + bias`. By default these are frozen — fixed
at quantization time. DWQ walks the model, finds quantized linears, and
flips just `scales` and `biases` to trainable. Integer weights stay
frozen. Training signal is KL distillation against an fp16 teacher
(`dwq.py:114`, temperature=2.0).

**Why it matters for nanochat.** Our current STE path in
`nanochat/quant.py` derives the per-group scale on every forward as
`scale = groups.abs().mean(dim=-1)` (lines 53, 69-70 for binary; 107 for
ternary). The scale therefore *does* move during training — but only as
a **closed-form function of the latent fp32 weight**, not as a separate
parameter with its own gradient signal. The optimizer never directly
touches the scale. DWQ is external evidence from the PTQ literature
that promoting the scale to its own learnable parameter (initialized
from the magnitude estimate) is one of the highest-leverage moves at
low bits: it preserves quality where the closed-form scale alone is
insufficient.

**This is not a novel suggestion — it's external validation for a
deferred decision.** The operator already considered learnable scales
in Phase 1 design and deliberately omitted them. From
`nanochat/quant.py:31`:

> Exotic flags from the trx4mr port (`learned_scale`, `pre_norm`,
> `centralize`, `turbo`) are deliberately omitted — they're separate
> research knobs that can be re-introduced in a follow-up if the
> vanilla STE smoke test motivates them.

The trx4mr port lives at `~/projects-new/trx4mr/picoGPT/binary.py`. DWQ
adds a second independent data point: an Apple-blessed PTQ pipeline
treats the same parameter as the highest-priority thing to make
trainable, with a documented quality benefit. So the framing for Phase
2 of the binary/ternary thread shifts from "speculative knob" to "two
independent codebases (trx4mr, mlx-lm) treat this as the right move."

**What this is not.** DWQ itself is PTQ — integer weights frozen, scales
learned. Our case is QAT — both the latent weight and (potentially) the
scale would be learned, but the integer weights are still derived
deterministically from the latent at every forward. So we'd be lifting
the *technique* (let the optimizer touch the scale), not the *recipe*
(distillation against an fp16 teacher).

**Sketch of how to apply.**
- Re-introduce `learned_scale` in `nanochat/quant.py`. Cross-reference the
  trx4mr implementation for the gradient-flow shape; don't reinvent.
- Promote `scale` from a derived quantity (line 53) to an
  `nn.Parameter` initialized from the per-group magnitude estimate, so
  the starting state matches today's behavior.
- Decide whether the latent's magnitude continues to drive a separate
  closed-form scale (in which case the learned scale is a *correction*
  on top) or whether the learned scale fully replaces the closed-form
  one. The trx4mr port likely has a position; defer to it.
- Validate first on `--depth=3` smoke (the 1300-iter recipe in
  `docs/d3_smoke_recipe_2026-05-05.md`) before any d6 commitment.

**Falsification.** If learnable scales don't move val_bpb at d3, drop it.
If they help at d3, run a single d6 binary/ternary pretrain to see if
the improvement holds at scale.

**Cost.** Sub-day implementation if the STE module is well-isolated.
Validation is one d3 smoke (~30 min M3 Max) and one d6 pretrain (~3 h
M2) at most.

---

## (2) Sort-by-length batching with pad-to-multiple-of-32

**Where it lives:** `mlx-lm/mlx_lm/tuner/trainer.py:110-170`

```python
# Sort by length:
idx = sorted(range(len(dataset)), key=len_fn)
...
# Make the batches:
batch_idx = [
    idx[i + offset : i + offset + batch_size : step]
    for i in range(0, len(idx) - batch_size + 1, batch_size)
]
...
# Pad to one plus nearest multiple of pad_to or the maximum length
pad_to = 32
max_length_in_batch = 1 + pad_to * ((max(lengths) + pad_to - 1) // pad_to)
```

**Mechanism.** Two stacked tricks:

1. Pre-sort the dataset by sequence length. Form batches as adjacent
   slices of the sorted index. Within a batch, all sequences are roughly
   the same length, so padding waste is minimized.
2. Within a batch, pad up only to `1 + 32 * ⌈max_len / 32⌉`. Keeps the
   actual tensor shape multiple-of-32 (good for Metal/CUDA tile
   alignment) while not forcing every batch up to the global max.

The batch-order itself is shuffled (line 141: `np.random.permutation`)
each epoch, so training still sees randomized batch order — only the
*intra-batch* composition is length-stratified.

**Why it matters for nanochat.** Pretraining packs data into
fixed-length sequences (no padding bottleneck) — irrelevant there. But
SFT data is variable-length with explicit padding (see
`scripts/chat_sft.py`), and chat fine-tuning batches are demonstrably
padding-heavy. A length-stratified shuffler at SFT time would directly
reduce wasted compute per step.

**What this is not.** It's not free quality: by construction, sorted
batching reduces the diversity of any single gradient. The mitigation is
randomizing batch *order* (which mlx-lm does) so the optimizer still
sees varied work over a window. This needs validation, not just adoption.

**Sketch of how to apply.**
- In `nanochat/dataloader.py` (or wherever SFT batching lives), add a
  `length_sort=True` mode that pre-sorts indices by length, forms batches
  as adjacent strides, and shuffles batch order each epoch.
- Compute `pad_to_multiple = 32` and pad each batch to `1 +
  pad_to * ⌈max_len_in_batch / pad_to⌉` rather than the global SFT max.
- Validate on a short SFT run that loss curves and downstream eval match
  the unsorted baseline within seed variance.

**Falsification.** If validation loss diverges from unsorted baseline at
the same step count, abandon — the padding savings aren't worth gradient
quality loss. If it matches, measure the actual tokens-per-second
improvement; below ~10% it's not worth the dataloader change.

**Cost.** Few-hour implementation. One short SFT A/B (~20 min M2)
validates whether quality is preserved.

---

## (3) Gradient checkpointing via class-method monkey-patch

**Where it lives:** `mlx-lm/mlx_lm/tuner/trainer.py:25-38`

```python
def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(),
                                       *args, **kwargs)

    type(layer).__call__ = checkpointed_fn
```

**Mechanism.** Replaces the `__call__` method on the *class* of the
target layer (e.g. `TransformerBlock`) with a checkpointed version. Every
existing instance now performs activation checkpointing. The wrap closes
over the original method via `fn = type(layer).__call__` to preserve it.

The PyTorch equivalent — `torch.utils.checkpoint.checkpoint` — typically
forces you to wrap each layer's forward call site individually, or use
`checkpoint_sequential` which assumes a flat `nn.Sequential` structure.
Neither matches nanochat's hand-rolled block layout cleanly.

**Why it matters for nanochat.** M2 has 24 GB unified memory. Activation
memory at d12+ batch sizes already pushes against it. A clean
gradient-checkpointing toggle on `Block.__call__` (or
`Block.forward` in PyTorch) would let us trade a fixed compute overhead
(~30%) for a proportional reduction in activation memory — enabling
larger `device_batch_size` at the same depth, which is currently the
binding constraint for grad-accum schedules.

**What this is not.** It's not a wall-clock perf win — it's an *enabler*.
The win comes from being able to use a larger batch size, which (a)
reduces grad-accum step count for a given target effective batch size,
or (b) gets us closer to the optimal per-step batch geometry. The
benefit is conditional on us actually being memory-bound — which we are
on M2 24 GB at d12+.

**Sketch of how to apply (PyTorch port of the idea).**

```python
# In nanochat/transformer.py or similar:
import torch.utils.checkpoint as cp

def enable_grad_checkpoint(block_class):
    fn = block_class.forward
    def checkpointed_forward(self, *args, **kwargs):
        return cp.checkpoint(fn, self, *args, use_reentrant=False, **kwargs)
    block_class.forward = checkpointed_forward
```

Toggled by a single training-script flag (`--grad-checkpoint`).
`use_reentrant=False` is the modern/recommended path; reentrant has a
known footgun with `requires_grad` propagation.

**Falsification.** Run a d12 forward+backward step on M2 with and
without the toggle. Confirm activation memory drops (peak GPU memory
metric) and that step time grows by the expected ~30%. If the memory
drop doesn't materialize, abort — likely a bug in how the wrap composes
with our existing FSDP/grad-accum stack.

**Cost.** Hour or two. Validation is one short pretrain step at d12 with
the memory profiler enabled.

---

## (4) Top-K logit caching for distillation — file for later

**Where it lives:** `mlx-lm/mlx_lm/quant/dwq.py:59-63`

```python
idx = mx.argpartition(logits, kth=-1024, axis=-1)[..., -1024:]
logits = mx.take_along_axis(logits, idx, axis=-1)
file = path / f"{i:010d}.safetensors"
mx.save_safetensors(file, {"logits": logits, "indices": idx})
```

**Idea.** When distilling from a teacher, you don't need the full vocab's
logits — the bottom 99% are noise. Pre-compute the teacher's top-K
(K=1024) logits and indices per token, store on disk as safetensors, and
have the student `take_along_axis` to match. Cuts target storage by
~100× for vocab=100k, makes "one teacher, many distillation runs"
practical.

**When this becomes relevant for nanochat.** If we ever do KD on the SFT
or RL side (e.g. distilling a larger teacher into the d6 student), the
top-K cache pattern is a known good infra trick. Not actionable today —
no distillation in current roadmap. Filed here so it doesn't get lost.

---

## (5) `mx.compile` whole-step fusion — still parked

**Where it lives:** `mlx-lm/mlx_lm/tuner/trainer.py:248-262` and
`mlx-examples/transformer_lm/main.py:100-105`.

The training step is wrapped in `@partial(mx.compile, inputs=state,
outputs=state)`, fusing forward + backward + grad accumulation +
optimizer update + RNG advance into a single graph. Lazy execution means
nothing runs until `mx.eval(...)` is called. Closest PyTorch analogue is
`torch.compile`, but PyTorch typically compiles forward only and runs
backward/optimizer eagerly.

**Why this stays parked.** The 2026-05-01 doc set the gating criterion:
"an optimizer-only MLX Muon benchmark on fake d12 tensors beats PyTorch
by ≥ 1.5×." That benchmark hasn't been run. Without it, `mx.compile`
remains a speculative win, and the `mlx#3196` matmul gap (~1.19× slower
than PyTorch MPS) is a known regression that would have to be paid up
front. So this stays parked under the same conditions as the original
port doc.

If/when someone runs the Muon benchmark, this idea unblocks itself
automatically — it's a `requires (1,2) from mlx_port_evaluation_2026-05-01.md`
sort of dependency.

---

## Re-confirmed: PTQ ≠ QAT, mlx-lm doesn't serve QAT

The most important *negative* finding from this audit:

mlx-lm's quantization suite is comprehensive (DWQ, AWQ, GPTQ, Dynamic
Quantization, all documented in `mlx-lm/mlx_lm/LEARNED_QUANTS.md`) but
**it is all PTQ**. Each method assumes a high-quality fp16/bf16 model
exists, then crunches it down via:

- DWQ — fine-tune affine wrapper (scales/biases) by distillation
- AWQ — pre-quant scaling+clipping based on activation statistics
- GPTQ — error-minimization per-layer
- Dynamic — sensitivity analysis → mixed-precision allocation

Our binary/ternary thread is QAT-from-scratch — train at the target
precision from step 0. Different problem. Different infra. Lifting
mlx-lm wholesale would not give us QAT for free; we'd still be writing
the QAT path ourselves either way.

This doesn't reduce the value of the three transferable ideas above —
those are sub-techniques (learnable scales, length batching, checkpoint
monkey-patch) that work in either regime. But it sharpens the answer to
"is mlx-lm the right framework to port to": no. The framework boundary
is wrong for our problem. Borrow the ideas, stay on PyTorch.

---

## Suggested backlog entries (operator decides)

If lifted into `docs/project_notes/backlog.md`, these are candidates,
ranked by expected payoff for the current binary/ternary thread:

| # | Idea | Cost | Validation gate |
|---|---|---|---|
| 1 | `learned_scale` re-introduction (trx4mr port + DWQ external validation) | <1 day impl | d3 smoke val_bpb improves |
| 3 | Block.__call__ grad-checkpoint | ~2 h | d12 peak-memory drops, step time +30% |
| 2 | Length-stratified SFT batching | ~half day | tokens/sec up ≥10%, val loss matches baseline |

Filed-for-later (not in current scope):

- (4) Top-K logit caching — gated on adopting distillation
- (5) `mx.compile` whole-step fusion — gated on the Muon-only benchmark
  in `mlx_port_evaluation_2026-05-01.md`

---

## References

- `docs/mlx_port_evaluation_2026-05-01.md` — port verdict (don't port).
- `mlx-lm/mlx_lm/quant/dwq.py` — DWQ implementation.
- `mlx-lm/mlx_lm/tuner/trainer.py` — main training loop, sort-by-length
  batching, grad-checkpoint monkey-patch.
- `mlx-lm/mlx_lm/LEARNED_QUANTS.md` — overview of the four PTQ methods.
- `mlx-examples/transformer_lm/main.py` — minimal MLX training loop with
  `mx.compile`.
- `nanochat/quant.py:31` — current Phase 1 omission of `learned_scale`
  and other trx4mr knobs (the deferral DWQ now externally validates).
- `~/projects-new/trx4mr/picoGPT/binary.py` — trx4mr's reference
  implementation of `learned_scale` (origin of the nanochat port).
- `docs/project_notes/decisions.md` ADR-006/007 — current binary/ternary
  quant integration choices (relevant context for idea #1).
