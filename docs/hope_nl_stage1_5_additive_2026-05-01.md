# Hope/NL Stage 1-additive — MQAR probe result

**Date:** 2026-05-01
**Branch:** `experiment/hope-nested-learning`
**Architecture commit:** `f990b16`
**Companion writeup:** `docs/hope_nl_stage1_5_results_2026-05-01.md` (the original swap-vs-baseline finding)

## TL;DR

The **additive** variant (memory module added as a third residual alongside attn+MLP, instead of replacing the MLP) is **at least as slow as the swap** on MQAR convergence — and slightly slower in the early grokking window. This **falsifies the "lost MLP nonlinearity" hypothesis** from the original Stage 1.5 writeup. The bottleneck is upstream: it lives in the `LinearAttentionMemory` module's `W_o = 0` init, which creates a slow-start for the K/V/Q projections.

| step | baseline | swap (Stage 1) | **additive (NEW)** |
|---:|---:|---:|---:|
| 1 | 0.031 | 0.031 | 0.031 |
| 26 | 0.037 | 0.054 | 0.058 |
| 51 | 0.111 | 0.066 | 0.064 |
| 76 | **0.999** | 0.131 | **0.068** |
| 101 | 1.000 | 0.632 | 0.132 |
| 126 | 1.000 | 0.863 | 0.942 |
| 151 | 0.999 | **1.000** | **0.999** |
| 200 | 1.000 | 1.000 | 1.000 |

**Both swap and additive saturate at step ~151, ~2× later than baseline's ~76.** Additive is slightly slower in the early window (steps 76–101) but catches up faster once it gets unstuck (the 13.2% → 94.2% jump in one window).

## What this rules out and rules in

### Ruled out
- **"Replacing the MLP at L3 removed a useful ReLU² nonlinearity, slowing recall."** This was the leading hypothesis from the original writeup. If true, the additive variant — which keeps all MLPs intact and adds the memory pathway on top — should match baseline. It doesn't. Additive is just as slow.

### Ruled in
- **The slow start lives inside `LinearAttentionMemory`, specifically the `W_o = 0` init.** Both swap and additive use the same `LinearAttentionMemory` module with the same init. Both are slow. Baseline (no memory module at all) is fast.

### Mechanism

The `LinearAttentionMemory` forward is:

```
o = einsum('btd,bsd->bts', q, k) * scale  ; mask  ; einsum back to v  → o
y = W_o(o)
```

With `W_o = 0` at init:

- Output `y = 0` — block contributes zero to the residual stream. Model is bit-identical to a baseline that has the memory block "off." (This is the guarantee that made the v1 step-1 loss bit-identical across arms.)
- `∂L/∂W_o = grad_y · oᵀ` — non-zero, gradient flows from upstream loss. So `W_o` does start moving on iter 1.
- **But:** `∂L/∂W_{k,v,q} ∝ W_oᵀ · (...) = 0 · (...) = 0` initially. The K/V/Q projections cannot start learning until `W_o` becomes non-zero, because their gradient is gated by `W_o`.

So the memory block has a chicken-and-egg cold-start: `W_o` needs `o` to be a useful direction to move toward, but `o` won't become useful until `W_{k,v,q}` learn, which requires `W_o` to be non-zero. The model has to wait for `W_o` to escape the zero attractor before the memory pathway can start learning meaningfully. Once it does, convergence is rapid (the 13.2% → 94.2% jump).

### Why additive is slightly slower than swap in the early window

Both have the W_o=0 cold start. But:
- In the **swap**, the memory block IS the only thing in the FFN slot at L3. The model has nowhere else to put L3-FFN-shaped signal, so the gradient pressure on the memory pathway is sharp.
- In the **additive**, the MLP at L3 can absorb most of what the memory pathway might do. The gradient pressure to use the additive pathway is diluted by the MLP being available. So the additive memory takes longer to find its niche.

Once the model starts using the memory pathway (step 101→126 jump), the MLP+memory combo apparently converges *faster* than the memory-only swap (94% vs 86% at step 126). But both saturate at the same step.

## Implications for Stage 2 design

The Stage 1.5 writeup said Stage 2 has to recover the swap's ~2× gap. **It actually has to fix the upstream cause, which is the W_o=0 init.** Three concrete options:

1. **W_o non-zero init.** Same uniform `[-s, s]` as K/V/Q. Loses the bit-identical-to-baseline-at-init guarantee, but unblocks the gradient flow. Simplest fix.

2. **Tiny W_o init.** Uniform `[-s*0.01, s*0.01]` or so. Keeps the contribution near zero at init (won't perturb early loss meaningfully) but lets K/V/Q gradients flow non-trivially. Compromise.

3. **Identity-on-W_o + sigmoid gate.** Init W_o as identity-like, gate the output through a sigmoid that starts ~0.5. The path is "on" from step 0 but the gate can learn to close it. Most expressive, most invasive.

Option 2 looks like the right pragmatic trade. It's a one-line change in `init_weights`. Worth a probe arm to compare.

## Stage 2 (learned α/η) probably won't fix this on its own

If we add learned `α` and `η` (the planned Stage 2 work) but keep `W_o = 0` init, we still have the K/V/Q cold start. The learned gates would also be subject to the same gradient-gating: their gradients flow through W_o.

So Stage 2 should land **on top of** an init fix, not before it. The right ordering:

1. **Stage 1.5b: probe `W_o` non-zero init.** ~10 min experiment, single probe arm with the additive arch + small W_o init, compare to current additive trajectory. If saturation moves from step ~151 to step ~76 (matching baseline), the init was the whole story.
2. **Stage 2: learned α/η on top of the fixed init.** Now the gates have something to actually gate; gradient signal is present from step 1.

## Wall budget

- Additive arm wall: 6.22 min (vs swap's 8.07 min — slightly faster despite more params, probably MPS warming better on the second run)
- Total Stage 1.5 wall to date: ~50 min across three probe arms (baseline v1 killed, baseline v2, swap v2, additive)
- W_o non-zero init follow-up: ~10 min wall, ~5 min code

## What's saved

| file | what |
|---|---|
| `/tmp/probe_mqar_additive.log` | full additive arm trajectory |
| `nanochat/gpt.py` | `hope_additive_memory_layer` config + Block branch + init_weights branch |
| `nanochat/checkpoint_manager.py` | back-compat shim |
| `scripts/base_train.py` | `--hope-additive-memory-layer` CLI flag |
| `dev/probe_mqar.py` | `--hope-additive-memory-layer` CLI flag |
| `tests/test_memory_plumbing.py` | 4 new tests for the additive arm |

All committed in `f990b16`.

## Decision point

Open question for the operator: do we

(a) **run the W_o init probe arm next** (~15 min total) to confirm the init is the bottleneck — strongest signal-per-wall available now,
(b) **proceed to Stage 2** with W_o-zero init knowing it's a slow-start architecture — gates Stage 2 results behind an unfixed cold start, or
(c) **wrap** with the strong negative result (additive doesn't help, the issue is the init not the architecture topology) and revisit later?

Operator's lean would dictate.
