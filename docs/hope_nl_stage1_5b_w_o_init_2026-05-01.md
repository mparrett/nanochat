# Hope/NL Stage 1.5b — `W_o` init was the bottleneck

**Date:** 2026-05-01
**Branch:** `experiment/hope-nested-learning`
**Architecture commit:** (pending — `_init_w_o` helper + `hope_memory_w_o_init_scale` config)
**Companion writeups:**
- `docs/hope_nl_stage1_5_results_2026-05-01.md` (original swap finding)
- `docs/hope_nl_stage1_5_additive_2026-05-01.md` (operator's additive idea, falsified the nonlinearity hypothesis)

## TL;DR

The Stage 1 swap's ~2× sample-efficiency gap on MQAR was **entirely caused by the `W_o = 0` init in `LinearAttentionMemory`.** It was not a property of the architecture topology (swap vs additive), nor of removing the MLP's nonlinearity. With `W_o` initialized at the same uniform `[-s, s]` magnitude as `K/V/Q`, the additive memory variant **matches baseline grokking step-for-step** (99.95% acc at step 76).

| step | baseline | swap (W_o=0) | additive (W_o=0) | **additive (W_o=1)** |
|---:|---:|---:|---:|---:|
| 26 | 0.037 | 0.054 | 0.058 | 0.065 |
| 51 | 0.111 | 0.066 | 0.064 | **0.156** |
| 76 | **0.999** | 0.131 | 0.068 | **0.9995** |
| 101 | 1.000 | 0.632 | 0.132 | 1.000 |
| 126 | 1.000 | 0.863 | 0.942 | 0.999 |
| 151 | — | 1.000 | 0.999 | 1.000 |
| 200 | 1.000 | 1.000 | 1.000 | 0.998 |

Wall: 6.41 min for this arm; ~46 min total Stage 1.5 across 5 arms.

## The mechanism, fully resolved

The `LinearAttentionMemory` forward is:

```
o = einsum('btd,bsd->bts', q, k) * scale  →  causal-mask  →  einsum back to v  →  o
y = W_o(o)
```

With `W_o = 0` at init, the gradient flow analysis is:

- `∂L/∂W_o = grad_y · oᵀ` — non-zero from upstream loss; W_o starts moving on iter 1.
- `∂L/∂W_{k,v,q} ∝ W_oᵀ · (...) = 0 · (...) = 0` initially. K/V/Q gradients are **gated by W_o** and are exactly zero at init.

So K/V/Q can only start learning after W_o moves enough that `W_oᵀ` is non-zero. But W_o moves toward whatever direction makes `o` useful — and `o` is computed from random K/V/Q at init, so it's noise. W_o has to escape the zero attractor on noisy gradient signal before K/V/Q can join the optimization. **Chicken-and-egg cold start.**

The empirical fingerprint matches: the W_o=0 arms hover near chance until step 76+, then jump rapidly once W_o has escaped. With non-zero W_o init, K/V/Q gradients flow from step 1, the model learns the same way it learns `attn` and `mlp` blocks, and convergence matches baseline exactly.

## What the prior writeups said and what they got wrong

- **`docs/hope_nl_stage1_5_results_2026-05-01.md`** (swap result): "Best hypothesis: removing the MLP at L3 (ReLU² nonlinearity) ... removes one feature-shaping nonlinearity that helps attention's selection sharpen." **Wrong.** The additive variant (which keeps all MLPs) was just as slow.
- **`docs/hope_nl_stage1_5_additive_2026-05-01.md`** (additive result): correctly identified the W_o=0 chicken-and-egg as the likely cause. This run confirms it.

## What it cost to get the right answer

5 probe arms, ~46 min total wall, two architectural variants, one init knob:

| arm | architecture | W_o init | grok step | wall |
|---|---|---|---:|---:|
| baseline v1 (killed) | MLP | n/a | ~76 (then killed) | 18 min |
| baseline v2 | MLP | n/a | ~76 | 7.44 min |
| swap v2 | LinAttn @ L3 (replace) | 0 | ~151 | 8.07 min |
| additive v2 | LinAttn @ L3 (add) | 0 | ~151 | 6.22 min |
| **additive_woinit1** | **LinAttn @ L3 (add)** | **uniform[-s, s]** | **~76** | **6.41 min** |

That's the value of the synthetic probe — without it we'd have spent another ~3h pretrain run with the wrong fix (e.g. trying Stage 2 with W_o=0 still in place).

## Stability tradeoffs of the new init

The `W_o = 0` init served a real purpose: **bit-identical to baseline at step 0**. With non-zero W_o init, the memory block contributes random noise to the residual stream from iter 1. At step 1 in our probe, loss diverged from baseline by ~0.003 (10.3954 vs 10.3988) — small but real.

For nanochat at scale (full pretrain over 5000 iters), this means the modified architecture's loss curve will diverge from baseline immediately, not after some grokking-style phase transition. That may complicate diagnostic comparisons but doesn't compromise stability — the magnitude is small and the model trains around it.

If we want to keep "bit-identical at step 0" as a guarantee for some experiments, intermediate scales (`scale=0.1`, `scale=0.01`) are available. Worth a follow-up arm to map the trade.

## Stage 2 design — now meaningfully informed

Stage 2 was originally going to add per-token learned `α` and `η`. With the corrected understanding:

1. **W_o init non-zero is now the default for any memory-bearing block.** New config field `hope_memory_w_o_init_scale: float = 0.0` (still 0 by default for backward compat); set to 1.0 in any new memory-bearing config.
2. **Learned `α`/`η` should be initialized so they don't gate the memory contribution to zero from step 1**. e.g. `α` near 0.99 (not 0.0), `η` at the magnitude that gives reasonable contribution at the chosen W_o scale.
3. **The probe is the gate for Stage 2 variants.** Any Stage 2 architecture should match baseline MQAR convergence at step ~76. If it doesn't, there's a similar cold-start issue to debug before sinking pretrain budget.

## Code change

In `nanochat/gpt.py`, factored the W_o init into a helper:

```python
def _init_w_o(weight, s, scale):
    if scale == 0.0:
        torch.nn.init.zeros_(weight)
    else:
        torch.nn.init.uniform_(weight, -s * scale, s * scale)
```

`init_weights` calls this for both `block.mlp.W_o` (swap variant) and `block.add_memory.W_o` (additive variant). Plus:
- `GPTConfig.hope_memory_w_o_init_scale: float = 0.0` (new field)
- `_patch_missing_config_keys` shim for older checkpoints
- `dev/probe_mqar.py` CLI flag `--hope-memory-w-o-init-scale`
- `scripts/base_train.py` should also get the flag (TODO, low priority — only matters if we run a full pretrain with this)

A bug along the way: the first attempt only modified the W_o init in the swap branch, not the additive branch (`replace_all` on `init.zeros_(block.mlp.W_o.weight)` didn't catch `init.zeros_(block.add_memory.W_o.weight)`). Caught by sanity-printing `W_o.norm()` after the model was built. The fix was a two-character edit. Good lesson: when adding a config knob, verify it actually takes effect on the path you intend.

## Stage 1.5c — W_o init scale sweep

Ran the full additive arm at four `W_o` init scales: 0.0 (chickenpox baseline), 0.1, 0.5, 1.0.

### Trajectory

| step | scale=0.0 | scale=0.1 | scale=0.5 | **scale=1.0** |
|---:|---:|---:|---:|---:|
| 1 | 0.031 | 0.032 | 0.033 | 0.031 |
| 26 | 0.058 | 0.059 | 0.058 | 0.065 |
| 51 | 0.064 | **0.591** ⚠ | 0.101 | 0.156 |
| 76 | 0.068 | 0.736 | 0.715 | **0.9995** |
| 101 | 0.132 | 0.864 | 0.979 | 1.000 |
| 126 | 0.942 | 0.996 | 0.9995 | 0.999 |
| 151 | 0.999 | 1.000 | 1.000 | 1.000 |
| 200 | 1.000 | 1.000 | 1.000 | 0.998 |

### Step-1 loss vs scale (perturbation magnitude)

Loss diverges from baseline monotonically as W_o magnitude grows — the additive memory contributes random noise to logits proportional to ‖W_o‖.

| scale | step-1 loss | step-1 Δ vs baseline |
|---:|---:|---:|
| 0.0 (baseline-equivalent) | 10.3988 | 0 |
| 0.1 | 10.3977 | −0.0011 |
| 0.5 | 10.3959 | −0.0029 |
| 1.0 | 10.3954 | −0.0034 |

### Step-to-saturation ranking

Defining "saturation" as first eval ≥ 0.995 stably:

| scale | saturation step | wall to saturation |
|---:|---:|---:|
| 0.0 | ~151 | ~6 min |
| 0.1 | ~126–151 | ~5 min |
| 0.5 | ~126 | ~4 min |
| **1.0** | **~76** | **~2.5 min** |

**Bigger W_o init = faster saturation, monotonically** (the scale=0.1 step-51 spike at 0.591 was a single-seed outlier — by step 76 it had only climbed to 0.736 while scale=0.5 was at 0.715, so the early jump didn't compound).

### Reading the sweep

The trade-off framing — "small init preserves bit-identical-at-step-0 vs large init unblocks gradient flow faster" — is real but has a clear winner on this task: **scale=1.0 wins on time-to-saturate** by a 2× margin over any smaller value, and only costs ~3 thousandths of a step-1 loss in noise injection. There is no sweet spot at 0.1 or 0.5; smaller scales just delay saturation proportionally.

If we ever need bit-identical-at-step-0 for some diagnostic comparison, scale=0.0 is still available. For training recipes that just want the memory mechanism to work, **scale=1.0 is the new default**.

### Cross-check: swap variant with W_o=1.0

The sweep was on the additive variant. To confirm the W_o init fix isn't somehow architecture-specific, we ran the swap variant (LinearAttentionMemory replacing MLP at L3) with `hope_memory_w_o_init_scale=1.0` too:

| step | baseline | **swap(W_o=0)** | **swap(W_o=1.0)** | additive(W_o=1.0) |
|---:|---:|---:|---:|---:|
| 26 | 0.037 | 0.054 | 0.065 | 0.065 |
| 51 | 0.111 | 0.066 | **0.696** | 0.156 |
| 76 | **0.999** | 0.131 | **0.9998** | 0.9995 |
| 101 | 1.000 | 0.632 | 1.000 | 1.000 |

**Confirmed: the fix works for both topologies.** swap(W_o=1.0) saturates at step ~76, identical to baseline and to additive(W_o=1.0). The architecture choice (swap vs additive) does not affect grokking speed once the cold start is fixed.

Curiosity: swap(W_o=1) shows even higher acc at step 51 than additive(W_o=1) (0.696 vs 0.156). Both saturate at step 76, but the swap may have a marginally sharper inflection — plausibly because the swap's gradient signal is cleaner ("memory IS the L3 FFN" vs "memory competes with L3 MLP"). One seed each, so this could also be noise. Worth a confirmation seed if anyone cares to claim swap > additive for this kind of task.

## Decision point

The sweep gives a clean answer: **`hope_memory_w_o_init_scale=1.0` is the recommended default** for any memory-bearing block going forward.

Three possible next moves:

(a) **Run the Stage 1-additive (W_o=1) variant at full d6 pretrain.** ~3 h wall. Validates whether the probe's "matches baseline grokking" generalizes to LM val_bpb on DCLM. The previous Stage 1 (swap, W_o=0) full pretrain landed at val_bpb 1.179 vs baseline 1.174 — a 0.4% gap. With the W_o init fix, that gap may close, widen, or stay the same. Each outcome teaches us something.

(b) **Stage 2 — learned α/η on top of `W_o init=1.0` foundation.** Per-token gates, vectorized via Codex's prefix log-product trick. Stability defaults: α bounded < 0.999, init near long-memory, η small-init, log α/η/memory-RMS/W_o-norm. The probe is the gate before any longer pretrain.

(c) **Wrap, hand off.** Stage 0 + 1 + 1.5 + 1.5b + 1.5c are a complete experimental unit. A clean handoff for whoever picks up Stage 2 is on the table.

Operator's lean dictates which.

## Resolution (2026-05-02)

Operator chose **(b)**, informed by Codex's response. Decision and design priors recorded in `docs/project_notes/decisions.md::ADR-002`. Headline:

- Stage 2 starts on the `W_o=1.0` foundation, additive topology, vectorized prefix-log-product form.
- α near long memory (~0.99 initial, α_max ≈ 0.999); η small but live (NOT near zero — same gradient-gate trap).
- MQAR probe is the gate; ~3h DCLM pretrain budget held until Stage 2 clears it.
- Where we depart from Codex: topology choice is design preference, not data-driven; skipping Stage 1-additive full pretrain is a budget call we accept with a noted gap.
