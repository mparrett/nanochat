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

## Decision point

Open question:

(a) **Stage 1.5c: probe `W_o` init scale sweep** (0.0 / 0.1 / 0.5 / 1.0). Maps the bit-identical-init vs grokking-speed trade. ~25 min wall.

(b) **Stage 2 — learned α/η on top of `W_o` init=1.0.** The originally planned next architectural step, now with the cold-start fix as the foundation.

(c) **Run the Stage 1-additive (W_o=1) variant at full d6 pretrain.** Test whether the LM val_bpb on DCLM reflects the probe's "matches baseline" finding. ~3 h wall.

(d) **Wrap.** The probe arc has paid for itself; a clean handoff to a future Stage 2 / pretrain session is on the table.

Operator's call.
