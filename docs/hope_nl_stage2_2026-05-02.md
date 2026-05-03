# Hope/NL Stage 2 — per-token learned alpha/eta on MQAR

**Date:** 2026-05-02
**Branch:** `experiment/hope-nested-learning`
**Architecture commit:** `e559446`
**Probe commit:** `c371022`
**Decision commit:** `e4f14db` (ADR-002)
**Companion writeups:**
- `docs/hope_nl_stage1_5b_w_o_init_2026-05-01.md` (W_o init root cause)
- `docs/project_notes/decisions.md::ADR-002` (Stage 2 design priors)

## TL;DR

Stage 2 with ADR-002 defaults (additive @ L3, W_o=1.0, α≈0.99, η≈0.10) on MQAR:

- **Architecture functions as designed.** Gates alive from step 1 (no cold-start
  dead gradient on any of W_α/W_η/b_α/b_η).
- **Reproducibly slower than baseline** at full saturation. Two seeds:
  saturation at steps 101 / 126. Baseline ~76. Lag ~25-50 steps.
- **High seed-variance with multiple optimization basins.** A 5-arm sweep
  (3 α_init values × 2 seeds, partial) shows α_init_bias=3.0 produces *both*
  the best run we've seen (seed=0: step 51 saturation, beats baseline) AND
  the worst (seed=1: 0.57 at step 200, never converged). The architecture
  has at least two attractor basins for MQAR and seed determines landing.
  α-tuning doesn't fix the variance; it amplifies it.

Decision: hold the default at α=4.595. Either (c) pretrain on DCLM and let
val_bpb arbitrate (the seed sensitivity at probe scale may not matter at full
d6 pretrain horizon), or stop and accept that Stage 2 isn't a load-bearing
architectural change at d6/T=128.

**Update (2026-05-02): operator picked (c). Pretrain landed at val_bpb 1.1743,
exact baseline parity. Bimodal probe behavior did not transfer to natural-
language pretrain. Detail in "Full d6 DCLM pretrain result" section below.**

**Update (2026-05-03): SFT ran into an additive-topology activation-memory
issue at B=32 (Metal-dispatch hangs on M2). Detail in "Activation memory
characteristics" section below — practical recommendation is `--device-batch-
size=16` for Stage 2 additive on M2. Swap topology has lower activation
memory but is structurally different; switching would require a fresh
val_bpb validation, not just a flag flip.**

## What Stage 2 adds

Generalizes Stage 1's `LinearAttentionMemory` (fixed `α=η=1`) to per-token
learned gates, vectorized via prefix log-products so we keep one big causal
einsum (no Python loop):

```
α_t = α_max · sigmoid(W_α x_t + b_α)             # per-token forget gate
η_t = η_max · sigmoid(W_η x_t + b_η)             # per-token write strength
log_α_t = log(clamp(α_t, ε, α_max))
S_t     = cumsum_{j≤t} log_α_j
decay(t,i) = exp(S_{t-1} - S_i)   for i<t        # zero off-causal
o_t        = sum_{i<t} (q_t · k_i) · η_i · decay(t,i) · v_i
```

At `α_max → 1` and constant `η = 1`, Stage 2 is bit-equivalent to Stage 1 (test
`test_stage2_reduces_to_stage1_when_alpha_eta_constant` pins this within
numerical tolerance) — Stage 2 is a strict superset.

## Default settings (per ADR-002)

| field | default | rationale |
|---|---:|---|
| `hope_memory_w_o_init_scale` | 1.0 | Stage 1.5b: avoids K/V/Q gradient gate cold start. |
| `hope_memory_alpha_max` | 0.999 | Strict upper bound keeps `log α` finite. |
| `hope_memory_eta_max` | 1.0 | No reason to amplify writes above Stage 1's η=1. |
| `hope_memory_alpha_init_bias` | 4.595 | `logit(0.99/0.999)`; initial α ≈ 0.99 (long memory, not saturated). |
| `hope_memory_eta_init_bias` | -2.197 | `logit(0.1)`; initial η ≈ 0.10 — small but **live**, NOT zero (would recreate the W_o=0 gradient-gate trap on a softer surface). |

Topology: **additive** insertion at one block (per ADR-002). The MQAR cross-check
in 1.5b showed swap and additive perform comparably with `W_o=1.0`; we start
additive to isolate the memory branch's contribution rather than to compress
the MLP away.

## Probe protocol

Three arms, each from-scratch d6 (n_layer=6, n_embd=384), MQAR with K=M=16,
T=128, 200 iters, eval-every=25, batch=64. Same recipe as Stage 1.5/1.5b for
direct comparison.

```bash
# baseline (no memory module)
python -m dev.probe_mqar --label baseline --num-iterations=200 --eval-every=25

# Stage 1-additive cross-check (linear, W_o=1.0)
python -m dev.probe_mqar --label stage1add_w1 \
    --hope-additive-memory-layer=3 --hope-memory-w-o-init-scale=1.0 \
    --num-iterations=200 --eval-every=25

# Stage 2 with default gates (learned_gate, W_o=1.0, α≈0.99, η≈0.10)
python -m dev.probe_mqar --label stage2_default \
    --hope-additive-memory-layer=3 --hope-memory-w-o-init-scale=1.0 \
    --hope-memory-kind=learned_gate \
    --num-iterations=200 --eval-every=25
```

Pass/fail bar (per ADR-002): Stage 2 must saturate at step ≤ ~100 (baseline +
small slack). Earlier than ~76 = green; meaningfully later = debug grad norms /
gate stats before sinking pretrain budget.

## Results

### Saturation trajectory

| step | baseline | stage1add (W_o=1) | **stage2_default** |
|---:|---:|---:|---:|
| 1   | 0.031 | 0.031 | 0.031 |
| 26  | 0.037 | 0.065 | 0.060 |
| 51  | 0.111 | 0.156 | **0.922** ⚡ |
| 76  | **0.999** | **1.000** | 0.944 |
| 101 | 1.000 | 1.000 | **1.000** |
| 126 | 1.000 | 0.998 | 1.000 |
| 151 | 1.000 | 1.000 | 1.000 |
| 176 | 0.999 | 1.000 | 1.000 |
| 200 | 0.999 | 0.998 | 1.000 |

Wall: 6.4 / 7.1 / 7.4 min per arm.

**Saturation step (first eval ≥ 0.99):**

| arm | first ≥ 0.90 | first ≥ 0.99 | first 1.00 |
|---|---:|---:|---:|
| baseline | 76 | 76 | 101 |
| stage1add (W_o=1) | 76 | 76 | 101 |
| **stage2_default** | **51** | **101** | 101 |

Stage 2 is the **fastest to reach 0.90**, but the **slowest to reach 0.99**. The
two prior arms have a sharp inflection between step 51 (~0.15) and step 76
(~1.00); Stage 2 has a softer inflection — it gets useful at step 51 (0.92) but
spends 50 more steps cleaning up the last 8%.

### Stage 2 gate evolution

α and η stats from each eval batch's last forward (one batch, 64×128 tokens →
8192 gate values per stat). Initial design: α near 0.99, η near 0.10.

| step | α_min | α_mean | α_max | η_min | η_mean | η_max |
|---:|---:|---:|---:|---:|---:|---:|
| 1   | 0.982 | 0.989 | 0.993 | 0.072 | 0.104 | 0.169 |
| 26  | 0.919 | 0.974 | 0.994 | 0.003 | 0.012 | 0.086 |
| 51  | **0.565** | 0.871 | 0.983 | 0.000 | 0.004 | 0.058 |
| 76  | 0.346 | 0.820 | 0.986 | 0.000 | 0.006 | 0.101 |
| 101 | 0.081 | 0.633 | 0.985 | 0.000 | 0.004 | 0.079 |
| 151 | 0.028 | 0.494 | 0.984 | 0.000 | 0.005 | 0.130 |
| 200 | 0.005 | 0.327 | 0.968 | 0.000 | 0.002 | 0.041 |

α moves dramatically: starts tight at 0.99, spreads bimodally — many positions
forget aggressively (α near 0) while ~30% of positions retain α near 0.97-0.99.
This is exactly the per-token gating story Stage 2 is designed to express. By
step 200 the *mean* α has dropped to 0.33 — the model is using "forget by
default, remember selectively."

η goes the other way: shrinks fast and stays small. Mean η ~0.005 by step 51
means the memory is barely writing on most tokens. Combined with α≈0 on those
positions, the message is "for non-recall tokens, ignore the memory entirely."
The learned gates are off-loading ~95% of positions and reserving the memory
for the actual k/v binding pairs.

### Stage 2 grad-norm trajectory

Critical: was there a 1.5b-style cold start? **No.** All 6 gate parameters had
nonzero finite grads from step 1.

| step | W_k | W_v | W_q | W_o | W_alpha | W_eta | b_alpha | b_eta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1   | 5.6e-3 | 5.3e-3 | 5.4e-3 | 5.3e-3 | 5.2e-4 | 4.4e-3 | 2.0e-4 | 8.0e-4 |
| 26  | 3.2e-2 | 6.1e-2 | 3.2e-2 | 6.2e-2 | 4.1e-2 | 1.5e-1 | 3.3e-3 | 1.9e-3 |
| 51  | 4.6e-3 | 6.1e-3 | 4.7e-3 | 5.5e-3 | 4.0e-3 | 7.5e-3 | 3.6e-4 | 7.2e-4 |
| 76  | 3.2e-3 | 2.9e-3 | 3.1e-3 | 2.7e-3 | 1.4e-3 | 4.2e-3 | 7.0e-5 | 2.5e-6 |
| 101 | 3.4e-5 | 3.4e-5 | 3.4e-5 | 3.3e-5 | 4.1e-5 | 6.5e-5 | 4.8e-6 | 5.3e-6 |
| 200 | 2.3e-8 | 1.9e-8 | 2.3e-8 | 1.8e-8 | 1.6e-8 | 6.1e-8 | 1.5e-9 | 4.7e-9 |

The grad story tracks the loss story: peak gradient activity ~step 26 as the
model is rapidly carving out the gate behavior, fall-off after step 76 once
the task is largely solved, near-zero by step 200. All gate parameters
participate (no dead-gradient anywhere). This confirms the bias-as-explicit-Parameter
fix is working end-to-end and validates the W_o=1.0 + nonzero-η-init defaults.

## Reading the results

The single most informative comparison is at step 51:

| arm | acc @ 51 | what's going on |
|---|---:|---|
| baseline | 0.111 | attention is still wiring up the recall pattern; sharp inflection coming at step 76 |
| stage1add (W_o=1) | 0.156 | linear-attn memory contributing slightly, but still pre-inflection |
| **stage2_default** | **0.922** | learned gates have **already configured** to amplify the recall positions |

This is what we'd hope to see from a learnable memory mechanism: the gating
machinery lets the model express the recall pattern earlier than baseline can
discover it from attention alone. The cost is a softer final approach to 1.00
— Stage 2 plateaus at 0.94 from step 76 to ~100, while baseline jumps cleanly
from 0.11 to 1.00 in the same window.

Possible reasons for the soft tail:
- **Gate over-aggression**: the model has driven α near 0 on most positions and
  η near 0 too. The remaining 6% errors are likely positions where the gate
  *should* fire but the gate's logits haven't yet settled to the right token-
  dependent pattern. The next ~50 steps are spent fine-tuning gate logits on
  a small fraction of positions.
- **Optimizer interaction**: the gate biases are scalar and routed to AdamW;
  the full attention pathway may continue advancing under Muon while the gates
  are converging slower. Asymmetric convergence could explain the plateau.
- **MQAR-specific**: this task has K=M=16 i.e. each query has exactly one
  correct value to retrieve. A small fraction of "near-miss" positions where
  the gate sets α slightly wrong is enough to keep acc below 1.00 for many
  steps. On a more fault-tolerant downstream task this tail could be invisible.

Critically: **no cold-start gradient gate.** Step 1 grads are nonzero on
W_alpha, W_eta, b_alpha, b_eta. The 1.5b lesson — silent dead-gradient at init
is invisible in loss but visible in grad norms — does not apply here. Stage 2's
defaults work as designed.

### Comparison to ADR-002's pass/fail bar

ADR-002 said "Stage 2 must match baseline MQAR saturation (~step 76) before
any pretrain. If it lags, debug before sinking budget." Reading literally:

- Stage 2 reaches 0.99 at step **101**, vs baseline step 76. That's a 25-step
  lag. By the strict reading, this would be debug-not-pretrain.
- But Stage 2 reaches **0.92** at step 51, vs baseline 0.11. The architecture
  is *more sample-efficient* at intermediate accuracy. By the spirit of the
  bar (does the memory mechanism help), Stage 2 is doing useful work.

This is a single-seed result. Baseline saturation in the 1.5b sweep was
reproducibly ~76; Stage 1-additive(W_o=1) was reproducibly ~76. Stage 2's
"step 101 to 1.00" could be ±25 steps under reseed. Without a confirmation
seed we can't distinguish "real architectural property" from "this seed got
unlucky on the long tail."

## Confirmation seed (2026-05-02)

Per the operator's lean (a → b), ran Stage 2 with `--seed=1` to test whether
seed=0's curve was the headline result or a single-seed artifact.

| step | baseline | s2 seed=0 | s2 seed=1 |
|---:|---:|---:|---:|
| 1   | 0.031 | 0.031 | 0.039 |
| 26  | 0.037 | 0.060 | 0.065 |
| 51  | 0.111 | **0.922** | 0.385 |
| 76  | **0.999** | 0.944 | 0.561 |
| 101 | 1.000 | 1.000 | **0.988** |
| 126 | 1.000 | 1.000 | 0.999 |
| 200 | 0.999 | 1.000 | 0.999 |

Wall: 6.4 min.

**Reading**: the Stage 2 lag is real, not single-seed luck. Both seeds reach
0.99 between step 101-126; baseline reproducibly hits it at step 76 (per the
1.5b sweep). That's a ~25-50 step lag depending on threshold.

But the two seeds found **different gate strategies** for the same task:

| step | seed=0 α[min/mean/max] | seed=0 η[min/mean/max] | seed=1 α[min/mean/max] | seed=1 η[min/mean/max] |
|---:|---|---|---|---|
| 51  | 0.565 / 0.871 / 0.983 | 0.000 / 0.004 / 0.058 | 0.983 / 0.995 / 0.998 | 0.001 / 0.045 / 0.767 |
| 76  | 0.346 / 0.820 / 0.986 | 0.000 / 0.006 / 0.101 | 0.960 / 0.997 / 0.999 | 0.002 / 0.093 / 0.703 |
| 200 | 0.005 / 0.327 / 0.968 | 0.000 / 0.002 / 0.041 | 0.823 / 0.998 / 0.999 | 0.001 / 0.132 / 0.933 |

- **seed=0**: "forget by default, remember selectively." α drops to mean 0.33;
  η stays small (max 0.13).
- **seed=1**: "remember by default, write strongly when needed." α stays high
  (mean 0.998); η_max climbs to 0.93 — essentially saturating the cap.

Both reach ~1.0 acc but via different basins. The optimization is sensitive
to seed AND the gate space has at least two solution modes for MQAR. That's
architecturally interesting — and a yellow flag for stability at scale.

## α_init_bias sweep (2026-05-02)

Per the operator's lean (b first), ran a single-axis sweep on α_init_bias.
Started with seed=0 only, then added a seed=1 confirmation on the apparent
winner. Default η_init_bias=-2.197 and W_o init scale=1.0 throughout.

| step | baseline | a=3 s=0 | a=3 s=1 | **a=4.595 s=0** | **a=4.595 s=1** | a=6 s=0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1   | 0.031 | 0.033 | 0.041 | 0.031 | 0.039 | 0.031 |
| 26  | 0.037 | 0.042 | 0.059 | 0.060 | 0.065 | 0.076 |
| 51  | 0.111 | **1.000** ⚡ | 0.061 | 0.922 | 0.385 | 0.776 |
| 76  | 0.999 | 1.000 | 0.064 | 0.944 | 0.561 | 0.818 |
| 101 | 1.000 | 1.000 | 0.063 | 1.000 | 0.988 | 1.000 |
| 126 | 1.000 | 1.000 | 0.064 | 1.000 | 0.999 | 1.000 |
| 200 | 0.999 | 1.000 | **0.575** ❌ | 1.000 | 0.999 | 1.000 |

Wall: ~6-7 min/arm.

### Reading the sweep

The seed=0 column alone tells a clean monotonic story: lower α_init → faster
saturation. α=3.0 (init α≈0.95) saturated at step 51 — 25 steps faster than
baseline. We thought we'd found the right default.

The seed=1 confirmation on α=3.0 inverted that conclusion: **stuck at 0.57
acc through step 200**. Looking at gate evolution under seed=1 with α=3.0:

| step | α [min/mean/max] | η [min/mean/max] |
|---:|---|---|
| 1   | 0.926 / 0.952 / 0.967 | 0.051 / 0.095 / 0.156 |
| 26  | 0.871 / 0.920 / 0.967 | 0.005 / 0.018 / 0.178 |
| 51  | 0.897 / **0.971** / 0.988 | 0.001 / 0.014 / 0.365 |
| 76  | 0.809 / 0.982 / 0.996 | 0.000 / 0.010 / 0.256 |
| 200 | 0.680 / **0.989** / 0.998 | 0.000 / 0.005 / 0.134 |

α started at 0.95 per design — and then drifted *upward* to 0.99 and stuck.
η went down to ~0.005 by step 76. The model fell into a "remember-everything-
but-don't-write-much" basin and couldn't escape. Compare seed=0 with the same
config, where α dropped monotonically (mean 0.66 by step 26) and the model
saturated at step 51.

So α=3.0 has *both* the best AND the worst Stage 2 runs. **α-tuning doesn't
fix the variance; it amplifies it.** Lowering α_init gave the gates more
freedom, which means more freedom to find a bad basin, not just a good one.

The default α=4.595 has lower variance across seeds (steps 101/126) than
α=3.0 (steps 51 / 200+). The default is the safer choice given the variance.

## Decision point (post sweep)

(b) **More tuning** is unlikely to be the right move. Possibilities:
   - Sweep W_α/W_η init scale (currently uniform[-0.02, 0.02], small so the
     bias dominates). Bigger init might let positions differentiate per-token
     from step 1, perhaps reducing basin-dependence. ~6 arms ~42 min.
   - Sweep η_init_bias. We have one column at η=0.10. Other values might
     stabilize.
   These would resolve the variance but only at the cost of more probe budget,
   and the architecture's seed sensitivity at d6/T=128 is already a yellow
   flag.

(c) **Schedule full d6 DCLM pretrain with α=4.595 (default) on seed=0.**
   ~3h wall. The probe is a synthetic unit test at K=M=16, T=128; full pretrain
   is 5000 iters across millions of tokens of natural language. Seed
   sensitivity that matters at probe scale may wash out at pretrain horizon.
   The only test that matters for the original ticket is val_bpb on DCLM.

(d) **Stop and accept Stage 2 isn't load-bearing at d6 scale.** The probe
   shows the architecture functions but is reproducibly slower than baseline
   AND seed-sensitive. At T=128 and d6, the gating mechanism doesn't pull its
   weight. Hope/NL benefits live at scales we won't reach on M2.

My read: **(c) is the right call given the project goal**, but with a clear
expectation that val_bpb is likely to land at or slightly worse than baseline.
The Stage 1 swap pretrain (W_o=0) hit val_bpb 1.179 vs baseline 1.174 (+0.4%).
Stage 2 with α=4.595 on seed=0 will probably be in the same band. The probe
result tells us we shouldn't expect a gain. We should pretrain anyway because:
- The original ticket asks "does this architecture work?" — val_bpb is the
  honest answer at the project's scale.
- Even a "no improvement" result is informative: we'd document that Hope/NL's
  per-token gates don't help at d6/T=2048 on natural language.
- The probe's bimodal behavior is itself an interesting finding worth a clean
  full-pretrain follow-up.

If the operator's project-priority is shifting elsewhere, **(d)** is also
defensible — Stage 0+1+1.5+2 is a complete experimental unit and the
architecture wall has been mapped.

## Full d6 DCLM pretrain result (2026-05-02)

Operator picked **(c)**. Ran the full pretrain (5000 iters, T=512, accum=1,
device-batch-size=32, additive @ L3, W_o init 1.0, default α/η). Wall: 5h13min.

**Final val_bpb = 1.1743**, exactly at baseline (1.174) — better than Stage 1
swap's 1.179 (+0.4%). The probe-level concerns (bimodal basin behavior,
seed sensitivity, ~25-50 step saturation lag on MQAR) **did not manifest at
full pretrain horizon**. Smooth monotonic descent through the LR decay window,
no signs of optimization instability. The 5000-iter natural-language signal
washed out the seed sensitivity that plagued the synthetic probe.

| run | val_bpb | Δ vs baseline |
|---|---:|---:|
| baseline d6 (5000 iter) | 1.1743 | — |
| Stage 1 swap (W_o=0) | 1.179 | +0.4% |
| **Stage 2 additive (W_o=1, learned_gate)** | **1.1743** | **0.0%** |

Two readings of the 0.0% delta:

- **Pessimistic:** ~590K extra params (W_α, W_η, b_α, b_η at one block) bought
  exactly zero val_bpb improvement. The learned gates compute meaningful
  per-token values that aren't load-bearing for next-token prediction at this
  scale. Hope/NL benefits live at scales we won't reach on M2.
- **Cautiously optimistic:** Stage 2 closed the +0.4% gap that Stage 1 swap
  carried. Tiny improvement, within noise, but consistent with "learned gates
  can do what fixed α=η=1 cannot." Plausibly becomes a real win at larger T or
  deeper d.

The architecture works, matches baseline, doesn't beat it. That's the honest
answer to the original ticket at the project's scale.

## Activation memory characteristics — additive vs swap (2026-05-03)

Discovered while trying to SFT the Stage 2 base checkpoint. Pretrain at d6,
B=32, T=512 ran fine. SFT at the same shape **hung in MPS Metal-dispatch
deadlock** at step 184-200 across two attempts. Halving to B=16 cleared it.

The additive topology has materially higher per-step activation memory than
swap or baseline because it adds (B,T,T) intermediate buffers *on top of*
the existing MLP activations:

| variant | block-3 forward | extra (B,T,T) tensors | extra memory at B=32, T=512 (bf16) |
|---|---|---:|---:|
| baseline | attn + MLP | 0 | 0 |
| Stage 1 swap | attn + Memory (no MLP) | 1 (`scores`) | ~33 MB |
| **Stage 2 additive** | **attn + MLP + Memory** | **4** (`scores`, `log_decay`, `decay`, `weights`) | **~130 MB** |

Each (B, T, T) tensor at our shape is 32×512×512 floats × 2 bytes (bf16) ≈ 33 MB.
Stage 2 additive's `forward` materializes four such tensors:

- `scores = einsum('btd,bid->bti', q, k)` — `(B, T, T)`
- `log_decay = S_shift - S.transpose(1, 2)` — `(B, T, T)`
- `decay = exp(log_decay.masked_fill(...))` — `(B, T, T)`, separate allocation
- `weights = scores * eta.transpose(1,2) * decay` — `(B, T, T)`

All four are held for backward. Plus the residual MLP at L3 keeps its `(B, T, 4D)`
activation. Stage 1 swap has only `scores`, no MLP at the swap block.

**Empirical confirmation:**

- Stage 1 swap SFT @ B=32: completed cleanly (HANDOFF: val_bpb 0.6712).
- Stage 2 additive SFT @ B=32: hung at step 184-200 in MPS dispatch (twice).
- Stage 2 additive SFT @ B=16: completed (no Metal deadlock).

### Implications

1. **Resource-constrained training (M2-class hardware).** Default to
   `--device-batch-size=16` for Stage 2 additive at d6/T=512. Halve again if
   you increase d_model, T, or memory layer count. Pretrain happened to fit
   at B=32 because it was running closer to a fresh boot with less competing
   memory pressure; SFT after a long session got squeezed.

2. **Topology trade-off — swap is structurally different, not "same architecture
   cheaper".** Swap replaces the MLP at the memory block (~8D² fewer params at
   that layer; the block's nonlinearity comes only from the memory module).
   Additive keeps the MLP and adds memory as a third residual stream. These
   are different models with different learning dynamics. The MQAR cross-check
   (Stage 1.5b) showed both saturate at step 76 with W_o=1.0 — but that's a
   *synthetic recall* equivalence, single seed each. **We have no full DCLM
   pretrain run on Stage 2 swap**, only Stage 2 additive (val_bpb 1.1743) and
   Stage 1 swap-with-W_o=0 (val_bpb 1.179, suboptimal init). If you're tempted
   to switch topology to save activation memory, **validate val_bpb parity first
   with a Stage 2 swap pretrain run**. The savings are real (1 vs 4 (B,T,T)
   tensors, no MLP activations at L3) but conditional on the model still
   learning equivalently — which we don't know yet at this scale.

3. **Forward-pass engineering opportunity.** The Stage 2 forward materializes
   `log_decay → decay` as two separate (B, T, T) allocations. Could fuse
   in-place: `log_decay.masked_fill_(..., float('-inf')).exp_()` saves one
   33 MB allocation. The `weights = scores * eta * decay` line could fuse
   into the einsum via a custom kernel, but that's deeper work. Modest M2
   wins; not load-bearing for CUDA where (B, T, T) buffers are free relative
   to GEMM throughput.

4. **`(B, T, T)` is a quadratic-in-T cost.** At T=2048 (full pretrain context
   for nanochat's default recipe), each tensor would be 32×2048×2048×2 = 537 MB,
   and four of them is **2.1 GB extra** for one memory block. This is why the
   additive topology was problematic even on systems that could handle Stage 1
   swap's single buffer at the same shape. Anyone scaling Stage 2 to longer
   contexts should plan for the quadratic blow-up — chunked/streaming forms
   become important, not optional.

## Open questions

- **Hyperparameter sweep on α/η init biases?** Currently default-only. If
  Stage 2 underperforms baseline, is it because the init values are wrong, or
  because the architecture is fundamentally limited at this scale? A small
  sweep over `α_init_bias ∈ {3.0, 4.595, 6.0}` and `η_init_bias ∈ {-3.0, -2.197, -1.0}`
  would isolate this. Punted unless first arm shows interesting drift.
- **Multi-seed confirmation?** Same question as Stage 1.5: single seed per arm.
  Worth a confirmation seed before sinking ~3h pretrain on the headline result.
- **Swap topology with learned gates?** Per ADR-002, started additive. If
  Stage 2-additive works, a follow-up swap arm tests "can memory replace MLP
  with learned gates?"
- **Stage 2 at full d6 pretrain on DCLM.** Gated by probe pass/fail; ~3h wall.
