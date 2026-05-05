# Hope/NL track — synthesis (2026-05-05)

End-to-end read of the Hope / Nested Learning experimental track on
nanochat. Stage 0 → 1 → 1.5(a/b/c) → 2 → A1 → A2 → A3 → A3-prime, plus
the methodological lessons that came out of running it.

## TL;DR

We prototyped Behrouz et al.'s Hope / Nested Learning memory architecture
on nanochat, scaling from a stateless transformer block
(`output = f(x)`) to one that carries per-token mutable memory
(`output, M = f(x, M)`) with learned per-token gates. At d6/5000-iter
on ClimbMix, **the architecture is neutral on natural-language val_bpb
under recipe-controlled comparison** — modern-recipe vanilla d6 baseline
beats Stage 2 (0.6483 vs 0.6495 on SFT, 1.1686 vs 1.1712-1.1729 on
pretrain), all within the 0.0016 seed-noise spread we measured.

The original framing — "Stage 2 wins by 1.8% on SFT" — turned out to be
**recipe drift, not architecture.** ~0.0156 of "free" SFT improvement
had accumulated across 4 commits between when the historical baseline
was trained and when Stage 2 was trained, dwarfing the architectural
delta.

The probe (MQAR) showed the memory mechanism *does* learn: with the
right initialization (Stage 1.5b's `W_o` finding), Stage 2's additive
memory branch matches baseline synthetic-recall grokking step-for-step.
That capability just doesn't translate to LM bpb at this scale and
corpus.

Net contribution of the track is the working memory module, the
probe-first methodology, and the recipe-drift lesson — not an
architectural win.

## Setup

**Why this fork.** Investigation ticket from the trx4mr project (full
text: `~/projects-new/trx4mr/docs/idea-hope-nested-learning.md`).
Goal was to prototype Hope / NL on top of nanochat's well-tested CPU/MPS
pipeline rather than rebuild from picoGPT.

**Scale.** d6 (6 layers, n_embd=384, head_dim=64, max_seq_len=512),
5000-iter pretrain on a ~700 MB ClimbMix subset (8 train shards), 375-step
SFT on MMLU/GSM8K mix. 74M params. ~3-5h pretrain wall on M2 MacBook
Pro 24GB. ~600× smaller than the paper's 760M / 30B-token setup.

**Discipline.** nanochat is organized around a single `--depth` dial:
all hyperparameters (width, heads, LR, weight decay, training horizon)
are derived from depth. We followed the same discipline — every change
had to be principled enough to work across depths, not just at d6.
Probe-first: don't burn pretrain budget on hypotheses we haven't
falsified at probe scale.

## The arc

### Stage 0 — `memory_state` plumbing (commit `d2bbc0e`)

Plumbed an optional `memory_state` argument through `forward()` of the
nanochat block. Threaded gradient-flow assertions to confirm the new
return path didn't silently drop gradients. **No architecture change**
— purely the API contract that downstream stages would build on.
val_bpb unchanged.

### Stage 1 — swap one MLP for `LinearAttentionMemory` (commit `bc54858`)

First architectural change: at block index 3 of 6, swap the MLP for a
linear-attention memory module implementing the foundational equation
`M_t = α·M_{t-1} + v_t·k_t^T; y_t = M_t·q_t`. Single block, fixed α.

Full d6 pretrain → **val_bpb 1.179 vs baseline 1.174** (+0.4%, within
single-seed noise). SFT followed similarly. The architecture works at
d6 quality level — it doesn't break anything — but no clean win.

Open question at this point: is the memory module *actually* doing
memory-flavored work, or is it just acting as a peculiarly-parameterized
linear projection that the rest of the network learns to ignore?
val_bpb averaged over the SFT distribution can't disambiguate.

### Stage 1.5 — synthetic recall probe design (ADR-001)

Codex flagged the same concern: "Full LM val_bpb is too blunt for
Hope/NL behavior. Before Stage 2, add a tiny diagnostic task that
can tell whether memory helps at all."

Designed and built a multi-query associative recall (MQAR) probe
(`dev/probe_mqar.py`). Trains a small d6 from random init on a
binding-and-lookup task, measures recall accuracy at fixed step
intervals. Architecturally identical to nanochat except optional
swap/additive memory at layer 3.

### Stage 1.5 — MQAR results (swap and additive)

Both swap and additive-memory variants saturated at **step ~151** vs
**baseline ~76** — a ~2× sample-efficiency *gap*, the wrong direction
for an architecture that's supposed to *help* memory tasks.

This was the first piece of evidence that something deeper was wrong.
A clean architecture should at minimum match baseline on a task it's
ostensibly designed for. Two configurations losing by 2× pointed at a
shared cause, not at topology.

### Stage 1.5b — `W_o = 0` was the entire bottleneck

Hypothesized the cold-start problem: if `W_o` is initialized to zero
(common practice from "value head zero-init" patterns), the gradient
through `M·q` to `K`, `V`, `Q` is zero at step 0 — the memory pathway
can't *start* learning. The MLP path dominates until something tiny
breaks symmetry.

Added a `hope_memory_w_o_init_scale` knob, set it to 1.0 (uniform
`[-s, s]` like K/V/Q), re-ran the additive probe. **Saturation moved
from step ~151 to step ~76** — exact match to baseline. The 2× gap was
entirely the init.

This was the load-bearing finding of the track. Without it, every
downstream stage would have been measuring a debug artifact.

### Stage 1.5c — `W_o` init scale sweep

Confirmed monotonicity: bigger `W_o` init → faster saturation, no
sweet spot. scale=1.0 won by 2× over scale=0.5, by 4× over scale=0.1
(saturation steps 76 / 151 / 301). Locked `w_o_init_scale=1.0` as the
load-bearing default for any memory-bearing config from there on.

### Stage 1-additive — topology pivot (commit `f990b16`)

Codex's recommendation: switch from the original swap topology
(memory replaces MLP) to additive (memory inserted as a third residual
stream alongside attention + MLP). Cleaner experimental isolation —
isolates whether the memory branch *learns useful behavior* without
also paying the cost of *removing an MLP*. Cross-checked at probe
scale: swap(W_o=1.0) and additive(W_o=1.0) both saturate at step ~76
with comparable trajectories. Topology choice is a design preference,
not a data-driven elimination.

### Stage 2 — per-token learned α/η gates (commit `e559446`, ADR-002)

The headline architectural addition: replace the fixed scalar α (memory
decay) with a per-token learned `α_t = α_max · sigmoid(W_α x_t + b_α)`,
and add a per-token gain `η_t = η_max · sigmoid(W_η x_t + b_η)` on the
write. Vectorized via prefix-log-products: `log_α = log(clamp(α, eps,
α_max))`, prefix-sum, subtract to form causal decays. Same O(T²) cost
class as Stage 1.

Defaults from Codex's design priors: `α_max = 0.999`,
`alpha_init_bias = 4.595` (initial α ≈ 0.99 — long memory, not
saturated), `eta_init_bias = -2.197` (initial η ≈ 0.1 — small but
live), `w_o_init_scale = 1.0`.

Probe gate cleared (saturation step ~76). Full d6 pretrain →
**val_bpb 1.1743**. SFT → **val_bpb 0.6518**.

At the time, the comparison framing was **Stage 2 (0.6518) vs baseline
(d6_b_iso, 0.6639) = -0.0121 (~1.8% SFT improvement)**. We accepted
that headline at face value. ChatCORE on the SFT checkpoint came in at
0.1744, but 91% of that was SpellingBee template memorization that
would equally apply to any d6 SFT — the metric is diagnostically blunt
at this scale.

### A1 — `--inherit-from` recipe-parity infra (commit `ca9bc94`)

Pre-A3 audit step. Codex's metadata-audit pointed out that the queued
A3 commands had silent drift on `head_dim` (CLI default 128 vs reference
64) plus 5 other field mismatches. Without explicit re-passing of every
relevant flag, A3 would have trained a different model — 6h of compute
wasted.

Built `nanochat.common.load_inherit_config()` and `--inherit-from`
flag on `base_train.py` and `chat_sft.py`. Loads a reference run's
`user_config` as parser defaults *before* CLI parsing; CLI flags
override only what's intentionally different. Excludes per-run /
operational fields (`run`, `model_tag`, `seed`, `save_every`, etc.).

Validated end-to-end: 34 fields auto-loaded from the seed=42 reference
meta. The same mechanism was then load-bearing for A3, A3', and the
SFT runs.

### A2 — multi-seed SFT on the same Stage 2 pretrain

Cheapest step in the audit: re-ran SFT on the **same** Stage 2 pretrain
checkpoint with seeds 1 and 2.

| seed | SFT val_bpb |
|---:|---:|
| 42 (existing) | 0.6518 |
| 1 | 0.6516 |
| 2 | 0.6520 |

**Spread 0.0004.** ~30× smaller than the headline win we were
validating. SFT is highly seed-stable on this configuration. The
headline was not an SFT-seed lottery.

This narrowed the question: if there's any seed lottery in the
headline, it must live in pretrain. → A3.

### A3 — multi-seed Stage 2 pretrain

Two new pretrain seeds, recipe-pinned via `--inherit-from`.

| pretrain seed | val_bpb |
|---:|---:|
| 1 | 1.1729 |
| 2 | 1.1712 |

Inter-seed spread **0.0016**. SFT on top of seed=1 → val_bpb 0.6495 —
beat the A2 d6_stage2_s1 result (0.6516) by 0.0021. Stage 2 is
reproducible on pretrain seed.

A3 also caught the methodological gap that motivated A3-prime: A2's
tight cluster bounds the SFT-seed component of variance, A3's tight
cluster bounds the pretrain-seed component on Stage 2 — but neither
bounds the **baseline** d6 distribution, which we never measured.
Comparing Stage 2 (multi-seed, modern recipe) against baseline
(unpinned seed, stale recipe) is a confound stack, not a clean
architectural test.

### A3-prime — modern-recipe vanilla d6 baseline

The headline-overturning experiment. Re-ran vanilla d6 baseline (no
`hope_*` flags) under current `master` with seed=42 pinned. Recipe
matches Stage 2 hand-for-hand except the architecture.

**Pretrain val_bpb 1.1686** — **better than both Stage 2 seeds** by
0.0026-0.0043, and 0.0054 better than the historical 1.174. SFT on
top → **val_bpb 0.6483** — **better than Stage 2 SFT** (A3 seed=1:
0.6495) by 0.0012, and 0.0156 better than the historical 0.6639.

Side-by-side trajectory comparison (A3 seed=1 SFT vs A3' baseline SFT):
A3' starts ahead from initialization, the gap closes by ~step 100,
then converges back to ~0.001 ahead by step 375. Trajectories
interleave throughout — no point at which Stage 2 is cleanly ahead.

The 0.0121 SFT gap that originally read as "Stage 2 wins by 1.8%" was
overwhelmingly the 0.0156 recipe-drift component. Strip the confound
and the architectural delta is **+0.0012 in the wrong direction** —
Stage 2 loses by ~0.2%, well within the seed-noise spread we measured
on A3.

## What we learned

### 1. Probe-first was load-bearing

Stage 1.5b would have been invisible without the MQAR probe. A
2× synthetic-recall regression doesn't show up in d6 LM val_bpb
(the difference between Stage 1's 1.179 and baseline's 1.174 is
+0.4%, well below seed-noise). We could have spent another 6h of
pretrain budget chasing the gap before noticing the underlying
init bug.

The discipline that paid off here: **don't measure architecture-class
hypotheses with bulk LM bpb when a targeted synthetic probe is
available.** Probe → debug at probe scale → only then commit pretrain
budget. ADR-001 codified this pattern.

### 2. Recipe drift is silent and large

The single biggest finding of the closeout is operational, not
architectural. ~0.0156 of "free" SFT improvement accumulated across
4 commits between when the historical d6 baseline was trained and
when Stage 2 was trained. The most concrete component was
`840d3db` (move `ve_gate` from Muon → AdamW), but smaller drift in
optimizer/init/dataloader internals contributed too.

That 0.0156 dwarfs typical d6 architectural deltas (0.001-0.005).
Comparing a new architecture against an archived baseline number is
a confound stack, full stop. Future architecture-class experiments
on this codebase must:

- Pin recipe (current `master` commit hash).
- Pin seed (`--seed=N`, captured in meta).
- Re-baseline against current code on every comparison.
- Not retroactively compare against numbers in old writeups.

The `--inherit-from` mechanism (A1) is the technical enforcement of
the first point. The discipline of "re-baseline before claiming a
win" is the missing operational rule, and now lives here.

### 3. Codex as challenge-function delivered concrete unblockings

Three concrete examples where Codex's input changed the experiment
materially:
- ADR-001: pushed for the MQAR probe before Stage 2 budget commit.
  Without it, the W_o init bug stays invisible.
- ADR-002: pushed against `eta_init_bias = 0` (would have recreated
  a softer version of the gradient-gate problem). Stage 2 init
  defaults were chosen by his analysis, not by us.
- A3-prime: pushed for the modern-recipe baseline as the cheap
  compromise on bounding baseline-seed variance. Without it, the
  closeout would have shipped with the +1.8% headline intact.

Pattern: he reads as a *challenge* function, not an oracle (cf.
project memory `feedback_codex_collaborator.md`). Trust the
methodological pushes; verify the technical claims.

### 4. The single-depth-dial discipline made the experimental unit cohesive

nanochat's organizing constraint — every hyperparameter derived from
`--depth` — meant that any architectural change had to be principled
across depths, not just tuned at d6. We never accumulated d6-only
hacks. The cost is that some experiments (Stage 4 multi-block memory)
require careful per-depth tuning we didn't do; the benefit is that
the result transfers cleanly: "Stage 2 doesn't win at d6/5000-iter on
ClimbMix" is the actual statement, with no asterisks about which
hyperparameters were swept to make it work.

## What we didn't do

- **Stage 4 (multi-block memory).** Was conditional on Stage 2
  surviving. Condition not met. Defer.
- **Stage 5 (Hope-Attention, full CMS chain replacing MLP).** Was the
  natural next architectural block, contingent on Stage 2 producing a
  signal worth scaling. Same gate not met.
- **n=3 baseline pretrain (~13h).** Codex's full-bound proposal for
  baseline-seed variance. A3' is the cheap compromise (n=1 baseline
  + recipe parity). Result: "Stage 2 doesn't win" with high confidence;
  "Stage 2 loses" with lower confidence — the 0.0012-0.0026 magnitudes
  by which baseline beat Stage 2 could plausibly be on the lucky side
  of an unmeasured baseline-seed distribution.
- **Larger scale (d12+).** Outside M2 budget. The paper's results are
  at 760M / 30B (~600× our scale). Memory architectures may show
  structural value at scales we cannot test.
- **Long-context-rewarding evaluations.** Our SFT mix is MMLU/GSM8K-
  heavy, where long-context memory is not the bottleneck. A different
  evaluation could in principle reveal architectural benefit this
  one doesn't.
- **Forward-pass fuse opportunity in `LearnedGateLinearMemory`.**
  Performance only, not result-changing. Punted.

## Honest verdict

At **d6/5000-iter on ClimbMix with a MMLU/GSM8K SFT mix**, Hope/NL
Stage 2 (additive learned-gate memory at one block, current paper-
prior defaults) is **neutral, not net-positive**, on natural-language
val_bpb. The MQAR probe shows the mechanism works on synthetic recall;
the corpus + horizon don't reward it on natural text.

Cannot conclude:
- Stage 2 is bad (within seed-noise; "doesn't win" not "loses").
- Memory architectures are bad in general (single architecture, single
  scale).
- Memory wouldn't help at d12+, on long-context tasks, or with a
  different SFT mix.

Can conclude:
- The original "+1.8% SFT win" claim was overwhelmingly recipe drift.
- Probe-first methodology caught a load-bearing init bug that would
  have been invisible to bulk LM bpb.
- The track produced a working memory module, a working probe, and
  reusable recipe-parity infrastructure (`--inherit-from`).
- Recipe drift is a real, ~0.015-bpb-per-month silent contributor on
  this codebase. Future architecture experiments must re-baseline.

## Future work

If revived, in increasing order of cost:

- **Q1 length-stratified val_bpb** (no labels needed, ~10 lines of
  eval-loop code) — cheap probe of the memory hypothesis on long
  documents using current corpus. Could revive Stage 2 if the gain
  concentrates on long-context examples. See
  `docs/project_notes/data_investigations.md::Q1`.
- **F: CMS-Independent ablation** (Eq 74 head-wise CMS, ~3h pretrain).
  Cheap signal on whether multi-frequency memory at d6 produces any
  effect. NOT comparable to paper Table 6 numbers (those are at 760M).
- **n=2 baseline pretrain seed** (~5h). Bracket A3'. Promotes the
  "Stage 2 doesn't win" verdict to bounds on whether it loses.
- **D: §7.3 retrofit experiment** (~3-6h). Continue-pretrain Stage 2
  checkpoints at different CMS chunk schedules. Tests whether the
  retrofit mechanism even fires at d6.
- **Adopt PR #544** (dataloader remainder reuse) per ADR-003, with a
  clean `d6_v2` re-baseline. ~1.28× speedup at T=512 unblocks more
  iterations per wall-hour for any future runs.
- **Q2 from `data_investigations.md`** — adopt
  `ddudek/nanochat-climbmix-annotated` for labeled corpus; unlocks
  per-domain val_bpb (Q1 properly, Q4 per-domain CORE).
- **B/E: Stage 4 / Stage 5 (Hope-Attention)**. Defer indefinitely
  unless an architectural revival path appears or scale changes
  materially.

## Artifact inventory

### Per-stage writeups (chronological)
- `docs/hope_nl_stage0_2026-05-01.md` — `memory_state` plumbing
- `docs/hope_nl_stage1_2026-05-01.md` — swap topology, single block
- `docs/hope_nl_stage1_full_pretrain_2026-05-01.md` — Stage 1 d6 result (val_bpb 1.179)
- `docs/hope_nl_stage1_5_probe_design_2026-05-01.md` — MQAR probe design
- `docs/hope_nl_stage1_5_results_2026-05-01.md` — swap probe, ~step 151 saturation
- `docs/hope_nl_stage1_5_additive_2026-05-01.md` — additive probe, ~step 151
- `docs/hope_nl_stage1_5b_w_o_init_2026-05-01.md` — root cause: `W_o = 0` init
- `docs/hope_nl_stage2_2026-05-02.md` — learned α/η gates, full pretrain + SFT
- `docs/hope_nl_stage2_chatcore_2026-05-03.md` — ChatCORE 0.1744 (mostly SpellingBee)
- `docs/hope_nl_stage2_seed_variance_2026-05-04.md` — A2 SFT-seed variance (0.0004)
- `docs/hope_nl_a3_a3prime_2026-05-05.md` — A3 + A3-prime combined result
- `docs/hope_nl_phase_audit_2026-05-03.md` — cross-stage planning + Codex sync
- `docs/hope_nl_track_synthesis_2026-05-05.md` — this document

### Project notes
- `docs/project_notes/decisions.md` — ADR-001 (MQAR probe), ADR-002
  (Stage 2 priors), ADR-003 (defer PR #544)
- `docs/project_notes/data_investigations.md` — Q1-Q4 backlog
- `docs/project_notes/key_facts.md` — d6/M2 training recipes (canonical)
- `docs/project_notes/bugs.md` — known issues + fixes from the track

### Checkpoints (`~/.cache/nanochat/`)
- `base_checkpoints/d6_stage1/model_005000.pt` — Stage 1 swap (val_bpb 1.179)
- `base_checkpoints/d6_stage2/model_005000.pt` — Stage 2 reference (val_bpb 1.1743)
- `base_checkpoints/d6_stage2_pretrain_s1/model_005000.pt` — A3 seed=1 (1.1729)
- `base_checkpoints/d6_stage2_pretrain_s2/model_005000.pt` — A3 seed=2 (1.1712)
- `base_checkpoints/d6_baseline_modern/model_005000.pt` — A3' baseline (1.1686)
- `chatsft_checkpoints/d6_stage2/model_000375.pt` — Stage 2 SFT (0.6518)
- `chatsft_checkpoints/d6_stage2_s{1,2}/model_000375.pt` — A2 SFT seeds (0.6516, 0.6520)
- `chatsft_checkpoints/d6_stage2_pretrain_s1_sft/model_000375.pt` — A3 seed=1 SFT (0.6495)
- `chatsft_checkpoints/d6_baseline_modern_sft/model_000375.pt` — A3' baseline SFT (0.6483)

### Code
- `nanochat/gpt.py` — `LinearAttentionMemory`, `LearnedGateLinearMemory`,
  `hope_*` config fields
- `nanochat/common.py::load_inherit_config()` — recipe parity helper
- `dev/probe_mqar.py` — MQAR synthetic probe
- `scripts/base_train.py`, `scripts/chat_sft.py` — `--inherit-from`,
  `--seed`, `--sft-tag`, cooperative pause hook

### Branch
- `experiment/hope-nested-learning` (off `master 0aaca56`); the track's
  full commit history is on this branch. Not merged to `master`.
