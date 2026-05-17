# Paper notes — δ-mem: Efficient Online Memory for LLMs (Lei et al., May 2026)

**Date filed:** 2026-05-17
**Branch:** `experiment/hope-nested-learning` (paused)
**Paper:** `~/Library/Mobile Documents/com~apple~CloudDocs/Downloads/mem-2605.12357.pdf` (10 pages, arxiv 2605.12357)
**Cited GitHub orgs:** Declare-lab, MindLab-Research (NOT verified to contain runnable code; cited only in masthead)

## What δ-mem is

A trainable memory module bolted onto a **frozen** Transformer backbone.
The backbone runs unchanged; δ-mem injects via additive low-rank
steering of attention. Three load-bearing pieces:

**1. State.** An 8×8 matrix `S` of associative memory, updated via a
gated delta-rule:
```
S_t = Diag(λ_t) S_{t-1} + Diag(β_t) (v_t^m − S_{t-1} k_t^m) k_t^m^T
```
where `λ_t, β_t = sigmoid(W·x_t + b)` are **dimension-wise** gates from
the current hidden state, and `q_t^m, k_t^m, v_t^m` are r-dim
projections of the hidden state via small W_q^m, W_k^m, W_v^m.

**2. Read-then-steer.** Memory output is `r_t = S_{t-1} q_t^m`, then
**low-rank corrections** to BOTH query and output of the frozen
attention:
```
q̃_t = q_t^0 + (α/r) W_q^Δ r_t        # corrected query
ỹ_t = a_t   + (α/r) W_o^Δ r_t        # corrected output
```
The frozen backbone runs unchanged; memory injects via additive
low-rank steering. Very LoRA-shaped at the API level.

**3. Writing granularity.** Three variants ablated:
- TSW (Token-State Write) — update state at every token
- SSW (Sequence-State Write) — update once per message segment
- MSW (Multi-State Write) — N parallel sub-states, concat their reads

**Trainable params**: W_q^m, W_k^m, W_v^m, W_q^Δ, W_o^Δ, W_β, W_λ —
all small rank-r. State is 8×8. **Backbone is frozen.** Trained with
standard SFT loss.

## Why this matters to us specifically

This is essentially the architecture we sketched on 2026-05-13 and
parked. The strategic-pivot doc (`docs/strategic_pivot_2026-05-13.md`)
explicitly named this direction: "continuous-learning components
grafted onto a frozen 4–8B base." We discussed the LoRA-like-but-with-
state framing. Then we decided we didn't have the capacity to pursue
it on M2 and pivoted to using bonsai-as-tool instead.

**δ-mem is exactly that architecture, on the exact backbone we've now
canonically benchmarked.** Specifically:

- **Backbone: Qwen3-4B-Instruct** — same model family as our
  Bonsai-4B (int2 ternary distillate). We've measured fp-Qwen3-4B-8bit
  ChatCORE at 0.7656 (`docs/qwen3_4b_quantization_cost_2026-05-17.md`).
- **Mechanism: gated delta-rule with dimension-wise retention/write
  gates.** Structurally near-identical to our paused Hope/NL Stage 2,
  which was "per-token learned α/η, vectorized via prefix log-products."
  Same gated delta-rule family. We built this from scratch; they took
  the same equation as a graft module.
- **State size 8×8.** Truly tiny. Designed for exactly the memory-
  constrained regime we're stuck in.
- **Trainable subgraph is small** (rank-r adapters + 8×8 state).
  Backward only flows through that subgraph. Backbone frozen means
  no full-finetune memory cost.

## Reproduction feasibility on M2

**Likely feasible for the FP backbone.** Forward through frozen
Qwen3-4B-bf16/fp16 fits comfortably on M2 (~8 GB weights, ~12 GB
peak with KV cache). Backward only through rank-r adapters adds
<1 GB. Adam optimizer state only for the adapter params, not the
backbone. Should fit in the 24 GB unified memory budget without
gradient checkpointing.

**Probably NOT feasible for the int2 Bonsai backbone, at least not
naively.** MLX's int2 quantized weights aren't directly backprop-
through in standard tooling. The δ-mem math would still work
(low-rank corrections added to q and o, no gradient needed through
the frozen weights), but practical issues with how MLX exposes
gradients to user code on top of a quantized backbone are unverified.
Worth a probe before committing.

**What we'd need to actually reproduce:**

1. **Verify the cited code repos exist and contain runnable code.**
   The paper masthead cites "Github: Declare-lab & MindLab-Research"
   but this is a citation, not a verified-public repo. Need to check.
2. **PyTorch + MPS path for the backbone** — already verified the
   model loads (or will be, when we re-run the smoke that crashed
   2026-05-16). δ-mem needs PyTorch for the rich gradient tooling
   they presumably use.
3. **Their training data mixture** — LongSFT-3 family, exact mixture
   unclear from the abstract-level read; appendix would need a proper
   readthrough.
4. **Their evals** — LoCoMo, MemoryAgentBench, HotpotQA, plus general
   IFEval/GPQA-Diamond. Different from our six-task ChatCORE.
   Their numbers can't be directly cross-compared to our bonsai-eval
   table; we'd be measuring a different thing.

## Honest reservations

- **Numbers are modest.** δ-mem TSW lifts Qwen3-4B-Instruct from
  46.79 → 51.66 average across their suite. +4.87 absolute. The
  MemoryAgentBench +9.31 and LoCoMo +6.13 are larger but on memory-
  specific tasks. **Not earth-shattering capability lift; modest
  improvement on memory-heavy workloads.**
- **General capability essentially preserved, not improved.** IFEval
  and GPQA-Diamond are roughly flat — the module doesn't *hurt*, but
  doesn't help on non-memory tasks either. So the value proposition
  is specifically "long-history conversational and agent tasks," not
  "make the model smarter generally."
- **Conversational/agent benchmarks are different from ours.** Their
  paper validates on retrieval-and-history scenarios. Our ChatCORE
  is single-turn reasoning + knowledge. δ-mem might show essentially
  zero on our suite even if it works as advertised on theirs.
- **The 8×8 state is small but the *adapters* aren't free.** rank-r
  adapters at every layer, plus W_q^Δ and W_o^Δ at every attention
  block — paper mentions rank-8 configurations, so total trainable
  params probably in the low millions but not negligible at training-
  data scale.
- **The "Github: Declare-lab & MindLab-Research" line could be a
  fishing-for-citation that doesn't actually contain working code.**
  Common pattern in recent papers — code-coming-soon or partial-
  release. Needs verification before any reproduction work starts.

## What this changes for us

1. **The parked Hope/NL graft direction now has a published reference
   architecture.** The strategic-pivot doc said this thread was "not
   killed, just parked." It's now also "not original" if we ever do
   pursue it — we'd be reproducing or building on δ-mem, not
   inventing.
2. **The fp-Qwen3-4B-8bit baseline we just measured is the right
   reference point** for any δ-mem reproduction. Their paper uses
   Qwen3-4B-Instruct; our ChatCORE on the same architecture is
   0.7656. If we reproduce δ-mem with comparable trainable budget,
   we can ask "does the graft improve our ChatCORE, leave it flat,
   or hurt it." The δ-mem paper doesn't run our six tasks, so we'd
   be adding net signal not just replicating.
3. **The "graft onto bonsai" question is now empirically harder.**
   Our quantization-cost finding (2026-05-17) shows int2 Bonsai
   already preserves reasoning capability essentially for free vs
   fp 8bit. So a δ-mem graft on int2 Bonsai would need to lift
   *memory/knowledge* specifically — which is the area where int2
   already loses 7-25pp. δ-mem-on-int2 may be fighting the wrong
   battle; δ-mem-on-fp-4B is the cleaner experiment.
4. **No immediate-action change.** Branch still paused. This is
   reference material for whenever a "actually try graft architectures"
   thread re-opens. The bonsai-eval arc gave us the measurement
   infrastructure (`chat_eval_mlx.py`) that any δ-mem reproduction
   would also use to score itself.

## Pointers

- **Paper PDF:** `~/Library/Mobile Documents/com~apple~CloudDocs/Downloads/mem-2605.12357.pdf`
- **Strategic pivot context:** `docs/strategic_pivot_2026-05-13.md`
- **Hope/NL Stage 2 (the closest from-scratch precedent):** `docs/hope_nl_stage2_*.md`
- **Backbone benchmark:** `docs/qwen3_4b_quantization_cost_2026-05-17.md`
- **Cross-arch ChatCORE:** `HANDOFF.md` Day 2026-05-17
- **Backlog entry:** `docs/project_notes/backlog.md` (added 2026-05-17)
