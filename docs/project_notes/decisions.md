# Decisions

## ADR-001: Insert "Stage 1.5" synthetic recall probe before Stage 2 (2026-05-01)

**Context**: Stage 1 (one block's MLP swapped for `LinearAttentionMemory`) finished its full d6 pretrain at val_bpb 1.179 vs baseline 1.174 — within ~1% of baseline at iso-token. SFT on top landed at 0.6712 vs `d6_b_iso` 0.6639, also within ~1%. The natural next step from the trx4mr ticket is **Stage 2: per-token learned `α` and `η`**, but we have no way to tell whether the Stage 1 memory block is *actually doing something memory-flavored* — val_bpb is the average over the SFT distribution and doesn't isolate positions where carrying state across the sequence matters. We could be staring at noise either way.

Codex flagged the same concern explicitly: "Full LM val_bpb is too blunt for Hope/NL behavior. Before Stage 2, add a tiny diagnostic task/probe that can tell whether memory helps at all: associative recall, delayed copy, key-value recall across 512 tokens, or a synthetic in-context binding task."

**Decision**: Insert a **Stage 1.5** between Stage 1 and Stage 2 — design and implement a synthetic **multi-query associative recall (MQAR)** probe that trains a small d6 from random init on a binding-and-lookup task, then measures recall accuracy. Run it for both the unmodified MLP architecture and the Stage 1 `LinearAttentionMemory@L3` architecture, with identical training budget. The architectural delta on this probe is the load-bearing input to whether Stage 2 is worth implementing and how to tune its α/η stability defaults.

Detailed probe design lives in `docs/hope_nl_stage1_5_probe_design_2026-05-01.md`.

**Alternatives**:
- *Multi-seed Stage 1 confirmation runs.* Tightens noise on the val_bpb gap but doesn't address the "is memory actually doing anything" question. Codex: "I would not spend M2 time on multi-seed Stage 1 confirmation unless someone wants to claim 'Stage 1 improves/regresses nanochat.'"
- *Jump straight to Stage 2.* We'd add per-token learned α/η, full pretrain, observe whatever val_bpb shift we get, and still not know if the gating is exploiting a memory signal that exists or sculpting noise. Bad experimental hygiene; bad debugging position when stability problems hit.
- *A full benchmark suite (CORE, ARC, GSM8K).* Real downstream evals are valuable but (a) noisy at d6 scale, (b) measure "is the model good at the task" rather than "does the memory mechanism contribute," and (c) require expensive eval runs. The synthetic probe is sharper for the architectural question.

**Consequences**:
- Adds ~3–4 h of design + implementation + run time before Stage 2 starts. Two synthetic-task training runs (~15 min each on M2 plugged in for d6 from-scratch on a small synthetic vocab) plus eval.
- Defers Stage 2 by one step but de-risks it materially. If the probe shows Stage 1 already gives meaningfully better recall than baseline, Stage 2 has a clear "make it better" target. If the probe shows no architectural delta, we know that introducing learned α/η is fighting a ceiling problem and we revisit the design before sinking 3 h into another full pretrain.
- Probe code lives in `dev/` (not core). Reusable for Stage 2+ to track whether learned gates improve recall at iso-budget.
- Sets a precedent: from now on, architectural changes get diagnosed against a targeted synthetic probe before full LM pretrain budget is spent.

**Status**: accepted 2026-05-01. Implementation pending.

## ADR-002: Move to Stage 2; design priors after Codex response (2026-05-02)

**Context**: Stage 1.5b resolved the swap arm's ~2× MQAR sample-efficiency gap to a single root cause: `W_o = 0` init creates a K/V/Q gradient gate, and `hope_memory_w_o_init_scale=1.0` closes the gap exactly. Stage 1 (swap, W_o=0) hit val_bpb 1.179 vs baseline 1.174 (+0.4%) at full d6 pretrain. The 1.5b writeup left three options on the table:

(a) Re-run Stage 1-additive (W_o=1.0) at full d6 pretrain (~3h) to see if probe-level parity generalizes to DCLM.
(b) Move to Stage 2 (per-token learned α/η) directly on the W_o=1.0 foundation, gated by MQAR.
(c) Wrap and hand off.

We solicited Codex's read. Summary of his response (`/tmp/pasteboard-2`):

- W_o=0 was the wrong default; mechanism analysis matches our finding.
- For Stage 2: keep `w_o_init_scale=1.0`; control initial perturbation via `eta`/read scale, not by zeroing W_o. **Don't init eta near zero** — recreates a softer gradient gate.
- α near long memory but not saturated — `0.995 - 0.999`; explicit logged logit bias.
- Vectorize via prefix log-products with `log(clamp(α, eps, 0.999))`.
- Log gradient norms for `W_q/W_k/W_v/W_o/W_α/W_η` on the probe — "stable" can hide dead-gradient startup.
- **Topology**: start additive, not swap — isolates whether the memory branch learns useful behavior without paying the cost of removing an MLP. Swap is a sharper compression experiment for later.
- Skip option (a). MQAR is doing its job; full DCLM pretrain waits until Stage 2 shows a probe-level win or no regression.

**Decision**: Move to Stage 2 (option b). Adopt the following design priors:

1. **`hope_memory_w_o_init_scale=1.0`** — load-bearing, inherit from 1.5b.
2. **`α_t = α_max · sigmoid(W_α x_t + b_α)`** with `α_max ≈ 0.999`; init `b_α` so initial α sits near 0.99 (long memory, not saturated). Logged.
3. **`η_t = η_max · sigmoid(W_η x_t + b_η)`** with small but live init (do NOT init at or near zero). Logged.
4. **Vectorized prefix-log-product form**: `log_α = log(clamp(α, eps, α_max))`, prefix sums, subtract to form causal decays. One causal-masked einsum. No per-token Python loop. Same O(T²) cost class as Stage 1.
5. **Probe diagnostics**: log gradient norms for `W_q/W_k/W_v/W_o/W_α/W_η`, α/η histograms, memory-read RMS, residual RMS, W_o norm. The 1.5b lesson — silent dead-gradient startup — generalizes.
6. **Topology**: start with **additive** insertion (memory as third residual stream alongside attn and mlp at one block). Swap can follow if additive shows a clean probe win.
7. **Probe is the gate**: Stage 2 must match baseline MQAR saturation (~step 76) before any pretrain. If it lags, debug before sinking budget.

**Where we hold our own view (departures or open carve-outs from Codex's input)**:

- *Topology preference (additive vs swap)*: Codex argues additive is the cleaner experimental isolation. We agree to start there, but flag that the 1.5b cross-check showed swap(W_o=1.0) and additive(W_o=1.0) both saturate at step 76 with comparable trajectories — and swap had higher acc at step 51 (0.696 vs 0.156, single seed, may be noise). Swap is not architecturally weaker on the data we have; the choice is a design preference about what we're trying to measure, not a data-driven elimination.
- *Skipping Stage 1-additive (W_o=1.0) full pretrain*: Codex's logic — "MQAR found a real bug in 6 min, save the 3h for Stage 2" — is sound budget management. We adopt it, but note that the probe and DCLM val_bpb measure different things (synthetic recall vs natural-language modeling). If Stage 2 ever shows an unexpected DCLM regression, "we don't have a W_o=1.0 Stage 1 baseline at full pretrain" will be a real gap. Acceptable trade for now; flag it if it bites.
- *α init range (0.995-0.999)*: Reasonable, but not a fixed value. Effective horizon depends on context length: at T=512, α=0.99 decays to ~0.006 by end; α=0.999 to ~0.6. For our probe (T=128), almost any α in this range carries fine; for SFT (T≤1024) and pretrain (T=2048) it matters more. Default to ~0.99 initial (per design prior #2) but treat as a knob, not a fixed.

**Status**: accepted 2026-05-02. Implementation starting.
