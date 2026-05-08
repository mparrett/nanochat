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

## ADR-003: Defer upstream PR #544 (dataloader remainder reuse) until after A3/A3-prime (2026-05-04)

**Context**: [karpathy/nanochat#544](https://github.com/karpathy/nanochat/pull/544) is an open PR that reuses cropped document tails (with prepended BOS) instead of discarding them in `nanochat/dataloader.py`. Reduces effective crop waste from ~35% → ~23%. Claimed wall-clock speedup on the d24 record run: 1.18× at T=2048; **1.28× at T=512** (our pretrain seq length, so the biggest gain bucket applies to us). 20-line change with a simulation test.

**Decision**: Do not adopt during the in-flight Hope/NL experiment. Re-evaluate after A3 + A3-prime wrap.

**Why**:
1. **Parity**: A3 is a seed-variance comparison against the stage2 baseline (trained on the old dataloader). Adopting mid-run means seed=1 (currently running) and seed=2 see different token streams from each other AND from stage2, destroying the measurement.
2. **Upstream not validated**: PR is OPEN, not merged. No reason to take canary risk while a real experiment is running.
3. **Baseline drift**: Even after A3 wraps, switching changes the effective per-step token mix. Future val_bpb numbers cannot be directly compared against the historical d6 baseline (1.174) without an asterisk — model is trained on a richer effective corpus per token-budget.

**How to apply**: After A3-prime, if upstream merges OR we independently validate, adopt with a clean re-baseline run at the new dataloader (call it `d6_v2`). Do not retroactively compare cross-dataloader numbers.

**Status**: accepted 2026-05-04. Parked.

## ADR-004: Z-loss as standby stability lever (2026-05-05)

**Context**: CS336 Lecture 3 (Tatsu, 2026) catalogues z-loss — a `(log Z)²`
regularizer on the softmax normalizer — as one of three lifesaving tricks
against mid-training loss spikes. DCLM and Olmo both ship it; the lecture
frames it as "more universal" than logit softcap, which it labels Gemma-only
and perf-costing. nanochat currently has logit softcap (`softcap=15`,
`gpt.py:741-745`) and no z-loss. At d6/5000 iters on ClimbMix we have not
seen a spike that softcap couldn't suppress, so the lever is unused; the
question is whether to leave it that way.

trx4mr filed the same audit against picoGPT and adopted the standby stance
(ADR-019) for the 1-bit/ternary STE regime where spikes are most likely.
Cross-project alignment ticket: `docs/project_incoming/feat_modernization_alignment.md::A1`.

**Decision**: Don't add z-loss now. Note that it exists, and reach for it
*first* if a deeper-depth run (d8+ per `feat_d8_extension.md`), longer
training horizon, or quantization experiment starts spiking — before
chasing LR / clip-grad / init-scale. Keep the bar low: one diagnostic run
with z-loss before assuming the spike is a deeper problem.

**Why**: Avoids accumulating unneeded knobs in stable regimes (consistent
with nanochat's "strong baseline, not configurable framework" stance per
CLAUDE.md), but keeps the lever discoverable for the regime where it
actually helps. Mirrors trx4mr ADR-019 so the rationale is symmetric across
the two projects sharing this audit.

**How to apply**: When invoked, add as `(log Z)²` term on the cross-entropy
softmax normalizer with a small coefficient (DCLM uses 1e-4); ~5 lines.
Pair with a control run at iso-config to confirm no regression on stable
training. If adopted, becomes the new default — do not add a feature flag
(per modernization-alignment ticket "what's not in scope").

**Status**: accepted 2026-05-05. Parked as standby; no code change.

## ADR-005: Adopt PR #544 (dataloader remainder reuse) as env-gated opt-in (2026-05-06)

**Context**: ADR-003 deferred [karpathy/nanochat#544](https://github.com/karpathy/nanochat/pull/544) until after A3/A3-prime. A3-prime is wrapped, so the deferral is lifted. The PR claims ~1.28× speedup at T=512 by recycling cropped document tails (with prepended BOS) into the doc buffer instead of discarding them. The PR is still **OPEN** upstream as of 2026-05-06.

We did *not* want to either (a) adopt-as-permanent-default, which forces a one-time `d6_baseline_modern` rebaseline (~4 h M2) to keep historical comparisons coherent; or (b) ignore the capability, which leaves a known speedup unavailable in source-data-bound regimes (8×H100 speedrun, multi-epoch on small corpora).

**Decision**: Adopt as a capability behind an env-var gate, not as a default.

```
NANOCHAT_DATALOADER_REUSE_REMAINDER=1   # opt in
(unset / 0)                              # baseline behaviour, default
```

`user_config["dataloader_variant"]` is captured in `meta_*.json` ("discard" | "remainder_reuse") so any future cross-checkpoint comparison can audit which packing produced which checkpoint. SFT/RL are unaffected — they use a separate bestfit-pad loader at `scripts/chat_sft.py:221`. Wiring landed in `2173dab` (capability) + `2c2c54e` (banner unpack fix); ~21 lines total. Tests from the upstream PR (315-line simulator) were not adopted — capability only.

**Why** (empirical, not just upstream's claim — A/B at d3_tiny scale, T=256, 500 iters; n=2 seeds, 42 + 2):

| metric | discard (baseline) | remainder_reuse | Δ (mean) | per-seed Δ |
|---|---:|---:|---:|---:|
| **val_bpb (final, step 500)** | 1.6530 | 1.6619 | **+0.0089 (worse)** | +0.0050, +0.0128 |
| total_training_time | 382.9 s | 392.7 s | **+2.6 % slower** | +4.2 %, +0.9 % |
| train/loss (final) | 5.408 | 5.456 | +0.048 worse | +0.036, +0.060 |
| source-doc consumption (final `rg`) | 30 | 14 | **−53 % source tokens read** | identical both seeds |

val_bpb is the load-bearing metric (canonical for cross-checkpoint comparison
in this repo); the +0.005 to +0.013 regression is small but consistent across
seeds, not noise. Wall-time penalty looks ~+2 %, mostly Python overhead per
crop event. The −53 % source-token reduction is rock-solid.

The PR's gain is real but **regime-dependent**. At small-d / abundant-source
/ compute-bound (M2 + ClimbMix-400B) the gate is a small lose-lose: a few %
slower per step plus a small val_bpb regression, with no offsetting wall-time
return because we are not source-token-bound. At source-data-bound regimes —
`runs/speedrun.sh` on 8×H100 where I/O matters; multi-epoch on a small corpus
where fewer source-tokens-per-train-token means *more* unique data is
reachable per epoch — the same mechanism is a win.

Mechanism for the val_bpb gap: A/B's batches contain more partial documents
(recycled remainders), so within a fixed step budget the model sees the same
source content fragmented across more batches and slightly less unique
*ordering* of context. At small scale this is a marginal hit on
generalization. The gap should close as scale grows (more iters per source
doc, less sensitivity to per-batch composition); we have not measured that.

**How to apply**:

1. **Default OFF** for all M2 development runs. ADR-003's parity concern stays: any run intended to be cross-comparable with `d6_baseline_modern` (val_bpb 1.174) must use `discard`.
2. **Turn ON** when the run is source-data-bound:
   - `speedrun.sh` on 8×H100 (I/O bound at the ClimbMix scan rate)
   - Multi-epoch experiments where corpus exhaustion is in play
   - Any run where the meta records `dataloader_variant: "remainder_reuse"` for matching, e.g. an A/B against another `remainder_reuse` run
3. **Never mix** `discard` and `remainder_reuse` checkpoints in a comparison without flagging it. The meta captures the variant for exactly this audit.
4. **Removability**: gate is one commit to remove if upstream merges and we adopt as default (which would require a one-time `d6_baseline_modern_v2` rebaseline per ADR-003) or if we drop the capability.

**Where we depart from the upstream PR's framing**: the PR notes "The optimization is always-on with no configuration needed." For this codebase we explicitly disagree — abundant-source compute-bound regimes (which is most M2 work in this repo) net-lose from the change. Always-on would either silently regress those runs or force a global rebaseline. The env gate is a load-bearing departure from upstream until/unless the regime-dependence is resolved.

**Status**: accepted 2026-05-06. Capability committed in 2173dab + 2c2c54e. Default behaviour unchanged.

## ADR-006: Phase 2 quant integration — tentative working choices (2026-05-08)

**Context**: Phase 1 of the 1-bit-from-scratch direction landed (`nanochat/quant.py` — `BinaryLinear`/`TernaryLinear` + `apply_quant`, all CPU-validated). Phase 2 wires `apply_quant` into `scripts/base_train.py` for an actual short pretrain. Several integration knobs need a default before launch; none of them have a load-bearing rationale yet, but all need *some* answer to run.

**Decision**: Adopt the following **tentative** working defaults for Phase 2 short-validation runs. Each is "good enough to learn from now" and explicitly revisable once we have empirical data from the first short run.

1. **CLI surface in `scripts/base_train.py`**: add `--quant {none,binary,ternary}` (default `none` = bit-identical to current baseline) and `--quant-group-size 128` (matches Bonsai brief and trx4mr precedent). `apply_quant` runs after `init_weights()` and before `setup_optimizer()`.

2. **Optimizer routing for quantized latents**: keep nanochat's existing `setup_optimizer` shape — quantized latent weights go to **Muon** by virtue of being 2D matrix params, the same as the fp baseline. trx4mr/blabberverse precedent uses plain AdamW, but Muon-on-binary-latent is mathematically defensible (gradient flows through STE unchanged; Muon orthogonalizes the latent gradient before applying it). If the short run shows Muon-on-binary-latent diverges or stalls, this is the first knob to flip — switch matrix params to AdamW for binary runs.

3. **Phase 2 escape hatches**: keep `lm_head`, `transformer.wte`, `value_embeds`, `smear_gate`, `ve_gate` in fp. Bonsai's "no escape hatches anywhere" claim is interesting but not load-bearing for the d6 trunk-binary smoke test. Lighting up embeddings/lm_head is a separate axis we can light on a follow-up if and only if trunk-binary lands clean.

4. **Recipe**: mirror `d6_baseline_modern` for apples-to-apples comparison. Differences: `--model-tag=d6_binary_validate_short` (or d3 equivalent), `--num-iterations` reduced for short validation, `--save-every` matches `--num-iterations` so we keep only the final checkpoint, `--save-keep-last-n=2` for disk discipline. bf16 (already validated as parity-safe at d6 in commit `129c219`).

**Why tentative not load-bearing**: each of these is a "value for current phase, flexible later" choice. The *interesting* questions about quant integration (Muon vs AdamW for binary, escape hatches yes/no, group_size sweep) are research questions the short-validation runs will inform. Picking defaults too early — before any pretrain wall-clock data — would lock in answers we don't have.

**Revisit when**: the first short binary run (d3 1500-iter or d6 1000-iter) produces a val_bpb. If it tracks fp32 baseline, defaults stand. If not, the diagnostic table tells us which knob (optimizer, group_size, escape hatches, init scale) to flip first.

**Status**: tentative 2026-05-08. Phase 1 infra committed in 2b85b82 / 1547303 / fbf29be. Phase 2 wiring + short run pending operator green-light on depth choice.

## ADR-007: STE binary training does NOT save memory — strategic reframing (2026-05-08)

**Context**: Mid-launch of the d3 binary 1500-iter validation run (Phase 2 of the 1-bit-from-scratch direction), the operator asked the obvious-in-retrospect question: "what's the compute or memory penalty for binary training?". Working through the answer surfaced that the backlog entry's strategic framing — "14× memory reduction means d12+ might fit on M2" — quietly conflates two different memory regimes.

The STE port we have (`nanochat/quant.py`, `BinaryLinear`/`TernaryLinear`) keeps fp32 latent weights as the only `nn.Parameter` on each module. The optimizer (Muon for matrices) operates on those fp32 latents directly. Quantization happens in the forward — `weight → quantize() → cast(x.dtype) → F.linear` — and the quantized tensor is a transient that gets freed after the matmul. Optimizer state (Muon momentum) is also fp32, sized to the latent.

So during training, memory consumption is:
- fp32 latent `weight`: same as fp32 baseline
- fp32 transient quantized tensor during forward: ≥ 0 extra
- fp32 optimizer state: same as fp32 baseline
- activations / gradients: same as fp32 baseline

**Net training memory: ≥ fp32 training memory.** The ~14× win in the backlog framing is realised only at *inference time*, by serializing the weight as packed sign bits + per-group fp16 scales (the BitNet/Bonsai inference scheme). We don't have a "pack to 1-bit" inference path yet; building one is its own piece of infra.

**Decision**: Make the training-vs-inference distinction explicit and adopt the following framing.

1. **Phase 2's load-bearing question stays the same**: "Can STE binary training produce a d3/d6 model within shouting distance of fp32 val_bpb?" The compute cost (~same as fp), memory cost (~same as fp), and quality outcome are all that matters here. Result: still TBD pending the d3_binary_validate run.

2. **The strategic prize moves**: from "binary unlocks d12+ training on M2" (which it does NOT) to "binary unlocks d12+ *inference* on M2 once we add a packed-weight inference path" (which it could). This is closer in spirit to the C arm of `lora_proposal_2026-05-06.md` (fp LoRA on a frozen 1-bit base) than to the from-scratch training direction originally framed.

3. **What we don't have but might want**:
   - A packed-weight inference path: serialize `BinaryLinear.weight` as `(bool sign_bits, fp16 group_scales)` — saves ~14× on disk and at inference; doesn't help training. Not Phase 2 work.
   - Bonsai's native-1-bit training method (whitepaper-public, code-proprietary): *might* train with packed weights, in which case d12+ training on M2 becomes back on the table. Backlog has a separate "reverse-engineer Bonsai" entry for this.
   - A "binary embeds + lm_head" extension to `apply_quant`: Phase 2 deliberately leaves these in fp; turning them on *increases* training memory (more fp32 latents) but reduces inference-pack size further. Phase 3 axis.

4. **Empirical compute cost**: **+32.5 %** integrated wall penalty for d3 binary vs fp32 baseline (whole-run total_training_time, 2613.74 s vs 1973.16 s). An earlier mid-run reading at steps ~280-300 read +16 %; the integrated number includes the warmup, compile, and steady-state phases. Either way, within the "essentially the same compute" envelope vs the fp32 alternative — but the gap is wider than the steady-state sample suggested. Detailed breakdown in `docs/quant_d3_validate_2026-05-08.md`.

**Why this didn't surface earlier**: the backlog entry was written before any binary code existed in this repo; the trx4mr precedent (`picoGPT/binary.py`, `blabberverse/phase7_arch.py`) is small-scale and the memory accounting was never the load-bearing question there. The Phase 1 smoke ran on CPU where memory wasn't tight. Phase 2's launch-prep is the first time real-on-M2 memory pressure becomes a variable to think about, and the operator's question was the prompt.

**What this changes for the active d3 run**: nothing. The val_bpb result is independent of memory framing. We're still going to learn whether STE training produces a sensible model.

**What this changes for next steps**:
- Don't promise "d12+ on M2 via this code path" without qualifying training vs inference.
- Phase 3 of the original plan (d12+ on M2 *if* d6 works) needs to be split into "d12+ training on M2" (still gated on memory ceiling, no win from binary STE) and "d12+ inference on M2 from a packed checkpoint" (live as soon as someone produces a d12 binary checkpoint — own training, Bonsai's, etc.).

**Status**: accepted 2026-05-08. Captured in backlog.md's 1-bit-from-scratch entry as a "Memory accounting clarification" subsection cross-referencing this ADR.

