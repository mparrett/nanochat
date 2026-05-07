# Handoff: Hope / Nested Learning experiment fork

**Date opened:** 2026-04-29
**Branch:** `experiment/hope-nested-learning` (off master `0aaca56`)
**Upstream:** `karpathy/nanochat` — keep `master` clean for sync.

## Why this fork exists

Investigation ticket from the trx4mr project. Goal: prototype Hope /
Nested Learning (Behrouz et al., NeurIPS 2025, arXiv:2512.24695) on top
of nanochat's well-tested CPU/MPS pipeline rather than rebuilding from
picoGPT.

Full ticket with rationale, staged plan, and risks:
**`~/projects-new/trx4mr/docs/idea-hope-nested-learning.md`**

Related project memory:
- `~/.claude/projects/-Users-matt-projects-new-trx4mr/memory/MEMORY.md`

## What Hope/NL changes

The conventional transformer block is stateless: `output = f(x, params)`.
Hope makes blocks carry **input-driven, per-token mutable memory**:
`output, memory_state = f(x, params, memory_state)`. The block also
returns an updated memory state. This breaks nanochat's clean forward
pass abstraction and is the core implementation challenge.

Foundational equation we'd start from (from the paper, §2):
```
M_t = α_t · M_{t-1} + v_t · k_t^T   (rank-1 fast-weight update)
y_t = M_t · q_t                      (read)
```

Read the trx4mr ticket for the staged 0–6 implementation plan.

## Current state of the fork

- Just cloned. No modifications yet.
- Branch `experiment/hope-nested-learning` created off `master` at
  `0aaca56` (latest CPU fix PR — relevant for our M2/MPS target).
- Other available branches on origin: `master`, `fp8_attempt_fail`,
  `moe`. Not relevant for our work.
- Environment not yet set up. `runs/runcpu.sh` is the CPU/MPS reference
  script; it's tuned for ~30 min pretraining + ~10 min SFT on M3 Max.

## Recommended next steps

1. **Get the rest of the paper.** The local PDF
   (`~/Library/Mobile Documents/com~apple~CloudDocs/Downloads/Hope-titans-2512.24695.pdf`)
   is only 13 pages — the foundational/theoretical portion. The Hope-
   specific equations, CMS pseudocode, M3 optimizer, and chunk-parallel
   form live in §4–§9, which aren't in that file. Try arXiv 2512.24695
   in case the full paper is there, or NeurIPS 2025 proceedings, or any
   Google Research code release.

2. **Run the CPU baseline as-is.** Confirm `runs/runcpu.sh` works
   end-to-end on this M2 (24GB RAM). This validates our environment
   before any modifications. Expected: ~30 min pretraining of a
   depth=6, head_dim=64 model on tokenized DCLM data. No code changes.

3. **Read `nanochat/gpt.py`** (~26KB). The architecture is more modern
   than picoGPT — likely RoPE, RMSNorm, SwiGLU. We need to understand:
   - The block structure (where to insert / replace memory)
   - Whether the forward pass already accepts state-like args (KV
     cache lives in `nanochat/engine.py`)
   - How `--depth` propagates to width/heads/lr/etc. (single complexity
     dial — important for our scaling story)

4. **Implement Stage 0 + Stage 1** per the trx4mr ticket. Stage 0 is
   pure plumbing (return `None` for memory_state initially, no behavior
   change). Stage 1 swaps the FFN in **one** block for a fixed-η/α
   linear-attention memory (Eq 15), reset per sequence. Goal:
   "does it train, and is the loss curve reasonable vs unmodified
   baseline?"

## Important context

- **Single complexity dial.** nanochat auto-derives width/heads/lr/etc.
  from `--depth`. Our changes must respect this — don't hardcode
  dimensions; read them from config.
- **CPU/MPS only here.** No CUDA expected. Stick with `uv sync --extra cpu`
  per the runcpu.sh setup.
- **Chess work stays in trx4mr.** Don't migrate KPK experiments into
  this fork. picoGPT remains the right tool for those.
- **Don't push to upstream.** This is a local research fork. Treat
  origin/master as read-only — only sync from it.

## Open questions / decision points

(From the ticket — copied here for convenience. Update as resolved.)

- **Test domain**: nanochat's pretraining data (DCLM) gives real LM
  loss curves. We could also add a synthetic in-context recall task to
  diagnose whether the memory mechanism is doing what we think.
- **Reset cadence**: per-sequence reset is the safe Stage 1 choice.
- **Backprop boundary**: detached fast-weights (cheap) vs full
  backprop through memory updates (faithful). Start detached.
- **Baseline comparison**: same `--depth` config with vs without
  memory mod, matched on training compute.

## Files in this fork worth knowing

- `nanochat/gpt.py` — the model. Where we'll add memory blocks.
- `nanochat/engine.py` — inference w/ KV cache. Will need updating
  for memory-state passing in generation.
- `runs/runcpu.sh` — the CPU/MPS baseline command sequence.
- `scripts/base_train.py` — pretraining entry point.
- `nanochat/dataloader.py`, `nanochat/dataset.py` — tokenized DCLM data.

## Status checklist

- [x] Cloned to `~/projects-new/3p/nanochat/`
- [x] `experiment/hope-nested-learning` branch created
- [x] Environment set up (`uv sync --extra cpu`, now on torch 2.11.0)
- [x] CPU baseline run confirmed — full pretrain + SFT done end-to-end on M2
- [x] Read `nanochat/gpt.py` (and most of optim.py)
- [ ] Full paper §4–§9 obtained
- [x] Stage 0 implemented (memory_state plumbing) — commit `d2bbc0e`
- [x] Stage 1 implemented (one block linear-attention memory swap) — commit `bc54858`
- [x] Stage 1 vs baseline comparison run — val_bpb 1.179 vs 1.174 (+0.4%)
- [x] Stage 1.5 MQAR probe — sample-efficiency gap found and explained
- [x] Stage 1-additive variant — falsified the "lost MLP nonlinearity" hypothesis
- [x] Stage 1.5b/c W_o init knob + sweep — root cause + recommended default
- [x] Stage 2 implemented (per-token learned α/η, vectorized via prefix log-products) — commit `e559446`
- [x] Stage 2 MQAR probe + α_init sweep — paused for operator review
- [x] Stage 2 vs baseline DCLM pretrain — picked (c), val_bpb 1.1743 (baseline parity, beats Stage 1 swap's 1.179)
- [x] Stage 2 SFT — val_bpb **0.6518** (beats baseline 0.6639 by 1.8%, Stage 1 swap 0.6712 by 2.9%)
- [x] Investigation post-mortem captured at `docs/sft_oom_investigation_2026-05-03.md`
- [x] ChatCORE on final SFT checkpoint — **ChatCORE = 0.1744** at d6_stage2/375 (2026-05-03). 91% of metric is SpellingBee 95.31%; rest at noise floor. Full writeup at `docs/hope_nl_stage2_chatcore_2026-05-03.md`. Eval-loop fragmentation fix production-tested over ~14k MMLU batches with no OOM.
- [x] **A2 SFT-seed-variance disambiguation (2026-05-04)** — three SFT seeds on the same Stage 2 pretrain produced val_bpb 0.6518 / 0.6516 / 0.6520 (spread 0.0004, ~30× smaller than the 0.0121 headline win). SFT is seed-stable; the headline is not an SFT-seed lottery. Variance, if any, lives in pretrain → A3 justified. Writeup: `docs/hope_nl_stage2_seed_variance_2026-05-04.md`.
- [ ] **A3 multi-seed Stage 2 pretrain (queued)** — two more pretrains with --seed=1 and --seed=2 on the d6 Stage 2 config. Launch commands captured in the A2 writeup. ~6h wall total.

## Status update — 2026-04-30 (end of session)

The "set up the pipeline" Step 0 work expanded into a longer perf-investigation
detour because (a) Karpathy's recipe is tuned for M3 Max and assumes ~30 min
pretrain, (b) we hit two real bugs along the way that took priority. We're now
ready for actual Hope/NL implementation work, with a much sharper picture of the
hardware envelope than we started with.

### What got done in this session

**Phase 1 — pipeline working end-to-end** (`docs/m2_pipeline_2026-04-30.html`)
- Full `runs/runcpu.sh`-style pipeline ran on M2 24GB. Pretrain 3.6h, SFT ~1h.
- Hit and fixed two real bugs: cross-entropy NaN on fully-masked SFT batches
  (applied `karpathy/nanochat#610`) and SFT bestfit dataloader buffer lockup
  (our patch). Both documented in `docs/project_notes/bugs.md`.
- Smoke test passes — model knows Paris, Eiffel Tower, Louvre.

**Phase 2 — closing the M2 ↔ M3 Max wall-clock gap**
(`docs/phase2_perf_2026-04-30.html`)
- Net **~17% throughput improvement** (2.3 → 1.9 s/iter). Headline win was
  pinning optimizer scalars to the parameter device + replacing `add_(alpha=)`
  with inline multiply.
- Profiled with `torch.profiler` (CPU-side): optimizer is 79% of every iter.
  The Muon polar express Newton-Schulz matmul loop is the bottleneck.
- Ruled out: bigger `device_batch_size` (memory wall at 32 on 24GB),
  `PYTORCH_MPS_PREFER_METAL=1` (5× slower), `PYTORCH_MPS_FAST_MATH=1` (40%
  slower), `torch.compile` on muon kernel (CPU-only win, no GPU fusion).

**Phase 3 step 1 — torch 2.11 + bf16 audit**
- Upgraded torch 2.9.1 → 2.11.0 (`8b596d5`). Marginal baseline win (~3%).
- Patched `optim.py` to fix bf16 mixed-precision crashes on MPS (`7e21999`).
  bf16 now works end-to-end; bf16 + `device_batch_size=48` is now usable
  (was OOM at fp32). On M2 the throughput delta is small (no hardware bf16),
  but unlocks larger batch and fixes a real upstream bug.
- **Filed the bf16 fix as upstream PR `karpathy/nanochat#741`** with a
  regression test on a clean branch (`fix/mps-bf16-mixed-precision`).

**Phase 3 step 2 — GPU-side profiling** (`docs/phase3_gpu_profiling_2026-04-30.md`)
- Used `torch.mps.profiler.profile()` + `log show --signpost` to get the
  actual GPU dispatch counts. **319 dispatches per training iter.** ~250 of
  them in the optimizer. Per-dispatch wall cost ~8 ms.
- Confirmed: no Metal-level fusion; bottleneck is dispatch count not GEMM
  speed. The biggest remaining software lever is fusing the polar express
  loop into a single MSL kernel via `torch.mps.compile_shader`.

### Open follow-ups (in priority order)

1. **Codex consultation** — sent him a markdown brief covering the perf
   findings; asking for fresh-eyes suggestions especially on MSL kernel
   fusion and MLX-port tradeoffs. Reply pending.
2. **Phase B GPU profiling** — `torch.mps.profiler.metal_capture()` for
   per-kernel timing. Requires `MTL_CAPTURE_ENABLED=1` and Xcode for the
   `.gputrace` viewer. ~60–90 min of work. Would identify the single hottest
   dispatch (worth fusing first) vs the dispersed dispatches.
3. **`ns_steps=3` validation** — empirically identical loss to `ns_steps=5`
   over 9 SFT steps with ~12% optimizer speedup. Needs full pretrain A/B
   to commit. Ticket: `docs/project_incoming/feat_muon_ns_steps_validation.md`.
4. **Token-weighted loss EMA** — instrumentation idea documented in
   `docs/project_incoming/feat_sft_loss_instrumentation.md`. Cheap. Worth
   doing before any longer SFT runs to make the displayed loss honest.

### Where everything lives

- **Phase 1 writeup:** `docs/m2_pipeline_2026-04-30.html`
- **Phase 2 writeup:** `docs/phase2_perf_2026-04-30.html`
- **Phase 3 step 2 writeup:** `docs/phase3_gpu_profiling_2026-04-30.md`
- **Bug log:** `docs/project_notes/bugs.md`
- **Open tickets:** `docs/project_incoming/`
- **Profilers:** `dev/profile_pretrain_iter.py` (CPU), `dev/profile_mps_signpost.py` (GPU)
- **Upstream PR:** https://github.com/karpathy/nanochat/pull/741

### Where the hardware wall actually is

After all Phase 2/3 work, on M2 24GB with torch 2.11:
- **Per-token throughput**: ~125 μs (was ~150 μs at session start)
- **Per-iter wall**: 1.9 s steady state (was 2.5 s)
- **Full pretrain**: ~3.0 h projected (was 3.6 h)
- **Per-iter GPU dispatches**: ~319, sequential, ~8 ms each

The remaining floor is set by Apple Silicon's compiler stack — Metal has no
Triton-equivalent for fusing tight matmul loops the way CUDA does. The biggest
software lever is custom MSL kernels (weeks of work) or MLX port (1–2 weeks).

### When you actually start Hope/NL work

The Phase 1/2/3 work doesn't change the implementation plan in the original
ticket. But two things to keep in mind:
- The optimizer is the wall, not the model forward. Hope/NL adds memory state
  that flows through the forward — that adds dispatches in *forward*, which
  is currently only ~50 of the 319/iter. So the perf hit may be smaller than
  it would be on CUDA.
- We have a clean baseline checkpoint at `~/.cache/nanochat/base_checkpoints/d6/model_005000.pt`
  to compare against. Don't accidentally overwrite — use a different `--model-tag`
  for Hope/NL experiments.

## Status update — 2026-05-02 (end of session 2)

Closed Phase 3 perf detour and built Stage 0 + Stage 1 + a synthetic-probe
arc that resolved an architectural surprise into an actionable init fix.
Ready to start Stage 2.

### What got done this session

**Phase 3 step 3** (`docs/phase3_step3_grad_accum_2026-05-01.md`)
- `ve_gate` → AdamW: 319 → 176 GPU dispatches/iter (-45%), 12% iter wall (commit `840d3db`).
- SFT accum=4 A/B: B-iso (accum=4, 375 opt steps) val_bpb 0.6639 vs A 0.7568. Locked SFT recipe at accum=4. Found and fixed `chat_sft.py --num-iterations` bug (was counting micro-batches, gave ~N/accum opt steps silently). Commit `a2d56ef`.
- Found and fixed `chat_sft.py` model_config asdict bug — was hand-listing fields, missed `hope_memory_layer`. Commit `f143da2`.

**Phase 3 step 4** (`docs/phase3_step4_pretrain_accum_derisk_2026-05-01.md`)
- Pretrain accum=4 derisk: confirmed accum=4 doesn't transfer to base pretrain on M2. ~14% per-token slower because the streaming parquet+BPE dataloader serializes with accum>1. Locked pretrain at accum=1/16384. Commit `bc7f9a2`.

**Hope/NL Stage 0** (`docs/hope_nl_stage0_2026-05-01.md`)
- `memory_state` plumbing through `GPT.forward` and `Block.forward`. Conditional-tuple at GPT boundary keeps all ~11 existing call sites unchanged. `reset_memory()` returns fresh `[None]*n_layer`. 6 contract tests pin bit-identical loss/logits with vs without memory_state. Commit `d2bbc0e`.

**Hope/NL Stage 1** (`docs/hope_nl_stage1_2026-05-01.md`, `docs/hope_nl_stage1_full_pretrain_2026-05-01.md`)
- `LinearAttentionMemory` module (causal linear attention parallel form, alpha=1, eta=1, strict-causal mask). Swapped MLP at one block via new `hope_memory_layer` config field. Commit `bc54858`.
- Full d6 pretrain at L3: val_bpb **1.179** vs baseline 1.174 (+0.4% rel, within single-seed noise). 274 min wall. Commit reference in writeup.
- SFT on top with accum=4 (post-bug-fix): val_bpb 0.6712 vs `d6_b_iso` 0.6639. Chat assess via `chat_cli` and side-by-side `chat_web` confirmed the architecture works at d6 quality level.

**Hope/NL Stage 1.5 — synthetic probe** (`docs/hope_nl_stage1_5_*.md`)
- Codex steered: "val_bpb is too blunt, build a synthetic probe." ADR-001 captured the decision (`docs/project_notes/decisions.md`).
- Built `dev/probe_mqar.py` — Multi-Query Associative Recall, K=M=16, T=128. From-scratch random init, both architectures, identical training budget.
- v1 (full attention, 1000 iters): killed early — both arms saturate by step ~76 because attention solves recall on its own; the architectural swap isn't load-bearing on this task.
- v2 (200 iters, eval-every=25, convergence-speed): **swap is ~2× slower-to-grok than baseline.** Baseline saturates by step 76; swap by step 151.
- **Operator's additive-insertion idea:** keep the MLP, add memory as a third residual stream. Falsified the "lost MLP nonlinearity" hypothesis — additive was just as slow. Pointed at the real cause: `W_o = 0` init creates K/V/Q gradient-gating cold start. Commits `f990b16`, `4157e6e`.
- Stage 1.5b: added `hope_memory_w_o_init_scale` config knob. With scale=1.0 (uniform init at K/V/Q magnitude), additive memory matches baseline grokking step-for-step (99.95% at step 76). The W_o=0 init was the entire ~2× gap. Commit `3f54bc6`.
- Stage 1.5c sweep across scale ∈ {0.0, 0.1, 0.5, 1.0}: monotonic, scale=1.0 wins at 2× the second-best on time-to-saturate. Recommended default for any memory-bearing config. Commit `07e3312`.

**Tooling additions** that survived the session:
- `dev/probe_mqar.py` — reusable synthetic-recall gate for any future architectural variant. ~6 min/arm.
- `--hope-memory-layer` and `--hope-additive-memory-layer` CLI flags on `base_train.py`.
- Cooperative pause-by-touch in `base_train.py` and `chat_sft.py` (`touch /tmp/pause-nanochat`). Commit `245fd09`.
- `chat_sft.py` `mby:` instrumentation — surfaces microbatch yield count alongside opt-step count to immediately catch any future drift between the two.

### Where everything lives (added this session)

- **Phase 3 step 3 writeup:** `docs/phase3_step3_grad_accum_2026-05-01.md`
- **Phase 3 step 4 writeup:** `docs/phase3_step4_pretrain_accum_derisk_2026-05-01.md`
- **Stage 0 writeup:** `docs/hope_nl_stage0_2026-05-01.md`
- **Stage 1 writeup:** `docs/hope_nl_stage1_2026-05-01.md`
- **Stage 1 full pretrain:** `docs/hope_nl_stage1_full_pretrain_2026-05-01.md`
- **Stage 1.5 design + ADR:** `docs/hope_nl_stage1_5_probe_design_2026-05-01.md`, `docs/project_notes/decisions.md::ADR-001`
- **Stage 1.5 results (swap):** `docs/hope_nl_stage1_5_results_2026-05-01.md`
- **Stage 1.5 additive:** `docs/hope_nl_stage1_5_additive_2026-05-01.md`
- **Stage 1.5b W_o init + sweep:** `docs/hope_nl_stage1_5b_w_o_init_2026-05-01.md`
- **Field Report III narrative:** `docs/phase3_to_stage1_5_2026-05-01.html`
- **MLX evaluation (parked):** `docs/mlx_port_evaluation_2026-05-01.md`
- **Bug ticket archive:** `docs/project_archived/bug_chat_sft_num_iterations_micro_batch_semantics.md`

### Where the architecture wall is now

For Hope/NL specifically:
- **`W_o = 0` init is a known cold-start trap.** Use `hope_memory_w_o_init_scale=1.0` for any memory-bearing block. Backward-compat shim handles older checkpoints.
- **MQAR probe is the gate.** Any new architectural variant should match baseline's step ~76 saturation before getting a full d6 pretrain budget.
- **At d6 / T=512 / full attention, memory mechanisms are not load-bearing for recall.** Hope-flavored benefits (long-context, in-context recall at scale, test-time adaptation) live at scales we won't reach on M2.

### Stage 2 starting point

When you pick up Stage 2 next:

1. **Read first**:
   - `docs/hope_nl_stage1_5b_w_o_init_2026-05-01.md` (the W_o init story)
   - `docs/hope_nl_stage1_5_additive_2026-05-01.md` (the additive variant)
   - Codex's response in the conversation history that set up Stage 1.5

2. **Architectural design** (Codex's outline):
   - Per-token learned `α_t = sigmoid(W_α(x_t))` and `η_t = η_max * sigmoid(W_η(x_t))`.
   - **Vectorized via prefix log-products** to avoid token loops:
     ```
     log_prefix_alpha[t] = sum_{j<=t} log(alpha_j)
     decay(t, i)         = exp(log_prefix_alpha[t-1] - log_prefix_alpha[i])
     o_t                 = sum_{i<t} (q_t · k_i) * eta_i * decay(t, i) * v_i
     ```
     One big causal-masked einsum, same O(T²) cost class as Stage 1, no Python loops.

3. **Stability defaults** (per Codex):
   - Inherit `hope_memory_w_o_init_scale=1.0` (Stage 1.5b default — load-bearing).
   - Bound `α < 0.999` (clamp or sigmoid scaling).
   - Initialize `α` near long memory (e.g. start near 0.99), not random forgetting.
   - Cap or small-init `η`.
   - Log α histograms, η histograms, memory-read RMS, residual RMS, W_o norm.

4. **Pass/fail bar via probe before any pretrain**:
   - Run `dev/probe_mqar.py` with the Stage 2 architecture and `--hope-memory-w-o-init-scale=1.0`.
   - If saturation step is ≥ baseline (~76), Stage 2 is competitive with Stage 1 — proceed to a real pretrain.
   - If meaningfully later, debug the gates before sinking ~3h pretrain budget.

5. **Operator preferences** that came up:
   - One commit per discrete change, with multi-paragraph commit message explaining why.
   - Markdown writeup first (insurance against context exhaustion), HTML narrative second (publication-style).
   - Cooperative pause hook is wired in (`touch /tmp/pause-nanochat`); use it on long runs.

### Open questions punted for a future session

- The Stage 1-additive (W_o=1) variant has not been run at full d6 pretrain. Open whether the probe's "matches baseline" finding generalizes to LM val_bpb on DCLM.
- The always-final-layer-L constraint in `_compute_window_sizes` blocks any "make the probe attention-bottlenecked" follow-up. Lifting it is a small patch; would let us re-run MQAR with restricted attention to genuinely test whether memory blocks can do recall.
- No Hope-specific behavioral probe exists yet (in-context binding, parity, counting). MQAR was the chosen synthetic; others might surface different architectural tradeoffs.
- Codex's third Stage 2-related steer ("multi-seed Stage 1") was rejected for Stage 1; whether to seed-confirm any Stage 2 result is open.

## Status update — 2026-05-02 (end of session 3)

Stage 2 implemented end-to-end and probed. Architecture works as designed but
shows reproducible probe-level lag and high seed-variance with bimodal
optimization. Paused for operator review before deciding whether to spend ~3h
DCLM pretrain budget.

### What got done this session

**Stage 2 design + commit** (`docs/project_notes/decisions.md::ADR-002`,
`e4f14db`)
- Captured Codex's design priors response (additive topology, eta-not-near-zero,
  alpha 0.995-0.999 init, prefix-log-product, gradient-norm logging).
- Recorded our carve-outs: topology choice is design preference (data-neutral),
  skipping Stage 1-additive full pretrain is a budget call, alpha init range
  is a knob not fixed.
- New feedback memory: treat Codex as challenge function, not oracle. Critical
  read of his input before adopting verbatim. Saved by user request.

**Stage 2 module** (`nanochat/gpt.py`, `e559446`)
- `LearnedGateLinearMemory` subclasses `LinearAttentionMemory`. Adds W_alpha,
  W_eta projections and explicit b_alpha, b_eta scalar parameters (nanochat's
  custom Linear class skips bias even when bias=True, so the bias-as-Parameter
  pattern is necessary).
- Vectorized via prefix log-products: `decay(t,i) = exp(S[t-1] - S[i])` for
  `i<t`, computed via cumsum + outer subtraction + masked_fill(-inf) + exp.
  Same O(T²) cost class as Stage 1, no Python loop.
- New config kind switch: `hope_memory_kind ∈ {linear, learned_gate}` routes
  in `Block.__init__` for both swap and additive paths.
- Optimizer fix: 1D gate biases (b_alpha/b_eta) crash Muon (expects 2D), so
  `setup_optimizer` filters them out by name and routes to AdamW alongside
  the existing `ve_gate` carve-out.
- 5 new tests in `tests/test_memory_plumbing.py` (19 total, all passing).

**Probe instrumentation** (`dev/probe_mqar.py`, `c371022`)
- Forward pre-hook captures memory module input for gate-stat computation.
- Two new diagnostic lines per eval step when memory module present:
  - `grad: W_k=... W_v=... W_q=... W_o=... [W_alpha=... W_eta=... b_alpha=... b_eta=...]`
  - `gate: alpha[min/mean/max] eta[min/mean/max]`
- Generic — works for any memory variant; baseline (no memory module) emits
  no diagnostic lines.

**Stage 2 probe** (`docs/hope_nl_stage2_2026-05-02.md`, `03449b9` `a3b008c`
`016c5b4`)
- Initial 3-arm comparison (baseline, stage1add@W_o=1, stage2_default@seed=0):
  Stage 2 saturates at step 101 vs baseline 76. Marginal pass on ADR-002 bar.
  Different convergence shape — Stage 2 hit 0.92 at step 51 vs baseline 0.11.
- Confirmation seed (stage2 seed=1): step 126 saturation. Lag is real, not
  single-seed. Two seeds even adopted different gate strategies (seed=0
  drove alpha down, seed=1 kept alpha high and ran eta_max to 0.93).
- Single-axis alpha_init_bias sweep on seed=0: alpha=3.0 (init alpha~0.95)
  saturated at step 51 — beats baseline! Looked like the right new default.
- alpha=3.0 seed=1 confirmation INVERTED the conclusion: 0.57 acc at step
  200, never converged. Same config as seed=0's best-ever run. Architecture
  has bimodal optimization at d6/T=128.
- alpha-tuning amplifies the variance, doesn't fix it. Default alpha=4.595
  is the lowest-variance choice.

### Where the architecture wall is now (post-Stage-2)

- Stage 2 architecture **functions as designed**: gates alive from step 1
  (no W_o=0 cold-start trap), prefix-log-product math correct, optimizer
  routing clean, end-to-end probe + (would-be) pretrain pipeline working.
- At d6 / T=128 / MQAR / W_o=1.0 / default alpha=4.595, Stage 2 is
  **reproducibly slower than baseline** (~25-50 step lag on saturation) AND
  **seed-sensitive** (alpha=3.0: best AND worst observed).
- The bimodal basin behavior is interesting architecturally but a yellow
  flag for stability. May or may not matter at full pretrain horizon (5000
  iters, T=2048, 50× more data).

### Stage 2 state — paused for operator decision

Three options on the table, with my read documented in
`docs/hope_nl_stage2_2026-05-02.md`:

(b) **More probe tuning** — sweep `W_alpha/W_eta` init scale or `eta_init_bias`.
    ~42 min for 6 arms. Might resolve variance but the d6 sensitivity is
    itself a yellow flag.

(c) **Schedule full d6 DCLM pretrain at default alpha=4.595, seed=0.** ~3h
    wall, pause hook in place. Probe is synthetic at T=128; pretrain is
    5000 iters of natural language. Seed sensitivity at probe scale may
    wash out at pretrain horizon. **val_bpb is the only test that matters
    for the original ticket.**

(d) **Stop.** Stage 0+1+1.5+2 is a complete experimental unit. Architecture
    wall mapped. Hope/NL benefits live at scales we won't reach on M2.

My read: (c) — the original ticket asks "does this architecture work?", and
val_bpb on DCLM is the honest answer at the project's scale. Expect val_bpb
at or slightly worse than baseline (similar to Stage 1's +0.4%). Even a
"no improvement" result is informative. Operator paused to think before
committing 3h of pretrain budget.

### Files added this session

- **Stage 2 module:** `nanochat/gpt.py` (LearnedGateLinearMemory + Block dispatch)
- **Stage 2 ADR:** `docs/project_notes/decisions.md::ADR-002`
- **Stage 2 writeup:** `docs/hope_nl_stage2_2026-05-02.md`
- **Probe diagnostics:** `dev/probe_mqar.py` (grad-norm + gate stats hook)
- **Tests:** `tests/test_memory_plumbing.py` (+5 Stage 2 tests, 19 total)
- **CLI flags on `scripts/base_train.py`:** `--hope-memory-w-o-init-scale`,
  `--hope-memory-kind`, `--hope-memory-alpha-max`, `--hope-memory-eta-max`,
  `--hope-memory-alpha-init-bias`, `--hope-memory-eta-init-bias` — full
  end-to-end plumbing for a Stage 2 pretrain when greenlit.

### Quick-resume checklist for option (c)

```bash
# Pause hook armed (already in base_train.py:422):
#   touch /tmp/pause-nanochat   to pause between opt steps
#   rm /tmp/pause-nanochat      to resume

# Stage 2 pretrain command (mirrors Stage 1 recipe, with Stage 2 flags):
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=6 --device-batch-size=32 \
    --hope-additive-memory-layer=3 \
    --hope-memory-kind=learned_gate \
    --hope-memory-w-o-init-scale=1.0 \
    --run=dummy
# (note: on M2, torchrun bare-python; --num-iterations defaults to chinchilla)
```

Compare against:
- `~/.cache/nanochat/base_checkpoints/d6/model_005000.pt` — baseline (val_bpb 1.174)
- Stage 1 swap pretrain — val_bpb 1.179 (commit reference in stage1 writeup)

## Status update — 2026-05-03 (end of session 4)

Session 4 was the Stage 2 SFT debugging and infrastructure-hardening session. Stage 2 pretrain was already complete from session 3 at val_bpb 1.1743 (baseline parity); this session was about getting SFT to actually run on M2 + landing durable diagnostic infrastructure for the next round.

### What got done this session

**Stage 2 SFT crash investigation** (`docs/sft_oom_investigation_2026-05-03.md`)
- Three SFT runs hung silently at "step 200" before we figured out the cause. Two blocking issues compounded: (a) Python's default block-buffering hid the actual error message in the local log; (b) `sample <pid>` showed Metal stuck on `_MTLCommandBuffer waitUntilCompleted`, which we wrongly read as a deadlock when it was actually a thread waiting on an already-failed command buffer.
- Once `python -u` + the empty_cache fix at save let runs progress further, the actual error printed: `kIOGPUCommandBufferCallbackErrorOutOfMemory`. Root cause: ChatCORE eval at step 200 (chat_sft.py default `--chatcore-every=200`) — not the model save we kept blaming.
- Two layers of misdiagnosis before reaching the data-supported reality. Methodology lessons captured in the post-mortem.

**Falsified the T² hypothesis** (`dev/chatcore_prompt_lengths.py`)
- After the OOM error surfaced, I hypothesized the cause was ChatCORE prompts being routinely longer than the model's training `max_seq_len=512`, blowing up Stage 2's 4× (B,T,T) activation tensors via T² scaling. Operator pushed for cheap empirical validation.
- The script counts prompt token lengths across all 6 ChatCORE tasks. Result: prompts are nearly all under 512 (only 1 in 500 MMLU prompts barely exceeded). Hypothesis falsified by data in ~1 min.

**Confirmed the cumulative-fragmentation hypothesis** (`dev/mps_fragmentation_bench.py`)
- After the T² miss, reframed toward MPS allocator fragmentation across hundreds of variable-shape forwards. Synthetic reproducer that loads the Stage 2 model and runs N forwards at variable T, logging per-iter MPS allocator state.
- Without intervention: 10 iters → driver cache climbs 6.7 GB → 14.8 GB (4 GB from the 19.1 GB ceiling).
- With `torch.mps.empty_cache()` between iterations: cache stays bounded at 1-2 GB indefinitely.
- Hypothesis confirmed in 30 seconds. The bench doubles as an A/B platform for future architectural changes.

**Eval-loop fix landed** (commit `f27a6db`)
- Factored the synchronize+empty_cache pattern into `nanochat.common.mps_release_cache()` (no-op on non-MPS).
- Wired into `run_categorical_eval` (every batch) and `save_checkpoint` (replaced inline calls).
- `chat_sft.py --chatcore-every=200` (default) is now safe to use on M2.

**MPS metrics in wandb** (commit `f4f5066`)
- `nanochat.common.mps_metrics()` returns `mps/{allocated,driver,cache,recommended_max}_gb`. Wired into all three training scripts' eval-step wandb log calls.
- Charts the allocator climb fingerprint that precedes any future fragmentation event. Sub-microsecond cost; no new deps.

**CLAUDE.md updates and CLAUDE.md → DEV.md split**
- Lifted CLAUDE.md from gitignored to tracked (commit `42e4f9d`).
- Trimmed from 155 → 53 lines per Anthropic's "skinny CLAUDE.md" guidance. Spilled architecture detail and full command reference into new `DEV.md` (commit `d64e2a5` + `36152cf` + `2832049`).
- Lifted MPS-orphan-worker hygiene rule (previously buried in `m2_pipeline_2026-04-30.md:102`) into CLAUDE.md so it's auto-loaded.
- One-liner on triaging long runs (three signal sources: log file, wandb, ps/vm_stat) in CLAUDE.md.

**Disk hygiene** (commit `1d4fbc2`)
- Run 1 of this session's SFT was killed by macOS at step 184 — actual cause was disk OOM from 19 GB of accumulated Stage 1 intermediate checkpoints (`--save-every=200` with no rotation).
- Cleaned up ~27 GB of stale checkpoints. Added `--save-keep-last-n` rolling-cleanup flag to all three training scripts. Default behavior unchanged; opt-in.
- Recommended pretrain pattern: `--save-every=1000 --save-keep-last-n=2` (caps at ~2.4 GB).

**Pre-flight checkpoint guard** (commit `e7852b6`)
- Earlier in the session, the Stage 2 pretrain was kicked off without `--model-tag` and silently overwrote the canonical baseline at `d6/`. Added `assert_checkpoint_dir_safe()` called at training startup; aborts cleanly if the target dir already has `model_*.pt`. `--force-overwrite` is the explicit escape hatch.
- ADR-002 already documented the swap-vs-additive design priors. CLAUDE.md now also has the "always pass --model-tag" rule.

**`dev/wandb_status.py`** (commit `01e4f4f`)
- Live snapshot of any wandb run by URL or `entity/project/run-id`. Bypasses buffered local logs.
- Auto-detects metric column names; works for base_train, chat_sft, chat_rl without per-script tweaks.

### Stage 2 SFT result (2026-05-03)

SFT completed cleanly in 79 min on M2 (wandb run `2df5c88o`). All infrastructure landed this session worked: `python -u` for live logs, `mps_release_cache()` before save, `--save-keep-last-n=2` rolling cleanup, `--chatcore-every=-1` to skip the OOM path, pre-flight checkpoint guard. Final checkpoint at `~/.cache/nanochat/chatsft_checkpoints/d6_stage2/model_000375.pt`.

**Final val_bpb = 0.6518.** Beats baseline (0.6639) by 1.8% and Stage 1 swap (0.6712) by 2.9%. Single-seed but well above noise floor.

| arm | pretrain val_bpb | SFT val_bpb |
|---|---:|---:|
| baseline | 1.174 | 0.6639 |
| Stage 1 swap | 1.179 | 0.6712 |
| **Stage 2 additive** | **1.1743** | **0.6518** ⭐ |

Stage 2 isn't just at parity — it actually wins on the SFT loss. The learned per-token gates appear to give a small but real benefit on the SFT mixture. Pretrain was at parity; SFT shows real separation. Surprising and worth documenting.

### Files added this session (durable)

- `docs/sft_oom_investigation_2026-05-03.md` — full debug post-mortem with the methodology lessons (block-buffering hides errors, hypothesis falsification, "same arch" is a strong claim, project memory must be findable).
- `dev/wandb_status.py` — live wandb run viewer.
- `dev/chatcore_prompt_lengths.py` — falsifies the T² hypothesis empirically.
- `dev/mps_fragmentation_bench.py` — synthetic MPS-fragmentation reproducer + A/B platform.
- `nanochat/common.py::mps_metrics()` — wandb-shippable allocator stats.
- `nanochat/common.py::mps_release_cache()` — synchronize+empty_cache helper, used at save and in eval loops.
- `CLAUDE.md` (now tracked) + `DEV.md` (new depth reference).
- `docs/project_notes/bugs.md` updated with full SFT-hang root-cause entry.

### Next session: pick up here

1. **Surface SFT result.** `python -m dev.wandb_status matt-parrett/nanochat-sft/2df5c88o` after completion. Local log at `/tmp/sft_stage2.log`. Final checkpoint at `~/.cache/nanochat/chatsft_checkpoints/d6_stage2/model_000375.pt`.
2. **Optional ChatCORE on the final SFT checkpoint** — now that `run_categorical_eval` has the empty_cache fix, this should run cleanly even on M2. Compares Stage 2 SFT to Stage 1 SFT and baseline on real downstream tasks.
3. **chat_cli smoke test** — `python -m scripts.chat_cli --model-tag=d6_stage2 -p "What is the capital of France?"` for qualitative comparison.
4. **Optional upstream PR** — `karpathy/nanochat` users on Apple Silicon would hit the same OOM if running chat_sft with `--chatcore-every` enabled. The `mps_release_cache()` fix is generally applicable; could be filed as an issue or small PR.

### Open questions still punted

- Stage 2 swap topology has never been pretrained. Stage 1.5b's MQAR cross-check showed swap and additive equivalent at the synthetic level, but full DCLM val_bpb on Stage 2 swap is unknown. Cheap to answer (~3h pretrain) if interesting.
- Stage 1-additive (W_o=1) full pretrain. Probe found parity with baseline; whether DCLM val_bpb agrees was punted.
- Multi-seed confirmation of Stage 2 pretrain val_bpb 1.1743 — currently single-seed.
- The forward-pass fuse opportunity in `LearnedGateLinearMemory` (combine log_decay→decay into one in-place op; fuse weights computation). Modest M2 wins; not load-bearing.

## Status update — 2026-05-04 (end of session 5)

Session 5 was the experimental-result + paper-bootstrap session. ChatCORE on
Stage 2 SFT closed the only unchecked Stage 2 box; the full NeurIPS 2025 paper
unblocked Stages 3/5/6; the seed-variance experiment (A2) confirmed the Stage 2
SFT win is not a seed lottery. A3 (multi-seed pretrain) is queued for next
session with all prerequisites landed.

### What got done this session

**ChatCORE on Stage 2 SFT** (`docs/hope_nl_stage2_chatcore_2026-05-03.md`, commit `54610fb`)
- First downstream-task signal on Stage 2: ChatCORE = **0.1744** at d6_stage2/375.
- Per-task: ARC-Easy 25.80%, ARC-Challenge 28.67%, MMLU 26.98%, GSM8K 0.76%, HumanEval 0.00%, SpellingBee 95.31%.
- SpellingBee dominates 91% of the metric (0.9531 / 6 = 0.159 vs 0.1744 total). Non-SpellingBee tasks all hug their baselines.
- Honest read: at d6 scale ChatCORE is mostly a measurement of "did SFT memorize the SpellingBee template" — true for any d6 SFT regardless of pretrain architecture. Real architectural signal lives at d12+ which is outside our M2 budget.
- Eval-loop fragmentation fix (`mps_release_cache`, commit `f27a6db`) production-tested over ~14k MMLU batches, no OOM. Session-4 infrastructure validated end-to-end.

**Full Hope/NL paper read** (40pp, NL.pdf)
- Previously had only the 13pp truncated version. Full paper unblocks the previously-missing equations for Stages 3/5/6.
- **CMS** (Eq 70-71, three variants Eq 72/73/74): chain of MLP blocks at different update frequencies with chunked-gradient-sum updates. Independent / head-wise variant (Eq 74) is the simplest test.
- **Hope architecture** (Eq 94-97): Self-modifying Titans → CMS chain. Includes a **Hope-Attention variant** that swaps self-modifying Titans for plain attention — directly applicable to nanochat.
- **M3 optimizer** (Algorithm 1): Adam + Muon + CMS combined. Paper Figure 12 confirms M3 is slower than Muon at 140M and 1.3B; proof-of-concept only.
- **§7.3 retrofit recipe**: initialize CMS blocks from existing pretrained MLPs and continue pretraining. Means we don't have to retrain from scratch to test CMS at d6.
- **Ablation Table 6**: w/o CMS = +6.5% ppl; w/o DGD = +9.6%; w/o inner-q ≈ 0% (could be dropped to simplify); w/o inner-v = +13.5%. Numbers are at 760M / 30B; ~600× our d6, transfer unknown.

**Phase audit doc** (`docs/hope_nl_phase_audit_2026-05-03.md`, commits `d62063c` + later edits)
- Stage matrix table (Stages 0-6 status at a glance).
- Codex sync integration (challenge-function role, refined A1→A2→A3 order with SFT-seed-variance step inserted before pretrain).
- Source bootstrap section with full equations from the paper.
- Three new options on the table alongside A/B/C: D (§7.3 retrofit on Stage 2 d6), E (Hope-Attention as Stage 5 target), F (CMS-Independent ablation).
- Refined order: **A1 → A2 → A3 → F → (D or E) → C**.

**ChatCORE eval-scheduling followup ticket** (`docs/project_incoming/feat_chatcore_eval_scheduling.md`, commit `aaf94da`)
- Codex's proposal to add smoke / fastfail / full eval modes — without changing the canonical ChatCORE metric. ~80 of today's 87-min eval was spent on confirmed-floor GSM8K and HumanEval. Filed for pickup before Stage 4.

**Seed plumbing patch** (commits `29146e7` + `fc48d9c`)
- `--seed` CLI flag on `base_train.py` and `chat_sft.py`; default 42 preserves existing behavior. Plumbed into `nanochat/common.py:compute_init()`; auto-flows into `meta_*.json` via `vars(args).copy()`. Closes Codex's metadata-audit gap.
- `--sft-tag` flag separates SFT save dir from load dir (`--model-tag`). Required for multi-seed SFT on the same pretrain checkpoint; cleaner than symlink hacks.

**Training recipe lifted to key_facts.md** (commit `f6467ff`)
- The seed=1 SFT launch hung 60min into a default-eval-tokens (20M) val_loss eval before we caught the recipe mismatch. The recipe was already documented in `sft_oom_investigation_2026-05-03.md:181` (today's earlier post-mortem) but lived inside narrative, not in any auto-loaded surface.
- Lifted canonical d6/M2 pretrain + SFT recipes into `docs/project_notes/key_facts.md`. Future sessions auto-load it. Specifically calls out: both `--eval-every` and `--eval-tokens` are load-bearing for fair val_bpb comparison; default `eval-tokens=20M` silently hangs M2 SFT.

**Cross-project coordination** (note in `~/projects-new/trx4mr/docs/m2-shared-gpu-coordination.md`, user-committed as `8f82509` on trx4mr side)
- Documented the pre-flight + wait-for-PID monitor pattern used by nanochat. trx4mr could (and now does) run the same check the other direction. User hooked it into trx4mr's CLAUDE.md preflight surface.
- Tested in practice mid-session: trx4mr orchestrator's busy-check correctly detected our SFT process and waited.

**A2 SFT-seed-variance experiment** (`docs/hope_nl_stage2_seed_variance_2026-05-04.md`, commit `f8a2551`)
- Question: is the Stage 2 SFT headline win (val_bpb 0.6518 vs baseline 0.6639 = 0.0121) reproducible across SFT seeds, or an SFT-seed lottery?
- Re-ran SFT on the **same** Stage 2 pretrain checkpoint with seeds 1 and 2 (~2.5h total).
- Result: 3/3 seeds (42, 1, 2) → val_bpb 0.6518 / 0.6516 / 0.6520. Spread **0.0004** — ~30× smaller than the headline win.
- Trajectories tracked each other within ≤0.0024 at every checkpointed step. Optimization paths converge, not diverge.
- Conclusion: SFT is highly seed-stable on this configuration; the headline is not an SFT-seed lottery; variance, if any, must live in pretrain → A3 is justified.

### Files added this session (durable)

- `docs/hope_nl_phase_audit_2026-05-03.md` — cross-stage synthesis with Codex sync + paper bootstrap
- `docs/hope_nl_stage2_chatcore_2026-05-03.md` — ChatCORE 0.1744 writeup
- `docs/hope_nl_stage2_seed_variance_2026-05-04.md` — A2 result (this session's headline)
- `docs/project_incoming/feat_chatcore_eval_scheduling.md` — Codex's smoke/fastfail/full eval proposal, filed for pickup
- `docs/project_notes/key_facts.md` — appended d6/M2 training recipes section
- `~/.cache/nanochat/chatsft_checkpoints/d6_stage2_s1/model_000375.pt` (val_bpb 0.6516)
- `~/.cache/nanochat/chatsft_checkpoints/d6_stage2_s2/model_000375.pt` (val_bpb 0.6520)

### Next session: pick up here

1. **Launch A3** — two Stage 2 pretrains with `--seed=1` and `--seed=2`. Launch commands captured at the bottom of `docs/hope_nl_stage2_seed_variance_2026-05-04.md`. ~6h wall total. Compare val_bpb at step 5000 against the seed=42 reference (1.1743).
2. **Decision after A3**: tight cluster (similar to A2's 0.0004) → architecture's pretrain win is robust → proceed to **F** (CMS-Independent ablation, ~3h, cheapest direct test of "does multi-frequency memory help at d6"). Wide cluster (≥ 0.012) → headline was a pretrain-seed lottery → pivot to wrap-up.
3. **Codex sanity-check pending** — note at `/tmp/pasteboard-3` was sent for a critical pass on my paper-equation extraction. Reply may have actionable corrections for the audit doc; check before committing to F or E.

### Open questions still punted (carried from session 4 + new)

- Stage 2 swap topology has never been pretrained.
- Stage 1-additive (W_o=1) full pretrain.
- Multi-seed Stage 2 pretrain — **A3, queued for next session**.
- Forward-pass fuse opportunity in `LearnedGateLinearMemory`.
- ChatCORE on baseline d6 SFT and Stage 1 SFT for direct comparators (baseline pretrain checkpoint is gone; cost ~5h to regenerate).
- Codex's third sanity-check pass on paper equations.

### Addendum 2026-05-04 (post-session-5, pre-A3)

After the session-5 wrap, Codex's pre-A3 sanity-check came back. Two
concrete outcomes integrated before any next-session A3 launch:

**Q1 verdict (methodological)**: option (a) — A2's tight cluster only
bounds the SFT-seed component of variance, not the pretrain-seed
component, and not the baseline d6 distribution (which we never
measured). A3 will support "Stage 2 pretraining is seed-stable and
reaches baseline-like pretrain bpb" but NOT "Stage 2's architecture
effect is real vs natural d6-seed noise." Codex's cheaper compromise:
**A3-prime** — after A3 if tight, run ONE fresh baseline d6 pretrain
+ SFT with the modern recipe (~4h). Doesn't fully bound variance but
catches the biggest risk (historical baseline being stale/lucky).

**Q2 verdict (operational)**: implement **`--inherit-from`** on training
scripts before A3. Reason: I caught a head_dim default drift (128 vs
reference 64) plus 5 other field mismatches in the original queued A3
commands. Without explicit re-passing every relevant flag, A3 would
have silently trained a different model — 6h of compute wasted.

**Implementation landed (commit `ca9bc94`)**:
- `nanochat/common.py::load_inherit_config()` helper
- `--inherit-from=<meta_path>` flag on `base_train.py` and `chat_sft.py`
- Loads reference `user_config` as parser defaults *before* CLI parsing;
  CLI flags override only what's intentionally different
- Excludes per-run / operational fields (`run`, `model_tag`, `seed`,
  `resume_from_step`, `force_overwrite`, `save_every`, `save_keep_last_n`)
- Verified end-to-end: 34 fields auto-loaded from the seed=42 reference
  meta, including `head_dim=64`, `max_seq_len=512`, `window_pattern=L`,
  `num_iterations=5000`, `eval_every=100`

**Updated A3 launch commands** (`cca0f1c`): now ~5 lines instead of
~20, with architectural-config parity by construction. The config-
parity audit table stays in the writeup as a verification step, not
the prevention mechanism. Pre-Codex queued commands had the head_dim
bug; post-fix they would have worked but been brittle to future drift.

**Refined recommended order**: A1 ✓ → A2 ✓ → **A3** → **A3-prime**
(conditional) → F → (D or E) → C. Documented in audit doc.

## Status update — 2026-05-05 (end of session 6)

Session 6 ran A3 (multi-seed Stage 2 pretrain), SFT on top of A3 seed=1,
A3-prime (modern-recipe vanilla d6 baseline pretrain + SFT). Combined
result closes the architectural question on the Hope/NL track at
d6/5000-iter on ClimbMix.

**Headline:** the original "Stage 2 wins by 1.8% on SFT" claim was
overwhelmingly recipe drift, not architecture. Modern-recipe vanilla d6
baseline beats both Stage 2 seeds on pretrain val_bpb AND beats Stage 2
on SFT val_bpb. All deltas within the 0.0016 seed-noise spread we
measured on A3 — call it a tie, not a Stage 2 loss — but the +1.8% claim
does not survive recipe-controlled comparison.

### What got done this session

**A3 — multi-seed Stage 2 pretrain** (~10h wall, sequential)
- seed=1: val_bpb 1.1729 (4h49m)
- seed=2: val_bpb 1.1712 (~5h)
- Inter-seed spread: **0.0016**
- Both beat the historical d6 baseline (1.174). Confirms Stage 2
  pretraining is seed-stable.
- `--inherit-from` mechanism worked end-to-end: 34 fields auto-loaded
  from `d6_stage2/meta_005000.json`, including head_dim=64, alpha/eta
  init biases. Hardware-config parity by construction.

**SFT on A3 seed=1** (~1h)
- val_bpb 0.6495 — beats A2 d6_stage2_s1 (0.6516) by 0.0021. Confirms
  Stage 2 pretrain quality survives SFT.

**A3-prime — modern-recipe vanilla d6 baseline** (~4h pretrain + 1h SFT)
- Pretrain val_bpb **1.1686** (vanilla d6, seed=42, current `master`)
- SFT val_bpb **0.6483** (sft_seed=1, identical SFT recipe)
- **Beats both Stage 2 seeds on pretrain val_bpb** by 0.0026-0.0043
- **Beats Stage 2 SFT (A3 seed=1) by 0.0012** — the headline-overturning
  number
- The ~0.0156 SFT gap to the historical 0.6639 baseline is now
  attributable to recipe drift (most prominently `840d3db`: ve_gate
  Muon → AdamW move) accumulated over 4 commits since the historical
  baseline was trained.

**A3 + A3' writeup** (`docs/hope_nl_a3_a3prime_2026-05-05.md`, commit `070818a`)
- Combined experimental result, with full pretrain & SFT trajectory
  tables, headline-revision section, what-this-means, caveats,
  artifacts list. The synthesis doc for the architectural question.

**ADR-003** (`docs/project_notes/decisions.md`, commit `e407e41`)
- Defer upstream PR #544 (dataloader remainder reuse) until after A3'
  to preserve A3's parity comparison. Adopt later only with a clean
  re-baseline run; do not retroactively compare cross-dataloader
  numbers. The PR's claimed 1.28× speedup at our T=512 makes this
  worth revisiting after the writeup wraps.

**Data investigations backlog** (`docs/project_notes/data_investigations.md`, commit `f2c6cd7`)
- Q1 (length-stratified val_bpb), Q2 (adopt
  `ddudek/nanochat-climbmix-annotated` for labeled corpus), Q3
  (within-corpus dedupe sanity check), Q4 (per-domain CORE).
- Skip list (re-filtering curated data, DoReMi at 81M tokens,
  trillion-scale dedupe wins that won't materialize).
- All gated on Hope/NL writeup wrap to avoid invalidating comparisons.

**Disk hygiene** — pruned ~9 GB of stale step-5/10/50 debug checkpoints
from `base_checkpoints/d6_stage2/` (only step-5000, the load-bearing
checkpoint for A3 inherit-from, retained). Disk back to 37 GB free.

### What this means

1. **Stage 2 architecture is neutral, not net-positive, at d6/5000-iter
   on ClimbMix.** Both pretrain val_bpb and SFT val_bpb show A3' baseline
   slightly ahead of Stage 2. Magnitudes within seed-noise.
2. **MQAR probe story remains true** — additive memory branch *learns*
   on synthetic recall (saturation by step ~76 in Stage 1.5b). Just
   doesn't translate to LM bpb at this scale/corpus.
3. **Stage 4 (multi-block memory) was conditional on Stage 2 surviving.
   Condition not met.** Defer.
4. **Recipe drift is real and silent.** ~0.0156 of "free" SFT
   improvement accumulated across 4 commits between historical and
   modern code, dwarfing the architectural delta. Future
   architecture-class experiments must pin recipe + seed and re-baseline,
   not compare against archived numbers.

### Caveats (carried into the writeup)

- **n=1 baseline pretrain seed.** Result reads "Stage 2 doesn't win"
  with high confidence; "Stage 2 loses" with lower confidence.
- **Stage 2 SFT only run on A3 seed=1.** Bracketing across both Stage 2
  pretrain seeds × SFT was deemed redundant given A3'.
- **d6 / 5000-iter / ClimbMix-only conclusion.** Memory architectures
  may show structural value at larger scale, longer horizons, or with
  long-context-rewarding evaluations. Cannot exclude.
- **MMLU/GSM8K-heavy SFT mix** doesn't reward long-context memory.
  Different mix could in principle show Stage 2 benefit.

### Files added this session (durable)

- `docs/hope_nl_a3_a3prime_2026-05-05.md` — combined A3+A3' result + revised headline
- `docs/project_notes/data_investigations.md` — data backlog (Q1-Q4 + skip list)
- `docs/project_notes/decisions.md` — appended ADR-003 (PR #544 deferral)
- `~/.cache/nanochat/base_checkpoints/d6_stage2_pretrain_s1/model_005000.pt` (val_bpb 1.1729)
- `~/.cache/nanochat/base_checkpoints/d6_stage2_pretrain_s2/model_005000.pt` (val_bpb 1.1712)
- `~/.cache/nanochat/base_checkpoints/d6_baseline_modern/model_005000.pt` (val_bpb 1.1686)
- `~/.cache/nanochat/chatsft_checkpoints/d6_stage2_pretrain_s1_sft/model_000375.pt` (val_bpb 0.6495)
- `~/.cache/nanochat/chatsft_checkpoints/d6_baseline_modern_sft/model_000375.pt` (val_bpb 0.6483)
- wandb runs: avvd9uov, pajx70rk, 2m6exejl, 3fprtaef, s4x7im7t

### Track status (post-A3')

| | Status | Outcome |
|---|---|---|
| **A1** Pre-A3 audit / `--inherit-from` plumbing | ✅ done | head_dim drift caught; multi-seed unblocked |
| **A2** Multi-seed SFT-on-d6_stage2 | ✅ done | 3/3 within 0.0004 — SFT-seed not the noise source |
| **A3** Multi-seed Stage 2 pretrain | ✅ done | spread 0.0016, both beat historical baseline |
| **A3'** Modern-recipe baseline + SFT | ✅ done | Beats Stage 2 — overturns the headline |
| **B** Stage 4 multi-block memory | ❌ blocked | Conditional on A3 surviving; condition not met |
| **F** CMS-Independent ablation (Eq 74 head-wise) | open | Cheap (~3h), separate question from Stage 2 |
| **D** §7.3 retrofit (continue-pretrain at CMS levels) | open | ~3-6h, exploratory stress test |
| **E** Hope-Attention Stage 5 (full CMS chain) | open | ~1-1.5 weeks, real design block |
| **C** Wrap and write up | active | A3+A3' writeup committed; full track synthesis pending |

### Next session: pick up here

**Recommended path: C (wrap and synthesize).**

The architectural question has a clean answer ("Stage 2 doesn't win at
d6/5000-iter on ClimbMix"). F/D/E either ask different questions
(multi-frequency memory, retrofit mechanism, Hope-Attention) or scale up
substantially. The honest move is the synthesis doc + close-out:

1. **Track-level synthesis writeup** (~2-4h)
   - Stage 0 (plumbing) → 1 (swap) → 1.5 (probe + W_o root cause) → 2
     (learned gate) → A1/A2/A3/A3' → conclusion.
   - Pull from the existing per-stage docs; the experimental record is
     already complete.
   - End with the honest finding + the four "what this means" points
     above.
2. **HTML narrative writeup** for publication (per CLAUDE.md convention:
   markdown first, then HTML).
3. **Update the trx4mr ticket** with the result, so the originating
   project knows the d6 answer.

**Optional revisits** (not the recommended path, but available):
- **F** (CMS-Independent ablation, ~3h) — cheap signal on whether
  multi-frequency memory at d6 helps. Even at neutral, gives one more
  data point in a different memory-kind direction.
- **PR #544 adoption + clean re-baseline** — `d6_v2` corpus would unlock
  ~1.28× speedup at T=512 for any future runs. Per ADR-003.
- **Q1 from `data_investigations.md`** (length-stratified val_bpb) —
  cheap probe of the memory hypothesis on long docs even with current
  corpus. Could revive Stage 2 if memory turns out to help on long-doc
  subset specifically.
- **n=2 baseline seed** — bracket the n=1 A3' baseline. ~5h.

**Defer indefinitely:** B (Stage 4 multi-block) and E (Hope-Attention
Stage 5) until either an architectural revival path appears or compute
scale changes the picture.

### Open questions still punted (carried + closed)

- ~~Multi-seed Stage 2 pretrain — A3.~~ ✅ done.
- Stage 2 swap topology has never been pretrained. (still open, low priority post-A3')
- Stage 1-additive (W_o=1) full pretrain. (still open, low priority post-A3')
- Forward-pass fuse opportunity in `LearnedGateLinearMemory`. (still open; perf-only, not result-changing)
- ChatCORE on baseline d6 SFT and Stage 1 SFT for direct comparators.
  (now partially answerable — A3' baseline + SFT checkpoint exists, ChatCORE could be run on it for ~1h)
- Codex's third sanity-check pass on paper equations. (still open; relevant only if E is revived)

### Addendum 2026-05-05 (post-session-6 closeout)

Three commits landed after the session-6 status entry was written —
the synthesis pass + two future-work tickets:

- **`6561a4c`** — `docs/hope_nl_track_synthesis_2026-05-05.md`. The
  full Stage 0 → A3' arc as a single read. End-to-end narrative,
  honest verdict, four "what we learned" points (probe-first
  load-bearing; recipe drift silent + large; Codex challenge function;
  single-depth-dial discipline). Use as the canonical document for
  anyone picking up the track cold. Closes path **C** in the audit doc.
- **`6561a4c`** + **`92d0233`** — `docs/project_incoming/feat_d8_extension.md`.
  Tickets the "Cannot exclude memory architectures help at larger
  scale" carve-out from the synthesis. Three-phase plan + Phase 0 d4
  methodology rig (added in `92d0233`). Notes d3's separate role
  (pure plumbing/harness derisk; not for architectural-question work
  because memory layer is 33% of network at d3 vs 16% at d6).
  Total floor 13-21h, ceiling 23-36h.
- **`ae2fcb3`** — `docs/project_incoming/feat_modernization_alignment.md`.
  Cross-project sync with trx4mr's CS336/Tatsu lecture audit. nanochat
  is ~85% on-template for 2026 production conventions; 3 deliberate
  departures (ReLU² over GLU, aspect_ratio=64 vs ~100, softcap-yes /
  z-loss-no / qk_norm-no); 3 ticketed additions (A1 z-loss-as-standby
  ADR; A2 qk_norm at d8; A3 GLU sweep) in priority order.

**Next session pickup** (revised after these landings):

The track is wrapped. Three forward-looking lines exist as tickets:

1. **`feat_d8_extension.md`** — pick this up if you want to keep the
   architectural question open. Phase 0 (d4 rig) is the ~6-9h cheap
   first move that bounds baseline-seed variance; Phase 1+2 (d8
   derisk + headline) is ~11-17h committing to the scale question.
2. **`feat_modernization_alignment.md::A1`** — the zero-cost move:
   record z-loss as a stability-standby ADR in `decisions.md`,
   mirroring trx4mr's ADR-019. Independent of d8 path. Do regardless.
3. **`feat_modernization_alignment.md::A2/A3`** — gated on d8 path
   activating; slot in alongside Phase 1/2 of d8 if they happen.

Default: if no clear architectural revival appetite, do A1 only and
let the track stay closed. The synthesis is the canonical answer.

**Status of the synthesis prediction "C is the recommended path":**
✅ done. Branch is `experiment/hope-nested-learning`, not merged to
master. Whether to keep the branch open or close it is a separate
project-governance decision; the experimental record is complete on it.

### Addendum 2026-05-05 (A1 of modernization-alignment landed)

- **`b04f04f`** — `docs/project_notes/decisions.md::ADR-004` ("Z-loss as
  standby stability lever"). Mirrors trx4mr ADR-019, reframed for
  nanochat's spike-risk regime (d8+ extension, longer horizons,
  quantization) rather than trx4mr's STE regime. Park-don't-ship
  stance: reach for it *first* if a future run spikes, before
  LR/clip-grad/init-scale; if invoked, becomes the default (no
  feature flag, per modernization-alignment scope rules); ~5 lines,
  DCLM-style 1e-4 coefficient.
- Closes **A1** of `feat_modernization_alignment.md` — the zero-cost
  do-regardless item. Track is fully wrapped; only forward-work items
  left are `feat_d8_extension.md` and A2/A3 of modernization-alignment
  (both gated on d8 activating).

### Addendum 2026-05-06 (chat-quality investigation + cosine-NN diagnostic)

Big day, three threads, 19 commits, all local. Canonical read for picking
up cold: **`docs/chat_quality_arc_2026-05-06.html`** — publication-style
narrative covering the day's investigation arc end to end.

**Thread 1 — PR #544 dataloader capability** (`2173dab`-`a02db98`,
`e32c17b`). Adopted upstream PR #544 (cropped-remainder reuse) as an
env-gated capability behind `NANOCHAT_DATALOADER_REUSE_REMAINDER=1`,
default OFF. Validated with n=2 A/B at d3_tiny scale; structural
val_bpb regression of +0.005 to +0.013 paired with -53% source-token
consumption. Net loss-loss on M2 (compute-bound, abundant ClimbMix);
indicated for I/O-bound regimes (8×H100 speedrun, multi-epoch on small
corpora). ADR-005 records the decision; `meta_*.json` captures
`dataloader_variant` for cross-checkpoint auditability. **Closes**
ADR-003's deferral on PR #544.

**Thread 2 — chat-quality investigation** (`af2d86a`-`3442db0`).
Operator reframed from architectural memory to user-pain ("d6 chatbot
is kinda bad, especially after turn 1"). Four full-parameter levers
tested on a pre-registered 7-prompt multi-turn rubric:

| lever | run | val_bpb | multi-turn |
|---|---|---:|---:|
| baseline | `d6_baseline_modern_sft` | 0.6483 | 1/7 |
| architecture | `d6_stage2_pretrain_s1_sft` | 0.6495 | 1/7 (tie) |
| decoding (5 configs) | best: greedy | — | 1.5/7 (no win) |
| SFT-rebalance | `d6_baseline_chatmix_a` (cut MMLU/GSM8K) | **0.5952** | 0/7 |
| SFT-additive | `d6_baseline_smoltalk_2x` (2× SmolTalk) | 0.6027 | 0/7 |

Cleanest finding: **chatmix_a moved val_bpb by -0.053 (huge by this
codebase's standards) and multi-turn rubric by -1/7 in the opposite
direction**. SFT val_bpb is not predictive of chat quality at d6 —
documented quantitatively for the first time. Each rebalance produced
*differently broken*, not *less broken*; the model has finite
template-learning capacity at 74M and rebalancing redistributes which
templates dominate.

Promotes the synthesis verdict: Hope/NL Stage 2 is **also** neutral on
the multi-turn-coherence task class (not just val_bpb). The architecture
verdict closes on the user-relevant test class, not just the academic one.

`scripts/chat_sft.py` got a new `--smoltalk-epochs` flag (`20baaac`)
during this thread; `nanochat/engine.py` got a `repetition_penalty`
kwarg behind default=1.0 (`d13f5c4`). Both are reusable infra.

**Thread 3 — proposals + cosine-NN diagnostic** (`92eb7fe`-`ac6c0a0`).
Per the operator's interest in both *external loop* (Feedback Descent)
and *internal architecture* (LoRA / PEFT) directions, two proposals:

- `docs/feedback_descent_proposal_2026-05-06.md` — five FD directions
  (A diagnostic, B internal-vs-external memory, C paper replication,
  D system-prompt optimization, E asymmetric local evaluator)
- `docs/lora_proposal_2026-05-06.md` — three LoRA directions
  (L1 persona-LoRA, L2 multi-LoRA mixture, L3 FD→LoRA distillation).
  Notes Karpathy's nanochat ships no PEFT path; implementing it is a
  real contribution for the laptop / continuous-learning use case.

Then the trx4mr Phase 5 retro arrived
(`~/projects-new/trx4mr/experiments/blabberverse-phase5-impl-and-takeaways.html`)
with a categorical diagnostic: **two-axis debug framework** —
right-context drift (FP-flavored, fix = coarsen codebook) vs
left-context collision (binary-flavored, fix = increase d_model).

Ran the cosine-NN probe on `d6_baseline_modern_sft` (`6f81014`):
17 query tokens sampled from today's degenerate transcripts. Aggregate:
**~65 % right-context drift, ~18 % left-context collision, ~18 % mixed.**
The dominant signal is unambiguous. Specific mechanisms identified:
- Math-mode reflex: `<|python_start|>` embeds with bracket-syntax
  tokens (`,[`, `(f`, `[`); the path "digit input → number-state →
  bracket-syntax-state → emit `<|python_start|>`" runs through pure
  embedding-space proximity. Explains why even 4× reduction in GSM8K
  epochs (chatmix_a) didn't kill the template — embedding geometry
  preserves the trigger.
- Sydney attractor: places cluster densely; in any "predict capital
  city" state, Sydney is sticky. Self-reinforcing right-context drift
  loop, which user-injected facts ("Canberra, not Sydney") don't
  alter geometrically.

Per the framework's fix-direction table, ~65 % right-context drift
points at **codebook coarsening (binary/ternary)** as the indicated
fix; ~18 % left-context collision points at **capacity (M4 / d8+)**.
Both warranted, in those proportions.

Updated proposals (`b6434d1`):
- LoRA L1 restructured as a **two-arm A/B**: `L1-d6` (74M fp + LoRA)
  vs `L1-bonsai` (Bonsai 4B 1-bit + fp LoRA). Diagnostic-supported.
- FD Direction B adds a side-question: cosine-NN on
  `d6_stage2_pretrain_s1_sft` to test whether internal recurrent
  memory shifts the axis distribution (does Stage 2 attenuate
  right-context drift specifically?). ~1 minute marginal compute.

Resource-economics caveat captured honestly: STE-based binary
*training* on M2 doesn't save resources (latent fp32 + quantize op +
unoptimized MPS kernels). The path to access Phase 5's
regime-robustness benefits at our scale is **Bonsai (already paid the
QAT cost on bigger compute) + small fp LoRA adapters**, not
retraining binary from scratch.

**Disk audit** (`28 GB free → 30 GB free` after Tier-0 dupe trim):
removed 9 intermediate save-every checkpoint files (`d6_baseline_modern`
step 4000, `d3_smoke` step 1000, `d3_tiny` step 250). 1.3 GB
recovered; no load-bearing artifact lost. Tier 1-3 trim opportunities
documented but not run (operator decision deferred).

### Next session pickup (revised after 2026-05-06)

The architectural track stays wrapped; the chat-quality track is now
also wrapped at full-parameter / full-precision interventions. The
diagnostic gives a **directional preference** for next moves:

1. **Cosine-NN on `d6_stage2_pretrain_s1_sft`** — ~1 min compute,
   decoupled from any other experiment. Tests whether internal
   recurrent memory shifts the axis distribution. Intrinsically
   interesting whether or not Stage 2 wins on chat. Cheapest move
   available; do this first if poking around.
2. **Bonsai 1-bit forward + fp LoRA infrastructure** — ~1 day of
   focused infra work; gating for the diagnostic-supported L1-bonsai
   experiment. Touches tokenizer compatibility, QLoRA-style flow on
   top of frozen 1-bit weights, inference path that combines Bonsai
   1-bit forward + fp LoRA forward.
3. **L1-d6 first** as the baseline LoRA test — ~3-5h. Could run
   independently of Bonsai infra if you want to validate the LoRA
   wiring before committing to the bigger Bonsai investment.
4. **Widen the cosine-NN probe** — ~30 min compute. Stratified 200+
   tokens to tighten the ~65/18/18 estimate. Useful if you want the
   diagnostic's percentages more defensible before betting on the
   Bonsai direction.
5. **FD path independently** — directions A/D in the FD proposal are
   the cheapest probes (1h gating; 2-3h system-prompt FD on existing
   chatbot). Orthogonal to Bonsai-LoRA in mechanism; could compose via
   L3 (FD-as-data → LoRA-as-internalization) if both produce real
   signal.

**Default if no clear appetite:** the architectural and chat-quality
tracks are both wrapped at d6/74M with available data. The HTML
narrative (`docs/chat_quality_arc_2026-05-06.html`) is the canonical
record. If picking up days/weeks later, read that first — it
synthesizes the day's findings without requiring the per-experiment
markdown writeups.

**Standing operational rules** (per
`~/.claude/projects/-Users-matt-projects-new-3p-nanochat/memory/feedback_local_only.md`):
the branch is local-only; no pushing or PR creation without explicit
per-task approval. All 19 of today's commits are local on
`experiment/hope-nested-learning`.

## Day 2026-05-07: LoRA infrastructure + L1 persona-LoRA + scale knob

The cosine-NN diagnostic from 2026-05-06 pointed at parameter-efficient
adaptation as the cheapest probable next win on the dominant
right-context-drift failure axis. This session opened a fifth lever
(beyond yesterday's four full-parameter ones), landed end-to-end LoRA
infrastructure in the codebase, ran two LoRA experiments on a curated
persona-retention dataset, and characterised the resulting failure
modes well enough to point at v3 directions. Today's commits:
`79e4b3c` → `d7ca704` (12 commits, all local).

### Headline outcome

**Persona-retention is solved at d6 by a 0.4 %-of-parameters LoRA
adapter.** Held-out 30-prompt eval, all_three (recall name AND role
AND location): base 0.0 % → LoRA v2 63.3 %. The original failure mode
yesterday's multi-turn rubric caught (`"What's my name and job?" →
"Sydney" / "nanochat" / "Java J"`) is now mostly working.

The 7-prompt rubric pass rate went 0/7 → 2/7. No prompt the base
passes regresses. Two new wins: persona_retention (the proposal
target) and self_correction (incidental).

### Threads

**Thread 1 — LoRA infrastructure** (`79e4b3c`, `f75a90d`).
`nanochat/lora.py` (~280 lines) + `tests/test_lora.py` (~250 lines, 11
contract tests). Module surface:
- `LoRALinear(base, rank, alpha)` — wraps a frozen `nn.Linear`. Forward:
  `base(x) + (x @ A^T) @ B^T * (alpha/rank) * scale`. A is Kaiming-init,
  B is zero-init, `scale` defaults to 1.0. Bit-equivalent to base at
  apply time. Mirrors `nanochat.gpt.Linear`'s master-fp32 / matmul-in-
  input-dtype convention.
- `apply_lora(model, target, rank, alpha)` — walks
  `model.transformer.h`, replaces matching attention projections with
  `LoRALinear` wrappers. Errors if it would inject zero adapters.
- `lora_state_dict` / `load_lora_state_dict` — adapter-only save/load
  with strict-key validation.
- `apply_lora_from_tag(model, lora_tag, scale=1.0)` — one-call helper
  for inference scripts that reads the meta json and loads weights.
- `set_lora_scale(model, scale)` — runtime attenuation knob.

`scripts/chat_sft_lora.py` (~290 lines) — focused training entry point.
Single CustomJSON dataset, plain AdamW over LoRA params only (no
Muon — rank-r is too small to benefit from orthogonalisation), no
torch.compile. Reuses `tokenizer.render_conversation` and
`evaluate_bpb` so val numbers are directly comparable to chat_sft runs.
Saves to `$NANOCHAT_BASE_DIR/lora_checkpoints/<lora_tag>/`.

`scripts/chat_cli.py` + `scripts/chat_web.py` gain `--lora-tag`,
`--lora-step`, `--lora-scale`. Inference engine path is untouched —
`LoRALinear` is a regular `nn.Module`, so `Engine` and the KV cache
work without modification.

**Thread 2 — Persona-retention curator via Claude subscription**
(`da67b49`). First version used `anthropic` SDK with API key. After a
pointer to the sibling `~/projects-new/elixir-explore/pulse` project
(`lib/pulse/claude/{suggested_responses,sentiment,surprise}.ex`), the
curator was rewritten to use the **`claude` CLI in headless mode** with
the operator's Claude Code subscription auth — no `ANTHROPIC_API_KEY`
needed:

```sh
claude -p '<prompt>' \
    --model haiku --tools "" --no-session-persistence \
    --output-format json \
    --system-prompt '<system>' \
    --json-schema '<json-schema>'
```

Reads the result envelope's `structured_output` field. The curator
builds Pydantic-validated `Conversation` objects per batch, validates
client-side that T1 contains all persona substrings and T6 recalls
them all, tracks a running diversity list of used names/roles/locations
fed back into each batch's prompt as an "avoid these" constraint.

Output (under `$NANOCHAT_BASE_DIR`, not committed):
- `persona_retention_v1.jsonl` — 300 train rows (bare list per line,
  CustomJSON-compatible)
- `persona_retention_v1_eval.jsonl` — 30 held-out rows with `_persona`
  metadata for substring eval
- `persona_retention_v1.log` — generation log

Cost: **~$3.13 against the Claude subscription**, ~74 % acceptance
rate (113 rejects / ~430 attempts; most rejects were schema-substring
mismatches where Claude's persona dict claimed a role not exactly in
T1's text). Token-length distribution: min=82, median=120, p90=148,
max=207 — well under `max_seq_len=512`.

This pattern is reusable for any future LoRA dataset curation. The
sibling project's choice of CLI-headless over SDK is a real upgrade —
the auth model fits a developer-machine workflow with no separate API
key management.

**Thread 3 — L1 v1: rank-8 Q+V** (`6fa5b34`, `fa1e921`, `614890c`).
Training: 300 iters, 50.4 s wall on M2 mps, val_bpb 1.243 → 1.077.
73,728 trainable params (0.1 % of base). Adapter checkpoint 296 KB.

Eval (held-out 30, greedy temp=0):

| metric | base | v1 | delta |
|---|---:|---:|---:|
| name_recall | 6.7 % | 20.0 % | +13.3 pp |
| role_recall | 33.3 % | 26.7 % | -6.7 pp |
| location_recall | 13.3 % | 26.7 % | +13.3 pp |
| **all_three** | **0.0 %** | **13.3 %** | **+13.3 pp** |

Real signal but not a solve. ~50 % of v1 outputs collapsed to a
generic opener ("That's a great way to work") — mode collapse from
low rank / narrow target / limited training. Markdown writeup
(`docs/lora_l1_persona_2026-05-07.md`) and HTML narrative
(`docs/lora_l1_persona_2026-05-07.html`) committed; HTML mirrors
yesterday's `chat_quality_arc_2026-05-06.html` aesthetic.

The v1 writeup's headroom table predicted **rank=16 + Q+K+V+O target**
as the cheapest probable v2 win (addressing the two most-likely
undersizing failures with no new data, ~3 min wall).

**Thread 4 — L1 v2: rank-16 Q+K+V+O** (`eadcfb8`).
Setup delta from v1: target `c_q,c_k,c_v,c_proj` (was `c_q,c_v`),
rank 16 (was 8), alpha 32, 600 iters (was 300), lr 3e-4 (was 1e-4).
Same dataset, same seed. Trainable params 73,728 → **294,912** (~0.4 %
of base). Wall: **1.95 min**. val_bpb 1.243 → **0.8197** (vs v1's
final 1.077). Adapter 1.14 MB.

Eval (same harness, same 30 held-out personas):

| metric | base | v1 | **v2** | v2 vs base |
|---|---:|---:|---:|---:|
| name_recall | 6.7 % | 20.0 % | **80.0 % (24/30)** | **+73.3 pp** |
| role_recall | 33.3 % | 26.7 % | **86.7 % (26/30)** | **+53.3 pp** |
| location_recall | 13.3 % | 26.7 % | **83.3 % (25/30)** | **+70.0 pp** |
| **all_three** | **0.0 %** | **13.3 %** | **63.3 % (19/30)** | **+63.3 pp** |

**4/30 → 19/30 all_three.** Almost 5× v1's win rate. Mode collapse
gone — v2's reflexive opener is `"You're [Name], a [Role] in
[Location]..."`, which reads T1 and recalls. The headroom claim was
correct *and* understated.

Cleanest single non-trivial v2 failure: **Moscow → Houston substitution**
on Russian-name personas. Both Elena Volkova (botanist) and Dmitri
Volkov (quantum physicist) — both Moscow ground truth — get relocated
to "Houston" in v2's output. Base would never produce that
substitution; the LoRA is partially overwriting city-cluster geometry
on Russian-name + Moscow inputs toward an American-name attractor.
Worth a follow-up cosine-NN probe. (Not run in this session.)

**Thread 5 — Catastrophic-forgetting check** (`9f42f3e`, `6a9bc14`).
Extended `dev/multi_turn_eval.py` with a `--lora-tag` flag so arm 2
becomes base+LoRA instead of Stage 2. Ran the 7-prompt rubric.

| prompt | base | v2 | result |
|---|---|---|---|
| persona_retention | fail | PASS | +1 win (target) |
| reference_resolution | fail (loop) | fail | both fail; LoRA: "You're Mittens, a cat named Mittens" (treats user as cat) |
| numerical_thread | fail | fail | LoRA: "You're Emily, a apples..." (hallucinates name) |
| topic_stickiness | fail | fail | both vague |
| constraint_accumulation | fail | fail | both vague |
| self_correction | fail | PASS | +1 incidental — template happens to fit |
| open_drift | fail | fail | LoRA: "You're the most relevant part of the project..." |

Aggregate: **0/7 → 2/7. No prompt regresses.**

The qualitative pattern is unmissable: v2's "You're [X], a [Y]..."
opener fires on every prompt regardless of fit. **Template-bleed**:
the LoRA learned the persona-recall pattern strongly enough that it
stamps that shape onto unrelated prompts. Strict-sense catastrophic
forgetting (capacity erasure on math, reference-tracking, etc.) — no,
those were already broken in base. Net: rubric +2, qualitative cost
real but not measurable on the binary rubric.

The fix is **dataset diversity**, not architecture: v2 saw only
persona-retention shapes during training, so of course it applies that
shape to everything.

**Thread 6 — `--lora-scale` runtime knob** (`d7ca704`). Added a
`scale` attribute on `LoRALinear` (default 1.0); forward becomes
`base(x) + (x @ A^T) @ B^T * (alpha/rank) * scale`. `scale=0.0` is
bit-equivalent to base (with a fast-path that skips the matmul);
`scale=0.5` is halfway. `set_lora_scale(model, scale)` and
`apply_lora_from_tag(scale=...)` propagate. Wired through all four
entry points (`chat_cli`, `chat_web`, `eval_persona_retention`,
`multi_turn_eval`). Tests pin scale=0 ≡ base bit-exact, scale=0.5
sits closer to (base+full)/2 than to either endpoint.

`eval_persona_retention.py` gained a sweep mode: pass
`--lora-scale 0.0,0.25,0.5,0.75,1.0` to run all five arms in one
invocation with table-formatted output.

Sweep on the 30 held-out:

| scale | name | role | location | all_three |
|---:|---:|---:|---:|---:|
| 0.00 | 6.7 % | 33.3 % | 13.3 % | 0.0 % |
| 0.25 | 20.0 % | 23.3 % | 20.0 % | 10.0 % |
| 0.50 | 43.3 % | 46.7 % | 56.7 % | 26.7 % |
| 0.75 | 80.0 % | 86.7 % | 83.3 % | 63.3 % |
| 1.00 | 80.0 % | 86.7 % | 83.3 % | 63.3 % |

Two clean signals: (a) scale=0.0 reproduces base exactly (validates
runtime fast-path), (b) **the LoRA saturates by scale=0.75** — same
all_three at 0.75 and 1.0. Attenuation only buys reduction below ~0.75.

Rubric at scale=0.5 — same 2/7 pass rate as scale=1.0 *without
template-bleed*:
- persona_retention: PASS (Alex + software engineer + startup)
- self_correction: PASS — actually *cleaner* than at scale=1.0:
  `"No, the capital of Australia is Canberra, not Sydney."` instead of
  the persona-template wrap.
- Reference: no longer "You're Mittens, a cat named Mittens" — vague reply.
- Numerical: no longer "You're Emily, a apples" — degenerate +0+0+0 loop instead.
- Open drift: no longer "You're the most relevant part of the project" — generic positive reply.

Trade-off: held-out persona all_three drops 63.3 % → 26.7 % at
scale=0.5. The rubric persona is simple ("Alex / software engineer /
startup"); the held-out 30 has trickier personas (Marina Rossi,
Marcus Thompson, etc.) where attenuation costs recall.

### Cleanest findings of the day

1. **A 0.4 %-of-parameters LoRA solves persona-retention on held-out.**
   0/30 → 19/30 all_three. The diagnostic-supported direction
   (capacity-routing on the dominant right-context-drift axis) is real-
   signal-positive at d6 fp scale.

2. **Template-bleed is a real failure mode of single-task LoRAs at
   small data.** The v2 LoRA's "You're [X]" opener fires on every
   prompt regardless of fit — including the cat, the apples, the
   project chat. The strict-sense catastrophic-forgetting binary
   misses this; the qualitative hit is real.

3. **Runtime LoRA attenuation is a legitimate trade-off knob.**
   scale=0.5 keeps the rubric pass rate (persona + self-correction
   both pass) and removes template-bleed entirely. Costs 36.6 pp on
   held-out persona all_three. If the deployment goal is "default-on
   adapter that doesn't break free chat", scale=0.5-ish is the right
   answer until v3 lands.

### Where this leaves the project

The L1 arm of the LoRA proposal's two-arm comparison is settled:
**L1-d6 works.** The diagnostic-supported direction has measurable
signal at d6 with fp adapters and a small curated dataset, with two
characterised failure modes (template-bleed and Moscow → Houston
substitution) that point at concrete v3 directions.

Bonsai-LoRA priority is preserved as the next-level arm of the
L1-d6 vs L1-bonsai comparison — the question now is whether 1-bit
base + fp LoRA closes the remaining ~37 % all_three gap and
generalises better. v2's success strengthens the prior; the
template-bleed observation hints that *adapter shape* matters as much
as *which base it sits on*.

### Today's commits

```
d7ca704  nanochat/lora: --lora-scale runtime attenuation knob
6a9bc14  docs: L1 catastrophic-forgetting addendum — template-bleed found
9f42f3e  dev/multi_turn_eval: --lora-tag flag for catastrophic-forgetting checks
eadcfb8  docs: L1 v2 addendum — Q+K+V+O at rank 16 lands 0/30 → 19/30 all_three
614890c  docs: HTML narrative for L1 — fifth lever moves the metric
fa1e921  docs: L1 persona-LoRA verdict — 0/30 → 4/30 all_three on held-out (v1)
6fa5b34  dev: persona-retention eval — base vs base+LoRA on held-out personas
da67b49  dev: persona-retention curator via claude CLI (sibling pulse pattern)
f75a90d  chat_sft_lora + --lora-tag inference: end-to-end LoRA training pipeline
79e4b3c  nanochat/lora: LoRA wrapper, apply walker, state-dict round-trip + tests
```

### Next session pickup (revised after 2026-05-07)

The L1 arc is wrapped on the d6 fp side. Three live directions:

1. **L1 v3 — diverse training mix.** Targets template-bleed at the
   training-time root cause. ~$2 curation (re-use the curator with a
   different system prompt, ~80 % persona-retention + ~20 %
   SmolTalk-flavoured chit-chat where the right answer is *not*
   "You're [X]") + ~5 min training + ~5 min eval. Cheapest probable
   path to a default-on-quality LoRA without the trade-off.

2. **Moscow → Houston cosine-NN probe.** Run `dev/cosine_nn_probe.py`
   on `d6_baseline_modern_sft + d6_l1_persona_lora_v2`'s embeddings
   for Russian/Moscow-cluster tokens. Tests whether the LoRA shifted
   the city-cluster geometry or whether this is decoding-loop noise.
   ~30 min compute. Intrinsically interesting follow-up to yesterday's
   diagnostic.

3. **Bonsai-LoRA Phase 2 infrastructure.** ~1-2 days. Bonsai 1-bit
   forward + fp LoRA on top, tokenizer-compatibility investigation,
   QLoRA-style flow. Now strengthened — d6 LoRA demonstrably moves
   persona-retention; the question is whether 1-bit base + fp LoRA
   closes the remaining gap and generalises better.

**Default if no clear appetite next session:** read
`docs/lora_l1_persona_2026-05-07.html` (full L1 arc with all three
addenda) and pick whichever direction sounds appealing. The
infrastructure is in place; running v3 from end to end takes ~15
minutes wall.

### Postscript — free-chat web smoke caught a wider template-bleed

After committing the day, smoked the v2 LoRA through `scripts/chat_web`
at scale=0.5 to verify end-to-end web wiring. Engine + streaming + LoRA
all worked; the persona-recall test ("Hi I'm Alex…" → recall) returned
*"You're Alex, a software engineer at a small startup."* cleanly.

But on a free-chat session ("hi there" → "What's the capital of
France?" → "Why is the sky blue?" → "Who are you?"), the LoRA at
scale=0.5 produced:

```
"Hello! How can I help you with that?"
"You can find the capital of France."
"You can find the sky blue by the trees."
"You can find the sky blue in France!"
```

So **template-bleed at scale=0.5 isn't gone, it shape-shifted.** The
"You're [X]" stamp from scale=1.0 isn't there, but the LoRA emits a
different reflexive opener (*"You can find [thing]..."*) that also
doesn't actually answer most questions. The 7-prompt rubric I ran
didn't trigger this shape — likely because rubric T-finals are mostly
recall-shaped or recap-shaped, while free-chat T-finals are
question-shaped, and the LoRA picks a different attenuated template
on each shape.

**Updated honest verdict on the scale knob**: the eval-rubric "no
bleed at scale=0.5" claim was rubric-narrow. On free chat the bleed
is still present at all scales > 0; it just shifts shape under
attenuation. Practical inference recipe until v3 lands:

- `--lora-scale 1.0` only for persona-recall demos where the template fits
- `--lora-scale 0.0` (or no `--lora-tag`) for general chat
- No good middle ground exists without v3 (dataset diversity)

**This sharpens the v3 case** — dataset diversity isn't optional,
it's the actual fix. The runtime knob is useful for A/B and for
turning the LoRA off in mixed-traffic deployments, but it can't fix
the shape problem at training time.

**Standing operational rules** unchanged: branch is local-only; no
pushing or PR creation without explicit per-task approval.

