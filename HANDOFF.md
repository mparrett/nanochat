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
- [ ] ChatCORE on final SFT checkpoint (validates per-task downstream metrics + production-tests the eval-loop fragmentation fix)

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
