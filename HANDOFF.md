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
- [ ] Stage 0 implemented (memory_state plumbing)
- [ ] Stage 1 implemented (one block linear-attention memory)
- [ ] Stage 1 vs baseline comparison run

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

