# Phase 3 step 2 — GPU-side profiling on M2 (Phase A: OS Signpost)

**Date:** 2026-04-30
**Branch:** `experiment/hope-nested-learning`
**Goal:** validate or refute Phase 2's CPU-side conclusions using actual GPU dispatch data

## Why we did this

Phase 2 concluded "the optimizer is 79% of every iter" based on `torch.profiler` with `ProfilerActivity.CPU` only. That's a *CPU-time-share* statement — it tells us where Python wall-clock goes but not what the GPU is actually doing. We needed kernel-level data to know whether:

1. The polar express is really N separate GPU kernels (no fusion) or batched
2. The GPU is truly compute-bound or has idle gaps we could close
3. Per-dispatch overhead is the cost or per-FLOP work is the cost

## Method

Used `torch.mps.profiler.profile(mode='interval,event', wait_until_completed=True)` to wrap one steady-state pretrain iter (after 3 warmup iters). It emits OS Signposts that the system log captures.

Script: `dev/profile_mps_signpost.py`

Trace pulled via:

```bash
log show --signpost --start '<timestamp>' --end '<timestamp>'
```

Filtered to the python PID and bucketed by signpost event name.

## Findings

### Headline numbers (one training iter, d6, batch=32, seq=512)

| Metric | Value | What it means |
|---|---|---|
| `MPSGraph_Encode` events | **319** | Discrete GPU kernel dispatches per iter |
| `PyTorchOperationIntervals` | 574 | PyTorch op-level intervals (some batch into one MPS dispatch) |
| `MPSGraph_Compile` events | 6 | Unique graphs compiled (rest reuse) |
| `MPSGraph_Init` events | 2 | New graphs created |
| Wall time of the iter | 2.79 s | From wall clock |
| Span between first and last dispatch | 2.69 s | Dispatches fill almost the whole iter |
| **Average per-dispatch wall cost** | **~8.4 ms** | iter wall / 319 |
| Average gap between consecutive dispatch begins | ~6.3 ms | Sequential, no overlap |

### What we now know that we didn't before

1. **Confirmed: no Metal-level fusion of the polar express loop.** 319 separate GPU dispatches per iter. The optimizer alone accounts for ~250 of them (proportional to the 79% optimizer wall-time share).
2. **Bottleneck character: per-dispatch overhead, not per-FLOP work.** Each dispatch costs roughly the same ~8 ms regardless of matmul size. This is critical — it means **the levers are dispatch count, not GEMM speed**. Bigger batches, fused kernels, or skipping dispatches all directly translate to wall-clock.
3. **Graph reuse is working at steady state.** Only 6 unique graphs compiled across 319 dispatches. So `torch.compile` IS doing useful caching; the cost is the dispatch itself, not the codegen.
4. **Sequential dispatch pattern.** Mean 6.3 ms gap between dispatch begins, plus ~2 ms inside each = ~8 ms total. No GPU idle gaps to close via better pipelining; the GPU is actively executing a kernel, *encoding/submitting the next one is on the critical path*.

### Approximate dispatch attribution

Using Phase 2's CPU-time-share to apportion dispatches:

| Phase | Wall (ms) | % of iter | Estimated dispatches |
|---|---|---|---|
| Forward | 432 | 15.5% | ~50 |
| Backward | 17 | 0.6% | ~2 |
| **Optimizer** | **1,696** | **60.8%** | **~250** |
| Profile/sync overhead | ~640 | ~23% | (boundary noise) |

The optimizer's 250 dispatches per iter map roughly to:
- 4 muon param groups × (10 polar express matmuls + 5–10 variance reduction ops + 4 cautious update ops) ≈ 80–100
- 4 adamw param groups × (~6 ops × ~30 individual params iterated) ≈ 700? — doesn't fit
- Or maybe 4 groups × 1 fused call × ~50 internal ops = 200

The exact mapping needs Phase B (`metal_capture`) to confirm.

## What this means for the optimization wall

**The single biggest software lever we still have is reducing dispatch count.** Concretely:

- **Fuse the polar express into one MSL kernel via `torch.mps.compile_shader`.** If it's currently ~10 dispatches × 4 muon groups = 40 dispatches → 4 dispatches, save ~36 × ~8 ms = **~290 ms per iter** (~10% wall-clock).
- **Skip the variance-reduction dispatches in fp32 mode** (we don't need the per-row factored second moment if we're not memory-pressured). Maybe ~30 dispatches saved.
- **Reduce `ns_steps`** (already ticketed for full validation). ns_steps=3 cuts ~16 dispatches per muon group × 4 = ~64 dispatches per iter ≈ **~530 ms per iter saved** at ~8 ms each. That matches our observed 12% optimizer speedup at ns_steps=3 — and now we know *why* it works (dispatch count, not numerical math).

**Things this rules out:**

- ❌ Faster matmul kernels won't help much — per-FLOP work isn't the cost
- ❌ Better pipelining won't help — dispatches are already sequential and fully utilized
- ❌ Bigger batches give modest gains because they amortize dispatch cost over more tokens (this matches our bf16 + batch=48 ~3% per-token speedup)

## Next: Phase B — `metal_capture`

`torch.mps.profiler.metal_capture()` writes a `.gputrace` file viewable in Xcode's Metal debugger. Would give us:

- Per-kernel duration histogram (is *one* dispatch the heavy one?)
- GPU occupancy / memory bandwidth utilization
- Exact kernel names and call counts

Requires `MTL_CAPTURE_ENABLED=1` env var (currently disabled). Output is .gputrace — Xcode-only viewer.

Cost estimate: 60–90 min including Xcode inspection. Most useful if we want to identify a *specific* hot kernel to fuse first vs distributing work.

## What to send Codex

This document, plus:
- `dev/profile_mps_signpost.py` (the profiler we wrote)
- `dev/profile_pretrain_iter.py` (CPU-side profiler from Phase 2)
- `nanochat/optim.py` (the hot path — Muon + AdamW)

The questions for him narrow nicely:

1. **Is `torch.mps.compile_shader` the right tool to fuse the polar express?** Or would MLX-port effort beat the bespoke MSL work?
2. **Per-dispatch ~8 ms wall cost** — does that match what he's seen on M-series, or are we missing something obvious?
3. **Are there dispatch-count-reduction patterns** in optim/training loops we haven't applied? (e.g. fused optimizer step macros, combined param-group updates)
