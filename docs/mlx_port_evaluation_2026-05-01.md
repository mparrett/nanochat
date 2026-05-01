# MLX port evaluation — running notes

**Started:** 2026-05-01
**Status:** Living doc. Revisit when the conditions at the bottom flip.

## Why we're considering MLX

After Phase 2/3 perf work on M2 24GB:

- The optimization wall is **dispatch count, not GEMM speed**. 319 GPU dispatches / iter at ~8 ms each (`docs/phase3_gpu_profiling_2026-04-30.md`).
- Metal has no Triton-equivalent for fusing tight matmul loops. PyTorch MPS gets graph-level fusion via MPSGraph but doesn't fuse the polar express NS loop into one kernel.
- Two software levers remain: (a) write custom MSL kernels via `torch.mps.compile_shader`, (b) port to MLX, which has training-graph compile (`mx.compile`), lazy eval, and unified-memory primitives.

Codex's framing (consult on 2026-04-30): "1–2 days first for an optimizer-only MLX Muon benchmark on fake d12 tensors. If that doesn't beat PyTorch meaningfully, don't port yet."

## The case AGAINST an MLX port (current evidence)

### 1. Raw GEMM is *slower* on MLX than PyTorch MPS

[ml-explore/mlx#3196](https://github.com/ml-explore/mlx/issues/3196) — opened 2026-03 by Anemll, motivated explicitly by NanoChat training. M5 hardware with Neural Accelerator (NA) support enabled.

| shape | PyTorch MPS | MLX | ratio |
|---|---|---|---|
| 1280×1280 bf16 matmul | 5.11 ms | 6.19 ms | **1.21× slower** |
| 1280×320 bf16 matmul | 0.82 ms | 1.02 ms | 1.24× slower |
| 5120×1280 bf16 matmul | 15.19 ms | 17.02 ms | 1.12× slower |

Average ~1.19× slower across NanoChat-relevant Muon shapes. The reporter notes the `addmm` *fusion* is faster than PyTorch now — the gap comes from plain `X @ X.T`, which is exactly what the polar express does most of.

Status: OPEN, MLX maintainer (jagrit06) said "I will get to addressing this issue" but nothing has landed. Improvement possible over time, but not banked.

**Implication:** Porting Muon as-is to MLX would *regress* iter throughput by ~20% before any compile/fusion wins. Any MLX-port pitch has to assume we *also* rewrite the polar express for MLX's compile graph, not just translate it.

### 2. The dispatch-count problem doesn't disappear in MLX

MLX's `mx.compile` and lazy eval *can* fuse, but only on graphs where the lazy evaluation has visibility. The polar-express loop already runs as Python-level for-loops with intermediate tensors fed back in. If we want fusion, we have to express it as one compiled function — same intellectual lift as writing a fused MSL kernel via `torch.mps.compile_shader`. The MLX win comes only if MLX's compile is *better* than PyTorch + custom MSL at fusing this specific pattern, which we have no evidence of yet.

## The case FOR an MLX port (current evidence)

### 1. MLX upstream accepts custom Metal kernels

[ml-explore/mlx#3328](https://github.com/ml-explore/mlx/pull/3328) — TurboQuant SDPA: native Metal kernel for KV cache compression, 1.5–4.9× SDPA speedup at 1K–16K context on M4 Pro.

Status: OPEN, mergeable: CONFLICTING (not landed). Standalone repo at [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx).

**Implication:** MLX is not a closed garden. A fused polar-express kernel could land upstream if it's good. Cuts against "MLX is just lazy-eval glue, no real kernel work happens there."

### 2. Unified-memory model fits Apple Silicon better than PyTorch MPS

PyTorch MPS still treats the GPU as a separate device with explicit `.to(device)` and `torch.mps.synchronize()`. MLX's unified-memory model removes the host/device boundary, which on Apple Silicon is a software fiction anyway. This *might* save dispatch overhead for many-small-tensor workloads, but we have no measured number for our workload.

### 3. Training-graph compile

MLX claims full training-graph compile including optimizer state. We've never seen it close the dispatch-count gap on a Muon-like loop. This is the experiment Codex flagged as the prerequisite.

## What would it take to flip the verdict

Concrete prerequisites for "yes, port":

1. **MLX matmul reaches ≥ PyTorch MPS speed for the d6/d12 shapes.** Track #3196.
2. **An optimizer-only MLX Muon benchmark on fake d12 tensors beats PyTorch by ≥ 1.5× on dispatch count or wall.** This is the 1–2 day experiment Codex outlined. We haven't run it.
3. **`mx.compile` demonstrably fuses the polar express loop into a single kernel** (vs PyTorch's 30+ dispatches per group).

Until at least (2), an MLX port is speculative. Until (1), it's actively worse for steady-state throughput.

## Inference-side angle (separate calculus)

If/when nanochat inference gains long-context use cases:
- TurboQuant-style 3-bit KV cache compression buys 4.6× memory bandwidth and ~5× SDPA speedup at 16K context.
- That stays in Python-MLX land — no Hope/NL coupling.
- Probably a stronger MLX argument than training is, but unrelated to our current bottleneck.

## Decision today

Don't port. Continue on PyTorch MPS for Hope/NL Stage 0–6. Park MLX as a "revisit if (1) lands or someone else publishes the optimizer-only benchmark" project. Re-check MLX matmul perf after each MLX release that touches MPS / Metal / NA paths.

If we want a defensible MLX experiment, the cheapest one is the **1–2 day Muon-only benchmark** Codex sketched: write the polar express in MLX with `mx.compile`, time against the current `nanochat/optim.py` muon path on identical fake d12 tensors. If that beats PyTorch by ≥ 1.5×, the port becomes plausible. If not, parked.

## References

- [ml-explore/mlx#3196 — addmm/matmul slower than PyTorch on M5](https://github.com/ml-explore/mlx/issues/3196)
- [ml-explore/mlx#3328 — TurboQuant SDPA Metal kernel](https://github.com/ml-explore/mlx/pull/3328)
- Codex's first review: `docs/project_incoming/codex-2026-04-30-2046.md`
- Our dispatch-count analysis: `docs/phase3_gpu_profiling_2026-04-30.md`
- MLX overview: [ml-explore.github.io/mlx](https://ml-explore.github.io/mlx/build/html/index.html)
- MLX compile docs: [ml-explore.github.io/mlx — compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
