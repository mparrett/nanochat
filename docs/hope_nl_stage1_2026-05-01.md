# Hope/NL Stage 1 — LinearAttentionMemory at one block

**Date:** 2026-05-01
**Branch:** `experiment/hope-nested-learning`
**Commit:** `bc54858`
**Goal:** swap the FFN in **one** block of nanochat's d6 with the simplest fast-weight memory that exercises Hope's "block carries mutable state" thesis. Verify the model trains, the loss curve is reasonable, and the architecture leaves room for Stages 2–6 to extend.

## TL;DR

- New `LinearAttentionMemory` module in `nanochat/gpt.py`, sibling of `MLP`, used in place of MLP at one configurable block.
- Implements the trx4mr Stage 1 spec exactly:
  ```
  k_t, v_t, q_t = W_k x_t, W_v x_t, W_q x_t
  M_t = M_{t-1} + v_t k_t^T            (alpha=1, eta=1, fixed)
  o_t = M_{t-1} q_t                    (read uses *previous* memory)
  y_t = W_o o_t
  ```
- Uses the **parallel form** `o_t = sum_{i<t} (k_i · q_t) v_i` instead of a Python token loop, per Codex's "avoid token loops from day one" warning. Equivalent to the recurrence above when `alpha=1, M_0=0`. Stages 2–3 will revisit when learned `alpha`/`eta` make the parallel form harder.
- Selected via `GPTConfig.hope_memory_layer: int | None` (default `None` = baseline). CLI: `--hope-memory-layer=N` on `base_train.py`.
- Sanity check: 50-iter d6 pretrain stub trains cleanly. Loss 10.40 → 7.44 monotonic, no NaN, ~1.7 s/iter steady-state (no wall regression vs MLP baseline at the same dimensions).
- 4 new contract tests, full suite: **23 passed, 10 platform-skipped**.

**Pass/fail bar from the trx4mr Stage 1 spec** ("does it train, and is the loss curve reasonable vs unmodified baseline") **met for the short stub.** The full 5000-iter iso-token comparison vs the d6 baseline (val_bpb 1.174) is the next milestone and is deferred to a separate session.

## Architecture

### LinearAttentionMemory module

```python
class LinearAttentionMemory(nn.Module):
    def __init__(self, config, d_mem=None):
        super().__init__()
        d_model = config.n_embd
        self.d_mem = d_mem if d_mem is not None else d_model
        self.W_k = Linear(d_model, self.d_mem, bias=False)
        self.W_v = Linear(d_model, self.d_mem, bias=False)
        self.W_q = Linear(d_model, self.d_mem, bias=False)
        self.W_o = Linear(self.d_mem, d_model, bias=False)
        self._scale = self.d_mem ** -0.5

    def forward(self, x):
        T = x.size(1)
        k = self.W_k(x); v = self.W_v(x); q = self.W_q(x)
        scores = torch.einsum('btd,bsd->bts', q, k) * self._scale
        mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril(diagonal=-1)
        scores = scores.masked_fill(~mask, 0.0)
        o = torch.einsum('bts,bsd->btd', scores, v)
        return self.W_o(o)
```

`d_mem` defaults to `d_model` (= 384 for d6). Parameter count: `4 * d_model * d_mem = 4 * 384^2 ≈ 590K` per memory block, vs `8 * d_model^2 ≈ 1.18M` for the MLP it replaces. Roughly half the params; not matched on purpose. Tunable via the constructor if we want exact param parity later.

### Why this exact form

The trx4mr ticket gives the recurrence; the parallel form is a derived optimization. With `alpha=1`, `eta=1`, `M_0=0`:

```
M_t = sum_{i<=t} v_i k_i^T
o_t = M_{t-1} q_t = sum_{i<t} v_i (k_i · q_t) = sum_{i<t} (k_i · q_t) v_i
```

Three concrete decisions that follow:

1. **Strict-causal mask (`diagonal=-1`).** The spec reads `o_t = M_{t-1} q_t` — token `t` reads memory built from tokens `1..t-1`, not including itself. So the mask zeros the diagonal *and* the upper triangle. This makes the first token's output exactly zero, matching the recurrence with `M_0=0`.

2. **`1/sqrt(d_mem)` scaling on `(q · k)`.** Standard attention-style stability scaling. With `d_mem=384` and KVQ inits at `Uniform(-s, s)` where `s = sqrt(3/d_model)`, individual `q · k` values can otherwise have magnitude `~sqrt(d_mem)`. Without scaling, post-softmax attention masks this; we don't have a softmax, so we scale up front.

3. **`einsum` parallel form, not a Python `for t in range(T)` loop.** Codex's Phase 2 advice was explicit: token loops would "be catastrophic on MPS." The parallel form is `O(T^2)` which at `T=512` is `262K` ops per batch element — cheaper than the `512 * (a few small matmuls)` a token loop would dispatch.

### Init scheme

```python
elif isinstance(block.mlp, LinearAttentionMemory):
    torch.nn.init.uniform_(block.mlp.W_k.weight, -s, s)
    torch.nn.init.uniform_(block.mlp.W_v.weight, -s, s)
    torch.nn.init.uniform_(block.mlp.W_q.weight, -s, s)
    torch.nn.init.zeros_(block.mlp.W_o.weight)
```

KVQ get the same `Uniform(-s, s)` init as attention's KVQ projections. **W_o starts at zero** by deliberate choice — same convention as `attn.c_proj` and `mlp.c_proj` in the rest of nanochat. The block contributes exactly zero to the residual at step 0, so the modified architecture is **bit-identical to the baseline at initialization**. Divergence kicks in only as W_o trains away from zero. This is a stability decision: we don't want an untrained memory block to perturb the early loss landscape.

## API plumbing

Three config touchpoints, all fully backward-compatible:

1. **`GPTConfig.hope_memory_layer: int | None = None`** — new field, default `None` preserves prior behavior.
2. **`scripts/base_train.py --hope-memory-layer=N`** (default `None`) — CLI knob that flows into `GPTConfig`.
3. **`nanochat/checkpoint_manager.py::_patch_missing_config_keys`** — fills in `hope_memory_layer=None` when loading checkpoints saved before this change. The existing d6 base checkpoint (`val_bpb 1.174`) loads cleanly without modification.

`Block.__init__` does the swap:

```python
if config.hope_memory_layer is not None and config.hope_memory_layer == layer_idx:
    self.mlp = LinearAttentionMemory(config)
else:
    self.mlp = MLP(config)
```

Block forward stays unchanged — both `MLP` and `LinearAttentionMemory` expose the same `forward(x) -> y` shape.

### Stage 0 plumbing usage

Stage 1 doesn't actually use the `memory_state` parameter we plumbed in Stage 0. The reason: with per-batch reset (memory initialized to zero at the start of every forward) and the parallel form, all state lives inside one `forward()` call and never persists across calls. The `memory_state` slot is reserved for Stage 4+ when blocks may need to externalize state for cross-call persistence, multi-process sharding, or document-ordered streaming.

This is fine — Stage 0 was always meant to provide the API surface, not force every later stage to use it. The contract tests in `tests/test_memory_plumbing.py::test_*_bit_identical_*` still pass with `hope_memory_layer=None`, confirming the plumbing remains a no-op for the baseline.

## What we tested

| test (in `tests/test_memory_plumbing.py`) | what it pins |
|---|---|
| `test_stage1_memory_block_present_at_correct_layer` | the swap lands at the configured `layer_idx`; other layers stay `MLP` |
| `test_stage1_loss_differs_from_baseline_after_perturbation` | confirms the architectures are loss-identical at init by design (W_o=0) and diverge once W_o is perturbed (the forward path actually does something different) |
| `test_stage1_forward_runs_without_error_and_is_deterministic` | repeated forward on the same input gives bit-identical output |
| `test_stage1_backward_produces_finite_gradients` | gradients flow through the memory block; no NaN/Inf on any param |

Plus the 6 Stage 0 tests still pass with `hope_memory_layer=None`.

## Sanity stub: 50-iter d6 pretrain

```bash
python -u -m scripts.base_train \
    --depth=6 --head-dim=64 --window-pattern=L --max-seq-len=512 \
    --device-batch-size=32 --total-batch-size=16384 \
    --eval-every=-1 --core-metric-every=-1 --sample-every=-1 \
    --num-iterations=50 \
    --hope-memory-layer=3 \
    --model-tag=d6_pretrain_stub_stage1 --run=dummy
```

Memory block at layer 3 (mid-stack of 6). Plugged in.

| step | Stage 1 (mem at L3) | A1 baseline (50-step slice from earlier 200-iter stub) |
|---:|---:|---:|
| 1 | 10.396 | 10.396 |
| 10 | 10.326 | 10.328 |
| 25 | 8.765 | 9.037 |
| 40 | 7.747 | 7.588 |
| 49 | 7.439 | 7.133 |

Wall: **1.42 min** for 50 iters = **~1.7 s/iter** steady-state. No wall regression vs the MLP baseline (~2.4 s/iter on the 200-iter A1 stub) — actually slightly faster, consistent with the lower param count of the memory block.

**The step-by-step comparison is misleading** because the two runs use different `num_iterations` so the LR schedules are differently shaped (Stage 1 was warming up *and* down within 50 steps; A1 was 50/200 of the way through a 200-step schedule, still in warmup at step 25, finishing warmup at step 40). What the table actually shows is "both descend monotonically from the same init in the same general shape" — which is the falsifiable claim Stage 1's pass/fail bar makes.

What it does NOT show:
- Whether the architecture reaches a competitive val_bpb at iso-token over a full 5000-iter pretrain.
- Whether the loss-vs-tokens curve is meaningfully different from the MLP baseline.
- Whether the memory block is actually using its memory (vs degenerating to a glorified linear projection).

Those questions are deferred.

## Wall economics for the next experiment

The full Stage 1 verification is a 5000-iter d6 pretrain at the unchanged baseline recipe (accum=1, batch 16384, plugged in). Expected wall: **~3 h** (matches the unmodified baseline; per-iter wall is ~unchanged at 1.7–2.4 s).

That's the same ~3 h the unmodified d6 baseline took. We already have that baseline checkpoint at `~/.cache/nanochat/base_checkpoints/d6/model_005000.pt` with val_bpb 1.174, so the comparison is one new run away.

Output goes to a new `--model-tag=d6_stage1` so the baseline is preserved untouched.

## Caveats and known limitations

1. **alpha=1 is "no forgetting" memory.** With infinite-horizon accumulation, `M_t` grows in magnitude with `t`. The `1/sqrt(d_mem)` scaling on `q·k` provides some control but doesn't bound `M`. Stage 2 will introduce learned `alpha < 1` for actual decay.

2. **eta=1 is "fixed update strength."** Same caveat — Stage 2 will learn it.

3. **No memory normalization.** Some related work (RWKV, RetNet, Mamba) normalizes the memory state explicitly. We're not. If training instability shows up, normalization is the first thing to add.

4. **Causal linear attention has known expressivity limits.** This isn't full Hope — it's the equation 15 starting point. The interesting Hope/NL claims (CMS, self-modifying updates, M3 optimizer) live in Stages 4–6.

5. **One block out of six is a small lever.** Stage 4 adds memory at *several* layers. We picked layer 3 (mid-stack) as the cleanest single-block test; nothing principled says it's optimal.

6. **No backprop-through-memory question yet.** Because the parallel form is a single einsum, gradients flow through it in the standard way. The "fast-weight detached" question becomes meaningful only when we have an actual recurrence (Stage 2+).

## Files changed

| file | change |
|---|---|
| `nanochat/gpt.py` | +65: `LinearAttentionMemory` module, `GPTConfig.hope_memory_layer` field, conditional swap in `Block.__init__`, conditional init in `init_weights` |
| `nanochat/checkpoint_manager.py` | +3: backward-compat shim for `hope_memory_layer` in old checkpoints |
| `scripts/base_train.py` | +2: CLI flag and config wiring |
| `tests/test_memory_plumbing.py` | +96: 4 new Stage 1 contract tests |

## Next milestone

**Full d6 pretrain at iso-token vs the existing baseline.** That's the actual "is the loss curve reasonable" answer. Three possible outcomes:

- **Stage 1 within ~5% of baseline val_bpb.** Memory block works, architecture is sound, proceed to Stage 2 (learned `alpha`/`eta`).
- **Stage 1 noticeably worse (e.g. 10–30% higher val_bpb).** Architecture is plumbed correctly but the memory block underperforms an MLP at this scale. Proceed to Stage 2 anyway — learned gates may be what's missing.
- **Stage 1 NaN/diverges over the full run.** Stability work needed. The first knobs to turn would be normalization on the memory readout and bounding `M` magnitude.

Wall budget: ~3 h plugged in. Output target: `--model-tag=d6_stage1`. Keep `~/.cache/nanochat/base_checkpoints/d6/` as the untouched baseline.
