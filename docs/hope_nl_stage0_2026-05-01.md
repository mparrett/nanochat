# Hope/NL Stage 0 — memory_state plumbing

**Date:** 2026-05-01
**Branch:** `experiment/hope-nested-learning`
**Commit:** `d2bbc0e`
**Goal:** thread an optional `memory_state` parameter through `forward()` so that Stage 1+ can wire actual mutable per-sequence state into one block without touching any other call site.

## TL;DR

- Added `memory_state=None` arg to `GPT.forward()` and `Block.forward()`, plus a `GPT.reset_memory()` helper.
- **Zero behavior change** when no caller passes `memory_state` (the Stage 0 default everywhere). Loss and logits are bit-identical to pre-Stage-0 across the with/without paths.
- Conditional tuple return at the `GPT.forward()` boundary keeps all ~11 existing call sites (`base_train`, `chat_sft`, `chat_eval`, `chat_rl`, `engine.py`, `loss_eval`, `core_eval`, dev profilers, tests) unmodified.
- New test file `tests/test_memory_plumbing.py` pins the contract — 6 tests, all green.
- Total wall: ~45 min, well under the 2.5–4h estimate (the conditional-tuple choice avoided the ~11 caller updates that an always-tuple approach would have required).

## API surface

### Block.forward — always-tuple

```python
def forward(self, x, ve, cos_sin, window_size, kv_cache, memory_state=None):
    x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
    x = x + self.mlp(norm(x))
    return x, memory_state  # Stage 0: passthrough; Stage 1+ memory blocks override
```

Inner contract is consistent always-tuple. Stage 0 blocks pass `memory_state` through unchanged. Stage 1+ memory-bearing blocks will read the entry, update it, and return the new state.

### GPT.forward — conditional-tuple

```python
def forward(self, idx, targets=None, kv_cache=None, memory_state=None, loss_reduction='mean'):
    ...
    if memory_state is None:
        return out                        # original behavior; bit-identical
    return out, new_memory_state          # opt-in path
```

Outer contract is backward-compatible. No existing caller had to change.

### GPT.reset_memory()

```python
def reset_memory(self):
    return [None] * self.config.n_layer   # Stage 0: no memory blocks; Stage 1+ initializes per-block
```

Canonical way to opt into memory threading without knowing the layer count or how many blocks carry memory.

## Why this design

### Always-tuple at GPT boundary (rejected)
- Pro: most explicit return type, no conditional shape.
- Con: every existing caller (training loops, evals, inference, dev profilers, tests) must `loss, _ = model(x, y)` instead of `loss = model(x, y)`. ~11 call sites to update across the codebase, plus user-facing tools (`chat_cli`, `chat_web`).
- Verdict: too much blast radius for a refactor whose entire payload is "function accepts a None".

### Block-internal-only (rejected for Stage 0+)
- Pro: zero API change; Stage 1 just makes one block stateful and calls `model.reset_memory()` between sequences.
- Con: no path to externalize state for inspection, checkpointing, multi-process sharding, or persistence across calls — all of which Hope/NL Stages 2–6 likely need.
- Verdict: solves Stage 1 but punts on Stage 2+, which would re-trigger the same plumbing decision later.

### Conditional-tuple at GPT, always-tuple at Block (chosen)
- Existing callers see no API change at all.
- Stage 1+ callers opt into threading by passing `memory_state` and unpacking the tuple.
- Block contract is uniform always-tuple — clean inner abstraction, no per-block dispatch logic in `GPT.forward()`.
- Externalizes state from day one (lives in the Python list returned by `reset_memory()`), which keeps Stage 2+ doors open.

## What's load-bearing

The Stage 0 guarantee that everything else hinges on:

> When `memory_state is None` (the default for every existing caller), `GPT.forward()` returns exactly what it returned pre-Stage-0, computed on exactly the same code path.

This is what `tests/test_memory_plumbing.py::test_loss_bit_identical_with_and_without_memory_state` and `test_logits_bit_identical_with_and_without_memory_state` pin via `torch.equal(...)`.

If this ever breaks, all existing training/eval/inference is silently affected. The tests fail loudly via bit-equality (no tolerance).

## What this *doesn't* do yet

- No block actually carries memory. The `memory_state` slots are passed through untouched.
- No training/inference loop opts in. Existing pipelines pretend Stage 0 didn't happen.
- No `reset` cadence wired anywhere. `reset_memory()` is just a constructor for empty state.
- No backprop boundary policy decided (detached vs. through-memory updates).

These all become relevant in Stage 1 when one block actually starts using its slot.

## Test coverage (`tests/test_memory_plumbing.py`)

| test | what it pins |
|---|---|
| `test_reset_memory_returns_list_of_nones` | shape contract of the helper |
| `test_forward_without_memory_returns_single_tensor` | pre-Stage-0 callers see exactly the original return type |
| `test_forward_with_memory_returns_tuple` | opt-in callers get `(out, new_state)` |
| `test_loss_bit_identical_with_and_without_memory_state` | **load-bearing**: training loss is unchanged |
| `test_logits_bit_identical_with_and_without_memory_state` | **load-bearing**: inference logits are unchanged |
| `test_forward_rejects_wrong_length_memory_state` | API misuse caught at the boundary, not silently |

Stage 1 will add a memory-bearing block; the bit-identical tests should still pass for every block-position that doesn't carry memory. The opt-in path will diverge meaningfully from the no-state path once memory is wired (and that divergence is the point).

## Verified

- Full pytest suite: **19 passed, 10 skipped** (same 10 platform-gated skips as before).
- `dev/profile_mps_signpost.py` runs the real d6 base checkpoint through warmup + profiled iter at the same wall (~2.5 s/iter steady-state) as pre-Stage-0.

## Files changed

| file | change |
|---|---|
| `nanochat/gpt.py` | +46/-8: Block.forward signature + always-tuple return; GPT.forward signature + conditional tuple return + per-block state threading; new `reset_memory()` method |
| `tests/test_memory_plumbing.py` | new file: 6 contract tests |

## Next

Stage 1: pick one block (per the trx4mr ticket: a mid-stack block, swapping its FFN), give it a fixed-η/α linear-attention memory module (Eq 15 from the Hope paper) that reads/writes its `memory_state[i]` slot. Reset per sequence. Backprop policy: detached fast-weights to start. Pass/fail bar: does the modified d6 train, and is the loss curve reasonable vs. the unmodified baseline (val_bpb 1.174)?

Stage 1 will exercise the plumbing for the first time. If the tests in `test_memory_plumbing.py` start failing for the memory-bearing block position, that's a good failure — it means the slot is doing something. We'll add Stage-1-specific tests there.
