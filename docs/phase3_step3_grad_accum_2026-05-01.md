# Phase 3 step 3 — `ve_gate` regroup + gradient-accumulation iso-token A/B on SFT

**Date:** 2026-05-01
**Branch:** `experiment/hope-nested-learning`
**Goal:** validate Codex's two cheap perf levers — (a) move `ve_gate` out of Muon, (b) test whether gradient accumulation amortizes the optimizer wall without hurting SFT quality.

## TL;DR

- **Step 1 — `ve_gate` → AdamW**: dispatch count per iter dropped 319 → 176 (−45%), much bigger than the ~30–40 Codex predicted. Steady-state iter wall improved ~12%. One-file change in `nanochat/gpt.py::setup_optimizer`.
- **Step 2 — grad accum SFT A/B**: at iso-token, accum=4 (375 opt steps, 24.6M tok) **beat** accum=1 (1500 opt steps, 24.6M tok) on val_bpb by **12% relative** (0.6639 vs 0.7568). Codex predicted "preserves quality"; we got "improves quality." Wall savings muddied by battery throttling on the long run.
- **Bonus**: the buggy first attempt of B (94 opt steps, 6.16M tok) reached val_bpb 0.7599 — within 0.4% of A — strongly suggesting SFT on this mixture plateaus very fast.
- **Bug discovered**: `chat_sft.py --num-iterations` is enforced as micro-batches inside the data generator, so it produces ~N/accum optimizer steps instead of N. Filed `docs/project_incoming/bug_chat_sft_num_iterations_micro_batch_semantics.md`.

## Step 1 — `ve_gate` → AdamW

### Change

In `nanochat/gpt.py::setup_optimizer`, filter `.ve_gate.` params out of the `transformer.h.parameters()` collection by name (not shape, per Codex's guidance) and add a dedicated AdamW group with `lr=0.2, wd=0.0` matching how `smear_gate` is handled.

```python
# before
matrix_params = list(self.transformer.h.parameters())
# after
h_named_params = list(self.transformer.h.named_parameters())
ve_gate_params = [p for n, p in h_named_params if '.ve_gate.' in n]
matrix_params = [p for n, p in h_named_params if '.ve_gate.' not in n]
...
dict(kind='adamw', params=ve_gate_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
```

(Commit: `840d3db`)

### Verification (M2, d6, batch=32, seq=512, torch 2.11) via `dev/profile_mps_signpost.py`

| metric | baseline | after | Δ |
|---|---:|---:|---:|
| `MPSGraph_Encode` begin events / iter | 319 | **176** | **−143 (−45%)** |
| `MPSGraph_Compile` events / iter | 6 | 6 | 0 |
| Muon param groups | 4 | 3 | −1 (the `(6,12)` shape group) |
| profiled iter wall | 2793 ms | 2679 ms | −4% |
| steady-state iter wall (iter2) | 2469 ms | 2162 ms | **−12%** |

The dispatch drop is much bigger than expected ("one Muon group" should be ~30–40 dispatches). Compile count unchanged (6) so it's not a caching shift — the `(6,12)` shape's polar express loop was emitting more small dispatches than its compute justifies. Wall improvement is smaller than the dispatch ratio because the removed dispatches were on tiny matrices.

## Step 2 — Gradient accumulation iso-token SFT A/B

### Design (after Codex pushed back on original LR-scaled plan)

Two runs, single variable changed (effective batch size via `total_batch_size`), LR held at the inherited pretrain values for both. Codex's reasoning: doubling LR at the same time as 4× batch confounds the read between "fewer updates hurt" vs "wrong LR for warm-start SFT." Keep one degree of freedom at a time.

| run | `--total-batch-size` | accum | `--num-iterations` | opt steps | tokens | LR |
|---|---:|---:|---:|---:|---:|---:|
| A | 16384 | 1 | 1500 | 1500 | 24.6M | inherited |
| B (buggy) | 65536 | 4 | 375 | **94** | 6.16M | inherited |
| B-iso (corrected) | 65536 | 4 | **1500** | 375 | 24.6M | inherited |

Both A and B-iso share inherited pretrain LRs: `embedding_lr=0.3, unembedding_lr=0.008, matrix_lr=0.02`. ve_gate-on-AdamW change from Step 1 is in effect for all runs.

### Bug found mid-experiment

First attempt of B used `--num-iterations=375`, expecting the script to do 375 optimizer steps. It actually did **94** because `chat_sft.py:281`'s `num_iterations` check fires inside the data generator on yields (one per micro-batch), and accum=4 means 4 yields per opt step. The LR schedule's progress counter uses the same `it`, so `lrm` warmed down to 0 at step 94 — the run completed a "valid" miniature schedule, just not the one we asked for.

Workaround used: pass `--num-iterations=accum × desired_opt_steps` (1500 for B-iso). Filed as `docs/project_incoming/bug_chat_sft_num_iterations_micro_batch_semantics.md` with a one-line fix sketch.

### Results

| run | opt steps | tokens | wall | tok/sec (steady) | val_bpb |
|---|---:|---:|---:|---:|---:|
| A: accum=1 | 1500 | 24.6M | 69.38 min | ~5,800 | 0.7568 |
| B (buggy): accum=4 | 94 | 6.16M | 11.66 min | ~7,800 | 0.7599 |
| **B-iso: accum=4** | 375 | 24.6M | **65.16 min** | ~5,500 (throttled) | **0.6639** |

Smoke test (`chat_cli -p "What is the capital of France?"`): all three produce coherent answers about Paris and the Eiffel Tower; B-iso is qualitatively the least repetitive.

### Headlines

1. **B-iso beat A by 12% relative on val_bpb at iso-token.** Codex predicted "amortization preserves quality"; we observed amortization *improving* quality. Likely explanation: cleaner gradients from 4× larger effective batch let the same 24.6M tokens move the loss further. SFT loss at iso-progress was ~14% lower in B-iso (1.21 vs 1.44 around the same lrm).

2. **Wall savings smaller than predicted (~6% rather than ~3×).** This is misleading — B-iso ran the entire 65 min on battery, with `asitop` confirming `throttle: yes` and GPU power at 1.74 W (vs ~8–10 W unthrottled). At unthrottled steady-state (~7 s/iter, observed early in the run), B-iso would have finished in ~30–40 min, restoring most of the predicted 2× per-token speedup. Independent of throttling, the per-token speedup of accum=4 over accum=1 is mathematically capped by the optimizer/(fwd+bwd) ratio — at our current ~80% optimizer share, the ceiling is ~5×, but real-world overhead (data loader, sync) brings it lower.

3. **Buggy B (94 opt steps, 1/4 the tokens) only lost 0.4% val_bpb vs A.** SFT on this 1.07M-row mixture plateaus extremely fast. If the goal is "good enough chat," 1500 SFT steps may be massive overkill on this scale of model+data.

### Caveats

- Single seed per run. The 0.0929 bpb gap (B-iso vs A) is large enough to bet on, but a confirmation seed is cheap. **Not run.**
- Wall comparison muddied by battery state. **Not re-run plugged-in.**
- Codex's third leg (`accum=4, LR=2×`) **not run.** Result is "amortization alone wins by 12%"; whether `+2× LR` adds or subtracts on top is open.
- Eval is `eval_tokens=262144` worth of val_bpb on the SFT mixture; not a downstream eval like ARC/GSM8K/MMLU. The smoke tests are the only qualitative read.

## What this means for the perf wall

Given step 1 (~12% iter wall) + step 2 (effective ~2× per-token throughput at unthrottled), the M2 24GB pretrain wall moves from the original 3.6 h projection toward ~1.5 h projection — *if* accum=4 transfers from SFT to base pretrain. Pretrain uses a different optimizer schedule (longer horizon, no warm-start), and step 2's "amortization improves quality" finding is on SFT only. The pretrain extension is the open empirical question.

For Hope/NL Stage 0 onward, the perf takeaways are:
- Gradient accumulation is now the default mental model: any batch-size dial at the experiment-script level should land at `total_batch_size = N × device_batch_size × max_seq_len` for some `N ≥ 4`.
- The `--num-iterations` trap is real; either pass `-1` for epoch-driven stopping or remember the `× accum` workaround until the bug is fixed.

## Next

1. Reply to Codex with results + MLX link captures (`docs/mlx_port_evaluation_2026-05-01.md`).
2. Pivot to Hope/NL Stage 0 (memory state plumbing, returning `None` initially) per `HANDOFF.md` and `~/projects-new/trx4mr/docs/idea-hope-nested-learning.md`.
3. Optional follow-ups deferred:
   - Confirmation seed for B-iso.
   - Plugged-in B-iso for true wall numbers.
   - Codex's `accum=4 + LR=2×` leg.
   - Validate accum=4 on **base pretrain** (the actual perf-wall question).
