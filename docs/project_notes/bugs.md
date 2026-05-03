# Bug Log

## 2026-05-03 - Stage 2 additive SFT hangs at step 200 (M2 24GB) [ROOT-CAUSED]

**Root cause** (confirmed 2026-05-03 with `python -u` unbuffered logging plus the new `torch.mps.empty_cache()` fix that got us *past* the save and into the actual error):

```
Error: command buffer exited with error status.
  Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

The hang was **not** the model save. It was **ChatCORE eval at step 200** running out of GPU memory on M2 24GB. `chat_sft.py --chatcore-every=200` (default) triggers a full benchmark pass (ARC-Easy → ARC-Challenge → MMLU → GSM8K → HumanEval → SpellingBee). Stage 2 additive's higher activation memory + ChatCORE's evaluation batches pushed past the GPU's allocation ceiling. Process state went U with `_MTLCommandBuffer waitUntilCompleted` because the failed command buffer never returns.

Previous runs at bs=32 and bs=16 *also* OOM'd here — we just couldn't see the error message because we lacked unbuffered logging (`python -u`) and the runs were killed before the error printed.

**Fix(es) landed**:

1. **`torch.mps.synchronize() + torch.mps.empty_cache()` before save** in `checkpoint_manager.save_checkpoint()` (commit `0379888`-ish). This is a real win — without it, the save's MPS→CPU transfer would face additional pressure. With it, step 100 save and the train+val_bpb portion of step 200 ran cleanly on Stage 2 at bs=32. Keeps benefit even though it didn't fix the ChatCORE OOM.

2. **MPS allocator metrics in wandb** (`nanochat.common.mps_metrics()`, commit `f4f5066`). Will surface allocator climb in future runs *before* the OOM hits.

**Workaround** for Stage 2 SFT on M2: pass `--chatcore-every=-1` to skip the heavy benchmark eval during training. Run ChatCORE separately after SFT completes via a dedicated eval script, or once at the final step only (`--chatcore-every=<num_iterations>`). Reducing `--chatcore-max-cat` (default -1, i.e. unbounded) would also help if ChatCORE-during-SFT is desired.

**Why pretrain @ bs=32 succeeded**: pretrain doesn't run ChatCORE. `base_train.py` has `--core-metric-every=-1` by default. Its only eval is `evaluate_bpb` which is bounded by `--eval-tokens`.

**Architectural takeaway** (for the writeup): the (B, T, T) tensor accounting in Stage 2 additive is real but not *individually* OOM-causing at our shapes. The OOM is the architecture's memory pressure *plus* ChatCORE's task-eval batches *together* exceeding the M2's GPU ceiling. Anyone running Stage 2 SFT on M2 should disable ChatCORE during training.

## 2026-05-03 - Stage 2 additive SFT hangs in Metal dispatch at step 200 (M2 24GB) [SUPERSEDED — see ROOT-CAUSED entry above]

**Issue**: `scripts.chat_sft` on a Stage 2 additive checkpoint (Hope/NL `LearnedGateLinearMemory` at one block, alongside MLP) hangs deterministically around step 184-200 on Apple M2 24GB. Process state goes to U (uninterruptible kernel sleep), `sample <pid>` shows the active thread stuck on `_MTLCommandBuffer waitUntilCompleted`. Hit three times across two batch sizes:

- `--device-batch-size=32`: hung at step ~184-200 (twice)
- `--device-batch-size=16`: hung at step ~200 — **during the save**, not the eval (eval logged val/bpb 0.7915, then step 200 model save deadlocked)

**Critical hygiene rule (from `docs/m2_pipeline_2026-04-30.md:102`)**: Before any MPS run, **always check for orphan workers**:

```bash
pgrep -lf python    # any leftover python from previous runs?
pgrep -lf wandb     # any leftover wandb-core?
```

A multiprocessing worker from an earlier debug probe stayed running for ~3h at 22% CPU after a kill, hogging the Metal context, causing the *next* SFT to hang at 0% CPU after step 1. The current Stage 2 SFT hangs are **not** explained by this — we verified no zombies present — but the rule remains load-bearing for any MPS run.

**Stage-2-specific hypotheses (empirical investigation pending)**:

1. **Activation memory peak.** Stage 2 additive forward materializes 4× (B, T, T) tensors (`scores`, `log_decay`, `decay`, `weights`) per memory-bearing block, *plus* the MLP's (B, T, 4D) since additive doesn't replace. At B=32, T=512 in bf16, that's ~130 MB extra activation memory vs Stage 1 swap (1× (B,T,T), no MLP at L3) which SFT'd successfully at the same batch size (`val_bpb 0.6712`). bs=16 halved that to ~67 MB but still hung *during the save*.

2. **Save-time MPS→CPU transfer under memory pressure.** `torch.save(model.state_dict())` requires copying all params from MPS to CPU, which triggers a Metal command buffer flush. If GPU has too many uncommitted ops or the allocator is fragmented, `_MTLCommandBuffer waitUntilCompleted` can deadlock. The eval at step 200 succeeded (logged val/bpb), then the save attempt hung — pointing at the save-specific copy as the trigger.

3. **MPS allocator fragmentation accumulates over training steps.** Pretrain @ bs=32 succeeded (5000 iters with single end-of-run save). SFT @ bs=32 fails despite shorter (375 iters) — the difference is the periodic `--save-every=100` saves at step 100, 200 etc. that exercise the MPS→CPU copy path repeatedly under accumulated fragmentation. Step 100 save succeeded, step 200 save hung — supports the "fragmentation builds up" hypothesis.

**Investigation plan** (priority order):

- Test A: reboot, baseline SFT @ bs=32, no Hope flags. If it completes, hang is Stage-2-specific. (~1h)
- Test B: Stage 2 SFT @ bs=32 with explicit `torch.mps.empty_cache()` between optimizer step and save. If completes, fragmentation is the cause. (~1h)
- Test C: Stage 2 SFT @ bs=32 with `--save-every=-1` (no intermediate saves). If completes (i.e. final save at step 375 is the only save and it succeeds), confirms intermediate-save-time pressure is unique. (~1h)
- Test D: minimal repro — load Stage 2 checkpoint, do one forward+backward, attempt `torch.save(model.state_dict(), ...)`. Iterate until hang reproduces in <5 min. (~30min)

**Workaround until root-caused**: bs=16 + `--save-every=-1` (skip intermediate saves) might run to completion. Risk: if the *final* save also hangs, we lose the entire SFT.

## 2026-04-30 - chat_sft NaN at step 4 on M2 MPS

**Issue**: `scripts.chat_sft` produces `loss: nan` deterministically at training step 4 on Apple M2 (24GB) with MPS backend. Steps 1–3 produce sensible loss values (~1.9–2.8), then step 4 onward is NaN forever. Pretrain (`scripts.base_train`) on the same device works fine for 5000 steps.

**Root Cause**: Unknown; not reproduced upstream — Karpathy's `runs/runcpu.sh` was tuned on M3 Max. Confirmed independent of:
- LR magnitude (tested 10× lower across embedding/unembedding/matrix LRs — still NaN at step 4)
- Optimizer state inheritance (`--load-optimizer=0` — still NaN at step 4)
- LR warmup (`--warmup-ratio=0.5` linear ramp — still NaN at step 4)

So NaN is structural to SFT's loss path on M2 MPS, not a hyperparameter issue. Likely candidates: SDPA on packed sequences with the SFT loss-mask layout, or a packed batch row where all targets are `-1` (ignore_index) producing 0/0 in `F.cross_entropy(reduction='mean')`.

**Solution**: Not yet fixed. Workaround for now is to skip SFT — the base model is usable via `python -m scripts.chat_cli -i base` (very undertrained, but proves the pipeline).

**Prevention / Next steps**:
- Run with `torch.autograd.set_detect_anomaly(True)` to pinpoint the op that first produces NaN.
- Check whether any micro-batch row has zero unmasked targets; if so, guard cross-entropy or filter the row.
- Try `--max-seq-len=256` to change the packing.
- File upstream issue if reproduced cleanly.

## 2026-04-30 - chat_sft bestfit dataloader locks up with over-length conversations

**Issue**: After 3–10 successful training steps with healthy loss, every subsequent step has `n_valid=0` (all targets masked). Confirmed by adding `(y != -1).sum()` instrumentation at the training loop in `chat_sft.py`. With `--max-seq-len=512 --device-batch-size=32`, our probe showed 46/49 steps fully-masked (median valid: 0%).

**Root Cause**: `sft_data_generator_bos_bestfit` in `scripts/chat_sft.py` maintains a `conv_buffer` of size 100. The greedy bestfit packer only `pop`s a conversation when it fits in `row_capacity = max_seq_len + 1 = 513`. If a conversation in the buffer has `len > 513`, it can never fit *any* row (since row_capacity is the max possible remaining). After the small conversations get drained from the buffer in the first few steps, the buffer fills permanently with over-length conversations from `refill_buffer`. Every subsequent row is fully padded with BOS (mask=0). The `consumed` counter never advances, so no progress is made — but `it` keeps incrementing, producing the appearance of training. Combined with PR #610's NaN fix returning 0 for fully-masked batches, the EMA decays to ~0 and the readout looks like fast convergence — but the model is never updated.

**Solution**: Patch `chat_sft.py` to detect buffer lockup and discard over-length convs. When `len(row) == 0` and no conv fits, all buffered convs must be `> row_capacity`. Drop them all (count as consumed) so `refill_buffer` can pull fresh data.

```python
elif len(row) == 0:
    # Fresh row and nothing fits: buffer is locked with over-length convs.
    n_dropped = len(conv_buffer)
    consumed += n_dropped * ddp_world_size
    conv_buffer.clear()
    continue  # retry; refill_buffer will repopulate
```

After this fix, n_valid ranges 16.7–36.4% per step (median 29%), 0 fully-masked steps. SFT trained from loss 1.89 → 1.40, smoke test produces coherent answers ("Paris... Eiffel Tower... Louvre").

**Prevention**: When using bestfit packing on datasets with conversations longer than the row capacity, you need an eviction policy for over-length items, otherwise the buffer locks. Worth filing upstream — current `chat_sft.py` is silently broken for users with `max_seq_len < median conversation length`.

## 2026-04-30 - PYTORCH_MPS_PREFER_METAL=1 and FAST_MATH=1 both make our workload slower

**Issue**: Tested the two MPS opt-in env vars from the PyTorch 2.11 docs as a Phase 2 perf lever. All combinations measurably slower than defaults:

| config | s/iter | vs baseline |
|---|---|---|
| baseline | 2.21 | 1.00× |
| `PYTORCH_MPS_PREFER_METAL=1` | 10.35 | **0.21× (5× slower)** |
| `PYTORCH_MPS_FAST_MATH=1` | 3.07 | 0.72× (40% slower) |
| both | 5.60 | 0.39× (2.5× slower) |

**Root Cause (hypothesis)**: For our workload (many small matmuls in Muon's polar express loop on stacked tensors), MPS Graph performs real fusion / batched dispatch that direct Metal kernels lose. `PREFER_METAL=1` bypasses Graph and forces per-op kernel launches; the dispatch overhead dominates. `FAST_MATH=1` may disqualify certain graph fusions or be tuned for fp16 rather than our fp32 path.

**Solution**: Don't set these env vars on M2 with this codebase. The MPS defaults are the right choice.

**Prevention**: When evaluating any MPS env var, measure on the actual workload — these knobs are tuned for inference-shaped workloads (single large matmul) rather than tight optimizer loops.

## 2026-04-30 - NANOCHAT_DTYPE=bfloat16 fails on M2 MPS (mixed-precision) [RESOLVED]

**Issue**: Setting `NANOCHAT_DTYPE=bfloat16` on M2 24GB caused Metal Performance Shaders Graph to fail at runtime:

```
'mps.multiply' op requires the same element type for all operands and results
%4 = "mps.multiply"(%arg2, %3) : (tensor<1xf32>, tensor<32768x384xbf16>) -> tensor<*xf32>
```

**Root Cause**: nanochat's bf16 path stores some params (wte, value_embeds) at bf16 to save memory but keeps the optimizer's shared scalar tensors at fp32. The polar express loop in muon also intentionally casts gradients to bf16 mid-function. CUDA implicitly promotes mixed-dtype operands; MPS hard-fails. Concrete crash sites:
- `adamw_step_fused`: `p.mul_(1 - lr_t * wd_t)` where `p` is bf16 (wte) and the scalars are fp32
- `muon_step_fused`: `second_momentum_buffer.lerp_(..., 1 - beta2)` where the buffer is fp32 but `beta2` came from `g.dtype` which became bf16 after the polar express loop

**Solution**: Two-part patch in commit `7e21999`:
1. In `adamw_step_fused`, cast scalar hyperparams to `p.dtype` at use sites
2. In `muon_step_fused`, cast `g` back to `stacked_params.dtype` after the polar express loop so subsequent ops are single-dtype

**Outcome**: bf16 now works end-to-end on M2 (forward + backward + optimizer.step). Per-token throughput is unchanged (~3% improvement) because M2 lacks native bf16 hardware (M3+ does). The real unlock is memory: bf16 + `device_batch_size=48` now works (was OOM at fp32). Numerical drift exists (~1e-3 in loss by step 9 vs fp32) — full A/B validation against the fp32 baseline (val_bpb 1.174 at step 5000) is still pending.

**Prevention / Followup**: A more rigorous bf16 validation run should be done before making bf16 the M2 default. Worth filing the patch upstream — nanochat's bf16 path was effectively broken on MPS for any mixed-dtype embedding setup before this fix.

## 2026-04-30 - chat_sft `--eval-every=-1` does NOT skip final eval

**Issue**: Passing `--eval-every=-1` to `scripts.chat_sft` skips intermediate val evaluations but still runs a full val eval at `last_step`. The condition at chat_sft.py:347 is `if last_step or (args.eval_every > 0 ...)` — `last_step` short-circuits the check.

**Root Cause**: Intentional design — Karpathy wants a final val number recorded. But the default `--eval-tokens=40*524288` (21M tokens) means on M2 MPS this final eval takes ~25–30 min before the checkpoint save block runs.

**Solution**: To actually skip the final eval, also pass `--eval-tokens=0` or a tiny number. Alternatively, accept the eval cost — it's just a one-time wait at the end.

**Prevention**: When running `chat_sft` on slow hardware where you don't care about the final val_bpb, pass both `--eval-every=-1 --eval-tokens=0`. (Verify `--eval-tokens=0` is honored; otherwise pass something like 16384.)

