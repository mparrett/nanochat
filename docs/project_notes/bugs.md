# Bug Log

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

## 2026-04-30 - chat_sft `--eval-every=-1` does NOT skip final eval

**Issue**: Passing `--eval-every=-1` to `scripts.chat_sft` skips intermediate val evaluations but still runs a full val eval at `last_step`. The condition at chat_sft.py:347 is `if last_step or (args.eval_every > 0 ...)` — `last_step` short-circuits the check.

**Root Cause**: Intentional design — Karpathy wants a final val number recorded. But the default `--eval-tokens=40*524288` (21M tokens) means on M2 MPS this final eval takes ~25–30 min before the checkpoint save block runs.

**Solution**: To actually skip the final eval, also pass `--eval-tokens=0` or a tiny number. Alternatively, accept the eval cost — it's just a one-time wait at the end.

**Prevention**: When running `chat_sft` on slow hardware where you don't care about the final val_bpb, pass both `--eval-every=-1 --eval-tokens=0`. (Verify `--eval-tokens=0` is honored; otherwise pass something like 16384.)

