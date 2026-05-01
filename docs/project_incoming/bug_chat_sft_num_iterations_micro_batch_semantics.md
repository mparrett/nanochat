---
status: open
assigned: claude-code
created: 2026-05-01
updated: 2026-05-01
project: nanochat
---
# Bug: `chat_sft.py --num-iterations` is enforced as micro-batches, not optimizer steps

## Summary

When `total_batch_size > device_batch_size * max_seq_len * world_size` (i.e. `grad_accum_steps > 1`), `--num-iterations=N` produces approximately `N / grad_accum_steps` optimizer steps, not `N`. This is silently wrong: training stops about `grad_accum_steps`× earlier than expected, with the LR schedule warmed-down on schedule for the truncated horizon.

## Reproduction (M2, d6 base checkpoint)

Two runs intended as iso-token, accum-only A/B:

- A: `--total-batch-size=16384 --num-iterations=1500` → 1500 opt steps, 24.6M tokens, 69.4 min wall (correct)
- B: `--total-batch-size=65536 --num-iterations=375` → **94 opt steps**, 6.16M tokens, 11.7 min wall (truncated)

Logs show the LR schedule warming down to `lrm=0.00` by step 94, with the per-step printout showing `100.53%` progress — the data generator's `it` counter hit `num_iterations` after 94 optimizer steps × 4 micro-batches/step.

## Root cause

`scripts/chat_sft.py:281`:

```python
# inside sft_data_generator_bos_bestfit, runs once per yielded micro-batch
it += 1
if 0 < args.num_iterations <= it and split == "train":
    last_step = True
```

The data generator yields one micro-batch per call, and the training loop calls `next(train_loader)` `grad_accum_steps` times per optimizer step (chat_sft.py:443–453). So the `it` counter increments per micro-batch, but `num_iterations` is documented and used by callers as if it counted optimizer steps.

The same `it` counter also drives `approx_progress = it / num_iterations` (line 288), so the LR schedule's warmdown fires on micro-batch progress rather than optimizer-step progress, completing exactly when the early `last_step` is asserted.

## Workaround (no code change)

Multiply `--num-iterations` by `grad_accum_steps`:

```bash
# accum=4, want 375 opt steps:
--total-batch-size=65536 --num-iterations=1500   # not 375
```

Verified: re-running with `--num-iterations=1500 --total-batch-size=65536` yields 375 optimizer steps and progress crosses 100% at step 375 (1500 yields = 5 prefetch + 374×4 + 1 + ε due to the prefetch ordering, but it lands on step 375 in practice).

## Suggested fix (one-line, preserves CLI semantics)

Change the guard at chat_sft.py:281 to use the optimizer step counter that the training loop already maintains, e.g. by passing `step` into the generator or asserting on `it / grad_accum_steps`:

```python
if 0 < args.num_iterations * grad_accum_steps <= it and split == "train":
    last_step = True
```

And similarly for `approx_progress`:

```python
approx_progress = it / (args.num_iterations * grad_accum_steps)
```

Either rename the flag or document the semantics if the current behavior is intentional (e.g. "iso-microbatch budget across accum settings"), but the current state surprises iso-token comparisons.

## Why it stayed hidden

`base_train.py` documents and uses `--total-batch-size` to set the gradient accumulation count; users typically vary that without touching `--num-iterations` (which often defaults to `-1` meaning "full epoch"). On `chat_sft.py` the default is also `-1` (epoch-driven stopping via `consumed >= dataset_size`), so the bug only bites when a user explicitly sets `--num-iterations` AND uses gradient accumulation to compare batch-size variants.

## Impact

Anyone running A/B comparisons across `--total-batch-size` values with `--num-iterations` set is silently doing a different experiment than intended: fewer optimizer steps, fewer tokens, full LR warmdown on the truncated horizon. Mostly bites researchers; the speedrun pipeline doesn't trip it because `runs/runcpu.sh` and friends pass `--num-iterations=-1`.

## Related

- The micro-batch counter doubles as the LR schedule clock, so any user-visible "progress %" in the per-step log is also wrong by `grad_accum_steps`× for run with explicit `--num-iterations`.
- Consider the same audit on `chat_rl.py` if it has the same pattern.
