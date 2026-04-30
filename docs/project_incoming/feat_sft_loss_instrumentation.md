---
status: open
assigned: claude-code
created: 2026-04-30
updated: 2026-04-30
---
# Feature: SFT loss instrumentation to diagnose EMA artifact

## Context

After applying PR #610's NaN-fix to `nanochat/gpt.py` (return `logits.sum() * 0.0` when all targets are `-1`), the displayed training loss EMA in `chat_sft` collapsed unrealistically low (e.g. 1e-5 by step 100). Two competing hypotheses:

1. **Genuine fast convergence.** d6 (~73M params) overfitting on heavily-templated SFT data — assistant tokens become highly predictable.
2. **Packing producing many fully-masked micro-batches.** SmolTalk conversations are long; `max_seq_len=512` causes rows containing only user-prompt prefixes (mask=0 throughout). PR #610's fix correctly emits 0 loss for these but pollutes the EMA.

Gradients are correct in both cases (fully-masked batches contribute zero update). The artifact only misleads humans monitoring the run.

## When to act

After the current SFT v4 run completes and the smoke test (`chat_cli`) tells us if the model is usable. Especially relevant if:
- Smoke test shows poor quality → strongly suggests hypothesis (2): we're not actually training on much real signal
- Smoke test shows decent quality → mostly hypothesis (1); instrumentation is still useful but lower priority

## Proposed instrumentation

Cheapest first, most useful last:

1. **Log n_valid per step.** In `chat_sft.py` after computing the mask, print `(y != -1).sum().item()` alongside the loss. Tells us immediately if fully-masked batches are common.

2. **Skip EMA update when fix triggers.** Have `gpt.py` return `(loss, n_valid)` or set a sentinel; `chat_sft.py` only updates the EMA when `n_valid > 0`. Honest readout.

3. **Token-weighted EMA.** Weight each step's contribution to the EMA by `n_valid`. A fully-masked batch (n_valid=0) contributes nothing. Partial batches contribute proportionally. This is the production-grade fix.

## Out of scope (for now)

Fixing this upstream — would need a clean PR to `karpathy/nanochat` and isn't blocking us tonight. If the diagnosis turns out interesting, worth filing.
