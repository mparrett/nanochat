---
status: open
assigned: claude-code
created: 2026-04-30
updated: 2026-04-30
---
# Feature: validate Muon `ns_steps=3` vs `ns_steps=5` on full pretrain

## Context

During Phase 2 perf work on M2 24GB, profiling showed `optimizer.step` dominates wall-clock time (~79% of each iter). The Muon `polar_express` Newton-Schulz loop runs `ns_steps=5` iterations of stacked matmuls, hardcoded in `nanochat/gpt.py:setup_optimizer`.

## What we found

Reducing `ns_steps` from 5 to 3 produced:

- **Speed**: ~12% reduction in optimizer wall-time (1.7s → 1.5s per iter on M2)
- **Quality**: Loss values for SFT steps 1–9 were **identical to 6 decimal places** vs. `ns_steps=5`. Newton-Schulz appears to converge to numerical precision in 3 iters for d6's matrix shapes.

## Why we didn't commit

- The polar express coefficients in `nanochat/optim.py` are explicitly tuned for 5 iters (per the comment, "computed for num_iters=5, safety_factor=2e-2, cushion=2"). Using only the first 3 coefficients is theoretically suboptimal even if empirically identical short-term.
- Drift could accumulate over 5000+ iterations in ways not visible in the first 9 SFT steps.
- The speedup is modest (~12%), so the risk/reward needs proper measurement.

## Validation plan (when done)

1. Run full pretrain with `ns_steps=5` (control) — already have this in `~/.cache/nanochat/base_checkpoints/d6/model_005000.pt`, val_bpb 1.174.
2. Run full pretrain with `ns_steps=3` (variant) under a different `--model-tag` like `d6_ns3`.
3. Compare:
   - Val bpb at step 5000
   - Wall time
   - SFT downstream quality on the same prompts ("Paris", "sky is blue", etc.)
4. If val_bpb is within ~1% and SFT quality is comparable, commit `ns_steps=3` as the M2 default.

## Implementation note

Could add `--ns-steps` to base_train as a CLI flag, threading through to `setup_optimizer`. Or just hardcode based on device type. Cleanest path: CLI flag, defaults to 5.
