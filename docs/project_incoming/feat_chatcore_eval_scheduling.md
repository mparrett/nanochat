# Feature: ChatCORE eval scheduling — fast-fail mode for d6

**Filed:** 2026-05-03
**Source:** Codex sync after first ChatCORE on d6_stage2

## Context

Today's ChatCORE on `d6_stage2` ran for 1h27min on M2 24GB. The
breakdown was wildly skewed:

| Task | Wall fraction | Result | Decision value |
|---|---:|---:|---|
| ARC-Easy | small | 25.80% | At baseline — pass/fail |
| ARC-Challenge | small | 28.67% | At baseline — pass/fail |
| MMLU | medium | 26.98% | At baseline — pass/fail |
| **GSM8K** | **~70 min** | **0.76%** | **Confirmed floor — known a priori at d6** |
| **HumanEval** | **~10 min** | **0.00%** | **Confirmed floor — known a priori at d6** |
| SpellingBee | ~15 min | 95.31% | Dominant signal (91% of metric) |

The two generative tasks (GSM8K, HumanEval) consumed ~80 of the 87
minutes and returned essentially zero — not surprising at d6 scale,
since multi-step arithmetic and Python synthesis are out of reach for
a 600K-param model. That ~80 min is paid every time we want to
report a ChatCORE number.

## Codex's proposal

Don't change the canonical ChatCORE metric — keep that stable so
historical comparisons are valid. But add eval-scheduling modes that
**run tasks in decision-value order and can stop early** when the
remaining tasks are confirmed-floor:

- **`chatcore-smoke`** — `SpellingBee + ARC-Easy` only. Cheap canary
  for "is the SFT broken?" (~20 min on M2). Catches tokenization,
  checkpoint, and obvious mode-collapse issues quickly.

- **`chatcore-fastfail`** — full ordered suite, stops if early canaries
  fail. Order by decision value:
  1. SpellingBee (cheap, dominant signal)
  2. ARC-Easy (quick broad sanity)
  3. ARC-Challenge (paired with ARC-Easy)
  4. MMLU (broader but weak at d6)
  5. GSM8K
  6. HumanEval

  Early-stop rule (suggested for d6, tunable): if `SpellingBee` regresses
  *materially* vs the calibration run **or** `ARC-Easy` centered score
  is at ~0, skip GSM8K and HumanEval. Report a marked "ChatCORE-partial"
  number rather than the full metric — explicitly *not* directly
  comparable to canonical.

- **`chatcore-full`** — the canonical current suite, no early stop.
  Use when reporting a comparison-quality number.

## Why this matters for our project

For the next phase (multi-seed Stage 2 + Stage 4 multi-block memory),
we'll want fast turn-around on architecture variants that may break
SFT entirely. Burning ~80 min on confirmed-floor tasks is cheap
insurance against false negatives but expensive on iteration speed.
Smoke-test mode would let an architecture variant get rejected (or
provisionally accepted) in ~20 min instead of 90.

## Implementation sketch

Add a `--mode` flag to `scripts/chat_eval.py` with three values:

```python
MODE_TASK_ORDERS = {
    'full':     ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee'],
    'fastfail': ['SpellingBee', 'ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval'],
    'smoke':    ['SpellingBee', 'ARC-Easy'],
}

# In fastfail mode, after each task check early-stop predicate.
# Predicate is configurable via two flags:
#   --fastfail-spelling-floor=0.50    (regress below 50% → skip late tasks)
#   --fastfail-arc-easy-floor=0.27    (centered acc ~0 → skip late tasks)
# Output records which tasks were skipped and why.
```

Reporting:
- `full` → canonical "ChatCORE: 0.1744"
- `fastfail` (all tasks ran) → same canonical metric
- `fastfail` (early-stop triggered) → "ChatCORE-partial: 0.XXX (skipped: GSM8K, HumanEval — reason: ARC-Easy below floor 0.27)"
- `smoke` → "ChatCORE-smoke: 0.XXX (SpellingBee+ARC-Easy only)"

The partial-result label is **load-bearing** — it prevents anyone
from accidentally comparing a fastfail-truncated number to a full
number months later.

## Open design questions

1. **Calibration**: what numbers should the early-stop floors be?
   Today's run is the only d6 ChatCORE we have. We need at least
   one more (baseline d6 SFT) to set principled thresholds. Until
   then, defensive defaults: SpellingBee floor at 0.80, ARC-Easy
   centered floor at 0.05.

2. **Where the calibration lives**: hardcoded in chat_eval.py? In a
   YAML? In meta.json from a "golden" run? Probably a small
   constants block in chat_eval.py is fine for now — this is research
   code.

3. **Should smoke-test failures be a non-zero exit code?** Probably
   yes for CI use, but for interactive runs we want full output
   regardless. Add `--strict` flag.

## Effort estimate

~1.5 hours for an MVP:
- 30 min: refactor `chat_eval.py` task loop to be order-driven
- 30 min: add early-stop predicate + mode flag
- 30 min: testing on the d6_stage2 checkpoint (smoke + full both
  produce the right numbers; fastfail with deliberately broken
  thresholds early-stops correctly)

## Priority

Low-medium. Not blocking A2 (SFT-seed-variance) or A3 (multi-seed
pretrain) — those still want full ChatCORE for comparability. But
high-value before Stage 4 (multi-block memory), where we'll be
iterating on configs and want fast smoke-test feedback.
