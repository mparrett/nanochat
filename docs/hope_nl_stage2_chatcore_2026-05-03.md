# Hope/NL Stage 2 — ChatCORE evaluation (2026-05-03)

ChatCORE eval on the final Stage 2 SFT checkpoint
(`~/.cache/nanochat/chatsft_checkpoints/d6_stage2/model_000375.pt`,
val_bpb 0.6518). First downstream-task signal on Stage 2; closes the
last unchecked Stage 2 box from the original ticket.

## Result

**ChatCORE metric = 0.1744**

| Task | Accuracy | Baseline | Centered |
|---|---:|---:|---:|
| ARC-Easy | 25.80% | 25% (1/4) | 0.0107 |
| ARC-Challenge | 28.67% | 25% (1/4) | 0.0489 |
| MMLU | 26.98% | 25% (1/4) | 0.0264 |
| GSM8K | 0.76% | 0% | 0.0076 |
| HumanEval | 0.00% | 0% | 0.0000 |
| SpellingBee | 95.31% | 0% | 0.9531 |
| **Mean (ChatCORE)** | — | — | **0.1744** |

Wall: 1h27min on M2 24GB. Process clean throughout — no MPS allocator
issues, no fragmentation events. The `mps_release_cache()` fix from
session 4 (commit `f27a6db`) production-tested through ~2400+ MMLU
batches, ~14k MMLU problems, no OOM.

## SpellingBee dominates

SpellingBee's 95.31% contributes 0.9531 / 6 = 0.159 to the average.
That's **91% of the 0.174 ChatCORE metric** coming from a single
task. Everything else hugs baseline:

- ARC and MMLU are 1-3 points above 25% chance — within multiple-choice
  noise floor for any d6 model.
- GSM8K (math word problems) and HumanEval (Python) are essentially
  zero. d6 doesn't have the parameter budget for multi-step reasoning
  or program synthesis.
- SpellingBee at 95% is consistent with the SFT chat assessment from
  earlier today (commit `55b5d3a`): the model overfit the SpellingBee
  template hard, triggering the spell-out-and-count CoT shape on any
  word-shaped prompt. That same overfit produces high accuracy on
  actual SpellingBee test problems.

So the headline number is mostly a measurement of how well SFT
memorized the SpellingBee training distribution, not a measurement
of general chat capability.

## What this does and doesn't tell us about Stage 2

**Does tell us:**
- Stage 2 SFT works end-to-end on real tasks, not just held-out
  cross-entropy. Tool use survives (the `<|python_start|>` scaffold
  in SpellingBee). The model is functional, just narrow.
- ChatCORE infrastructure is solid on M2 — the OOM fix landed in
  session 4 holds up under a full 6-task evaluation.

**Does NOT tell us:**
- Whether Stage 2 is *better* than baseline d6 SFT. We don't have a
  ChatCORE for the baseline (`~/.cache/nanochat/base_checkpoints/d6/`
  was overwritten and re-secured per HANDOFF; current state needs
  verification before SFT-then-eval-on-baseline is feasible).
- Whether Stage 2 is better than Stage 1 swap. Same situation.
- Whether the SFT val_bpb 1.8% win we observed translates to any
  downstream task. The non-SpellingBee tasks are all at noise; the
  SpellingBee win comes from the SFT mixture, not the architecture.

The ChatCORE number is honest but its diagnostic value at d6 is
limited: at this scale, the metric mostly reports "did SFT memorize
the SpellingBee template," and that's true for any d6 SFT
regardless of pretrain architecture.

## Implications for next steps

Per the audit doc (`docs/hope_nl_phase_audit_2026-05-03.md`), the
A1 → A2 → A3 → F → (D or E) → C sequence is the plan after Codex
sync + paper bootstrap. ChatCORE was A1; this completes that step.

Before A2 (SFT-seed-variance disambiguation), Codex's metadata-audit
point lands: **seed is hardcoded in `nanochat/common.py:234` as
`torch.manual_seed(42)`** with no CLI flag and no capture in
`meta_*.json`. A2 requires plumbing seed through CLI + meta first —
small patch, ~30 min of work, isolated change.

After A2, the question of whether to run ChatCORE on baseline d6 SFT
and Stage 1 swap SFT becomes relevant for any "Stage 2 wins" claim.
But that's a separate decision; the current ChatCORE on Stage 2
alone is informative even without the comparators.

## Where everything lives

- **Final SFT checkpoint:** `~/.cache/nanochat/chatsft_checkpoints/d6_stage2/model_000375.pt`
- **Eval log:** `/tmp/chatcore_d6_stage2.log`
- **Report card:** `~/.cache/nanochat/report/chat-evaluation-sft.md`
- **wandb run** (if needed): chat_eval doesn't write to wandb; report card is the persistent record
