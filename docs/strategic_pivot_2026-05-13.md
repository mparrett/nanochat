# Strategic pivot: pause Hope/NL from-scratch, shift to minimum-viable signal

**Date:** 2026-05-13
**Branch:** `experiment/hope-nested-learning`
**Context:** End of the d6/d8 Hope/NL pretrain arc. d8 SFT A2 (`val_bpb 0.6218`, ChatCORE 0.1622) cleared the under-tokened-SFT diagnostic on 2026-05-11. With the immediate diagnostic settled, we stepped back to reconsider whether the from-scratch Hope/NL thread still serves the larger vision.

## The reframe

Two motivations were on the table going in:
1. **Architectural research** — does Hope/NL's per-token mutable memory actually do what the paper claims?
2. **Engagement / qualitative validation** — a working chatbot at the end of the iteration loop matters; without it, motivation decays.

The current from-scratch nanochat pipeline (d6 / d8 pretrains, SFT, ChatCORE) was the obvious vehicle because it lets us **own every parameter end-to-end**, which is the right move when an architectural change requires retraining from the ground up. It is **not** the right move for a class of changes that don't — LoRA-style additions, gated graft modules, test-time-trained layers grafted onto a frozen base. For those, owning every parameter is overkill, and the d6-or-up pretrain cost (3–18 h wall) is paying for something we don't need.

The bigger vision the operator articulated:
- **Long arc:** continuous-learning components grafted onto a "bonsai" 4–8B model running locally — closer in spirit to LoRA than to from-scratch architecture design. We do not have the compute to pretrain that scale, ever. We never will on this machine.
- **Near term:** use existing local small-model inference (1-bit / quantized projects already in the wild — XOR-swap weights, etc.) as a *tool* in feedback loops or as a distillation source. **Modeling the base isn't ours to do; using the base is.**
- **Memory-mechanism research:** the architectural-signal question survives. Per-token mutable memory is still an interesting idea. But validating it at d6 LM-loss scale is the slowest possible way to get signal.

## The conclusion

**Pause the from-scratch nanochat Hope/NL thread.** The d6 and d8 result series stand as captured (see `docs/hope_nl_*` and `docs/d8_*` writeups). Future memory-mechanism work moves into a separate, much smaller iteration loop:

- **Smallest-possible transformers** (depth 2–4, dim 64–128).
- **Synthetic diagnostic tasks** that directly exercise the mechanism: MQAR, selective copy, induction heads, modular arithmetic where it adds value.
- **Iteration cycle in minutes**, not hours.
- **Goal: minimum viable signal** — does this mechanism do the thing it's supposed to do? — not language-model quality.

We do **not** pivot to the "graft onto frozen 1.5B base" path that came up earlier in the conversation. That was a reasonable framing, but the operator's actual interest in local bonsai inference is downstream (feedback loops, distillation) rather than as a graft target for new memory layers. The graft-on-base research thread stays parked — possibly revisited later as a separate effort, possibly never; it's not the spine.

**Revisit from-scratch chatbot-as-testbed if/when we have more compute** (a GPU box, a workstation, etc.). The d6/d8 work was educational and the infrastructure (preflight, wandb defaults, `--grad-checkpoint`, `--inherit-from`, ChatCORE harness) carries forward to that future.

## Possible next steps

Listed without commitment. The operator may pick up any of these threads, in any order, or none of them.

### Memory-mechanism playground (the obvious next thread)

Build a small `bench/` harness that pairs `nanochat/gpt.py` with synthetic-task dataloaders. Existing Stage 1.5 MQAR code (`dev/stage1_5_mqar*.py` and `docs/hope_nl_stage1_5*.md`) is the seed.

- Generalize the MQAR probe into a `bench/tasks.py` module with a `--task=` flag.
- Add **selective copy** (Mamba's go-to: present a sequence with content + noise tokens; model must copy only content). Tests learned gating.
- Add **induction heads** (1-layer setup; does it learn the bigram-copy circuit?). Cheap and well-instrumented in the literature.
- (Optional) **modular arithmetic / grokking** — tests generalization, less directly tied to memory.
- Reuse `enable_block_grad_checkpoint`, the Muon optimizer, the existing `Block` / `LinearAttentionMemory` modules. Don't reinvent.
- Iteration cycle target: **under 5 minutes per architectural variant.**
- Logging: keep wandb default-on (gate group by task, not by model — see `docs/project_notes/key_facts.md`).

This harness becomes the architecture playground. Anything that survives it might earn a d3 LM smoke run later; nothing earns a d6+ pretrain until compute is cheaper.

### Local bonsai inference as a tool (parallel thread)

Less specified but flagged by the operator. Candidate framings:
- **Feedback loops**: a local bonsai (Qwen-0.5B, TinyLlama, Phi-3.5-mini at int4, etc.) acts as a judge / critic / refiner in a multi-step inference pipeline.
- **Distillation source**: small fast model proposes; bigger remote model verifies; distill the bigger model's preferences back into the small one.
- **1-bit / XOR-swap projects**: survey what already exists rather than reinventing. The operator has seen specific projects; we don't currently have a list in the repo.

This thread does not need the nanochat pretrain pipeline at all. It might not even live in this repo. Flag here so future sessions know it's an active interest.

### Revisit when compute arrives

- The d6/d8 Hope/NL writeups (`docs/hope_nl_stage*`, `docs/d8_*`) are publication-grade evidence of mechanism behavior at LM scale on a constrained machine. Worth keeping clean.
- The "graft onto frozen 4–8B base" framing is parked, not killed. If a future result from the playground looks compelling, this is the obvious validation step.
- A canonical d8 base CORE evaluation (`--max-per-task=500`, ~5 h) is still queued from the 2026-05-10 work, independent of the strategic pivot. Worth doing if a precise d8-vs-d6_stage2 headline matters for future writeups.

## What we keep

Preserved as-is:
- d6 baseline (`base_checkpoints/d6/`, `chatsft_checkpoints/d6/`) — canonical reference.
- d6_stage2 (Hope/NL Stage 2) — best-of-d6 SFT result, val_bpb 0.6518, ChatCORE 0.1744.
- d8_overnight base + d8_overnight_a2 SFT — best-of-d8 results.
- All `docs/hope_nl_*`, `docs/d8_*`, `docs/project_notes/*` writeups.
- Infrastructure: `dev/preflight_memory.py`, `--grad-checkpoint`, `--inherit-from`, wandb defaults, ChatCORE harness.

Dropped:
- First-pass d8 SFT (`chatsft_checkpoints/d8_overnight/`) — 2.6 GB freed earlier today; the under-tokened result is fully captured in `docs/d8_sft_2026-05-11.md` and `key_facts.md`, the checkpoint itself was redundant once the A2 result landed.

## Branch state

No code changes today. Commits associated with this pivot:
- This document.
- ADR-008 in `docs/project_notes/decisions.md`.
- HANDOFF.md "Day 2026-05-13" entry.

Local-only per `feedback_local_only.md`. The `experiment/hope-nested-learning` branch enters a paused state — no active investigations, no in-flight runs. Future work either re-opens it (the playground could live here as `bench/`) or branches off elsewhere.
