# Decisions

## ADR-001: Insert "Stage 1.5" synthetic recall probe before Stage 2 (2026-05-01)

**Context**: Stage 1 (one block's MLP swapped for `LinearAttentionMemory`) finished its full d6 pretrain at val_bpb 1.179 vs baseline 1.174 — within ~1% of baseline at iso-token. SFT on top landed at 0.6712 vs `d6_b_iso` 0.6639, also within ~1%. The natural next step from the trx4mr ticket is **Stage 2: per-token learned `α` and `η`**, but we have no way to tell whether the Stage 1 memory block is *actually doing something memory-flavored* — val_bpb is the average over the SFT distribution and doesn't isolate positions where carrying state across the sequence matters. We could be staring at noise either way.

Codex flagged the same concern explicitly: "Full LM val_bpb is too blunt for Hope/NL behavior. Before Stage 2, add a tiny diagnostic task/probe that can tell whether memory helps at all: associative recall, delayed copy, key-value recall across 512 tokens, or a synthetic in-context binding task."

**Decision**: Insert a **Stage 1.5** between Stage 1 and Stage 2 — design and implement a synthetic **multi-query associative recall (MQAR)** probe that trains a small d6 from random init on a binding-and-lookup task, then measures recall accuracy. Run it for both the unmodified MLP architecture and the Stage 1 `LinearAttentionMemory@L3` architecture, with identical training budget. The architectural delta on this probe is the load-bearing input to whether Stage 2 is worth implementing and how to tune its α/η stability defaults.

Detailed probe design lives in `docs/hope_nl_stage1_5_probe_design_2026-05-01.md`.

**Alternatives**:
- *Multi-seed Stage 1 confirmation runs.* Tightens noise on the val_bpb gap but doesn't address the "is memory actually doing anything" question. Codex: "I would not spend M2 time on multi-seed Stage 1 confirmation unless someone wants to claim 'Stage 1 improves/regresses nanochat.'"
- *Jump straight to Stage 2.* We'd add per-token learned α/η, full pretrain, observe whatever val_bpb shift we get, and still not know if the gating is exploiting a memory signal that exists or sculpting noise. Bad experimental hygiene; bad debugging position when stability problems hit.
- *A full benchmark suite (CORE, ARC, GSM8K).* Real downstream evals are valuable but (a) noisy at d6 scale, (b) measure "is the model good at the task" rather than "does the memory mechanism contribute," and (c) require expensive eval runs. The synthetic probe is sharper for the architectural question.

**Consequences**:
- Adds ~3–4 h of design + implementation + run time before Stage 2 starts. Two synthetic-task training runs (~15 min each on M2 plugged in for d6 from-scratch on a small synthetic vocab) plus eval.
- Defers Stage 2 by one step but de-risks it materially. If the probe shows Stage 1 already gives meaningfully better recall than baseline, Stage 2 has a clear "make it better" target. If the probe shows no architectural delta, we know that introducing learned α/η is fighting a ceiling problem and we revisit the design before sinking 3 h into another full pretrain.
- Probe code lives in `dev/` (not core). Reusable for Stage 2+ to track whether learned gates improve recall at iso-budget.
- Sets a precedent: from now on, architectural changes get diagnosed against a targeted synthetic probe before full LM pretrain budget is spent.

**Status**: accepted 2026-05-01. Implementation pending.
