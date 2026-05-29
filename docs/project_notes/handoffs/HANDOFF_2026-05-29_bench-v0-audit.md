# Session Handoff

**Created:** 2026-05-29
**Session ID:** fb86ad93-0cb7-4841-8997-23cb8c5a65f8
**Working Directory:** /Users/matt/projects-new/3p/nanochat

## What to read first

`docs/bench_v0_audit_2026-05-28.md` (or its HTML twin) — this session's
load-bearing finding. The bench v0 "Stage 2 specializes" architectural
narrative cited as the "convergence" node in `docs/project_map_2026-05-20.html`
is dead. Three of four bench v0 headlines have material problems under
n=3 seed bracketing. The project map has a banner aside pointing at the
audit; the May-20 snapshot itself is preserved in place.

## Summary

Multi-day session (2026-05-24 → 2026-05-29) that produced a coherent
two-act falsification. Act 1: NTK-Mirror Path A cross-adapter
comparison on persona-retention came in at parity with L1 LoRA after
seed bracketing (single-seed +1/30 win collapsed to mean tie). Act 2:
applied the same bracketing methodology to bench v0's four headline
numbers; the "Stage 2 specializes" architectural narrative didn't
survive. Built `--n-seeds N` into `bench/run.py` mid-session to make
n≥3 bracketing one CLI flag away.

## Current State

Branch: `experiment/hope-nested-learning`. Local-only per
`feedback_local_only.md` (no push).

Session commits (in chronological order):

```
5713372  bench: NTK-Mirror persona-retention scripts — Qwen adapter, scoring, seed shuffle
eeb6293  result: NTK-Mirror Path A — parity with L1 LoRA after seed bracketing, Path B shelved
cc00b9b  result: bench v0 SC headline falsified — Stage 2 win was lucky seed
8f77ab1  feat: bench/run.py — --n-seeds N flag for built-in seed bracketing
1809904  result: bench v0 four-headline audit — architectural story dead, U-curve weakened, Stage 1 survives
```

Cross-session memory updates this session:

- `project_ntkmirror_gate_selection.md` (new) — NTK-Mirror's gate
  selection is order-sensitive; sets a ceiling on subsequent training.
- `feedback_bench_seed_default.md` (new on 2026-05-25, extended
  2026-05-28) — n=3 minimum for any architectural claim; catastrophic-
  failure rate is a separate property; shape claims require bracketing
  at every sweep point.

Both indexed in `~/.claude/projects/-Users-matt-projects-new-3p-nanochat/memory/MEMORY.md`.

## Uncommitted State / Untouched

**Uncommitted:**
- `bench/logs/` (untracked) — 14 new JSONL + 14 stdout logs from this
  session's runs, plus 3 smoke files (`smoke_default.jsonl`,
  `smoke_n2_s{0,1}.jsonl`). Per the established convention bench logs
  are not tracked. Don't add them.
- `knowledge/` (untracked) — pre-existing as of session start
  (`summary_eggroll_lowrank_es.md` from a different project's session
  2026-05-24). **Not mine; don't touch.**

**Untouched (deliberate):**
- `docs/project_map_2026-05-20.html` — banner aside added at top
  pointing at the audit; the body of the map left intact as a
  historical snapshot per the audit writeup's explicit decision.
  Do not rewrite the map in place.
- Original bench v0 writeups (`docs/bench_v0_*_2026-05-1{8,9}.md`)
  left as-is even though now superseded. They contain the original
  n=1 framings — useful as historical record and as evidence the
  writeups themselves flagged the n=1 risk.

## In Progress

Nothing carrying over mid-task. The session closed cleanly after
the audit writeup commit. Suggested next experiments are listed
under "Next Steps" below.

## Gotchas

- **Stage 2 MQAR has a 33% catastrophic-failure rate at d4/500-iter.**
  This is a load-bearing fact that mean/range reporting hides — seed 1
  of both η=0.1 and η=0.5 produced complete training failure (loss
  4.89 → 4.49 over 500 steps, acc never moves off random). If
  bracketing more Stage 2 / MQAR variants, expect ~1/3 of seeds to fail
  and report pass-rate alongside central tendency.
- **Reproducibility drift between May-18/19 and today.** Same seed
  produces ±30-50 step shifts on sat_step — likely torch/transformers
  minor version drift since the original bench v0. Doesn't change
  verdicts but means May-19 numbers can't be combined exactly with
  today's runs at the same seed value.
- **`--n-seeds N` auto-suffixes labels.** When N=1 (default), no
  suffix is added (backwards-compat). When N>1, labels become
  `{label}_s{seed}`. Don't use `--label foo` and `--seed 1` separately
  expecting `foo.jsonl` — at single-seed runs the existing
  `{label}.jsonl` convention is preserved only when you don't pass
  `--n-seeds`.
- **Don't kill operator daemons** for memory pressure
  (`feedback_kill_candidates.md`). bench/ at d4 isn't memory-bound;
  Qwen-0.5B inference can be. Use workload dials instead.
- **Local-only repo** (`feedback_local_only.md`). No pushes, no PRs.

## Next Steps

In priority order, all from the audit's "what's next" section:

1. **Bracket baseline MQAR n=3** (~50 min wall) — makes the Stage 2
   MQAR / η=0.5 comparisons fully rigorous (currently comparing to
   a single-seed baseline of sat=225 from May-18). Confirms whether
   catastrophic failure is mechanism-specific or affects baseline too.
2. **Bracket η=0.99 n=3** (~58 min wall) — the U-curve "best"
   endpoint at n=1=325. If it lands ≈ 412 (η=0.1's bracketed mean),
   the U-shape is entirely flat and the eta_init mechanism story
   dies completely. Use:

   ```
   python -m bench.run --task mqar --depth 4 \
     --K 64 --M 32 --T 256 --n-keys 128 --n-values 128 \
     --hope-additive-memory-layer 1 \
     --hope-memory-w-o-init-scale 1.0 \
     --hope-memory-kind learned_gate \
     --hope-memory-eta-init-bias 4.595 \
     --n-seeds 3 \
     --label stage2_d4_mqar_hard_eta099 \
     --log-dir bench/logs
   ```
3. **Re-think the architectural hypothesis with the operator.** Stage 1
   recall-shape + δ-mem are what's left of the "memory mechanisms are
   task-mechanism-routing" evidence base after the bench v0 audit took
   ~75% of it down. Open question: is the hypothesis still worth
   investing further bench compute in? If yes, the induction-heads
   probe (May-20 lower-priority list) might now be elevated as the
   third axis between recall and state-tracking.
4. **Update key_facts.md** in `docs/project_notes/` with the
   catastrophic-failure rate observation as a Stage 2 MQAR mechanism
   property. Currently it lives in the writeup and in cross-session
   memory but not in the standing project-memory facts file.

## Open Tickets

Not changed this session. Still on the books in `docs/project_incoming/`:

- `feat_ntkmirror_upstream_bash_pr.md` — tiny ~2-line bash strict-mode
  fix to upstream `leochlon/ntkmirror`; filed 2026-05-24, still open.
- `feat_muon_ns_steps_validation.md` — `ns_steps=3` vs `5` Muon
  validation (queued since 2026-04-30, ~12% optimizer speedup at
  bit-identical loss on a 9-step smoke).
- `feat_sft_loss_instrumentation.md` — log `n_valid` and skip EMA on
  fully-masked SFT batches (queued since 2026-04-30).
- `feat_ntkmirror_cross_adapter.md` — **closed in spirit by this
  session's NTK-Mirror Path A work** (`eeb6293`). Ticket file not
  yet moved to `docs/project_archived/`; the audit writeup +
  `docs/ntkmirror_persona_comparison_2026-05-25.md` are the
  canonical closeout. Move on cleanup pass.
