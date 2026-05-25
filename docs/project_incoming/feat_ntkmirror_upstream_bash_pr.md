# Feature: Upstream PR — bash fix for ntkmirror disjoint-composition runner

**Filed:** 2026-05-24
**Source:** Bug encountered during today's NTK-Mirror disjoint-composition
smoke (see `docs/project_notes/backlog.md::NTK-Mirror` and
HANDOFF Day 2026-05-24).

## Bug

`scripts/run_disjoint_composition.sh` line 43 expands `"${CARGS[@]}"`
on a possibly-empty array under `set -euo pipefail`. On macOS bash 5.x,
expanding an empty array's `@` under `set -u` triggers "unbound
variable" and aborts the eval loop.

**Subtle:** the script exits 0 because the trailing `python <<'PY'`
report block runs to completion regardless of the for-loop failure.
The user sees the composition report (compose step succeeded), but no
`eval_*.json` files exist (8 evals all skipped).

## Fix

Two-line diff, both at `ntkmirror eval` invocations in the for-loop:

```diff
-    "${CARGS[@]}" --eval "$OUT/data/gsm8k_eval.jsonl" \
+    ${CARGS[@]+"${CARGS[@]}"} --eval "$OUT/data/gsm8k_eval.jsonl" \
```

The bash idiom `${arr[@]+"${arr[@]}"}` expands to the array's contents
if defined, nothing otherwise — standard workaround for the strict-mode
empty-array trap.

**Local patch applied to:** `~/projects-new/3p/ntkmirror/scripts/run_disjoint_composition.sh`
(uncommitted; verify with `git status -s` in that repo).

## To do upstream

1. Fork `leochlon/ntkmirror` on a personal GitHub account.
2. Create branch from main, apply patch, commit with message
   `Fix unbound variable in run_disjoint_composition.sh eval loop`.
3. Open PR; reference macOS bash 5.x strict-mode empty-array behavior in
   the description.
4. Optionally include a small bash-syntax test to lock the contract.

Author is responsive (last commit 2026-05-24). Two-line PR; should
merge fast.

## Why not now

Deferred per operator decision 2026-05-24. The fork+PR mechanics aren't
blocked; just no urgency. `ntkmirror` is the upstream of a codebase we're
consuming for evaluation, not a hot dependency. If we adopt NTK-Mirror
into more workflows (cross-adapter comparison, graft), revisiting this
gets more important.

Note: nanochat's `feedback_local_only.md` rule is about the nanochat
fork specifically. ntkmirror is a separate upstream we cloned, so the
PR mechanism doesn't conflict with that rule. Still worth a brief check
before forging the personal-account remote.

## Pre-launch (when revisited)

- Make sure the local patch is still present in the ntkmirror clone
  (`git status -s` should show `M scripts/run_disjoint_composition.sh`).
- Forge a personal-account remote: `git remote add fork git@github.com:<you>/ntkmirror.git`.
- Push branch, open PR.

## Estimated effort

~30 min total: 10min setup + 5min commit + 10min PR description + 5min
buffer.
