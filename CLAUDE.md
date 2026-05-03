# CLAUDE.md

Operational guidance for Claude Code working in nanochat. Loaded into every session.

## Project

nanochat is a minimal, single-GPU-node training harness for an end-to-end ChatGPT-style LLM (tokenizer → pretraining → SFT → RL → eval → inference → web UI). It is a "strong baseline" codebase, not a configurable framework — keep changes minimal, cohesive, hackable. Avoid config objects, model factories, or wide if/else trees.

The codebase is organized around a single complexity dial: `--depth` (transformer layers). All other hyperparameters (width, heads, LR, training horizon, weight decay) are derived from depth so any depth produces a compute-optimal model. Any change has to be principled enough to work across depth values, not just at one scale.

## Where to look

- **`DEV.md`** — architecture, conventions, full setup + command reference.
- **`runs/runcpu.sh`** (M3 Max ~30 min, M2 ~3h) and **`runs/speedrun.sh`** (8×H100 ~3h) — canonical end-to-end pipelines.
- **`docs/project_notes/`** — `decisions.md` (ADRs), `bugs.md` (known issues + fixes), `key_facts.md` (ports, paths, environments).
- **`HANDOFF.md`** — current branch context (Hope/NL experiment in progress).

## Disk constraints (M2 development machine)

The dev machine runs at >90% disk usage by default. Training can blow up cache fast — a 5000-iter pretrain with `--save-every=200` left **20 GB of intermediate checkpoints** until trimmed.

- **Default to no `--save-every`.** A pretrain only needs the final checkpoint (~800 MB).
- **If you need recovery insurance, pair with `--save-keep-last-n=2`** (rolling cleanup; flag exists on all three training scripts). Without it, every save adds a permanent ~800 MB.
- **Pretrain pattern:** `--save-every=1000 --save-keep-last-n=2` caps at ~2.4 GB.
- **SFT pattern (375-step recipe):** `--save-every=100 --save-keep-last-n=2`.
- **Pre-flight:** `df -h ~/.cache/nanochat`. If <10 GB free, clean up first.
- **Disk-kill symptom:** training crashes silently mid-run (no traceback, possibly "leaked semaphore" warning). macOS kills processes when disk hits critical; wandb marks the run as a crash.

## Checkpoints — `--model-tag` is mandatory for any non-baseline run

Training scripts write to `$NANOCHAT_BASE_DIR/{base,chatsft,chatrl}_checkpoints/<model_tag>/`. **When `--model-tag` is unset, it defaults to `d<depth>` (e.g. `d6`)** — the canonical baseline location. Running any modified architecture without `--model-tag` will silently overwrite the baseline. `--run` (wandb name) does NOT affect the on-disk path — only `--model-tag` does.

**Always** pass `--model-tag=<descriptive-name>` for any non-baseline experiment (e.g. `d6_stage2`, `d6_b_iso`).

There is also a startup pre-flight guard in `checkpoint_manager.assert_checkpoint_dir_safe()` — aborts cleanly if the target dir already has `model_*.pt`. Override with `--force-overwrite` (intentional replace) or `--resume-from-step=<N>` (continue training).

The historical baseline numbers (val_bpb 1.174 at d6/5000 iter) are documented in `docs/hope_nl_stage1_full_pretrain_2026-05-01.md` and `HANDOFF.md` even when the on-disk checkpoint is gone.

## Conventions (essentials)

- One commit per discrete change, with multi-paragraph commit messages explaining why.
- Markdown writeup first (insurance against context exhaustion), HTML narrative second (publication-style).
- Cooperative pause hook: `touch /tmp/pause-nanochat` on long runs to pause cleanly between optimizer steps.

See `DEV.md` for the architectural conventions (the depth-sweep rule, CORE metric, etc.).

## Project Memory

When resolving bugs or making decisions, update `docs/project_notes/{bugs.md,decisions.md,key_facts.md}` so future sessions inherit the context.
