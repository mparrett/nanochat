# Key Facts

## Base Directory
- All artifacts (datasets, tokenizer, checkpoints, reports): `$NANOCHAT_BASE_DIR` (default `~/.cache/nanochat`)
- Source of truth: `get_base_dir()` in `nanochat/common.py`

## Checkpoints
- Layout: `$NANOCHAT_BASE_DIR/{base,chatsft,chatrl}_checkpoints/<model_tag>/`
- Files: `model_<step>.pt`, `meta_<step>.json`, `optim_<step>_rank<N>.pt`

## Compute / Precision
- `COMPUTE_DTYPE` auto-detected (bf16 on SM 80+, fp32 elsewhere)
- Override via `NANOCHAT_DTYPE=bfloat16|float16|float32`
- Master weights stay fp32; `Linear` casts in forward pass

## Logging
- wandb optional. `--run dummy` (default) → `DummyWandb` (no-op)
- Enable: `WANDB_RUN=<name>` env var or `--run=<name>` (run `wandb login` first)

## Key Metrics
- CORE (DCLM) on base model — headline number
- `val_bpb` (bits per byte) — loss proxy
- `time-to-GPT-2` — wall-clock until CORE > 0.256525 (speedrun leaderboard, see `dev/LEADERBOARD.md`)

## Branch
- Main: `master`
- Current working branch: `experiment/hope-nested-learning`
