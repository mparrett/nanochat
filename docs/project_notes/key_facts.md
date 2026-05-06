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

## Training recipes — d6 on M2 24GB

Canonical settings used for d6 baseline / Stage 1 / Stage 2 comparisons. Match
exactly across runs — val_bpb is only meaningful when eval setup is identical.

**Pretrain:** `--depth=6 --device-batch-size=32 --total-batch-size=16384`
(accum=1, ~3h wall, ~5000 iters at chinchilla budget). Default `--eval-every=250
--eval-tokens=524288`. `--core-metric-every=-1` is already the default.

**SFT:** `--num-iterations=375 --total-batch-size=65536` (accum=4 — does NOT
transfer to pretrain, see `phase3_step4_pretrain_accum_derisk` writeup).
**`--eval-every=50 --eval-tokens=524288`** — both load-bearing for fair
val_bpb comparison; defaults are wrong (`eval-tokens` defaults to 20M which
takes ~67min per eval on M2 and hangs runs). `--chatcore-every=-1` to skip
the categorical-eval OOM path; `--save-every=100 --save-keep-last-n=2`
for rolling cleanup.

**Multi-seed pattern:** `--model-tag=<base>` to load, `--sft-tag=<base>_s<N>`
to save somewhere different, `--seed=<N>` for the global RNG. seed lands
in `meta_*.json` automatically via `vars(args).copy()`.

**Source of these numbers:** seed=42 d6_stage2 SFT meta
(`chatsft_checkpoints/d6_stage2/meta_000375.json`) is the canonical reference;
SFT recipe story is captured in `docs/sft_oom_investigation_2026-05-03.md`.

## References

- **Cross-depth architecture audit (d3/d6/d12/d20):**
  `docs/architecture_audit_d3_d6_d12_d20_2026-05-05.md` — shape, params,
  auto-derived training hyperparams, M2 wall projections, tweakable knobs
  in 4 tiers, code-location pointers for every scaling formula.

## Branch
- Main: `master`
- Current working branch: `experiment/hope-nested-learning`
