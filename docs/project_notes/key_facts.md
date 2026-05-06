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

## Launch patterns on M2

**Use direct python, not `torchrun`.** Canonical pattern:
```bash
PYTHONUNBUFFERED=1 nohup uv run python -u -m scripts.base_train ...
```
Why: `torchrun --standalone --nproc_per_node=1` injects `RANK`/`LOCAL_RANK`/
`WORLD_SIZE` env vars, but `compute_init` in `nanochat/common.py:246` only
calls `dist.init_process_group()` when `device_type == "cuda"`. On MPS the
process group is never initialized, and the `MuonAdamW` optimizer's
`dist.get_rank()` at the first step crashes with "Default process group has
not been initialized." The d6_baseline_modern + A3 + A3' runs all use
direct python; matched by the d3 smoke recipe (`docs/d3_smoke_recipe_2026-05-05.md`).

**Use `--inherit-from`, not flag enumeration.** Canonical pattern for any
non-baseline run:
```bash
uv run python -u -m scripts.base_train \
    --inherit-from=$NANOCHAT_BASE_DIR/base_checkpoints/d6_baseline_modern/meta_005000.json \
    --depth=<N> --seed=<S> --model-tag=<tag> --run=<name> \
    [other intentional overrides]
```
Why: script defaults differ from canonical d6 (`head_dim=128` vs `64`,
`max_seq_len=2048` vs `512`, `window_pattern=SSSL` vs `L`). Forgetting any
one silently trains a different model — A1 of the Stage 2 audit caught a
head_dim drift exactly this way. `--inherit-from` loads 34 fields from a
reference meta as parser defaults; CLI flags override only intentional
deltas. Excluded from inheritance (must re-pass): `run`, `model_tag`,
`sft_tag`, `seed`, `resume_from_step`, `force_overwrite`, `save_every`,
`save_keep_last_n`. Mechanism landed in commit `ca9bc94`.

## References

- **Cross-depth architecture audit (d3/d6/d12/d20):**
  `docs/architecture_audit_d3_d6_d12_d20_2026-05-05.md` — shape, params,
  auto-derived training hyperparams, M2 wall projections, tweakable knobs
  in 4 tiers, code-location pointers for every scaling formula.

## Branch
- Main: `master`
- Current working branch: `experiment/hope-nested-learning`
