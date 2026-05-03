# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

nanochat is a minimal, single-GPU-node training harness for an end-to-end ChatGPT-style LLM (tokenizer → pretraining → SFT → RL → eval → inference → web UI). It is a "strong baseline" codebase, not a configurable framework — keep changes minimal, cohesive, and hackable. Avoid adding configuration objects, factories, or branching abstractions.

The codebase is organized around a single complexity dial: `--depth` (transformer layers). All other hyperparameters (width, heads, LR, training horizon, weight decay) are derived from depth so that any depth produces a compute-optimal model. Any change has to be principled enough to work across depth values, not just at one scale.

## Setup

```bash
uv sync --extra gpu                  # CUDA (A100/H100/etc.)
uv sync --extra cpu                  # CPU / MPS (e.g. Apple Silicon)
uv sync --extra gpu --group dev      # adds pytest, matplotlib, ipykernel, transformers
source .venv/bin/activate
```

`gpu` and `cpu` extras are mutually exclusive (declared in `pyproject.toml`). `torch` is pinned to 2.9.1.

## Common commands

Tokenizer:
```bash
python -m nanochat.dataset -n 8           # download N pretraining shards (~250M chars each)
python -m scripts.tok_train               # train BPE tokenizer (vocab 32768)
python -m scripts.tok_eval                # report compression ratio
```

Pretrain / eval / SFT / RL (use `torchrun --standalone --nproc_per_node=8` on multi-GPU; bare `python` works on single-GPU/CPU/MPS via auto gradient accumulation):
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --device-batch-size=16 --fp8
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval  -- --device-batch-size=16
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft   -- --device-batch-size=16
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval  -- -i sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl    -- ...
```

Talk to the model:
```bash
python -m scripts.chat_cli -p "Why is the sky blue?"   # CLI; omit -p for interactive
python -m scripts.chat_web                              # FastAPI WebUI; --num-gpus N for data parallel
```

Reference end-to-end pipelines (run as a single script):
- `runs/speedrun.sh` — full ~3h GPT-2-grade run on 8×H100 (the canonical pipeline)
- `runs/runcpu.sh` — small CPU/MPS demo, ~30 min on M3 Max
- `runs/scaling_laws.sh`, `runs/miniseries.sh` — research sweeps

Reports (used by speedrun): `python -m nanochat.report reset` then `... generate` writes `report.md`.

If OOM: lower `--device-batch-size` (32 → 16 → 8 → ...). Each shard is ~100MB compressed; full GPT-2 run uses ~150 shards.

## Tests

```bash
pytest                                # run full suite
pytest -m "not slow"                  # skip slow-marked tests
pytest tests/test_engine.py::test_x   # single test
```

Test config lives in `[tool.pytest.ini_options]` in `pyproject.toml` (`testpaths = ["tests"]`, `slow` marker registered).

## Architecture

**Two-layer split.** Library code lives in `nanochat/` (importable modules); entry points live in `scripts/` and run as `python -m scripts.<name>`. `tasks/` contains evaluation/training task definitions (ARC, GSM8K, HumanEval, MMLU, SmolTalk, SpellingBee, plus `customjson` for arbitrary JSONL conversations and `common.py` with `TaskMixture` / `TaskSequence`). `runs/` holds bash pipelines that compose the scripts.

**Model (`nanochat/gpt.py`).** Vanilla-ish GPT with rotary embeddings (no positional emb), QK norm, untied embed/lm_head, ReLU² MLP, RMSNorm without learnable params, no biases, GQA, and Flash Attention 3 on Hopper+ (with SDPA fallback in `nanochat/flash_attention.py`). Sliding-window attention is configured by `--window-pattern` (e.g. `"SSSL"` tiled across layers; final layer is always full).

**Optimizer (`nanochat/optim.py`).** Custom MuonAdamW (single GPU) and DistMuonAdamW (distributed). Different parameter groups get different LRs: `embedding-lr`, `unembedding-lr`, `matrix-lr` (Muon), `scalar-lr` (for `resid_lambdas`, `x0_lambdas`).

**Precision (`nanochat/common.py`).** No `torch.amp.autocast`. A single global `COMPUTE_DTYPE` is auto-detected (bf16 on SM 80+, fp32 elsewhere) and overridable via `NANOCHAT_DTYPE=bfloat16|float16|float32`. Master weights stay fp32; the custom `Linear` layer in `gpt.py` casts weights to `COMPUTE_DTYPE` in the forward pass. Embeddings are stored directly in `COMPUTE_DTYPE`. fp16 auto-enables a `GradScaler` in `base_train.py` (SFT also; RL does not).

**Distributed.** `compute_init()` / `compute_cleanup()` in `common.py` handle DDP setup from `torchrun` env vars. Optimizer state is sharded per-rank; checkpoints save model on rank 0 and `optim_<step>_rank<N>.pt` per rank (`checkpoint_manager.py`).

**Checkpoints (`nanochat/checkpoint_manager.py`).** Stored under `$NANOCHAT_BASE_DIR/{base,chatsft,chatrl}_checkpoints/<model_tag>/{model_<step>.pt, meta_<step>.json, optim_<step>_rank<N>.pt}`. `load_model("base"|"sft"|"rl")` resolves the right dir; missing model_tag → largest `d<N>` available; missing step → latest. `_patch_missing_*` shims handle older checkpoints with absent newer config keys.

**Data flow.** `nanochat/dataset.py` downloads ClimbMix-400B parquet shards on demand to `$NANOCHAT_BASE_DIR/base_data_climbmix/`. `nanochat/dataloader.py` provides the tokenizing distributed data loader (`tokenizing_distributed_data_loader_bos_bestfit` and a state-resumable variant).

**Inference (`nanochat/engine.py`).** KV-cache engine used by `chat_cli`, `chat_web`, and during eval. `nanochat/execution.py` lets the model call out to a Python interpreter as a tool (used by tasks like `gsm8k`/`humaneval`).

**Base directory.** All artifacts (datasets, tokenizer, checkpoints, reports) go under `$NANOCHAT_BASE_DIR` (default `~/.cache/nanochat`). `get_base_dir()` in `common.py` is the single source of truth.

**Logging.** wandb is optional — `--run dummy` (the default) routes through `DummyWandb` and logs nothing. Set `WANDB_RUN=<name>` env var or `--run=<name>` to enable real logging (run `wandb login` first).

## Conventions

- Don't introduce config objects, model factories, or wide if/else trees. Prefer adding a few flags or editing a script directly.
- Changes that touch training have to make sense across the depth sweep, not just one model size.
- The CORE metric (DCLM) on the base model is the headline number; `val_bpb` (bits per byte) is the loss proxy. `time-to-GPT-2` (CORE > 0.256525 wall-clock) is the speedrun leaderboard metric — see `dev/LEADERBOARD.md`.

## Long-running scripts: always use `python -u` when redirecting to a file

Python's stdout is **block-buffered** (4-8 KB chunks) when redirected via `>`, not line-buffered. A long training run's `tail -f /tmp/run.log` will appear silent for minutes at a time as the buffer fills, even though wandb is showing live progress and the GPU is busy. This makes the log look hung when it isn't, and obscures real hangs.

Always pass `-u` (or set `PYTHONUNBUFFERED=1`) when launching a training run that's redirected to a logfile in the background:

```bash
# wrong (buffered, log lags reality by minutes):
python -m scripts.base_train ... > /tmp/run.log 2>&1

# right (line-buffered, tail -f works):
python -u -m scripts.base_train ... > /tmp/run.log 2>&1

# also right:
PYTHONUNBUFFERED=1 python -m scripts.base_train ... > /tmp/run.log 2>&1
```

This applies to `base_train.py`, `chat_sft.py`, `chat_rl.py`, `dev/probe_mqar.py`, and any other long-running script. Some print statements in these use `flush=True` explicitly, but not all do — the `-u` flag covers everything uniformly.

## Disk constraints (M2 development machine)

The dev machine runs at >90% disk usage by default. Training runs can blow up cache fast — a 5000-iter pretrain with `--save-every=200` left **20 GB of intermediate checkpoints** until we trimmed it. Be aware:

- **Default to no `--save-every`.** A pretrain only needs the final checkpoint (~800 MB). The intermediates are recovery insurance, not artifacts to keep.
- **If you need recovery insurance, always pair with `--save-keep-last-n=2`** (or similar small N). This rolling-cleanup flag is on `base_train.py`, `chat_sft.py`, `chat_rl.py`. Without it, every save adds a permanent ~800 MB to disk.
- **Recommended pretrain pattern:**
  ```
  --save-every=1000 --save-keep-last-n=2     # caps at ~2.4 GB (2 intermediate + final)
  ```
- **Recommended SFT pattern (375-step recipe):**
  ```
  --save-every=100 --save-keep-last-n=2      # caps at ~2.4 GB; keeps recovery for OOM/disk kills
  ```
- **Free space check before kicking off long runs:** `df -h ~/.cache/nanochat`. If <10 GB free, pause and clean up first.
- **Disk-kill symptom:** training crashes silently mid-run (no Python traceback, possibly a "leaked semaphore" warning). macOS kills processes when disk hits critical. wandb will mark the run as a crash.

## Checkpoints — `--model-tag` is mandatory for any non-baseline run

`base_train.py` / `chat_sft.py` / `chat_rl.py` write to `$NANOCHAT_BASE_DIR/{base,chatsft,chatrl}_checkpoints/<model_tag>/`. **When `--model-tag` is unset, it defaults to `d<depth>` (e.g. `d6`).** The default `d6/` directory is where the canonical baseline lives. Running any modified architecture (Hope/NL, Stage 1/2/N, ablations, sweeps) without `--model-tag` will silently overwrite the baseline checkpoint. The `--run` flag (wandb name) does NOT affect the on-disk path — only `--model-tag` does.

**Always** pass `--model-tag=<descriptive-name>` for any non-baseline experiment. Examples that have been used:
- `--model-tag=d6_stage1` (Stage 1 swap)
- `--model-tag=d6_stage2` (Stage 2 learned-gate)
- `--model-tag=d6_b_iso` (SFT recipe A/B)
- `--model-tag=d6_pretrain_stub_a4` (perf derisk)

Before kicking off any training that takes more than a few minutes, **check that the target `<model_tag>` directory does not already contain a `model_<step>.pt` you care about**:
```bash
ls $NANOCHAT_BASE_DIR/base_checkpoints/<model_tag>/  2>/dev/null
```
If it exists with a checkpoint at the same `--num-iterations`, the run will overwrite it without prompting. SFT/RL inherit this footgun via the `chatsft_checkpoints/` and `chatrl_checkpoints/` paths.

The historical baseline numbers (val_bpb 1.174 at d6/5000 iter) are documented in `docs/hope_nl_stage1_full_pretrain_2026-05-01.md` and `HANDOFF.md` even if the on-disk checkpoint is gone.

## Project Memory

Memory files live in `docs/project_notes/`.

**Before proposing changes**: Check `decisions.md` for existing ADRs
**When encountering errors**: Search `bugs.md` for known solutions
**When looking up config**: Check `key_facts.md` for ports, URLs, environments

When resolving bugs or making decisions, update the relevant file.
