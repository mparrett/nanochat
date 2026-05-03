# DEV.md — Architecture & detailed reference

This is the depth reference for working in nanochat. Operational guidance (footguns, save discipline, `--model-tag`) lives in `CLAUDE.md`. This file is for "how the code is organized and why."

## Setup detail

```bash
uv sync --extra gpu                  # CUDA (A100/H100/etc.)
uv sync --extra cpu                  # CPU / MPS (e.g. Apple Silicon)
uv sync --extra gpu --group dev      # adds pytest, matplotlib, ipykernel, transformers
source .venv/bin/activate
```

`gpu` and `cpu` extras are mutually exclusive (declared in `pyproject.toml`). `torch` is pinned to 2.9.1 (some forks here use 2.11 — check `pyproject.toml`).

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

Reference end-to-end pipelines:
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

**Optimizer (`nanochat/optim.py`).** Custom MuonAdamW (single GPU) and DistMuonAdamW (distributed). Different parameter groups get different LRs: `embedding-lr`, `unembedding-lr`, `matrix-lr` (Muon), `scalar-lr` (for `resid_lambdas`, `x0_lambdas`, plus 1D gate biases like Stage 2's `b_alpha`/`b_eta` routed via name filter in `setup_optimizer`).

**Precision (`nanochat/common.py`).** No `torch.amp.autocast`. A single global `COMPUTE_DTYPE` is auto-detected (bf16 on SM 80+, fp32 elsewhere) and overridable via `NANOCHAT_DTYPE=bfloat16|float16|float32`. Master weights stay fp32; the custom `Linear` layer in `gpt.py` casts weights to `COMPUTE_DTYPE` in the forward pass. Embeddings are stored directly in `COMPUTE_DTYPE`. fp16 auto-enables a `GradScaler` in `base_train.py` (SFT also; RL does not).

Note: nanochat's custom `Linear` skips bias even when `bias=True` — its forward only uses `self.weight`. If you need a learnable bias on a projection, use a separate `nn.Parameter(torch.zeros(...))` and add it explicitly in forward (see `LearnedGateLinearMemory.b_alpha` / `b_eta` in `gpt.py`).

**Distributed.** `compute_init()` / `compute_cleanup()` in `common.py` handle DDP setup from `torchrun` env vars. Optimizer state is sharded per-rank; checkpoints save model on rank 0 and `optim_<step>_rank<N>.pt` per rank (`checkpoint_manager.py`).

**Checkpoints (`nanochat/checkpoint_manager.py`).** Stored under `$NANOCHAT_BASE_DIR/{base,chatsft,chatrl}_checkpoints/<model_tag>/{model_<step>.pt, meta_<step>.json, optim_<step>_rank<N>.pt}`. `load_model("base"|"sft"|"rl")` resolves the right dir; missing model_tag → largest `d<N>` available; missing step → latest. `_patch_missing_*` shims handle older checkpoints with absent newer config keys.

`save_checkpoint(...)` accepts an optional `keep_last_n` parameter for rolling cleanup of older intermediates — wired through to the `--save-keep-last-n` CLI flag on all three training scripts (see CLAUDE.md disk-constraint section).

**Data flow.** `nanochat/dataset.py` downloads ClimbMix-400B parquet shards on demand to `$NANOCHAT_BASE_DIR/base_data_climbmix/`. `nanochat/dataloader.py` provides the tokenizing distributed data loader (`tokenizing_distributed_data_loader_bos_bestfit` and a state-resumable variant).

**Inference (`nanochat/engine.py`).** KV-cache engine used by `chat_cli`, `chat_web`, and during eval. `nanochat/execution.py` lets the model call out to a Python interpreter as a tool (used by tasks like `gsm8k`/`humaneval`).

**Base directory.** All artifacts (datasets, tokenizer, checkpoints, reports) go under `$NANOCHAT_BASE_DIR` (default `~/.cache/nanochat`). `get_base_dir()` in `common.py` is the single source of truth.

**Logging.** wandb is optional — `--run dummy` (the default) routes through `DummyWandb` and logs nothing. Set `WANDB_RUN=<name>` env var or `--run=<name>` to enable real logging (run `wandb login` first). `base_train.py` uses `project="nanochat"`; `chat_sft.py` uses `project="nanochat-sft"`; `chat_rl.py` uses `project="nanochat-rl"` — each phase shows up in its own wandb project.

## Conventions (architectural)

- Don't introduce config objects, model factories, or wide if/else trees. Prefer adding a few flags or editing a script directly. Three similar lines is better than a premature abstraction.
- Changes that touch training have to make sense across the depth sweep, not just one model size.
- The CORE metric (DCLM) on the base model is the headline number; `val_bpb` (bits per byte) is the loss proxy. `time-to-GPT-2` (CORE > 0.256525 wall-clock) is the speedrun leaderboard metric — see `dev/LEADERBOARD.md`.

## Hope/NL experiment-specific (current branch only)

Branch `experiment/hope-nested-learning` adds Stage 0/1/1.5/2 of a Hope/Nested-Learning prototype. See `HANDOFF.md` for full status and `docs/hope_nl_stage*.md` for stage-by-stage writeups. Key architectural additions in `nanochat/gpt.py`:

- `LinearAttentionMemory` (Stage 1): Titans-style fast-weight memory with parallel form, `alpha=eta=1` fixed.
- `LearnedGateLinearMemory` (Stage 2): per-token learned `alpha`/`eta` via prefix log-products; subclasses `LinearAttentionMemory`.
- Config switch `hope_memory_kind ∈ {linear, learned_gate}`; insertion via `hope_memory_layer` (swap) or `hope_additive_memory_layer` (additive).
- `hope_memory_w_o_init_scale=1.0` is the recommended default for any memory-bearing config (the `=0` cold-start trap is documented in `docs/hope_nl_stage1_5b_w_o_init_2026-05-01.md`).
