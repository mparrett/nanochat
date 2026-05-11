# d8 SFT — kickoff handoff

**Written:** 2026-05-10, 22:40 PDT
**For:** next session / teammate kicking off SFT on the d8 baseline
**Base model:** `~/.cache/nanochat/base_checkpoints/d8_overnight/model_005000.pt`
  (val_bpb 1.076, CORE 0.081 quick; full writeup
  `docs/d8_baseline_2026-05-10.md`)

## Goal

Run canonical SFT on the d8 base model and produce a `d8_overnight`-flavored
chat checkpoint. Estimated wall: **3-4 h** on a clean M2, **5-7 h** if
paging-pressured.

## Pre-flight (do not skip)

```bash
# 1) Free as much memory as possible. Today's run showed background
#    workloads (Docker, server.py, faprox, browser tabs, vite/node dev
#    servers) can 2-4× the wall on M2 due to paging.
python3 dev/preflight_memory.py
#    → if exit 1 ("free <1 GB"): kill suspects from the list, re-run.
#    → if "warning": still recommend killing Docker / server.py before
#      a 3-4 h launch.

# 2) Confirm no orphan MPS workers from prior runs
pgrep -lf python | grep -v remind   # only `remind` daemon is OK

# 3) Check disk headroom
df -h ~/.cache/nanochat
#    → SFT saves are ~2-3 GB each. With --save-every=100 +
#      --save-keep-last-n=2 the on-disk ceiling is ~6-9 GB.
```

## Step 1 — port `--grad-checkpoint` into `chat_sft.py` (~10 min)

The pretrain side already has this flag (added 2026-05-09 in commit
`1be94e3`). SFT will hit similar memory pressure at d8; without the
flag, SFT either OOMs or runs with paging-strangled batch sizes.

**The wrap helper already exists** in `nanochat/gpt.py`:
`enable_block_grad_checkpoint()` (idempotent monkey-patch on
`Block.forward`).

Edits needed in `scripts/chat_sft.py`:

1. Add import:
   ```python
   from nanochat.gpt import GPT, GPTConfig, Linear, enable_block_grad_checkpoint
   ```
   (extend the existing `from nanochat.gpt import ...` line; current
   import doesn't include `enable_block_grad_checkpoint`)

2. Add argparse flag (group with memory/throughput flags, near
   `--device-batch-size`):
   ```python
   parser.add_argument(
       "--grad-checkpoint", action="store_true",
       help="enable activation/gradient checkpointing on transformer "
            "Block forwards. Trades ~30%% step time for activation-"
            "memory savings, enabling larger --device-batch-size on "
            "memory-bound machines (M2 24GB). Idempotent. See "
            "docs/grad_checkpoint_smoke_2026-05-09.md.")
   ```

3. Call the helper **before** `torch.compile(model)`. The script's
   compile site is the one that needs to see the patched forward.
   `grep -n 'torch.compile' scripts/chat_sft.py` to find it, then add
   immediately before:
   ```python
   if args.grad_checkpoint:
       enable_block_grad_checkpoint()
       print0("✓ Activation checkpointing enabled on Block.forward")
   ```

**Smoke test the port** before launching the real SFT:
```bash
# 2-iter sanity: should print "✓ Activation checkpointing enabled..."
# and complete two steps without OOM. Bonus: confirms imports + flag.
PYTHONPATH=. uv run python -u -m scripts.chat_sft \
  --model-tag=d8_overnight --model-step=5000 \
  --num-iterations=2 --device-batch-size=4 --total-batch-size=8192 \
  --eval-every=-1 --chatcore-every=-1 \
  --grad-checkpoint --run=dummy \
  --sft-tag=d8_sft_smoke_will_delete
# Clean up the smoke ckpt dir afterward:
rm -r ~/.cache/nanochat/chatsft_checkpoints/d8_sft_smoke_will_delete
```

Commit the port as one focused change. Suggested message:
`scripts/chat_sft: --grad-checkpoint flag (parity with base_train)`.

## Step 2 — capacity probe (~5 min, optional but recommended)

Before committing to a 3-4 h run, find d8 SFT's working batch ceiling
on this hardware. d6 SFT canonical was `--device-batch-size=32
--total-batch-size=65536` (grad-accum=4). At d8 we likely need to
reduce. The pretrain probe found d8 fits at batch=8 with compile + gc
and OOMs at batch=16; SFT has similar memory characteristics with
some extra cost from the longer context / different loss path.

```bash
# Probe batch=8 first (proven to fit at d8 pretrain):
PYTHONPATH=. uv run python -u -m scripts.chat_sft \
  --model-tag=d8_overnight --model-step=5000 \
  --num-iterations=4 --device-batch-size=8 --total-batch-size=16384 \
  --eval-every=-1 --chatcore-every=-1 \
  --grad-checkpoint --run=dummy \
  --sft-tag=d8_sft_probe_b8

# If batch=8 fits cleanly, optionally bracket up to batch=12 / 16
# (ascending until OOM, same strategy as the d10/d8 probes documented
# in HANDOFF.md Day 2026-05-09).

# Clean up after:
rm -r ~/.cache/nanochat/chatsft_checkpoints/d8_sft_probe_b8
```

Decide the final SFT batch from probe data. **If you're short on time,
just use `--device-batch-size=8 --total-batch-size=16384`** — it's
guaranteed to fit; the bigger-batch question is a future optimization,
not a kickoff blocker.

## Step 3 — canonical SFT launch

The canonical SFT recipe is 375 optimizer steps. With wandb on (per
the project preference codified in `key_facts.md`) and the conservative
batch from above:

```bash
# Pre-flight one more time (5-10 min between probe and real launch
# can be enough for browser tabs etc. to wake up):
python3 dev/preflight_memory.py

# Launch via nohup so the run survives session end:
nohup /usr/bin/time -l env PYTHONUNBUFFERED=1 PYTHONPATH=. \
  uv run python -u -m scripts.chat_sft \
    --model-tag=d8_overnight --model-step=5000 \
    --sft-tag=d8_overnight \
    --num-iterations=375 \
    --device-batch-size=8 --total-batch-size=16384 \
    --eval-every=50 --eval-tokens=524288 \
    --chatcore-every=-1 \
    --save-every=100 --save-keep-last-n=2 \
    --grad-checkpoint \
    --run=d8_sft_overnight \
  > /tmp/d8_sft.log 2>&1 &
echo "PID: $!" | tee /tmp/d8_sft.pid
```

**Flag rationale:**
- `--sft-tag=d8_overnight` writes to a separate dir
  (`chatsft_checkpoints/d8_overnight/`), keeps the base ckpt untouched.
- `--eval-every=50 --eval-tokens=524288`: load-bearing per
  `key_facts.md` SFT recipe. Defaults are wrong (eval-tokens=21M takes
  ~67 min per eval on M2 and stalls the run).
- `--chatcore-every=-1`: skip the categorical-eval OOM path during the
  run; eval ChatCORE on the final ckpt separately.
- `--save-every=100 --save-keep-last-n=2`: rolling cleanup keeps disk
  under ~6 GB.
- `--run=d8_sft_overnight`: per the wandb default preference.
  `wandb.init` is now defensive (wrapped in try/except — commit
  `1d06e6d`), so a wandb failure won't kill the run.
- **No `--total-batch-size=65536`** (the d6 canonical). At
  device-batch=8 that'd be grad-accum=4 ≈ 4× the per-step time. If
  capacity allows the bigger batch, prefer it (better gradient
  quality); otherwise single-grad-accum is fine for a first pass.

## Step 4 — evaluate

After SFT completes:

```bash
# ChatCORE on the final SFT checkpoint
PYTHONPATH=. uv run python -u -m scripts.chat_eval \
  --model-tag=d8_overnight --sft-tag=d8_overnight \
  --max-per-task=100 --run=d8_chat_eval

# Or for the canonical (more precise but slower) headline:
# --max-per-task=500
```

## Expected outcomes

- **Training loss**: should descend smoothly. d6 SFT typically hits
  train loss ~1.5-2.0 by step 375; d8 should track similarly or
  slightly better.
- **Validation loss**: improves through the run; check `--eval-every=50`
  output for the trajectory.
- **ChatCORE**: d6_stage2 SFT hit 0.1744 on M2 (per
  `docs/hope_nl_stage2_chatcore_2026-05-03.md`). d8 should match or
  slightly improve; if it regresses below ~0.15 something's off in
  the SFT recipe or the base model is underbaked.
- **Wall**: 3-4 h clean, 5-7 h if paging.

## Things to watch / known risks

- **Memory pressure rebuilds.** Even after a fresh start, browser
  tabs / IDE caches accumulate. If step time climbs noticeably mid-run,
  kill background processes — operator did this twice during the
  pretrain and each cleanup recovered ~2× throughput.
- **`chat_sft.py` may have other defaults that differ from the
  pretrain script.** Especially `--eval-tokens` default is much larger
  than what's reasonable on M2 — already handled in the command above
  but worth confirming `grep eval_tokens scripts/chat_sft.py` if
  anything looks off.
- **The base d8 model is sub-Chinchilla** (82 M tokens trained at
  pretrain). Don't expect SFT to perform miracles — the base model
  hasn't seen GPT-2-budget data yet. ChatCORE numbers should be
  interpreted with that in mind.

## If you get stuck

- `docs/d8_baseline_2026-05-10.md` — full pretrain writeup, caveats,
  artifact paths.
- `docs/grad_checkpoint_smoke_2026-05-09.md` — full validation of the
  pattern this SFT will rely on; explains the compile / no-compile
  interaction.
- `docs/project_notes/key_facts.md` — wandb default, preflight rule,
  SFT recipe canon.
- `docs/sft_oom_investigation_2026-05-03.md` — prior SFT OOM debugging
  (block-buffering + ChatCORE memory peak). Use `python -u`,
  `--chatcore-every=-1`, and an empty_cache around saves — these are
  already in the command above but were hard-won lessons.
- `HANDOFF.md` — day-by-day session log; today's section caps the d8
  pretrain story.
