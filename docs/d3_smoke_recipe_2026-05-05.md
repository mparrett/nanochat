# d3 smoke recipe — ready-to-launch

**Filed:** 2026-05-05 (pre-launch)
**Status:** Pre-flight passed. Ready to run later tonight.
**Purpose:** Fast-iteration harness/burn-in platform + entertainment chatbot.
**Origin:** Conversation continuing post-session-6 wrap. d3 carve-out
documented in `feat_d8_extension.md:91-108`.

## What this is (and isn't)

**Is:** the cheapest end-to-end nanochat run that produces a chat-able model.
~25-30 min pretrain + ~12-18 min SFT on M2 → ~45 min cold-to-chatbot.
Useful for harness validation, instrumentation burn-in, and seeing what
silly stuff a tiny LM produces.

**Isn't:** an architectural-question platform. At d3, memory layer is 33%
of the network (vs 16% at d6) — any architectural conclusion is distorted.
Numbers from this recipe are not quotable as Hope/NL evidence.

## Pre-flight (verified 2026-05-05)

- ✅ No orphan Python / wandb workers (`pgrep -lf python`, `pgrep -lf wandb`)
- ✅ All CLI flags exist on `base_train` and `chat_sft` (verified via `--help`)
- ✅ Disk: 33 GB free (`df -h ~`), well above d3 footprint (~300 MB total)
- ✅ No existing `d3_smoke` or `d3_smoke_sft` tag — no clobber risk
- ✅ `NANOCHAT_BASE_DIR` unset → defaults to `~/.cache/nanochat` (correct)

**Re-run before launch:**
```bash
pgrep -lf python    # if anything besides remind/non-nanochat → kill before MPS run
pgrep -lf wandb     # should be empty
df -h ~/.cache/nanochat
ls ~/.cache/nanochat/base_checkpoints/d3_smoke 2>/dev/null  # must be empty/missing
```

## Auto-derived d3 sizing

| | d3 | d6 (reference) |
|---|---:|---:|
| `n_embd` | 192 | 384 |
| `n_head` | 3 | 6 |
| `head_dim` | 64 | 64 |
| `n_layer` | 3 | 6 |
| Scaling params (excl embed) | ~1.3M | ~10.6M |
| Embedding table | ~6.3M | ~12.6M |
| Total params | ~7.6M | ~30M |
| Chinchilla iters @ ratio=12 | ~900-1500 | ~5000 |

## Recipe A — vanilla d3 pretrain + SFT (default; launch this)

**Memory: OFF.** No `--hope-*` flags. Plain transformer baseline.

```bash
# Pretrain — ~25-30 min wall on M2
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=1 \
    -m scripts.base_train -- \
    --depth=3 --device-batch-size=32 --total-batch-size=16384 \
    --num-iterations=1500 \
    --eval-every=100 --eval-tokens=524288 \
    --save-every=500 --save-keep-last-n=2 \
    --seed=42 \
    --model-tag=d3_smoke --run=d3_smoke \
    2>&1 | tee /tmp/d3_smoke_pretrain.log

# SFT — ~12-18 min wall on M2
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=1 \
    -m scripts.chat_sft -- \
    --num-iterations=375 --total-batch-size=65536 \
    --eval-every=50 --eval-tokens=524288 \
    --seed=1 \
    --model-tag=d3_smoke --sft-tag=d3_smoke_sft --run=d3_smoke_sft \
    2>&1 | tee /tmp/d3_smoke_sft.log
```

**Notes on the flags:**
- `PYTHONUNBUFFERED=1` is the session-4 hard-won lesson — without it, errors
  hide for hours behind block-buffered stdio. Always set for long-running runs.
- `--num-iterations=1500` overrides the chinchilla auto-calc (~900) to give
  the model a bit more horizon at d3 — pulls out of pure memorization
  territory enough to be entertaining. Adjust down to 1000 if you want
  faster.
- `--eval-every=100` gives ~15 val_bpb samples across the run for a clean
  curve.
- `--eval-tokens=524288` matches the d6 SFT recipe (`key_facts.md`).
  Default `20M` would silently hang.
- `--save-keep-last-n=2` caps disk at 2 intermediate + 1 final ≈ 90 MB.
- `--total-batch-size=65536` for SFT means accum=4 (per the post-bug recipe).

## Recipe B — d3 + memory (only if validating memory machinery)

**Use only when:** testing new memory plumbing (probe hooks, optimizer
routing, `--inherit-from` carrying memory flags). **Do not use** to
re-test "does Stage 2 work."

```bash
# Pretrain with Stage 2 memory at layer 1 (the only middle option at d3)
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=1 \
    -m scripts.base_train -- \
    --depth=3 --device-batch-size=32 --total-batch-size=16384 \
    --num-iterations=1500 \
    --eval-every=100 --eval-tokens=524288 \
    --save-every=500 --save-keep-last-n=2 \
    --seed=42 \
    --hope-additive-memory-layer=1 \
    --hope-memory-kind=learned_gate \
    --hope-memory-w-o-init-scale=1.0 \
    --hope-memory-alpha-init-bias=4.595 \
    --hope-memory-eta-init-bias=-2.197 \
    --model-tag=d3_smoke_stage2 --run=d3_smoke_stage2 \
    2>&1 | tee /tmp/d3_smoke_stage2_pretrain.log
```

Different `--model-tag` from Recipe A so the vanilla checkpoint stays intact.

## Post-run inspection — talk to the silly model

```bash
# CLI smoke prompts
python -m scripts.chat_cli --model-tag=d3_smoke_sft -p "What is the capital of France?"
python -m scripts.chat_cli --model-tag=d3_smoke_sft -p "Spell 'banana' letter by letter."
python -m scripts.chat_cli --model-tag=d3_smoke_sft -p "Once upon a time"

# Web UI for free-form play
python -m scripts.chat_web --model-tag=d3_smoke_sft
# Then browse to the printed http://localhost:... URL
```

## Expected output quality

At d3 / 1500 iters, the model trains on ~25M tokens — about 1/5 of d6's
budget on a model 1/8 the parameter count. Realistic forecast:

- **Bigrams + common phrasings**: yes. No "the the" loops. Sentence
  starters work.
- **Facts (Paris/France/Eiffel)**: probably no. d6 reliably knows these;
  d3 will likely substitute or drift.
- **Coherent multi-sentence reply**: unlikely. Topic drift within 1-2
  sentences.
- **SpellingBee**: maybe partial. SFT teaches the format; capacity may not
  match it. Hover well below d6's 95%.
- **Genuinely funny garbage on open-ended prompts**: yes. The
  entertainment value is real here.

## Things to record (if you want this to be a future reference)

- Final pretrain `val_bpb` (will be much higher than d6's 1.17 — this is
  fine, it's the d3 reference baseline).
- Actual wall per iter (calibrates the 25-30 min estimate above for
  future d3 plans).
- Final SFT `val_bpb`.
- Subjective notes on chat output quality (one or two sample exchanges).

Save as `docs/d3_smoke_results_2026-05-05.md` if it's worth keeping.

## Cleanup if you decide to discard

```bash
rm -rf ~/.cache/nanochat/base_checkpoints/d3_smoke
rm -rf ~/.cache/nanochat/chatsft_checkpoints/d3_smoke_sft
# (and d3_smoke_stage2 if Recipe B was run)
rm /tmp/d3_smoke_*.log
```

Total recovered: ~200-300 MB.

## Cooperative pause

Both training scripts honor `touch /tmp/pause-nanochat` (pause cleanly
between optimizer steps); `rm /tmp/pause-nanochat` to resume. Pre-armed
in `base_train.py:422` and `chat_sft.py`.

## Wandb run names

`d3_smoke` (pretrain) and `d3_smoke_sft` (SFT). Live status:
```bash
python -m dev.wandb_status matt-parrett/nanochat-base/d3_smoke
python -m dev.wandb_status matt-parrett/nanochat-sft/d3_smoke_sft
```
