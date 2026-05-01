# Resume here after /compact

**Branch:** `experiment/hope-nested-learning`
**Last commit:** `5ceaccc` (Phase 3 step 2 GPU profiling + handoff)
**Date:** 2026-04-30 (late)

## Read these first (in order)

1. `HANDOFF.md` — full project context, what's been done across Phase 1/2/3, where everything lives.
2. `docs/phase3_gpu_profiling_2026-04-30.md` — the GPU dispatch findings (319 dispatches/iter, ~8 ms each, no fusion).
3. `docs/project_incoming/codex-2026-04-30-2046.md` — Codex's first review of our perf work.
4. **The Codex follow-up reply** below in this doc (his second message after seeing our Phase A findings).

## Codex's second reply (verbatim, captured here so we don't lose it)

> Those signpost numbers are a very clean confirmation. If dispatch cost is flat-ish, then optimizer-step frequency is the main lever.
>
> One caution on the SFT accum A/B: make the comparison explicit as either iso-token or iso-update. For example:
>
> - `accum=1, 1500 steps`: baseline update count, baseline tokens.
> - `accum=4, 375 steps`: iso-token, fewer updates, tests whether amortization preserves quality per token.
> - `accum=4, 1500 steps`: iso-update, 4x tokens, tests best quality per wall-clock but is not apples-to-apples.
>
> Also watch LR scaling. `base_train.py` scales LR by `sqrt(B/B_ref)` when `total_batch_size` changes; `chat_sft.py` appears to mostly inherit or set LRs directly, so accum changes may not get equivalent tuning unless you do it intentionally.
>
> For `ve_gate`, I'd split it by module name, not shape, because the tiny shape alone may later collide with some other experimental matrix. Treat it like `smear_gate`: AdamW, small LR, probably no weight decay or very light decay. Then verify with a 1-iter signpost count before any long run. If optimizer dispatches drop by roughly one Muon group and loss step 1 is sane, it's worth an SFT probe.
>
> The Phase B capture is useful, but if accum lands at ~3x tok/sec, I'd make Phase B diagnostic-only. At that point the cost floor moves back toward model fwd/bwd, and Hope/NL design choices matter more than shaving another tiny optimizer dispatch group.

## The plan (in one paragraph)

We have two perf experiments to run. **Step 1** is small: move `ve_gate` from Muon to AdamW (filtered by name not shape, like `smear_gate`), verify the dispatch count drops by ~one Muon group via signpost, sanity-check loss. **Step 2** is the headline experiment: a gradient-accumulation iso-token A/B on SFT to test whether amortizing fewer optimizer steps preserves quality. If step 2 lands as predicted (~3× tok/sec), the perf wall moves back to fwd/bwd and we pivot to Hope/NL Stage 0 implementation rather than chasing more optimizer optimizations.

---

## Step 1 — `ve_gate` → AdamW

### What to change

In `nanochat/gpt.py`, `setup_optimizer()` (around line 379–410). Right now matrix params are collected then grouped purely by shape. We need to:

1. Find where `value_embeds` and their gates (`ve_gate`) are defined — `grep -n "ve_gate\|value_embed" nanochat/gpt.py`
2. In `setup_optimizer`, filter `ve_gate` params *out of* `matrix_params` and *into* an AdamW group (treat exactly like `smear_gate` — small LR, light/no weight decay).
3. Make the filter by **module name** (e.g. `'.ve_gate' in name`), not shape. Codex was explicit on this.

### How to verify

Run the existing signpost profiler:

```bash
source .venv/bin/activate
python -u -m dev.profile_mps_signpost > /tmp/signpost_after_vegate.log 2>&1
```

Then pull dispatch counts (template below — replace timestamps):

```bash
/bin/bash -c "log show --signpost --start '<START>' --end '<END>'" 2>&1 \
  | grep " <PID> " > /tmp/signposts.txt
grep -c "MPSGraph_Encode" /tmp/signposts.txt
```

**Expected outcome:** total dispatch count should drop from ~319 to roughly 280–290 (one fewer Muon group = 30–40 fewer dispatches). Also confirm loss-step-1 in the log is the standard `~10.39` (sanity check that we haven't broken anything).

### What to commit

Single small commit on the experiment branch:
```
optim: move ve_gate from Muon to AdamW

Phase 3 perf work, per Codex review. ve_gate is a tiny gating
matrix that doesn't benefit from Muon's orthogonalization but
costs a full Polar Express dispatch group (~30-40 dispatches/iter).
Move to AdamW with small LR and light weight decay, matching how
smear_gate is handled.

Verified: signpost dispatch count drops from ~319 to ~XXX per
iter on M2. Loss step 1 unchanged at ~10.39.
```

---

## Step 2 — Gradient accumulation iso-token A/B on SFT

### The two runs

**Baseline:** `accum=1, 1500 steps` (current default)

```bash
source .venv/bin/activate
PYTHONUNBUFFERED=1 caffeinate -i python -u -m scripts.chat_sft \
    --max-seq-len=512 --device-batch-size=32 --total-batch-size=16384 \
    --eval-every=-1 --eval-tokens=262144 --chatcore-every=-1 \
    --num-iterations=1500 \
    --model-tag=d6 --run=dummy 2>&1 | tee /tmp/sft_accum1.log
```

**Experiment:** `accum=4, 375 steps, LRs scaled by sqrt(4)=2×`

The pretrain checkpoint inherits these defaults: `embedding_lr=0.3, unembedding_lr=0.008, matrix_lr=0.02`. Multiply by 2:

```bash
source .venv/bin/activate
PYTHONUNBUFFERED=1 caffeinate -i python -u -m scripts.chat_sft \
    --max-seq-len=512 --device-batch-size=32 --total-batch-size=65536 \
    --embedding-lr=0.6 --unembedding-lr=0.016 --matrix-lr=0.04 \
    --eval-every=-1 --eval-tokens=262144 --chatcore-every=-1 \
    --num-iterations=375 \
    --model-tag=d6 --run=dummy 2>&1 | tee /tmp/sft_accum4.log
```

**Note:** we re-enabled `--eval-tokens=262144` (was 0 in some earlier runs). We *want* the final val_bpb for comparison.

**Important:** SFT writes to `~/.cache/nanochat/chatsft_checkpoints/d6/` and overwrites by step number. Back up the existing one first if you care about it:

```bash
cp -R ~/.cache/nanochat/chatsft_checkpoints/d6 ~/.cache/nanochat/chatsft_checkpoints/d6_baseline
```

Or use a different `--model-tag` for one of the runs if there's collision (chatsft_checkpoints uses the model_tag from the loaded base).

### What to compare

At end of each run, the log prints `Step XXXX | Validation bpb: Y.YYYY` (followed by save messages). Compare:

- **Final val_bpb at step 1500 (accum=1) vs step 375 (accum=4)** — same total token budget, both with final eval. If accum=4 val_bpb is within ~5% of accum=1, amortization preserves quality and we win.
- **Wall-clock time** — accum=4 should be ~3× faster per token. Watch for thermal throttling on the longer run.
- **Smoke test on each saved model:**
  ```bash
  python -m scripts.chat_cli -p "What is the capital of France?"
  ```
  Both should produce coherent answers. If accum=4 produces gibberish, training collapsed despite val_bpb being OK — escalate.

### What to commit

Don't commit code changes (there aren't any for step 2 — just CLI args). Commit a writeup:

```
docs: phase 3 step 3 - gradient accumulation A/B results

Result table here:
- accum=1, 1500 steps: val_bpb X.XX, wall YYY min
- accum=4, 375 steps:  val_bpb X.XX, wall YYY min, ZZ% per-token speedup

Smoke test: <coherent / gibberish / similar>

Decision: <accum=4 is the new SFT default | revert | needs more validation>
```

Also update `docs/phase3_gpu_profiling_2026-04-30.md` to link the result.

---

## After both steps done

1. **Reply to Codex** with a results-only message — "step 1 dropped dispatches from 319 to XXX; step 2 iso-token A/B showed val_bpb X.XX vs Y.YY at Z× wall speedup." Use the format from previous Codex exchanges (see `docs/project_incoming/codex-2026-04-30-2046.md` and our reply on clipboard).
2. **Decision point:** if accum delivers as predicted, Phase B (`metal_capture`) drops to "diagnostic-only" priority. Pivot to Hope/NL Stage 0 — start reading the original ticket at `~/projects-new/trx4mr/docs/idea-hope-nested-learning.md` and Stage 0 implementation per the plan in `HANDOFF.md`.
3. If accum *doesn't* deliver (val_bpb significantly worse), regroup. Possibilities: LR scaling was wrong, SFT distribution doesn't tolerate larger batches, or we hit a different bottleneck. Codex would likely have specific suggestions.

## State of things at handoff

- Working tree clean (just `.claude/scheduled_tasks.lock` untracked, irrelevant)
- Web UI not running (was earlier, killed)
- No background jobs
- torch 2.11.0 active in venv
- Base checkpoint preserved at `~/.cache/nanochat/base_checkpoints/d6/model_005000.pt` (val_bpb 1.174)
- SFT checkpoint at `~/.cache/nanochat/chatsft_checkpoints/d6/model_001499.pt` (most recent successful run, slightly stale though — gate move + accum runs will overwrite unless you back up)
- Upstream PR `karpathy/nanochat#741` is filed and independent of this branch
