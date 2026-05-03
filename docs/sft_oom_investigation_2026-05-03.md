# Stage 2 SFT OOM investigation post-mortem (2026-05-03)

**TL;DR:** chasing a "save-time deadlock" wasted ~6 hours and three SFT attempts on M2. The actual cause was a **GPU OOM during ChatCORE eval at step 200**. The error message had been printing to stdout the whole time but was hidden by Python's default block-buffering on a redirected log file. Once we added `python -u` and an unrelated MPS hygiene fix that got us past the save, the OOM message appeared on the very next run. The investigation produced four durable wins regardless: a CLAUDE.md MPS-hygiene rule, a checkpoint pre-flight guard, MPS allocator metrics shipped to wandb, and `torch.mps.synchronize + empty_cache` before each save.

This is a debugging post-mortem. It's here so we don't repeat the methodology mistakes — the technical fixes are documented in `docs/project_notes/bugs.md`.

## The symptom

Three Stage 2 additive SFT runs on M2 24GB, all using a ~590K-extra-param `LearnedGateLinearMemory` block alongside the existing MLP at L3:

| run | settings | observed failure |
|---|---|---|
| 1 | bs=32, eval-every=50, save-every=100 | killed at step 184 by macOS (disk OOM, separate issue) |
| 2 | bs=32 (after disk cleanup) | hung at step 184; process U-state, 0% CPU, no error message |
| 3 | bs=16, halved eval-tokens, `python -u` | hung at step ~200; same U-state, no error message |

The pattern fooled us: each run got slightly further than the last, suggesting the fix-of-the-moment was helping. We were tightening the wrong screws.

## The misdirections

### Misdirection 1: "save-time MPS→CPU deadlock"

`sample <pid>` showed the active thread stuck on `_MTLCommandBuffer waitUntilCompleted`. The buffered local log showed step 184 as the last entry. The chatsft checkpoint dir contained model_000100.pt but no model_000200.pt. From this we deduced the save at step 200 was where things broke.

This was **plausible but wrong**. The model save *would* have failed under those conditions, but it never even started — something earlier in step 200 had already broken the GPU.

### Misdirection 2: "(B, T, T) activation pile-up"

We accounted for Stage 2 additive's extra activation memory: 4× (B, T, T) tensors that swap doesn't have, plus the MLP isn't replaced. At B=32, T=512, bf16, that's ~130 MB extra. We hypothesized the save's MPS→CPU transfer was triggering pressure on top of fragmentation.

This is **architecturally correct but causally wrong**. The (B, T, T) tensors are real and add real activation memory. But they're released after each forward+backward step. They don't pile up across steps. The actual OOM trigger was a different memory event entirely.

### Misdirection 3: "MPS allocator fragmentation"

We added `torch.mps.synchronize() + torch.mps.empty_cache()` before every save in `checkpoint_manager.save_checkpoint()`. This was a good defense against a real but adjacent class of issues (heavy MPS workloads accumulating cached driver memory; `_MTLCommandBuffer waitUntilCompleted` deadlocks under fragmentation are a documented Apple Silicon issue). And it *did* let the next run get further — past the save and into ChatCORE eval, where it OOM'd loudly.

The empty_cache fix gets to *stay* — it's good hygiene independent of this bug. But it didn't fix the root cause; it cleared the fog around it.

## What finally surfaced the real error

Two changes, applied together on the third retry:

1. **`python -u`** (unbuffered stdout) — Python's default 4-8 KB block buffering on file redirects had been hiding error messages until they were swallowed by the kill. The CLAUDE.md system rule about `python -u` was added late in the session; if it had been there from run 1 we'd have seen the OOM message immediately.

2. **`torch.mps.synchronize + empty_cache`** before save — kept the run progressing through the save and into ChatCORE eval, where the actual error printed to the now-unbuffered log:

```
Error: command buffer exited with error status.
  Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

Total time-to-truth from "add the unbuffered flag and the save fix": about one ChatCORE eval cycle (~12 minutes). Cumulative time spent before that: roughly 6 hours across three failed runs.

## The actual root cause

`chat_sft.py --chatcore-every` defaults to **200**. At step 200, the script runs a full benchmark sweep:

### Why ChatCORE blows up — the T² hypothesis (falsified by data)

**Hypothesis (initially):** ChatCORE eval prompts are routinely longer than the model's training `--max-seq-len`, and `chat_eval.py::run_categorical_eval` pads each batch to the longest prompt in that batch:

```python
# chat_eval.py:107-115
prompt_ids = [tokenizer.render_for_completion(c) for c in conversations]
max_length = max(len(ids) for ids in prompt_ids)        # ← max over the WHOLE batch
padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
logits = model(prompt_ids)  # (B, T, V) where T = max_length
```

So one long MMLU few-shot prompt in a batch of 32 → the *whole* batch is forwarded at T=1500+. The model still runs (RoPE cache is `seq_len * 10 = 5120`, plenty of headroom; this is the ground covered by upstream issue #514) — but **activation memory scales as B × T²**, and at T=1500 vs the trained T=512 that's a 9× blow-up.

The numbers, at B=32 in bf16:

| component at one MMLU eval batch | T=512 (train) | T=1500 (eval) |
|---|---:|---:|
| one (B, T, T) tensor | 33 MB | **288 MB** |
| Stage 2's 4 of them at L3 | 132 MB | **1.15 GB** |
| MLP activation (B, T, 4D) | 25 MB | 73 MB |
| attention scores per layer × 6 | 198 MB | 1.7 GB |
| **total activation per fwd** | ~400 MB | **~3 GB** |

That ~3 GB stacks on top of model weights (~280 MB) + still-loaded AdamW optimizer state (~600 MB) + Engine KV cache + normal Python/Metal overhead. The M2's 24 GB unified-memory GPU budget runs out somewhere along the categorical-eval sweep — ARC-Easy (2376 problems, shorter prompts) succeeded; ARC-Challenge (1172 problems) succeeded; MMLU's longer few-shot prompts were the trigger.

**Validation result (2026-05-03):** ran `dev/chatcore_prompt_lengths.py` on all 6 ChatCORE tasks (500 problems sampled per task; 164 / 256 for the smaller ones). Result:

```
task             n    max    p50    p90    p99   >512    pct
ARC-Easy       500    200     71    103    129      0    0.0%
ARC-Challenge  500    209     81    115    163      0    0.0%
MMLU           500    598     95    248    472      1    0.2%
GSM8K          500    145     60     96    128      0    0.0%
HumanEval      164    426    137    248    341      0    0.0%
SpellingBee    256     40     17     28     36      0    0.0%
```

**Hypothesis falsified.** Almost no prompts exceed 512. The single MMLU outlier at 598 tokens is barely over and is one of 500. Activation memory at typical T (60–250) is *tiny* — one (B, T, T) tensor at B=32, T=100 is 0.64 MB at bf16. Even Stage 2's 4× of those is ~2.6 MB — not the OOM driver.

### Updated mechanism — cumulative MPS allocator state

If T² isn't the cause, what is? Most likely: **cumulative MPS allocator fragmentation across hundreds of ChatCORE forward passes.** With `--chatcore-max-cat=-1` (unlimited, default), MMLU runs all 14,042 problems → 439 batches at bs=32 → 439 forward passes. Each pass has different prompt-length composition (variable padding), so each allocates slightly differently. MPS's caching allocator generously holds onto freed chunks (we saw 4 MB of live tensors → 33 MB driver-allocated cache earlier). Over hundreds of variable-shaped forwards, the driver accumulates fragmented unusable chunks.

Combined with the still-loaded training state (model + AdamW optimizer state ~880 MB), the driver eventually can't satisfy the next allocation request. Hence `kIOGPUCommandBufferCallbackErrorOutOfMemory`.

This is consistent with why the `torch.mps.empty_cache()` we added before *save* helped *that* operation: it consolidates the allocator. It just doesn't fire during ChatCORE eval, where the same pressure builds up across hundreds of forwards.

**Predictive consequences:**

1. **Lowering `--chatcore-max-cat` and `--chatcore-max-sample` should help proportionally.** Fewer forwards → less fragmentation. A run with `--chatcore-max-cat=200` per task would do ~6 batches per categorical task, far fewer fragmentation cycles.

2. **Adding `torch.mps.empty_cache()` between ChatCORE tasks** (or every N forward passes inside the eval loop) should also help. Cheap; would test the fragmentation theory directly.

3. **bs=16 didn't help because** the memory savings per forward (B halved) is small relative to the cumulative cache over many forwards. The structural problem is "many forwards" not "big forwards."

**The Stage 2 deployment concern remains** — but reframed. Additive's 4× (B, T, T) tensors mean each forward has 4× more allocation churn than baseline. At training T=512 with grad accumulation, this releases cleanly per microbatch. At inference under MPS's caching allocator, the churn accumulates faster. For long-running inference workloads, this might still favor swap over additive — but the dominant Stage 2 deployment cost on M2 is the cumulative-allocations issue, not the per-forward T² blow-up.

Original task list at chat_sft.py:421:

```python
all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
```

### Validation: hypothesis falsified, mechanism reframed

(See "Validation result" subsection above.) `dev/chatcore_prompt_lengths.py` showed prompts are ~all under 512. T² scaling cannot be the dominant cause. Reframing toward cumulative-allocation-state.

**Next validation steps for the new mechanism:**

- **Cheap:** add `torch.mps.empty_cache()` calls every N batches inside `run_categorical_eval`'s loop, then run filtered ChatCORE on the existing Stage 2 model. If the OOM goes away, fragmentation is the cause. ~1h.
- **Diagnostic:** with the `mps_metrics()` we added in commit `f4f5066`, log allocator state per batch (not just per eval step). Watch for `mps/cache_gb` climbing monotonically vs `mps/allocated_gb` flat — that's the fragmentation fingerprint.
- **Prevention for future runs:** lower `--chatcore-max-cat` to bound the per-task forward count. e.g. `--chatcore-max-cat=200` runs ~6 batches per task instead of 439 for MMLU. Same accuracy signal, dramatically less allocator churn.

Each task instantiates fresh inference batches with KV cache via `nanochat/engine.py::Engine`. ARC-Easy and ARC-Challenge completed (logged 24.79% and 25.94% respectively, which is chance for 4-way multiple choice — model is barely trained at step 200). The third task (MMLU, by ordering) was where the OOM hit.

The math, roughly:
- d6 model + Stage 2 weights: ~280 MB
- AdamW optimizer state (m + v per param): ~600 MB
- Stage 2 forward/backward activations (peak): ~200 MB
- ChatCORE Engine KV cache @ batch=24, seq=512: ~120 MB per task's inference run
- Plus normal Python/Metal/wandb overhead

We're already at ~1.2 GB on the GPU under training. Adding ChatCORE's KV cache while the optimizer state still resides pushed past the M2's per-process GPU allocation ceiling.

**Why pretrain at the same shape worked:** `base_train.py --core-metric-every` defaults to `-1` (disabled). Pretrain only runs `evaluate_bpb` for its eval, which is bounded by `--eval-tokens` and reuses the training KV path. No multi-task benchmark sweep, no extra Engine instantiation. Same GPU ceiling, but only training memory needed at once.

**Why Stage 1 swap SFT worked at bs=32 earlier:** *it didn't, in the sense that we never tested it.* Phase 3 SFT explicitly disabled ChatCORE (`docs/phase3_step3_grad_accum_2026-05-01.md`: *"Eval is `eval_tokens=262144` worth of val_bpb on the SFT mixture; not a downstream eval like ARC/GSM8K/MMLU"*). Stage 1 SFT inherited that Phase 3 recipe. So ChatCORE-during-SFT had never been tested on this M2 24GB hardware before Stage 2.

The OOM is **not Stage-2-specific** — baseline SFT or Stage 1 swap SFT would also OOM if `--chatcore-every` were left at the default 200. The `Engine(orig_model, tokenizer)` instantiation at `chat_sft.py:420` allocates a fresh KV cache (~100+ MB at bs=24) on top of the still-loaded training model + AdamW optimizer state (~280 + 600 MB). Stage 2's extra ~590K params and (B, T, T) tensors *contribute* to the ceiling but aren't the deciding margin. The ceiling is the M2's 24 GB unified-memory GPU allocation budget, end of story.

**Related upstream issue:** karpathy/nanochat#592 (closed, fix in PR #593) — VRAM spike in `disable_fp8` context manager used by `base_train.py::evaluate_core`. We have the fix (`device="meta"` at `base_train.py:248`), but it's a different code path — used only on CUDA + `--fp8` runs, not our M2 SFT. The shape of the bug (eval-time fresh GPU allocation pushing past VRAM ceiling) is the same family as ours; their fix happens to not apply here because chat_sft's ChatCORE goes through `Engine`, not `disable_fp8`.

## The workaround and the fix

```bash
python -u -m scripts.chat_sft \
    --max-seq-len=512 --device-batch-size=32 --total-batch-size=65536 \
    --eval-every=50 --eval-tokens=524288 \
    --num-iterations=375 \
    --save-every=100 --save-keep-last-n=2 \
    --chatcore-every=-1 \
    --model-tag=d6_stage2 \
    --run=sft-stage2-d6-no-chatcore
```

The one new flag is `--chatcore-every=-1`. ChatCORE can be run separately after SFT completes via a dedicated eval script, or set it to fire at the final step only (`--chatcore-every=375`). Reducing `--chatcore-max-cat` and `--chatcore-max-sample` would also help if ChatCORE-during-SFT is desired.

## What we got out of this even though we chased ghosts

Four durable improvements are in main as a result:

1. **`python -u` rule lifted into the system-level CLAUDE.md.** Operator confirmed they added it. Future runs won't have block-buffering hide error messages from unbuffered logfiles.

2. **MPS hygiene rule lifted into project CLAUDE.md.** The `pgrep -lf python` check before any MPS run, and the orphan-worker context, was previously buried in `docs/m2_pipeline_2026-04-30.md:102`. Now in CLAUDE.md so every Claude session loads it.

3. **`assert_checkpoint_dir_safe()` pre-flight guard** in `checkpoint_manager.py`. Aborts at startup if the target `<model_tag>` directory already has `model_*.pt`. Stops the next "we forgot --model-tag and overwrote the baseline" disaster (commit `e7852b6`).

4. **MPS allocator metrics shipped to wandb** via `nanochat.common.mps_metrics()`. The four keys (`mps/{allocated,driver,cache}_gb` plus `recommended_max_gb`) fill the gap that wandb's built-in system monitor doesn't cover. Future runs will visibly chart the allocator climb that precedes a real fragmentation event, distinguishing it from a hard OOM.

5. **`torch.mps.synchronize + empty_cache` before every checkpoint save** in `save_checkpoint()`. Real save-time hygiene that benefits any MPS run, even ones not hitting this specific bug.

6. **`--save-keep-last-n` rolling cleanup** wired through all three training scripts. The disk-OOM that started the session (run 1's kill) was caused by 19 GB of Stage 1 intermediate checkpoints accumulating from `--save-every=200` with no cap.

## Methodology lessons

The technical lessons are in `docs/project_notes/bugs.md`. The methodology lessons we want to remember:

### When a process hangs silently, the error has *already happened* — find where it printed

Three SFT runs hung with no visible error. The error was visible in stdout the whole time; we just couldn't see it. The lesson is two-fold:

- **Default to `python -u`.** Block-buffering is the silent killer of post-mortem debugging.
- **`sample <pid>` only tells you where the *current* state is, not how it got there.** A thread stuck on `waitUntilCompleted` *might* be waiting on a deadlock, OR waiting on a command buffer that already failed and is now retiring with an error. The thread snapshot looks identical in both cases.

### Misdiagnoses follow shape, not first principles

We chased "save-time deadlock" because:
- Process hung "around" step 200 (where save-every triggered)
- `sample` showed Metal stuck (which we associated with save's GPU→CPU copy)
- The chatsft dir was missing model_000200.pt (so "the save didn't happen" felt obvious)

But none of that proved the save was the failure point. We needed to read the error message — which we couldn't, because of buffering. **When the data is consistent with multiple stories, get more data; don't pick the most plausible story.**

### Memory accounting is necessary but not sufficient

The (B, T, T) tensor accounting was correct math. It explained *part* of why Stage 2 additive uses more memory than swap or baseline. But it isn't what caused the OOM — that was ChatCORE's KV cache on top of the training model. Memory budget analysis should account for *all* allocators that share the device, including any periodic eval workloads, not just the training forward+backward.

### "Same arch" is a strong claim

Halfway through this session we'd written that swap and additive give "same accuracy" based on the MQAR probe. That's only synthetic-task evidence. The full pretrain hadn't been run on Stage 2 swap. Even at the same architecture-family scale, "same accuracy" requires equal-budget evaluation on the metric of interest. We softened the claim later, but it's a recurring failure mode worth flagging: **a probe match is not a benchmark match.**

### The missing rule was *already written down*

The `pgrep -lf python` orphan-worker rule existed in `docs/m2_pipeline_2026-04-30.md:102` from a prior debug session. We failed to find it during the new debug session because we didn't search "old writeups" before starting. Project memory works only if it's *findable*. Two operating hypotheses going forward:

- **CLAUDE.md is for rules that need to fire automatically.** Anything load-bearing for safety (don't overwrite, don't run on a dirty Metal context, don't redirect Python without -u) goes there even if it costs lines.
- **Old writeups are searchable too.** Before declaring a problem novel, grep `docs/` for prior occurrences. We did this eventually but late.

## Pending: actually finish Stage 2 SFT

This investigation was triggered by trying to SFT the Stage 2 base checkpoint. We have a working command and a clear cause for prior failures. Next session should run:

```bash
python -u -m scripts.chat_sft \
    --max-seq-len=512 --device-batch-size=32 --total-batch-size=65536 \
    --eval-every=50 --eval-tokens=524288 \
    --num-iterations=375 \
    --save-every=100 --save-keep-last-n=2 \
    --chatcore-every=-1 \
    --model-tag=d6_stage2 \
    --run=sft-stage2-d6-final
```

Expected: ~1h on M2, val/bpb landing around 0.66-0.68 (similar to Stage 1's 0.6712 and baseline's 0.6639). After completion, run ChatCORE separately on the final checkpoint via `chat_eval` for the comparable downstream metric.
