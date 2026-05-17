# Feature: δ-mem MLX port — graft published memory adapter onto fp-Qwen3-4B-Instruct

**Filed:** 2026-05-17
**Branch context:** `experiment/hope-nested-learning` (paused per `docs/strategic_pivot_2026-05-13.md`); this ticket is cross-cutting tooling/experiment that can land on the paused branch or its own.
**Operator preference:** **Path B first** (MLX port), Path A (PyTorch + MPS inference) as fallback if Path B blocks.

## Headline

Port the publicly-released δ-mem adapter (Lei et al., May 2026 — `docs/paper_delta_mem_2026-05-17.md`) from PyTorch/CUDA into MLX, apply it to `Qwen/Qwen3-4B-Instruct-2507` on M2, and evaluate via our existing `scripts/chat_eval_mlx.py` harness. Compare directly to the fp-Qwen3-4B-8bit baseline (ChatCORE **0.7656** at -x 200, lenient) to measure whether δ-mem's published-on-conversational-memory lift transfers to our six-task ChatCORE suite.

## Context

- We just completed a 14-commit bonsai-eval arc (2026-05-15 → 2026-05-17) that produced a publication-grade four-model cross-architecture ChatCORE table and a working harness (`scripts/chat_eval_mlx.py`) for evaluating any MLX-loadable model.
- The δ-mem paper landed in the operator's iCloud on 2026-05-16. It is **structurally near-identical to our paused Hope/NL Stage 2** (gated delta-rule on a small associative-memory matrix) but bolted on as a **graft to a frozen Qwen3-4B-Instruct backbone** rather than as a from-scratch block replacement.
- The strategic pivot doc (2026-05-13) explicitly named "continuous-learning components grafted onto a frozen 4–8B base" as the right long-term direction but parked it for lack of M2-feasible recipe. **δ-mem is exactly that recipe, on the exact backbone we just baselined.**
- The published GitHub code (`declare-lab/delta-Mem`, CC-BY-4.0, 101 stars, real implementation) is **CUDA-only for training** (mandatory FlashAttention + DeepSpeed), but **a pre-trained adapter is published**: `declare-lab/delta-mem_qwen3_4b-instruct` (rank-8 Q/O TSW variant, write length 8192). This bypasses the training requirement entirely.

## Headline question

Does δ-mem's published-on-LoCoMo/MemoryAgentBench lift transfer to our six-task ChatCORE suite (ARC-E, ARC-C, MMLU, GSM8K, HumanEval, SpellingBee)?

| Outcome | Interpretation | Next step |
| --- | --- | --- |
| ChatCORE ≥ 0.78 | Real lift on our suite. Graft architectures matter beyond memory-specific benchmarks. | Train our own adapter variants; pursue the grafted-memory direction more aggressively. |
| ChatCORE = 0.7656 ± noise | Graft works mechanistically per their published benchmarks but doesn't help our reasoning/knowledge tasks. Expected outcome. | Confirm by reading the per-task split; potentially try the SSW or MSW variants if the adapter exists. |
| ChatCORE < 0.7656 | The graft hurts. Falsifies "memory module is free to add." | Investigate whether the port has a bug; if confirmed, this is a real and surprising negative result. |

## Why this is non-trivial

1. **MLX has no direct port of the δ-mem implementation.** We'd be writing the forward in MLX from the paper + reference PyTorch implementation. Bounded but real work.
2. **Adapter weights need format conversion.** PyTorch safetensors → MLX format. `mlx_lm.convert` may or may not handle this cleanly for a partial-weight adapter; might need custom serialization.
3. **The δ-mem state is a separate runtime object**, not part of the model weights. Need to thread it through inference correctly: read from `S_{t-1}`, generate correction, then write `S_t` after attention. Our `chat_eval_mlx.py` doesn't have a hook for this — it assumes pure stateless forward passes.
4. **FlashAttention dependency in the reference code** likely permeates the adapter forward. Need to find the non-flash code path or substitute SDPA.
5. **The eval semantics for stateful inference differ.** δ-mem's TSW writes state on every token; our categorical eval does a single forward per prompt. The state evolution within the prompt matters, but writes shouldn't persist across problems. Need to verify state-reset semantics.

## Proposed sequence

### Phase 0 — Reconnaissance (~2-4h, mandatory)

**Goal:** Understand the reference implementation before committing to a port.

**Steps:**
1. Clone `declare-lab/delta-Mem` to a working directory (NOT inside the nanochat repo). Inspect `deltamem/core/delta.py` and `delta_impl.py` to extract the precise forward pass equations and where FlashAttention is invoked.
2. Identify the **non-FlashAttention fallback path** in their code, if one exists. If not, identify exactly where to swap in `torch.nn.functional.scaled_dot_product_attention`.
3. Download the pre-trained adapter weights from `declare-lab/delta-mem_qwen3_4b-instruct`. Inspect what tensors are in the adapter: which W_q^m, W_k^m, W_v^m, W_q^Δ, W_o^Δ, W_β, W_λ are present per layer, what their shapes are, what the 8×8 state initializer looks like.
4. Identify which Qwen3 layers the adapter targets. Paper says "all layers" performs best in their ablation; the published adapter is probably all-layers, but verify.
5. Disk + memory pre-check via `dev/preflight_memory.py`. The adapter download is small (rank-8, so ~MB scale, not GB). The Qwen3-4B-Instruct base is already cached at 7.5 GB (HF fp16).

**Gate to Phase 1:** clear picture of (a) which tensors are in the adapter, (b) where FlashAttention is in the forward, (c) whether a non-flash path exists.

### Phase 1 — MLX implementation (Path B) (~1-2 days)

**Goal:** Implement δ-mem forward in MLX, load the adapter weights, integrate with Qwen3-4B-Instruct in MLX.

**Steps:**

1. Convert Qwen3-4B-Instruct-2507 from HF to MLX format if needed (we have 7.5 GB HF cache; mlx-community publishes 8-bit only for this variant — see `docs/qwen3_4b_quantization_cost_2026-05-17.md`):
   ```bash
   python -m mlx_lm.convert --hf-path Qwen/Qwen3-4B-Instruct-2507 \
     --mlx-path ./qwen3_4b_instruct_2507_fp16 --dtype float16
   ```
   Disk cost: ~8 GB. Run preflight first.

2. Implement δ-mem module in MLX. Sketch:
   ```python
   # In nanochat/delta_mem.py or similar
   class DeltaMemLayer:
       def __init__(self, r, d_hidden, alpha):
           self.W_q_m = mx.zeros(...)  # r × d_hidden
           self.W_k_m = mx.zeros(...)
           self.W_v_m = mx.zeros(...)
           self.W_q_delta = mx.zeros(...)  # d_q × r
           self.W_o_delta = mx.zeros(...)  # d_o × r
           self.W_beta = mx.zeros(...)
           self.W_lambda = mx.zeros(...)
           self.alpha = alpha
           self.r = r

       def read(self, x_t, S_prev):
           q_m = mx.tanh(self.W_q_m @ x_t)
           q_m = q_m / mx.linalg.norm(q_m)  # L2 norm
           r_t = S_prev @ q_m  # (r,)
           dq = self.W_q_delta @ r_t * (self.alpha / self.r)
           do = self.W_o_delta @ r_t * (self.alpha / self.r)
           return r_t, dq, do

       def write(self, x_t, S_prev):
           k_m = mx.tanh(self.W_k_m @ x_t)
           k_m = k_m / mx.linalg.norm(k_m)
           v_m = self.W_v_m @ x_t
           beta = mx.sigmoid(self.W_beta @ x_t + b_beta)
           lam = 1.0 - beta
           # Gated delta-rule update
           S_new = mx.diag(lam) @ S_prev + \
                   mx.diag(beta) @ mx.outer(v_m - S_prev @ k_m, k_m)
           return S_new
   ```
   Exact details from `deltamem/core/delta.py`.

3. Patch the Qwen3 model in MLX to insert the δ-mem read-then-steer logic at every attention block:
   ```python
   # Inside attention forward
   r_t, dq, do = delta_mem[layer_idx].read(x_t, state[layer_idx])
   q_corrected = q + dq
   a_t = standard_attention(q_corrected, k, v)
   y_t = a_t + do
   state[layer_idx] = delta_mem[layer_idx].write(x_t, state[layer_idx])
   ```

4. Load adapter weights from safetensors into the MLX module. Map tensor names between PyTorch and MLX conventions.

5. **Verify forward output matches PyTorch reference** on a single sample. Generate one token with PyTorch reference (CPU-only mode if needed), then with our MLX port, and compare logits. **Tolerance: ~1e-3 max abs diff** (float16 numerical noise). If they don't match, debug.

**Gate to Phase 2:** logit-match against reference on at least 3 hand-chosen prompts.

### Phase 2 — Harness integration (~0.5 day)

**Goal:** Wire the MLX δ-mem model into `chat_eval_mlx.py` cleanly.

**Steps:**

1. Add a `--delta-mem` flag to `chat_eval_mlx.py` that, when set, wraps the loaded model with the δ-mem module.
2. Reset state at the start of each problem (categorical and generative paths both — δ-mem state shouldn't leak across problems).
3. Generative path: state updates per token during generation. Categorical path: state updates during the single prompt forward (state matters because writes happen mid-prompt). Verify both paths handle state correctly.
4. Smoke-test on `-x 5` ARC-Easy with and without `--delta-mem`. Expect accuracy within a few points of plain Qwen3 (small N, but mechanism sanity).

**Gate to Phase 3:** smoke passes; no crashes; numbers look like a real model.

### Phase 3 — Full eval (~1.5h wall)

**Goal:** Run the canonical x200 lenient comparison.

**Steps:**

1. `python3 dev/preflight_memory.py` first (memory headroom, no parallel jobs).
2. Run the harness:
   ```bash
   uv run python -u -m scripts.chat_eval_mlx \
     -m ./qwen3_4b_instruct_2507_fp16 \
     --delta-mem path/to/adapter \
     -x 200 --no-system-prompt --max-new-tokens 512 --lenient-extract \
     -o docs/qwen3_4b_delta_mem_x200_<date>.md
   ```
3. Compare per-task to the fp-Qwen3-4B-8bit baseline (`docs/qwen3_4b_quantization_cost_2026-05-17.md`).
4. Commit the result + a focused writeup analyzing the per-task pattern.

## Path A fallback (if Path B blocks)

If MLX porting hits an unforeseen blocker (adapter tensor format incompatible, MLX missing a primitive, logit-match fails persistently), fall back to PyTorch + MPS:

1. Verify the PyTorch + MPS load path for `Qwen/Qwen3-4B-Instruct-2507` (the smoke that crashed 2026-05-16 — see `feedback_parallel_workload_hygiene.md`).
2. Clone the reference δ-mem code, install requirements minus FlashAttention (`INSTALL_FLASH_ATTN=0 bash scripts/setup_uv_env.sh`).
3. Patch the inference path to substitute `torch.nn.functional.scaled_dot_product_attention` for FlashAttention calls. The deltamem package may need source edits — work in a clone, not pip-installed.
4. Build a thin PyTorch eval harness mirroring `chat_eval_mlx.py` semantics: same six tasks, same `apply_chat_template` logic, same lenient extractor. ~150 LOC.
5. Run the canonical x200 sweep.

**Path A estimated cost: ~1 day if no surprises; 2-3 days with the usual.**

## Falsification thresholds (per the headline question)

Run is meaningful if ChatCORE moves by **>0.02** from the baseline 0.7656 (well outside per-task stderr at n=200). Specific thresholds:

- **ChatCORE ≥ 0.79** (+0.024): real lift signal. Worth following up with SSW/MSW variants or trying it on Bonsai-int2 to see if it can recover the quantization-cost gap (the quantization-cost analysis suggests it might *not*, but empirics).
- **ChatCORE in [0.76, 0.78]**: no signal vs noise. Document the per-task pattern carefully — if HumanEval moves but MMLU is flat, that's interpretable. If everything is flat, the graft is essentially decoration on this evaluation suite.
- **ChatCORE < 0.74** (-0.026): graft is *hurting* the model. Almost certainly a port bug; investigate before publishing the negative result.

## Critical gotchas (read before starting)

1. **Parallel-workload hygiene** (`feedback_parallel_workload_hygiene.md`): always run `dev/preflight_memory.py` first; never queue a second GPU/MPS workload while one is running on M2 unified memory; check `df -h ~/.cache/huggingface` before pulling multi-GB models. Hard-rebooted 2026-05-16 from violating this.
2. **Disk budget**: current state should be ~20 GiB free. The fp16 Qwen3-4B-Instruct conversion adds ~8 GB. The adapter is small (~rank-8 × all layers, MB scale). Stay above 5 GiB free at all times.
3. **State semantics**: δ-mem state must reset between problems in evaluation. Don't let state leak across the eval loop or you'll get spurious in-context-learning effects.
4. **HumanEval is the wall**: full eval at x200 takes ~3h on fp Qwen3-4B-8bit. δ-mem adds compute per token (extra read+write+correction). Estimate ~3.5-4h for the full eval.
5. **Adapter is TSW only**: the published HF adapter is the Token-State Write variant. SSW and MSW variants aren't released as far as we can tell. So we measure one variant, not the family.

## References

**Paper + code:**
- Paper PDF: `~/Library/Mobile Documents/com~apple~CloudDocs/Downloads/mem-2605.12357.pdf` (10 pages, arxiv 2605.12357)
- Code repo: `https://github.com/declare-lab/delta-Mem` (CC-BY-4.0)
- Mirror: `https://github.com/MindLab-Research/delta-Mem` (identical content)
- Pre-trained adapter: `https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct` (TSW, rank-8, write length 8192)

**Project context:**
- Paper notes (precise mechanism walkthrough): `docs/paper_delta_mem_2026-05-17.md`
- Backlog entry that this ticket replaces: `docs/project_notes/backlog.md` (will mark "moved to incoming" on filing)
- Strategic pivot (why the graft direction was parked, why it now has a reference): `docs/strategic_pivot_2026-05-13.md`
- Backbone baseline (the comparison target): `docs/qwen3_4b_quantization_cost_2026-05-17.md`
- Cross-arch ChatCORE table: `HANDOFF.md` Day 2026-05-17
- Closest from-scratch precedent (gated delta-rule built into nanochat as a block replacement): `docs/hope_nl_stage2_*.md`

**Infrastructure (already in repo):**
- Harness: `scripts/chat_eval_mlx.py` (~250 LOC, lenient extractor included)
- Pyproject `[mlx]` extra: `uv sync --extra cpu --extra mlx`
- Preflight: `python3 dev/preflight_memory.py`

**Operator constraints / preferences (memory):**
- `feedback_parallel_workload_hygiene.md` — always run preflight; never parallel GPU jobs on M2
- `feedback_local_only.md` — never push or open PRs; commits stay local
- `feedback_driving_research.md` — when you're driving, own the actions; don't punt back to operator
- `feedback_debug_methodology.md` — empirical validation, synthetic reproducers, hypothesis falsification

## Status

**OPEN.** Phase 0 reconnaissance not yet started. Ready for a fresh session to pick up cold. The 14-commit bonsai-eval arc provides all the infrastructure; this ticket adds one new model class (δ-mem grafted onto fp-Qwen3-4B-Instruct in MLX) plus the integration glue.
