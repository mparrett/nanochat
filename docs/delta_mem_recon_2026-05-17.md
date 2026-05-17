# δ-mem MLX port — Phase 0 reconnaissance

**Date:** 2026-05-17
**Ticket:** `docs/project_incoming/feat_delta_mem_mlx_port.md`
**Branch:** `experiment/hope-nested-learning` (paused; ticket cross-cutting)
**Reference clone:** `/tmp/delta-Mem` (commit at clone time; CC-BY-4.0)
**Adapter download:** `/tmp/delta-mem_qwen3_4b-instruct` (11 MB, bf16)

## Phase 0 gate (from ticket)

> Clear picture of (a) which tensors are in the adapter, (b) where FlashAttention is in the forward, (c) whether a non-flash path exists.

All three answered. Phase 1 (MLX implementation) is **green-lit**.

## Adapter inventory

`/tmp/delta-mem_qwen3_4b-instruct/` contains:
- `delta_mem_config.json` (700 B) — `HFDeltaMemConfig` snapshot.
- `delta_mem_adapter.pt` (11 MB) — torch.save dict, **324 tensors = 9 × 36 layers**, all bf16.

All 36 Qwen3-4B layers wrapped (`target_layers=[]` → no filter; verified `min=0 max=35 count=36`).

### Per-layer tensor names and shapes (Qwen3-4B-Instruct-2507)

| Name | Shape | Role | Used in active path? |
| --- | --- | --- | --- |
| `memory_q_proj` | `(8, 2560)` | Project hidden → state-query | ✅ |
| `memory_k_proj` | `(8, 2560)` | Project hidden → state-key | ✅ |
| `memory_v_proj` | `(8, 2560)` | Project hidden → state-value | ✅ |
| `delta_q_proj`  | `(4096, 8)` | Low-rank correction to base query | ✅ (head `q` active) |
| `delta_o_proj`  | `(2560, 8)` | Low-rank correction to base attn output | ✅ (head `o` active) |
| `delta_k_proj`  | `(1024, 8)` | Inactive head — **all-zero across 36 layers** | ❌ |
| `delta_v_proj`  | `(1024, 8)` | Inactive head — **all-zero across 36 layers** | ❌ |
| `beta_proj`     | `(8, 2560)` | Per-rank gate from hidden | ✅ |
| `beta_bias`     | `(8,)`      | Gate bias (init −1.5) | ✅ |

Shape source-of-truth: Qwen3-4B-Instruct-2507 hidden=2560, q_proj.out=4096 (32 heads × 128), k_proj.out=v_proj.out=1024 (8 KV heads × 128), o_proj.out=2560. The 8 in shapes is `rank`.

**MLX port can skip `delta_k_proj` / `delta_v_proj` entirely** — they are present only because `reset_parameters` allocates all four delta heads then zeros the inactive ones; the published adapter's training set was `delta_heads=["q", "o"]`. Verified empirically: `abs().sum() == 0` for all 36 layers.

### Adapter config (`delta_mem_config.json`)

Non-default knobs:
- `rank: 8`, `alpha: 16.0` → effective δ-scaling = α/rank = **2.0**
- `delta_heads: ["q", "o"]` (q and o only, no k/v)
- `couple_lambda: true` → λ = 1 − β (no separate `lambda_proj`)
- `rankwise_gates: true` → gate_dim = rank = 8 (β is rank-dimensional, not scalar)
- `normalize_qk: true` → tanh + L2-normalize the memory-q and memory-k projections
- `state_update_mode: "standard"` → keep=λ, erase=β, write=β
- `num_state_heads: 1` → single (8×8) state matrix
- `memory_write_granularity: "token"` → TSW (write every token)
- `target_modules: ["self_attn"]`, `target_layers: []` → wrap every attention block
- `trainable_delta_scale: false`, `delta_o_rmsnorm: false` → simplest output path
- `output_init: "base_slice_fixed"` — only affects initialization, irrelevant at inference

## Where FlashAttention lives — and how to avoid it

Result: **the upstream forward has a clean non-flash path. Setting `attn_implementation='eager'` on the base model is sufficient — no source edits needed.**

### Forward attention dispatch (`delta_impl.py` line 2255)

```python
attention_interface = self.eager_attention_forward  # qwen3_eager_attention_forward
if not use_prefixed_memory and self.base.config._attn_implementation != "eager":
    attention_interface = ALL_ATTENTION_FUNCTIONS[
        self.base.config._attn_implementation
    ]
```

`self.eager_attention_forward` is imported from `transformers.models.qwen3.modeling_qwen3.eager_attention_forward` (line 518) — a pure-PyTorch SDPA-style implementation. The flash dispatch only kicks in when the base config explicitly requests it.

Repo-wide flash references (`grep flash_attn`):
- `deltamem/demo/run_chat_demo.sh:15` — defaults `ATTN_IMPLEMENTATION="flash_attention_2"`. Override at the env-var level.
- `deltamem/core/delta_impl.py:2261` — the dispatch line above. Only path that would invoke flash.
- `deltamem/tests/test_delta_mem_regressions.py:375,401` — explicitly set `config._attn_implementation = "eager"` for regression. **The non-flash path is a tested, supported path upstream.**
- `deltamem/tools/bench_scan.py:35` — same eager path used for scan benchmarks.

**Action for the MLX port:** N/A on the port itself; this matters for the Path A fallback (PyTorch reference for logit-match validation). Set `attn_implementation='eager'` when loading the base, and the wrapper stays in its eager dispatch — no FlashAttention import attempted.

### Triton kernel (the other CUDA dependency)

`_memory_affine_scan` (line 2022) prefers a Triton kernel (`triton_affine_scan`) when supported, falling back to `_memory_affine_scan_torch`. Triton-support probes the tensor backend and `scan_impl` env var:

```python
self.scan_impl = os.environ.get("DELTA_MEM_SCAN_IMPL", "auto")
```

On MPS / CPU, `triton_scan_support(...).supported` returns False → automatic fallback to the pure-PyTorch scan. Belt-and-suspenders: `DELTA_MEM_SCAN_IMPL=torch` forces the fallback unconditionally. For the MLX port we re-implement the scan in MLX from scratch anyway — but for Path A validation on PyTorch/MPS the torch scan is the right reference.

## The math the port has to reproduce

Reconstructed from `delta_impl.py`. All paths below assume the adapter's config: `couple_lambda=True`, `state_update_mode="standard"`, `rankwise_gates=True`, `num_state_heads=1`, `memory_write_granularity="token"`, `normalize_qk=True`, `trainable_delta_scale=False`, `delta_o_rmsnorm=False`, `delta_heads=("q","o")`.

### Per-layer state

State is `S ∈ R^{B × 8 × 8}`, zero-initialized on first forward (`_ensure_state`, line 764). Persists across forward calls until `reset_delta_mem_states(model)` zeros it.

### Projections from hidden states (`_memory_sequence_projections`, line 887)

```
x : (B, T, 2560)                    # hidden_states
q_m = F.normalize(tanh(x @ memory_q_proj.T), dim=-1, eps=1e-6)  # (B, T, 8)
k_m = F.normalize(tanh(x @ memory_k_proj.T), dim=-1, eps=1e-6)  # (B, T, 8)
v_m = x @ memory_v_proj.T                                       # (B, T, 8)   no normalize, no tanh
β   = sigmoid(x @ beta_proj.T + beta_bias)                      # (B, T, 8)
λ   = 1 - β                                                     # couple_lambda
```

### Coefficients (`_memory_update_coefficients`, line 949, `state_update_mode="standard"`)

```
keep_t   = λ_t                                                  # (B, T, 8)
erase_t  = β_t
write_t  = β_t
```

### Affine scan (per token, `_memory_affine_scan_torch`, line 1895)

```
S_0 = zeros((B, 8, 8))
for t in range(T):
    read_t = einsum("bij,bj->bi", S_{t-1}, q_m[:,t,:])           # (B, 8)
    pred_t = einsum("bij,bj->bi", S_{t-1}, k_m[:,t,:])           # (B, 8)
    write_outer = outer(v_m[:,t,:], k_m[:,t,:])                  # (B, 8, 8)
    pred_outer  = outer(pred_t,    k_m[:,t,:])                   # (B, 8, 8)
    S_t = keep_t[:,t,:,None] * S_{t-1}
        - erase_t[:,t,:,None] * pred_outer
        + write_t[:,t,:,None] * write_outer
```

Token-mask is folded in: invalid tokens get `read_t = 0` and `S_t = S_{t-1}`.

Algebraic check (couple_lambda case):
```
S_t = Diag(1-β) S_{t-1} + Diag(β) (v_m - S_{t-1} k_m) k_m^T
```
which is the paper's gated delta-rule (paper notes §2 → confirmed).

### Reads → q/o corrections (`_project_delta_head`, line 1101)

```
reads = stack of read_t                                          # (B, T, 8)
δ_q   = (reads @ delta_q_proj.T) * 2.0                           # (B, T, 4096), 2.0 = α/rank
δ_o   = (reads @ delta_o_proj.T) * 2.0                           # (B, T, 2560)
# δ_k, δ_v: None (inactive heads) → skipped at apply time
```

### Apply (`_apply_delta_qkv` line 855, then standard attention, then δ_o)

```
q = x @ q_proj.T + δ_q                                           # (B, T, 4096) → reshape (B, T, 32, 128)
k = x @ k_proj.T                                                 # standard GQA
v = x @ v_proj.T
# q_norm / k_norm (Qwen3 RMSNorm per head)
# RoPE
# KV cache update
# eager attention → attn_output (B, T, 4096)
# o_proj
attn_output = attn_output @ o_proj.T + δ_o                       # (B, T, 2560)
```

`δ_o` is added **after** `base.o_proj` — the additive correction is in residual-stream space, not pre-projection (confirmed line 2294).

## Stateful-eval semantics

State semantics for our six-task ChatCORE eval:
- State is module-level mutable (`self.delta_state`). Inference forward updates state in place as it walks the prompt; subsequent generation steps (KV cache active, seq_len=1 per step) continue to evolve state token-by-token.
- **State must be reset between problems** to prevent cross-sample leakage. Call `reset_delta_mem_states(model)` (or our MLX equivalent) at the start of each problem.
- Categorical-path correctness: even with no generation, the single prompt forward evolves S across the prompt tokens. The final-token logits depend on S_T which integrated over all prompt tokens — exactly what TSW is designed for.

## Pre-flight at Phase 0 close

- Disk: 30 GiB free at `/`; adapter cost 11 MB, reference clone ~3 MB, zero net change to `~/.cache/huggingface`. Above the 5 GiB floor.
- No GPU/MPS workload triggered during Phase 0 — pure code + small-file inspection.
- Memory: 0.34 GB free at session start, but no MPS workload during recon. Phase 1 will need a separate preflight before any MLX forward pass.

## Implications for Phase 1 (MLX implementation)

1. **Tensor inventory is fixed and small:** 7 active tensors per layer × 36 layers = 252 active weights. Total active param storage ~7.5 MB in bf16 (and the bookkeeping zeros for δ_k/δ_v cost another ~3 MB).
2. **No FlashAttention, no Triton, no CUDA dependency in the active path.** The port writes pure MLX equivalent of `_memory_affine_scan_torch` + `_memory_sequence_projections` + the linear additive corrections.
3. **mlx-community ships Qwen3-4B only in 8-bit.** Per `docs/qwen3_4b_quantization_cost_2026-05-17.md` we already converted HF fp16 ourselves. Reuse that conversion path for the δ-mem port baseline (apples-to-apples vs the fp-Qwen3-4B-8bit ChatCORE 0.7656 target requires running the **same precision** with and without δ-mem; do the comparison at whichever precision MLX-LM serves Qwen3 in — likely 8-bit, since that's our existing strong baseline).
4. **Adapter weights are bf16 .pt format.** MLX-LM uses fp16 / 8-bit safetensors by convention. Conversion: torch.load → numpy → mx.array. The 11 MB adapter is trivial to convert; can store as a `delta_mem_adapter.npz` alongside the base model dir or pack into `model.safetensors` with the base weights merged.
5. **State threading touches the attention forward.** We can't reuse mlx-lm's stock `Qwen3Attention` unchanged — we need a subclass (or monkey-patch) that, on each layer's forward, reads/writes the per-layer state. `chat_eval_mlx.py` currently treats the model as a black-box `mlx_lm.load(...)` — Phase 2 needs a state-reset hook.
6. **Logit-match validation strategy:** install upstream reference with `INSTALL_FLASH_ATTN=0 bash scripts/setup_uv_env.sh`, load base with `attn_implementation='eager'`, set `DELTA_MEM_SCAN_IMPL=torch`, run a single forward on CPU (or MPS if it works), diff vs MLX port. Tolerance target per ticket: 1e-3 max-abs in bf16.

## Phase 1 ready-to-start checklist

- [x] Reference repo cloned at `/tmp/delta-Mem`
- [x] Adapter downloaded at `/tmp/delta-mem_qwen3_4b-instruct/delta_mem_adapter.pt` (11 MB, 324 tensors)
- [x] Forward math reconstructed (this doc, §"The math the port has to reproduce")
- [x] Non-flash path confirmed (eager attention dispatch)
- [x] Inactive heads identified (δ_k, δ_v are zero — can be omitted entirely)
- [ ] Qwen3-4B-Instruct-2507 in MLX format on disk (decide: convert fp16 ourselves vs run against existing 8-bit baseline)
- [ ] MLX `DeltaMemLayer` module (memory_q/k/v projections, beta gate, affine scan, δ_q/δ_o linear)
- [ ] Patch / subclass of mlx-lm's Qwen3 attention to thread state through forward
- [ ] Adapter weight conversion (.pt → MLX-native)
- [ ] Logit-match harness vs PyTorch reference

## File pointers (for the next session)

- Reference forward (with comments to follow): `/tmp/delta-Mem/deltamem/core/delta_impl.py:2083` (forward), `:1895` (scan), `:887` (projections), `:949` (coefficients), `:1101` (δ-projection)
- Adapter weights: `/tmp/delta-mem_qwen3_4b-instruct/delta_mem_adapter.pt`
- Adapter config: `/tmp/delta-mem_qwen3_4b-instruct/delta_mem_config.json`
- Our existing baseline target: `docs/qwen3_4b_quantization_cost_2026-05-17.md` (fp-Qwen3-4B-8bit ChatCORE 0.7656 at -x 200 lenient)
- Our paper notes: `docs/paper_delta_mem_2026-05-17.md`
- Our ticket: `docs/project_incoming/feat_delta_mem_mlx_port.md`
