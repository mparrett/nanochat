# EGGROLL — Evolution Strategies at the Hyperscale

**Paper:** Sarkar, Fellows, Duque et al. (Oxford FLAIR / WhiRL, MILA, NVIDIA), arxiv **2511.16652**, Feb 2026.
**Site / code:** https://eshyperscale.github.io/
**Local source:** `~/.cache/nanochat/knowledge/2511.16652/`

---

## TL;DR

EGGROLL = **Evolution Guided GeneRal Optimisation via Low-Rank Learning**. Take Salimans-style OpenAI Evolution Strategies (ES), but replace the full-rank per-worker noise matrix `E ∈ R^{m×n}` with a low-rank factorisation `E = (1/√r) A Bᵀ` where `A ∈ R^{m×r}`, `B ∈ R^{n×r}` and typically **r = 1**. The forward pass for a perturbed worker becomes

```
u (M + σE)ᵀ  =  uMᵀ  +  (σ/√r) (uB) Aᵀ
```

which is exactly the **batched LoRA inference** trick used by vLLM. The mean `M` matmul is shared across all population members in a single GEMM, and each member only adds a cheap rank-r outer product. Arithmetic intensity stays in the compute-bound regime → up to **100× over naive ES** and **91% of pure batched-inference throughput** for billion-parameter models with large populations.

Result: a single GPU can simulate populations of ~10⁶ workers with cheap noise (counter-based RNG, regenerated on demand → no auxiliary memory), making ES competitive with backprop at hyperscale.

## Why it's interesting

Beyond the speed numbers, EGGROLL changes what's feasible to train end-to-end **without backprop**:

1. **Pure-integer pretraining.** They build "EGG" (Evolved Generative GRU), a 6L×256D char-level RNN with all weights in int8 and all activations in integer datatypes, **no floating point anywhere**. Nonlinearity comes only from int8 saturation/clipping (riffing on Foerster's 2017 nonlinear-via-rounding idea). With population 2²⁰ they beat a same-size bf16 Transformer on minipile (3.40 vs 3.58 bits/byte). MeZO-style two-point estimates (pop=2) drastically underperform — large populations are what unlocks pretraining.
2. **LLM fine-tuning.** Beats GRPO on Countdown and GSM8K with RWKV-7 at 1.5B and 7B, matches GRPO on Qwen3-4B-base / DeepScaleR. Pop sizes 1024–8192 (GRPO maxes at ~64–256 because the KV cache eats the memory; RWKV's constant-state recurrence frees that memory for population). At 14B, GRPO is infeasible (Adam state too big) but EGGROLL fine-tunes on 32 GPUs and pushes AIME24 13%→30%, AIME25 7%→33%.
3. **Int8 quantised LLM distillation.** Per-channel symmetric int8 quantisation of RWKV-7, then EGGROLL with Adam-driven proposals (Adam picks magnitude, ES picks direction; integer step via thresholding). Distill from non-quantised teacher via KL on teacher-forced GSM8K.
4. **Theory.** ES has a critical noise scaling `σ_d = o(d^{-1/2})`. Below it, the ES update provably linearises to the true gradient; above it it diverges. EGGROLL (even at r=1) inherits this linearisation in the high-dimensional limit and converges to full-rank Gaussian ES at rate **O(r⁻¹)**. The CLT also says the low-rank perturbation distribution → Gaussian as `r→∞`, justifying using the Gaussian score function `Ŝ(E) = -E` as the approximate score even though the true low-rank density has no analytic form.

## The actual algorithm (one-page version)

```
EGGROLL(rank r, lr α, noise σ, T steps, N workers)

initialise mean parameters M; broadcast known RNG seeds ς to all workers
for t in 1..T:
  for worker i in parallel:
    A_i ~ p(A), B_i ~ p(B)              # iid, zero-mean, unit-var, finite 4th moment
    E_i = (1/√r) A_i B_iᵀ
    f_i = fitness(M + σ E_i)            # forward pass with low-rank LoRA-style adapter
  all-gather scalar fitnesses f_1..f_N  # *only scalars cross the wire*
  for worker i in parallel:
    reconstruct all E_j from seeds ς    # no need to materialise / store noise
    M ← M + (α/N) Σ_j E_j f_j           # full-rank update of rank min(Nr,m,n)
```

Key implementation details:
- **Per-layer**, not per-tensor: each linear-layer weight matrix gets its own (A,B).
- The update `Σ E_j f_j` is computed without materialising individual `E_j`: when r=1, stack A into `R^{N×d_out}` and B into `R^{N×d_in}`, then it's `(diag(f) A)ᵀ B` — one GEMM.
- Counter-based RNG (Threefry/JAX) means noise costs ~0 memory; you regenerate `A_i, B_i` from the worker index when you need them.
- Update rank is `min(Nr, m, n)` → with N > m/r the parameter update is **full-rank**, so EGGROLL is *not* a low-rank fine-tuning method like LoRA, it's a low-rank-*sample* method whose update is dense.

## Connection to nanochat

nanochat is a single-node, depth-swept, backprop-trained pipeline: AdamW pretrain → SFT (full and LoRA variants in `scripts/chat_sft.py` / `chat_sft_lora.py`) → "GRPO" RL on GSM8K (`scripts/chat_rl.py`). EGGROLL touches four pieces of that pipeline:

### 1. RL post-training: a credible drop-in alternative to GRPO

`scripts/chat_rl.py` already runs a stripped-down GRPO: sample G completions per prompt, score with task reward, advantage = `reward − mean`, policy-gradient step. The paper's GSM8K story is exactly this loop but with parameter-space perturbations instead of action-space sampling:

- **Memory:** GRPO needs a frozen reference policy + Adam state for the policy. EGGROLL needs neither — just the mean weights `M` and per-worker low-rank `(A_i, B_i)`. At a `--depth=20` scale on M2 this is the difference between fitting and OOM.
- **Pop size vs group size:** chat_rl.py's `group_size` (default 16) maps to EGGROLL's `N_workers`. The paper shows pop=2 (MeZO regime) underperforms heavily; the win comes from N ≥ 1024. On 1 GPU with KV cache, that's tight for our Transformer (the paper specifically picks RWKV because the constant-state recurrence releases the KV memory back into the population budget).
- **Scoring is structurally identical:** their `z_ij = (s_ij − μ_qj)/σ̄` (group-relative advantage with global σ) is what chat_rl.py already computes up to a constant. The codepath that converts "reward → advantage" stays; what changes is *what* the advantage multiplies (an outer product `E_i f_i` instead of a log-prob gradient).

A nanochat experiment worth running: a fork of `chat_rl.py` (call it `chat_es.py`) that does EGGROLL r=1 on a checkpoint of `d20` and compares val accuracy on GSM8K against the GRPO baseline at matched wall-clock. Even at modest pop sizes this is a cheap test of whether ES is a viable post-training path in the depth-swept regime.

### 2. SFT-on-LoRA: cheaper because of vLLM-style batched inference

`scripts/chat_sft_lora.py` and `nanochat/lora.py` already implement LoRALinear with rank `r` and `alpha` scaling. The EGGROLL forward-pass identity `u(M+σE)ᵀ = uMᵀ + (σ/√r)(uB)Aᵀ` is exactly the LoRALinear.forward path. Practical implication: if we want to *sample* multiple LoRA adapters per batch (e.g. for hyperparameter search, multi-task fine-tuning, or model soups), the same kernel pattern in `lora.py` generalises to a batched-LoRA inference primitive with negligible extra plumbing. This is plumbing nanochat doesn't currently have and would be a small, principled addition consistent with the depth-sweep philosophy (works at any width/depth).

### 3. Quantised inference: distillation from a bf16 teacher

`nanochat/quant.py` exists. EGGROLL's int8 distillation recipe (KL between teacher and student on teacher-forced sequences, ES on the student's int8 weights with Adam-magnitude / ES-direction hybrid updates) is a recipe we could borrow if we ever want to bake post-training int8 into the pipeline. The Adam-as-proposal / ES-as-discretiser idea is the part that's actually novel and reusable.

### 4. Non-obvious architectural connection: state-tracking RNNs vs. Transformers

The paper's argument for the EGG architecture is that classic RNNs (LSTM/GRU/minGRU) handle simple state-tracking that Transformers and SSMs provably can't (Merrill 2024). nanochat is firmly Transformer (`nanochat/gpt.py`), but the Hope/NL experiment branch in this repo is explicitly exploring memory-modulated architectures. The EGGROLL recipe is what makes a *nonlinear* RNN trainable without BPTT or vanishing-gradient problems — i.e. ES sidesteps the reason classic RNNs were abandoned. If the Hope/NL line ever wants to test a non-linear-recurrence variant that backprop can't stably train, EGGROLL is the obvious optimiser to reach for.

## What I'd be cautious about

- **Cost comparison is GPU-hours-asymmetric.** Their pop=2²⁰ pretraining win costs ~180× the backprop baseline in GPU-hours. EGGROLL beats backprop on *quality at a given data budget*, not on *quality at a given compute budget*. For nanochat's "single node, compute-optimal" framing this is the wrong knob to optimise — unless the goal is specifically a data-limited regime (e.g. fine-tuning on a small reward signal).
- **The score function is a *heuristic***. They use `Ŝ(E) = −E` (the Gaussian score) on a non-Gaussian rank-r distribution. The CLT justifies this in the `r→∞` limit, and the high-dim linearisation theory makes it asymptotically right, but for r=1 and modest m,n it's an approximation. Their ablations show it works empirically; just be aware it's not exact.
- **RWKV-specific framing.** Their LLM wins lean heavily on RWKV's constant-state property freeing memory for population size. On a vanilla Transformer with KV cache, the achievable N per GPU shrinks substantially, and the GRPO-vs-EGGROLL trade looks less favourable to EGGROLL. nanochat is currently Transformer-only.
- **Single-GPU EGGROLL on M2 is untested in the paper.** All their throughput numbers are H100 with int8/bf16 tensor cores. MPS does not have the same low-precision matmul path; the arithmetic-intensity story may not carry over cleanly. Likely fine for r=1 LoRA-style inference (which already works on MPS), but the "saturate compute with thousands of perturbations" win is H100-shaped.

## Bottom line for nanochat

The most realistic experiment is an EGGROLL fork of `chat_rl.py` for GSM8K post-training, sharing the existing `LoRALinear` kernel as the batched-perturbation primitive. It would tell us whether ES-style RL — which avoids needing log-probs, KL-to-reference, or Adam state — is a viable post-training path at the depth-swept scales we care about. The pretraining and quantisation stories are intriguing but reach further outside the current pipeline.
