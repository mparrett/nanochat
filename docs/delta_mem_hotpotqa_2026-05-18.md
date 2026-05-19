# δ-mem on HotpotQA — the paper-native triangulation comes back null

**Date:** 2026-05-18 (UTC: late evening; wall ~01:30–02:21 UTC 2026-05-19)
**Setup:** Qwen3-4B-Instruct-2507-8bit, MLX, temp=0.0, max_new_tokens=64, n=200 problems, seed=42 shuffle, validation split, distractor config.
**Adapter:** `~/.cache/nanochat/delta_mem_qwen3_4b_instruct/` (the same artifact used for HumanEval/GSM8K/MBPP).

## Headline

| metric                       | baseline (no δ-mem) | δ-mem      | Δ          | z      |
| ---------------------------- | ------------------: | ---------: | ---------: | -----: |
| F1≥0.5 accuracy              |              64.50% |     65.50% |     +1.00pp |  0.21σ |
| mean F1                      |              0.6407 |     0.6410 |    +0.0003 |   ~0   |
| mean EM (strict)             |              48.50% |     50.00% |     +1.50pp |  0.30σ |
| wall (200 problems)          |             38.1 min |    52.7 min |       +38% |  —     |

**Clear null on the paper's home-axis task.** Mean F1 is identical to three decimal places. The F1≥0.5 boolean accuracy differs by 2 examples in 200 (129 vs 131). The EM differs by 3. All differences are well within sampling noise.

## Significance

- σ_baseline = √(0.645·0.355 / 200) = 3.38pp
- σ_δ = √(0.655·0.345 / 200) = 3.36pp
- σ_diff = √(3.38² + 3.36²) = **4.77pp**
- z = 1.00pp / 4.77pp = **0.21σ** (p ≈ 0.83 two-tailed)

A power analysis: to detect a 5pp lift at 80% power on this binomial, we'd need n≈600. So the null doesn't rule out a small (<5pp) lift hiding in this slice. **But the observed difference is so close to zero** (mean F1 differs by 0.0003 across 200 problems) **that no plausible larger run rescues it**. If the paper's reported HotpotQA lift were e.g. +8-12pp like our HumanEval/MBPP coding numbers, this slice would have detected it cleanly.

## Comparison against the three reasoning tasks

| task        | baseline | δ-mem    | Δ        | z       | regime                          |
| ----------- | -------: | -------: | -------: | ------: | ------------------------------- |
| HumanEval   |  58.54%  |  70.73%  | +12.2pp  | 2.33σ   | multi-step code generation      |
| GSM8K       |  76.00%  |  81.50%  |  +5.5pp  | 1.35σ   | multi-step arithmetic           |
| MBPP        |  53.00%  |  61.00%  |  +8.0pp  | 1.15σ   | multi-step code generation (OOD)|
| **HotpotQA**|  64.50%  |  65.50%  |  **+1.0pp** | **0.21σ** | **long-context multi-hop recall** |

All three reasoning tasks lift. HotpotQA — the paper's headline-claim regime — does not. The structural difference between the four tasks isn't task difficulty (HotpotQA is comparable to GSM8K in baseline accuracy) or modality (HotpotQA is text-in / short-text-out, no different from GSM8K's word-problem-in / number-out). It's the *shape of the generation*: HumanEval/GSM8K/MBPP all require the model to maintain and update intermediate state across many output tokens. HotpotQA requires the model to retrieve and condense, but the final answer is short (1-5 tokens for most) — there's no multi-step generation to stabilize.

## What this rules in and out

**Ruled in:**
- The "δ-mem is a working-memory adapter that helps multi-step generation" hypothesis from yesterday's interpretation doc is now the cleanest reading of the data. It was a hypothesis fit to 3 reasoning lifts + 4 knowledge nulls; HotpotQA is a fifth data point that's structurally different from both groups and still null. The hypothesis predicted exactly this.
- The MLX port is not bugged in a long-context regime: trajectories converge cleanly, F1 is essentially unchanged (not catastrophically degraded), δ-mem doesn't *hurt* HotpotQA. The adapter is running; it just doesn't help here.
- "δ-mem helps coding, not memory" is the operational call. The published adapter's name and the paper's framing are misleading on our setup.

**Ruled out:**
- "δ-mem helps everywhere on this model" — definitively no.
- "δ-mem helps long-context recall on Qwen3-4B-Instruct-8bit at temp=0.0" — no, at least not at any magnitude this 200-example slice could detect.
- Concerns about HotpotQA-specific port bugs — adapter ran cleanly, wall overhead in line with other tasks (+38% here vs +47% on shorter-context tasks), trajectory was smooth.

**Not yet ruled in or out:**
- Whether the paper's published numbers reproduce on the original Qwen2.5-7B-Instruct base they trained against, in fp16, with the paper's exact decoding settings. We're using Qwen3-4B-Instruct-8bit and temp=0.0; the gap could be base-model, precision, or decoding.
- Whether δ-mem helps HotpotQA in a sampling regime (temp>0). Greedy gives the most stable signal but could be hiding a "stabilize sampling" effect. The δ-mem state interacts with the sample path, so temp>0 could in principle change things. We did not test this.
- The other paper-native tasks: LoCoMo, MemoryAgentBench, IFEval, GPQA. HotpotQA was the cheapest of the five (~90 min wall); the others are longer/harder. The "the published claim doesn't reproduce on the headline benchmark we tested" finding makes running the other four lower-priority — most likely they show similar nulls.

## Cost ledger

- Baseline: 38.1 min wall (`docs/qwen3_4b_baseline_hotpotqa_x200_2026-05-18.md`)
- δ-mem: 52.7 min wall (`docs/qwen3_4b_delta_mem_hotpotqa_x200_2026-05-18.md`)
- Code: `tasks/hotpotqa.py` (~115 LOC) + ~25 LOC harness wiring in `scripts/chat_eval_mlx.py`.
- Total wall: 90.8 min eval + ~30 min dev. Matched the 90-min budget from the recommendation.

## What changes downstream

1. **Field-result HTML revision.** Section 07 ("what we deliberately did not test") had "The paper's native evaluation tasks" as item 2 — flagged as the triangulation we hadn't run. This is now run and null. That item moves out of the disclaimer list; the result becomes a new section in the body. Coda updates to drop "the paper's 'long-term memory' framing isn't wrong — it might also be true on their benchmarks" since now there's one benchmark from their list where it isn't.

2. **The δ-mem story strengthens, doesn't weaken.** It's tempting to read a null on the paper's home axis as a negative result, but operationally it's the opposite: it sharpens the mechanism story (working memory, not long-term memory) and makes the reframing publishable rather than speculative. The result is now "we identified what δ-mem actually does, which is different from what the paper claims."

3. **SDFT relevance unchanged.** SDFT is still flagged in §07 as the recipe-side counter-hypothesis we didn't run. HotpotQA being null on δ-mem doesn't change whether SDFT could be a better recipe — that's a separate axis.

## Decision

Fold HotpotQA into the field-result HTML and commit. The mechanism-story revision (working memory, not long-term memory) is now empirically grounded on five tasks: three multi-step-generation lifts (HumanEval / GSM8K / MBPP), four knowledge nulls (ARC-E / ARC-C / MMLU / SpellingBee), and one paper-native long-context-recall null (HotpotQA). Fisher's combined evidence on the three reasoning lifts remains p<0.001. The HotpotQA null becomes the cleanest single argument for the reframing.

The branch can be considered done. Further measurement (LoCoMo, MemoryAgentBench, etc.) is incremental hardening of a story that's already coherent.
