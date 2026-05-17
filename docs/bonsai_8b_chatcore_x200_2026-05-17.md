# MLX chat eval — prism-ml/Ternary-Bonsai-8B-mlx-2bit (canonical, -x 200)

Generated: 2026-05-17 01:16 UTC (final section), 23:11 UTC (first section)
Total wall: 132.4 min across three chained invocations
Args: max_problems=200 temp=0.0 max_new_tokens=512 no_system_prompt=True lenient_extract=True

Run as three sectional invocations per Option-1 chain
(`/tmp/8b_x200_chain.sh`):
1. Categoricals (ARC-Easy, ARC-Challenge, MMLU) — 478.5s = 8 min
2. GSM8K — 2285.1s = 38 min
3. HumanEval + SpellingBee — 5179.3s = 86 min

Section reports: `bonsai_8b_x200_{cat,gsm,gen}_2026-05-16.md`. This
doc merges them into the canonical ChatCORE.

## Results

| Task | Acc | n | Centered |
| --- | ---: | ---: | ---: |
| ARC-Easy | 0.9000 | 200 | +0.8667 |
| ARC-Challenge | 0.8150 | 200 | +0.7533 |
| MMLU | 0.5600 | 200 | +0.4133 |
| GSM8K | 0.7700 | 200 | +0.7700 |
| HumanEval | 0.7500 | 164 | +0.7500 |
| SpellingBee | 0.6800 | 200 | +0.6800 |
| **ChatCORE** | — | — | **0.7056** |

Computed as `(0.8667 + 0.7533 + 0.4133 + 0.7700 + 0.7500 + 0.6800) / 6 = 0.7056`.
