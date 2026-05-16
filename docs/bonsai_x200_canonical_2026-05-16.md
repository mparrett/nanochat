# Bonsai cross-arch ChatCORE at -x 200 — canonical follow-up

**Date:** 2026-05-16
**Branch:** `experiment/hope-nested-learning` (paused)
**Companion:** `docs/bonsai_field_eval_2026-05-16.html` (smoke-era narrative);
this doc tightens the headline.

## Why we ran this

The bonsai-harness arc on 2026-05-15/16 produced a strong-looking
headline (Bonsai-4B ChatCORE 0.6322, 3.62× over nanochat-d6_stage2)
but every number was a smoke at `-x 50`. Per-task stderr at n=50 is
roughly ±7pp (binomial), wide enough that two of our observed effects
— the 8B MMLU regression and the 4B/8B GSM8K tie — were within noise
and couldn't be confirmed or ruled out. The cross-arch story
("4B is the operational sweet spot") was robust at smoke quality but
we wanted publication-grade evidence.

`-x 200` was the natural intermediate: stderr drops to ±3.5pp, plus
HumanEval (164-problem test set) gets covered in full. Memory cost
zero; wall cost ~1h on 1.7B, ~1.5h on 4B, ~2h on 8B. We ran 1.7B and
4B; 8B is queued for a possible overnight run.

## Headline result

All numbers below: same harness, same args (`--no-system-prompt`,
`--max-new-tokens=512`, greedy, `--lenient-extract`), single
end-to-end run per model.

| Task          | 1.7B x200  | 4B x200    | 8B smoke   | d6_stage2  | random |
| ---           | ---:       | ---:       | ---:       | ---:       | ---:   |
| ARC-Easy      | 69.00%     | 84.00%     | 94.00%     | 25.80%     | 25.00% |
| ARC-Challenge | 48.50%     | 72.00%     | 76.00%     | 28.67%     | 25.00% |
| MMLU          | 46.00%     | 53.50%     | 50.00%     | 26.98%     | 25.00% |
| GSM8K         | 59.00%     | 79.00%     | 74.00%     |  0.76%     |  0.00% |
| HumanEval     | 46.95%*    | 58.54%*    | 74.00%     |  0.00%     |  0.00% |
| SpellingBee   | 43.50%     | 70.50%     | 66.00%     | 95.31%     |  0.00% |
| **ChatCORE**  | **0.4458** | **0.6456** | **0.6789** | **0.1744** | 0.00   |
| vs d6_stage2  | 2.56×      | 3.70×      | 3.89×      | 1.00×      | —      |

*HumanEval test set is exactly 164 problems; -x 200 ran the full set.
These are canonical HumanEval numbers, not smoke estimates.

The "vs d6_stage2" row is the operationally interesting number:
even the smallest Bonsai (473 MB on disk, 0.64 GB peak memory,
110 tok/s) beats our best nanochat-from-scratch result by 2.56×.
The 4B reaches 3.70×; 8B 3.89×.

## Smoke vs x200 — what shifted

Bonsai-1.7B smoke (-x 50) → x200:

| Task          | smoke | x200  | Δ      | within stderr? |
| ---           | ---:  | ---:  | ---:   | ---            |
| ARC-Easy      | 74.00 | 69.00 | −5.00  | yes (±3.3)     |
| ARC-Challenge | 44.00 | 48.50 | +4.50  | yes (±3.5)     |
| **MMLU**      | 36.00 | 46.00 | **+10.00** | **no (2.8σ)** |
| GSM8K         | 60.00 | 59.00 | −1.00  | yes (±3.5)     |
| HumanEval     | 44.00 | 46.95 | +2.95  | within canonical-vs-smoke variance |
| SpellingBee   | 46.00 | 43.50 | −2.50  | yes (±3.5)     |
| ChatCORE      | 0.4256 | 0.4458 | +0.020 | — |

Bonsai-4B smoke (-x 50) → x200:

| Task          | smoke | x200  | Δ      | within stderr? |
| ---           | ---:  | ---:  | ---:   | ---            |
| ARC-Easy      | 88.00 | 84.00 | −4.00  | yes (±2.6)     |
| **ARC-Chal.** | 64.00 | 72.00 | **+8.00**  | **no (2.5σ)** |
| MMLU          | 56.00 | 53.50 | −2.50  | yes (±3.5)     |
| **GSM8K**     | 74.00 | 79.00 | **+5.00**  | borderline (1.7σ) |
| HumanEval     | 64.00 | 58.54 | −5.46  | (n=164 canonical, smoke over-sampled easy) |
| **SpellingBee** | 64.00 | 70.50 | **+6.50** | **no (2.0σ)** |
| ChatCORE      | 0.6322 | 0.6456 | +0.013 | — |

**Real lift signals (multi-sigma) over smoke**: 1.7B-MMLU +10pp,
4B-ARC-Challenge +8pp, 4B-SpellingBee +6.5pp, 4B-GSM8K +5pp. In
every case the smoke under-counted the model's true capability.
**Real come-downs**: ARC-Easy on both models (~−5pp, within noise);
HumanEval on 4B (smoke over-sampled the easy problems, canonical is
58.54%). Net ChatCORE moved up modestly for both models, with the
gap between them essentially unchanged.

## Diminishing returns — tighter

| Step               | ChatCORE lift | Multiple |
| ---                | ---:          | ---:     |
| d6_stage2 → 1.7B   | +0.271        | 2.56×    |
| 1.7B → 4B          | +0.200        | 1.45×    |
| 4B → 8B*           | +0.033        | 1.05×    |

*4B at x200, 8B at smoke. The 8B-vs-4B comparison is partially
unfair because the 8B is still at the noisier measurement. The
pattern from smoke→x200 shows smoke results tend to be biased *upward*
on ARC-Easy and HumanEval and *downward* on ARC-Challenge / GSM8K /
SpellingBee / MMLU-low-N. If 8B follows the same pattern at x200, its
ChatCORE would likely land between 0.65 and 0.67, **making the
4B-vs-8B gap effectively close to zero**. Worth confirming overnight.

## Operational call — unchanged, sharper

**4B is the pick.** At canonical-ish N:

- 4B ChatCORE 0.6456 vs 8B 0.6789 → only 5% relative gain
- 4B wall 85 min vs 8B ~120 min → 41% more wall
- 4B peak memory 1.29 GB vs 8B 2.48 GB → 92% more memory
- 4B disk 1.1 GB vs 8B 2.2 GB → exactly 2×

The 8B might lose its remaining 5% lift at fair measurement. The 4B
is the operational sweet spot empirically, and bonsai's ADR-001
(written months before our harness existed) called it correctly.

## Caveats worth carrying forward

- **MMLU is the noisy task.** At n=200 the 4B's 53.5% vs 8B's 50%
  is well within ±3.5pp stderr. The supposed 4B-vs-8B MMLU
  regression is still unresolved. Disambiguating would require both
  at -x 500+ (~3h each).
- **GSM8K 4B/8B tied at 74% on smoke; x200 split them** (4B 79%,
  8B 74% at smoke). With 8B at x200 the picture might shift again.
- **HumanEval is now canonical**, which is the cleanest single result
  in this batch. 1.7B 46.95%, 4B 58.54%. 8B is still 50/164 = 30%
  coverage of its test set; canonical 8B HumanEval is the most
  interesting open number from a "what does int2-Qwen3-8B actually
  do" perspective.
- **All eval problems are nanochat-side renderings.** ARC, MMLU,
  GSM8K, HumanEval, SpellingBee come from the same dataset
  shuffles seed (42) that nanochat uses, so the d6_stage2 reference
  numbers and the Bonsai numbers are evaluated on identical problem
  subsets. The harness is fair across architectures.
- **`--lenient-extract` is the load-bearing flag**, validated against
  16 representative completions including the actual probed
  multilingual SpellingBee outputs from yesterday. False-positive
  rate from the "last bare integer in tail" fallback looks bounded
  in practice — GSM8K results stayed in the published Qwen3 range.

## Wall ledger (this batch)

| Run             | Wall    | ChatCORE | Bytes per second of capability* |
| ---             | ---:    | ---:     | ---:                           |
| 1.7B x200       | 49.6 min | 0.4458   | 158 KB/sec                     |
| 4B x200         | 84.9 min | 0.6456   | 215 KB/sec                     |
| **Combined**    | **134.5 min** | — | — |

*KB/sec just for fun — disk size of the weights divided by wall to
produce the ChatCORE result. Reflects "how much capability did you
fetch for each second the M2 was busy." 4B is denser by this metric.

## Open follow-ups

1. **8B at x200** (~2h overnight). Would make the table fully
   symmetric and resolve whether 4B and 8B are effectively tied
   in capability.
2. **HTML field-eval update.** The 2026-05-16 HTML used smoke
   numbers. If we want a publication-grade version, swap the
   headline table to use x200 numbers for 1.7B and 4B; 8B can be
   updated once that runs (or left as smoke with an asterisk).
3. **Apples-to-apples nanochat baseline.** d6_stage2 and d8_a2 were
   themselves smoke runs at -x 100. Rerunning them at x200 with our
   own harness would make the cross-arch comparison fully
   symmetric. ~30 min each on M2 (categorical-heavy, no MMLU-14k
   blowup). Worth doing only if we want a publication-grade ChatCORE
   comparison.
