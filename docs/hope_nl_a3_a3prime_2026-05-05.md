# Hope/NL A3 + A3-prime — pretrain seed-variance + modern-recipe baseline (2026-05-05)

The two-experiment finale of the Hope/NL track. A3 measures pretrain-seed
stability of Stage 2. A3-prime re-runs vanilla d6 baseline under modern
training code + pinned seed, removing the recipe-drift confound that
inflated the original "Stage 2 wins by 1.8% on SFT" headline. Together
they answer: does the Stage 2 architecture win at d6/5000-iter on this
corpus, or was the headline an artifact?

**TL;DR:** the headline was an artifact. Modern-recipe vanilla d6 baseline
beats Stage 2 (slightly) on both pretrain val_bpb and SFT val_bpb. All
differences are within the 0.0016 seed-noise spread we measured on A3.
Strip the recipe-drift confound and the Stage 2 architectural delta is
**neutral, not the +1.8% the original comparison suggested**.

## Experiment design

### A3 — multi-seed Stage 2 pretrain
Two new pretrain seeds (1, 2) on the Stage 2 (additive learned-gate)
architecture. Recipe inherited via `--inherit-from` from the original
Stage 2 reference meta — 34 fields including `head_dim=64`,
`max_seq_len=512`, `window_pattern=L`, `num_iterations=5000`,
`hope_additive_memory_layer=3`, alpha/eta init biases. Hardware-config
parity by construction (the `--inherit-from` mechanism added in
`ca9bc94`).

```bash
PYTHONUNBUFFERED=1 nohup uv run python -u -m scripts.base_train \
    --inherit-from=base_checkpoints/d6_stage2/meta_005000.json \
    --run=stage2-d6-pretrain-seed{1,2} \
    --model-tag=d6_stage2_pretrain_s{1,2} \
    --seed={1,2} \
    --save-every=1000 --save-keep-last-n=2
```

### A3-prime — modern-recipe baseline pretrain
Vanilla d6 baseline (no `hope_*` flags) under current `master`, seed=42
pinned. Recipe matches Stage 2 hand-for-hand except the architecture
(no memory branch). Goal: replace the unpinned-seed, stale-recipe
historical baseline (val_bpb 1.174, SFT 0.6639) with a fair comparator.

```bash
PYTHONUNBUFFERED=1 nohup uv run python -u -m scripts.base_train \
    --run=baseline-d6-modern \
    --model-tag=d6_baseline_modern \
    --depth=6 --head-dim=64 --max-seq-len=512 --window-pattern=L \
    --num-iterations=5000 --total-batch-size=16384 \
    --eval-every=100 --sample-every=100 \
    --core-metric-every=-1 --eval-tokens=524288 \
    --seed=42 \
    --save-every=1000 --save-keep-last-n=2
```

Recipe drift between historical baseline and A3-prime is small but real
— most prominently `840d3db` (move `ve_gate` from Muon → AdamW) — plus
optimizer/dataloader/init churn that's hard to enumerate. A3-prime locks
all of it under `master`.

### SFT
Same SFT recipe across all SFT runs (375 steps, batch 65536, seed=1)
inheriting from `chatsft_checkpoints/d6_stage2_s1/meta_000375.json`.
Run on top of A3 seed=1 pretrain and A3-prime baseline pretrain. (SFT
on A3 seed=2 was not run — bracketing across both stage 2 pretrain
seeds × SFT was deemed redundant given the A3' result.)

## Results

### Pretrain val_bpb @ step 5000

| pretrain | seed | val_bpb | Δ vs A3' baseline |
|---|---:|---:|---:|
| Historical d6 baseline (pre-`--seed`-flag, old recipe) | unpinned | 1.1740 | +0.0054 |
| Stage 2 (additive learned gate), A3 seed=1 | 1 | 1.1729 | +0.0043 |
| Stage 2 (additive learned gate), A3 seed=2 | 2 | 1.1712 | +0.0026 |
| **A3' vanilla d6 baseline (modern recipe)** | 42 | **1.1686** | — |

A3 inter-seed spread: **0.0016.**

A3-prime came in **0.0026-0.0054 below** every Stage 2 / historical
baseline number on pretrain val_bpb. The 0.0026 gap to the closer Stage 2
seed is ~1.6× the A3 seed-spread — within what an unmeasured baseline-seed
noise distribution could plausibly explain, but the direction is consistent
across both Stage 2 seeds.

### SFT val_bpb @ step 375

| pretrain → SFT | val_bpb (final) | Δ vs A3' baseline |
|---|---:|---:|
| Historical d6 baseline → SFT (`d6_b_iso`, Phase 3) | 0.6639 | +0.0156 |
| Stage 2 (`d6_stage2`) → SFT (A2 sft_s1) | 0.6516 | +0.0033 |
| Stage 2 A3 seed=1 → SFT | 0.6495 | +0.0012 |
| **A3' baseline → SFT** | **0.6483** | — |

A3-prime baseline SFT beat the closer Stage 2 SFT (A3 seed=1) by 0.0012.
Beat A2 historical-recipe Stage 2 SFT (`d6_stage2_s1`) by 0.0033. Beat
the historical baseline SFT by 0.0156 — the latter is the gap the original
"+1.8% Stage 2 win" was built on, and it's now revealed to be entirely
recipe drift.

### SFT trajectory comparison (step → val_bpb)

| step | A3 seed=1 → SFT | A3' baseline → SFT | Δ |
|---:|---:|---:|---:|
| 0 | 1.0266 | 1.0209 | -0.0057 |
| 50 | 0.8330 | 0.8287 | -0.0043 |
| 100 | 0.7939 | 0.7952 | +0.0013 |
| 150 | 0.7690 | 0.7692 | +0.0002 |
| 200 | 0.7618 | 0.7612 | -0.0006 |
| 250 | 0.7287 | 0.7267 | -0.0020 |
| 300 | 0.6898 | 0.6889 | -0.0009 |
| 350 | 0.6565 | 0.6552 | -0.0013 |
| 375 | 0.6495 | 0.6483 | -0.0012 |

A3-prime baseline starts ahead from initialization (better pretrain),
loses the gap by ~step 100, then converges back to ~0.001 ahead by step
375. Trajectories interleave throughout — no point at which Stage 2 is
cleanly ahead.

## Headline check (revised)

The "Stage 2 SFT win" headline (originally **0.6518 vs baseline 0.6639 =
0.0121, ≈1.8%**) was framed against a **stale-recipe historical baseline**.
Replacing the comparator with a modern-recipe baseline:

| quantity | original framing | revised framing |
|---|---:|---:|
| Stage 2 SFT | 0.6518 | 0.6495 (A3 seed=1) |
| Baseline SFT | 0.6639 (historical) | 0.6483 (A3' modern) |
| Δ | **-0.0121 (Stage 2 wins ~1.8%)** | **+0.0012 (Stage 2 loses ~0.2%)** |
| Within A3 seed-noise (0.0016)? | no, ~7.5× spread | yes, ~0.75× spread |

**The architectural win evaporates under recipe-controlled comparison.**
The 0.0121 gap was overwhelmingly recipe drift, not architecture.

## What this means

1. **Stage 2 architecture is neutral, not net-positive, at d6/5000-iter
   on ClimbMix.** Both pretrain val_bpb and SFT val_bpb show A3' baseline
   slightly ahead. Magnitudes are within seed-noise — call it a tie, not
   a Stage 2 loss — but the +1.8% claim does not survive.

2. **The MQAR probe story remains true.** Stage 2's additive memory
   branch *does* learn memory-flavored behavior on synthetic recall
   (saturation by step ~76 vs baseline never; W_o init scale is the
   load-bearing knob, per Stage 1.5b). What does NOT happen at this
   scale is translation of that capability to natural-language val_bpb
   gain. The mechanism works; the corpus / horizon doesn't reward it.

3. **Stage 4 (multi-block memory) was conditional on Stage 2 surviving
   the comparison.** That condition is not met. Without a clean Stage 2
   win to anchor on, scaling memory across multiple blocks is poorly
   motivated — there's no signal to amplify.

4. **Recipe drift is real and silent.** 4 commits between the historical
   baseline and modern code accumulated ~0.0156 of "free" SFT improvement
   without any architecture work. Future Hope/NL-style experiments must
   pin recipe + seed and re-baseline against modern code, not against
   archived numbers. A3-prime's existence catches this category of risk.

## Caveats

- **n=1 baseline pretrain seed.** A3-prime did not measure baseline-seed
  variance. The 0.0012-0.0026 magnitudes by which baseline beat Stage 2
  could plausibly be on the lucky side of a baseline-seed distribution.
  For a tight bound, n=3 baseline (~13h) was Codex's preferred plan;
  A3-prime is the cheap compromise that catches recipe drift but not
  baseline-seed luck. **The result reads "Stage 2 doesn't win" with high
  confidence; "Stage 2 loses" with lower confidence.**
- **Stage 2 SFT only run on top of A3 seed=1.** Bracketing across both
  pretrain seeds × SFT was deemed redundant given A3' result.
- **d6 is small.** Memory architectures may show structural value at
  larger scale or longer training horizon. This experiment cannot
  exclude that hypothesis — only the d6/5000-iter / ClimbMix one.
- **SFT mixture is MMLU/GSM8K-heavy.** Tasks that don't reward
  long-context memory. A different SFT mix (e.g. multi-doc QA,
  long-form summarization) could in principle reveal architectural
  benefit that this evaluation doesn't.

## Decision

Wrap. The Hope/NL Stage 0 → 1 → 1.5(a/b/c) → 2 → A1/A2/A3/A3' arc has
delivered:

- A working additive memory module (Stage 1.5b root cause + fix on `W_o`
  init = 1.0).
- A learned per-token gating mechanism (Stage 2) that *does* learn on
  the MQAR probe.
- A clean experimental discipline (probe-first, multi-seed, recipe-
  pinned, config-parity-by-construction via `--inherit-from`).
- Honest finding: at d6/5000-iter on ClimbMix, the architecture does
  not improve LM val_bpb under recipe-controlled comparison.

Defer Stage 4 / multi-block memory. The conditional ("Stage 2 survives")
is not met. Revisit at larger scale or different evaluation if compute
opens up.

## Artifacts

- Pretrain checkpoints:
  - `base_checkpoints/d6_stage2_pretrain_s1/model_005000.pt` (A3 seed=1)
  - `base_checkpoints/d6_stage2_pretrain_s2/model_005000.pt` (A3 seed=2)
  - `base_checkpoints/d6_baseline_modern/model_005000.pt` (A3' baseline)
- SFT checkpoints:
  - `chatsft_checkpoints/d6_stage2_pretrain_s1_sft/model_000375.pt`
  - `chatsft_checkpoints/d6_baseline_modern_sft/model_000375.pt`
- wandb runs:
  - A3 seed=1: https://wandb.ai/matt-parrett/nanochat/runs/avvd9uov
  - A3 seed=2: https://wandb.ai/matt-parrett/nanochat/runs/pajx70rk
  - A3' baseline: https://wandb.ai/matt-parrett/nanochat/runs/2m6exejl
  - SFT on A3 seed=1: https://wandb.ai/matt-parrett/nanochat-sft/runs/3fprtaef
  - SFT on A3' baseline: https://wandb.ai/matt-parrett/nanochat-sft/runs/s4x7im7t
- Logs: `/tmp/pretrain_stage2_s{1,2}.log`,
  `/tmp/pretrain_baseline_modern.log`, `/tmp/sft_stage2_pretrain_s1.log`,
  `/tmp/sft_baseline_modern.log`.
