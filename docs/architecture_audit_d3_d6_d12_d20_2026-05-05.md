# Architecture audit — d3 / d6 / d12 / d20

**Date:** 2026-05-05
**Origin:** Conversation post-d3 smoke run, asking "what changes across depths,
and what's tweakable beyond `--depth`?"
**Purpose:** Single-table reference for what each depth costs, what auto-derives
from `--depth=N`, and which knobs are escape hatches with their re-baseline cost.

All numbers below assume **canonical d6 settings**: `head_dim=64`,
`aspect_ratio=64`, `max_seq_len=512`, `window_pattern=L`. These match
`d6_baseline_modern` and the A3 Stage 2 runs. **The script defaults differ**
(`head_dim=128`, `max_seq_len=2048`, `window_pattern=SSSL`); see "Tweakable
knobs" below.

## Architecture shape

| | d3 | d6 | d12 | d20 |
|---|---:|---:|---:|---:|
| `n_layer` | 3 | 6 | 12 | 20 |
| `n_embd = depth × 64` | 192 | 384 | 768 | 1280 |
| `n_head = n_embd / 64` | 3 | 6 | 12 | 20 |
| `head_dim` | 64 | 64 | 64 | 64 |
| value_embed layers `⌈n_layer/2⌉` | 2 | 3 | 6 | 10 |

Depth-sweep rule (`scripts/base_train.py:165-167`):
```python
base_dim   = depth * args.aspect_ratio
model_dim  = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads  = model_dim // args.head_dim
```

`n_embd` rounds up to the next multiple of `head_dim` (FA3 alignment), then
`n_head = n_embd / head_dim`. Width and head count rise linearly with depth;
`head_dim` stays fixed.

## Params (~M)

| group | d3 | d6 | d12 | d20 |
|---|---:|---:|---:|---:|
| `wte` (token embed: 32k × n_embd) | 6.29 | 12.58 | 25.17 | 41.94 |
| `lm_head` (untied) | 6.29 | 12.58 | 25.17 | 41.94 |
| `value_embeds` (each: 32k × n_embd) | 12.58 | 37.75 | 151.0 | 419.4 |
| `transformer_matrices` (12 · n_embd² · n_layer) | 1.33 | 10.62 | 84.93 | 393.22 |
| **total** | **~26.5** | **~73.5** | **~286** | **~896** |
| **scaling_params** (`transformer_matrices + lm_head`) | 7.62 | 23.20 | 110.10 | 435.16 |

**Two non-obvious things in this table:**

- **`value_embeds` are a parameter monster.** At d20 they're nearly half the
  model. They live on alternating layers (last layer always included; the
  rule is `has_ve(i, n_layer) → i % 2 == (n_layer-1) % 2` at `gpt.py:86-88`).
  Each one is a full `vocab × n_embd` table, same shape as `wte`.
- **`scaling_params` excludes embeddings + value_embeds.** This is what
  nanochat uses for Chinchilla budget calculations (`base_train.py:309`).
  It scales roughly as `depth³` — one factor from `n_layer`, two from
  `n_embd²` — which is what makes wall-time blow up at d20.

The per-block matrix count of 12·n_embd² breaks down as:
- Attention: QKV (3·n_embd²) + W_o (n_embd²) = 4·n_embd²
- MLP (ReLU² style, no GLU): c_fc (4·n_embd²) + c_proj (4·n_embd²) = 8·n_embd²
- Total: 12·n_embd² per block × n_layer blocks

## Auto-derived training hyperparams

All four scaling formulas come from `scripts/base_train.py:300-360` and
`nanochat/gpt.py:638`. d12 is the reference depth; everything else scales
toward or away from it.

| | d3 | d6 | d12 | d20 |
|---|---:|---:|---:|---:|
| Chinchilla `target_tokens` (ratio=12) | 91M | 278M | 1.32B | 5.22B |
| optimal `total_batch_size` (Power-Lines D^0.383) | ~32k | ~64k | 524,288 (`B_REF`) | ~1M |
| Chinchilla iters @ optimal B | ~2,800 | ~4,300 | 2,520 | ~5,000 |
| `dmodel_lr_scale = (n_embd/768)^-0.5` | 2.000 | 1.414 | 1.000 | 0.775 |
| weight_decay scale `(D_REF/D) · √(B/B_REF)` | depth-dependent (decreases with D, increases with B) | | | |

So for AdamW groups (embeddings, lm_head, scalars, gates), the actual LR at
d3 is `2.0 × base_lr`, and at d20 it's `0.78 × base_lr`. Muon's matrix LR
doesn't get the `dmodel_lr_scale`, only the batch-size scale `√(B/B_REF)`.

The references:
- `B_REF = 2**19 = 524,288` — hardcoded as "optimal batch size at d12,
  measured empirically" (`base_train.py:317`).
- `D_REF = ratio × scaling_params(d12)` — recomputed per-run (`base_train.py:316`).
- Power-Lines `B^0.383` from arxiv 2505.13738.
- T_epoch framework for weight decay from arxiv 2405.13698.

## On M2 specifically

| | d3 | d6 | d12 | d20 |
|---|---:|---:|---:|---:|
| Per-iter wall (MPS dispatch-bound) | ~1.0 s | ~1.9 s | ~5-7 s | ~30-60 s (est) |
| Full Chinchilla pretrain wall on M2 | ~1 h | ~3-4 h | ~25-40 h | infeasible (>200 h) |
| Memory pressure at `device_batch_size=32` | tiny | safe | tight | OOM risk |

**d12 is plausibly attemptable** with reduced `device_batch_size` and ~30 h
wall. **d20 is not an M2 model** — needs a real GPU box.

---

## What's tweakable

**Karpathy's stance and why it's load-bearing:** every other knob defaults
to a value that makes `--depth=N` coherent. The moment you tweak one, you
owe a re-baseline because results across depths stop being apples-to-apples.
Today's d3 smoke run hit exactly this — three script defaults differed
from canonical d6 and the run blew up at the assertion. The fix wasn't to
remember more flags; it was `--inherit-from <known-good-meta>`.

### Tier 1 — defensible escape hatches (with re-baseline)

| flag | default | canonical d6 | what changes |
|---|---|---|---|
| `--target-param-data-ratio` | 12 | 12 | data budget. Try 20 (true Chinchilla) or 6 (over-trained) for scaling-law studies. |
| `--seed` | 42 | various | bracket variance. A2/A3 already validated SFT-seed-stable, pretrain spread ~0.0016 at d6. |
| `--num-iterations` | -1 (auto) | 5000 | shorten for fast smoke, lengthen for over-Chinchilla. |

### Tier 2 — modernization knobs (real architectural changes)

Documented in `docs/project_incoming/feat_modernization_alignment.md`:

| flag/change | now | proposed | reference |
|---|---|---|---|
| `--aspect-ratio` | 64 | 100 (CS336 "production" ratio) | deliberate-departures section |
| MLP nonlinearity | ReLU² | SwiGLU/GeGLU | A3 of modernization-alignment, deferred |
| qk_norm | absent | RMSNorm on Q and K before matmul | A2, gated on d8 |
| z-loss | absent | `(log Z)²` softmax-normalizer regularizer | **ADR-004 standby** |
| `--window-pattern` | `SSSL` (default) / `L` (canonical d6) | `SSSL` is Llama 3 / Gemma 2 / Olmo 2 pattern | currently mixed |

### Tier 3 — direct LR/optimizer tweaks

`--matrix-lr=0.02`, `--embedding-lr=0.3`, `--unembedding-lr=0.008`,
`--scalar-lr=0.5`, `--weight-decay=0.28`, `--warmup-steps=40`,
`--warmdown-ratio=0.65`, `--final-lr-frac=0.05`. All tuned at d12 and
muP-style transferred. Departing from these needs scaling-law evidence,
not intuition.

### Tier 4 — almost certainly shouldn't tweak unless you know why

- `vocab_size = 32,768` — property of the BPE tokenizer, not a free knob.
- `head_dim` script default of 128 — canonical is 64; the script default
  is the **trap** (causes silent recipe drift).
- `B_REF = 524,288` — hardcoded "d12 reference batch size, measured
  empirically." Moving this invalidates every scaling formula.

---

## Recommendations (post-A3, M2 dev box)

1. **Promote canonical defaults to be the script defaults.** `head_dim=64`,
   `max_seq_len=512`. Pure recipe-drift prevention. Zero risk, eliminates
   the class of bug d3 smoke hit today.
2. **z-loss as standby** — already done, ADR-004.
3. **qk_norm at d8** — only if d8 activates. Cheap (~3 lines). The
   lecture's "more universal" claim is testable there.
4. **aspect_ratio sweep** — defer. The "is 100 better than 64" question
   only matters if we go deeper than d8, where it might affect M2 fit.

Everything else is research-class compute spend. The depth dial is principled
because the `1/√dmodel` + Power-Lines + T_epoch corrections form a coherent
system. Break one and you're solo-debugging at every depth. Keep them
coherent and you can sweep.

## Where the numbers come from

- Architecture shape: `scripts/base_train.py:160-184` (build_model_meta)
- Param counting: `nanochat/gpt.py:580-585` (num_scaling_params)
- value_embeds policy: `nanochat/gpt.py:86-88` (has_ve)
- Chinchilla / Power-Lines / T_epoch: `scripts/base_train.py:300-348`
- 1/√dmodel LR scaling: `nanochat/gpt.py:637-639` (setup_optimizer)
- Reference checkpoint: `~/.cache/nanochat/base_checkpoints/d6_baseline_modern/meta_005000.json`
