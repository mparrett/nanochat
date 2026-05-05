# Data Investigations

Backlog of data-side questions and experiments that don't fit the architecture
track. **Defer all of these until the Hope/NL writeup wraps** — switching the
training corpus or eval composition invalidates comparisons against existing
val_bpb / SFT numbers (historical d6 baseline 1.174, A3 seeds 1.1729/1.1712,
A3' baseline TBD).

## Open questions

### Q1: Per-domain / per-source val_bpb during eval
**Why:** Tells us where memory layer 3 helps vs hurts. Bulk val_bpb averages
over the entire SFT distribution and hides the structural question (does the
recurrent state actually pay off on long-context, recall-heavy, or
domain-specific text?).

**Blocker on current corpus:** Our `base_data_climbmix/` parquets ship as a
single `text` column — no source/domain provenance. ClimbMix is delivered
post-dedup-and-strip.

**Cheap workaround (no labels needed):** stratify val_bpb by document length —
short / medium / long. ~10-line eval-loop change. Doesn't tell us
"wikipedia vs reddit" but does directly probe the memory hypothesis (gain
should concentrate on long docs if recurrent state is doing real work).

**Better option (with labels):** see Q2.

### Q2: Adopt ddudek/nanochat-climbmix-annotated as the labeled corpus
**Resource:** [`ddudek/nanochat-climbmix-annotated`](https://huggingface.co/datasets/ddudek/nanochat-climbmix-annotated)

Drop-in replacement for ClimbMix (nanochat-compatible parquet, `text` column
preserved + row-group layout) plus per-row metadata:

| Field | Type | Notes |
|---|---|---|
| `embedding` | float16[768] | Pre-computed sentence embedding |
| `topic_id` | int32 | Topic classifier ID |
| `topic_str` | string | Human-readable topic |
| `format_id` | int32 | Format classifier ID |
| `format_str` | string | Human-readable format (e.g., "code", "forum") |

**Scale:** 200 shards × ~86K rows = ~17M docs (vs ClimbMix's 6543 shards).
At d6/5000-iter (81M training tokens) we use 8 train shards, so 200 is
plenty.

**What it unlocks:**
- Per-topic / per-format val_bpb during pretrain eval (Q1, properly).
- Topic-balanced or topic-weighted training mixes (DSIR / DoReMi-lite).
- Embedding-space data dedup (Q3) without re-running an encoder ourselves.
- Per-topic SFT comparison ("Stage 2 wins on topic X but loses on Y").

**Cost to adopt:** Replace `base_data_climbmix` with this dataset, regenerate
tokenized shards, retrain everything we want to compare. Mid-experiment
switching breaks comparisons — defer until A3-prime + writeup are done.

**Risk:** corpus quality/diversity may differ from current ClimbMix in ways
we don't notice until we've sunk N hours of compute. Sanity check before
committing: train one quick d6 baseline on the new corpus and confirm
val_bpb is in the same ballpark.

### Q3: Within-corpus dedupe sanity check
**Why:** ClimbMix is upstream-deduped, but it's a *mix* of sources. Cross-shard
near-duplicates are plausible. At our 81M-token horizon dedupe wins are
typically small (literature gain is at trillion-token scale), but a sanity
check is cheap and would put the question to bed.

**Approach:** sample ~5K docs per shard, compute MinHash signatures
(`datasketch` lib, ~10 min compute), report cluster stats. Acceptance
criterion: <1% near-dupe clusters → confirmed clean, stop worrying.

### Q4: Per-domain / per-format CORE on base model
**Why:** Same logic as Q1 but for the headline metric. Currently CORE is a
single number averaged over all DCLM tasks. Per-task already exists in the
report — what doesn't exist is per-format on val pretrain text.

**Blocker:** Same as Q1 (no labels). Requires Q2.

## Things to skip

- **Re-running a quality classifier on top of FineWeb-Edu / ClimbMix.**
  Diminishing returns on already-curated data; the literature wins
  (DCLM-baseline, FineWeb-Edu) come from filtering raw CC, not re-filtering
  curated mixes.
- **DoReMi / DSIR domain reweighting.** Too heavy for the 81M-token budget;
  proxy model + reweighting pipeline costs more than the experiment is worth
  at our scale.
- **Token-level dedupe with MinHash-LSH.** Same reasoning — typical wins
  (Lee et al 2022) are 2-4% perplexity at trillion-token scale, ≪1% at ours.

## Priority

If time/compute opens up after the Hope/NL writeup:
1. Q1 length-stratified — cheapest, directly probes memory hypothesis
2. Q3 dedupe sanity check — fast, builds confidence in current corpus
3. Q2 corpus swap to annotated ClimbMix — biggest leverage but biggest cost
4. Q4 per-domain CORE — depends on Q2
