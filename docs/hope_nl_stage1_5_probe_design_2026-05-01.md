# Hope/NL Stage 1.5 — synthetic recall probe design

**Date:** 2026-05-01
**Branch:** `experiment/hope-nested-learning`
**Decision:** see `docs/project_notes/decisions.md::ADR-001`
**Goal:** isolate the architectural question — *does Stage 1's `LinearAttentionMemory` block actually carry memory across positions, vs the unmodified MLP?* — using a task whose pass/fail bar is unambiguous, before spending more pretrain budget or implementing Stage 2.

## Why a probe (and not more pretrain)

LM val_bpb on DCLM/SFT mixtures is a sum across token positions; positions where carrying state matters are a small fraction of the total. A 1% val_bpb gap can be either real architectural divergence on the load-bearing positions or noise on the bulk. We can't tell from one number, and cranking more seeds doesn't sharpen the question — it just narrows the noise band on the same blunt instrument.

A synthetic task where **every position contributes to the question** (and the question is binary: did the model recall the right token?) is the sharper instrument.

## Task: Multi-Query Associative Recall (MQAR)

Standard from the linear-attention literature (Mamba paper uses a variant). The model sees a list of (key, value) pairs in the prefix, then a sequence of query keys; for each query, it must produce the value bound to that key in the prefix.

### Sequence format

```
<bos> k_1 v_1 k_2 v_2 ... k_K v_K <sep> q_1 a_1 q_2 a_2 ... q_M a_M
```

- `K` distinct (key, value) pairs in the prefix (the "lookup table")
- `M` queries in the suffix; each `q_j ∈ {k_1..k_K}` (chosen with replacement)
- `a_j` = the value bound to `q_j` in the prefix (the ground-truth answer)
- `<bos>` and `<sep>` are existing tokenizer special tokens (we reuse `bos_token` for `<sep>` if no dedicated separator is desired — single-token marker is enough)

Total length: `2K + 2M + 2`. With `K=16, M=16` that's 66 tokens; we pad to `T=128` for batching.

### Loss / eval

- **Training:** standard teacher-forced cross-entropy at every position (positions inside the lookup contribute too — they're "predict the next random token," roughly uniform-noise loss; the model can't do anything with them. Only the `a_j` positions carry signal.)
- **Eval (the load-bearing measurement):** at each query position, `accuracy = (argmax(logits) == a_j)`; also report mean log-prob of the correct answer.

Pad positions get `target = -1` (cross-entropy `ignore_index`) — same convention as SFT.

### Vocabulary and token ID choices

Reuse the existing nanochat tokenizer (`vocab_size = 32768`) so model construction is unchanged. For the probe sequences we draw keys and values from disjoint token-ID ranges that are unlikely to be common subwords:

- **Keys:** token IDs in `[1024, 1024 + N_KEYS)` with `N_KEYS = 32`
- **Values:** token IDs in `[2048, 2048 + N_VALUES)` with `N_VALUES = 32`
- **Special:** `bos_token` for `<bos>`; reuse for `<sep>` (or pick a third disjoint token)

Range starts (1024, 2048) avoid the lowest IDs which tend to be common BPE merges. `N_KEYS = N_VALUES = 32 ≥ K = 16`, so we always have enough distinct keys/values per sequence.

This is essentially using the full transformer model on a synthetic mini-task — most of the embedding table goes unused, but that's fine. Both architectures pay the same overhead.

### Data generator

```python
def generate_mqar_sequence(K, M, T, key_range, value_range, bos_id, sep_id, rng):
    keys = rng.choice(key_range, size=K, replace=False)
    values = rng.choice(value_range, size=K, replace=False)
    lookup = dict(zip(keys, values))
    query_indices = rng.choice(K, size=M, replace=True)
    seq = [bos_id]
    for k, v in zip(keys, values):
        seq.extend([int(k), int(v)])
    seq.append(sep_id)
    for qi in query_indices:
        seq.extend([int(keys[qi]), int(values[qi])])
    targets = seq[1:] + [-1]  # next-token targets, last position has no target
    # Mask everything except the query-answer positions
    mask = [-1] * len(targets)
    answer_position_offset = 1 + 2 * K + 1  # offset of first a_j in `seq`
    for j in range(M):
        ans_pos = answer_position_offset + 2 * j + 1
        # The TARGET at position ans_pos - 1 is what predicts seq[ans_pos]
        mask[ans_pos - 1] = targets[ans_pos - 1]
    # Pad to T
    pad = max(0, T - len(seq))
    seq = seq + [bos_id] * pad
    mask = mask + [-1] * pad
    return seq[:T], mask[:T]
```

(Sketch — final impl will batch and tensorize.)

The "training-time mask" choice: only score the answer positions, not the entire sequence. This focuses gradient on the actual signal. Padding positions and lookup positions all get `-1`. Same convention as SFT; reuses the existing cross-entropy ignore-index handling and the upstream PR #741 NaN-safety patch we already shipped.

## Experiment design

Two arms, identical training budget:

| arm | architecture |
|---|---|
| **A: baseline** | d6, all MLP blocks (existing nanochat) |
| **B: Stage 1** | d6, `LinearAttentionMemory` at L3 (existing `bc54858`) |

**From-scratch initialization** (not warm-started from existing pretrained checkpoints) — isolates the architectural question from pretraining-prior artifacts. Both arms see the same random batches via fixed seed; the only varying input is `hope_memory_layer`.

### Hyperparameters

- d6 (n_layer=6, n_embd=384) — matches our actual experiments
- `T = 128`, `K = 16`, `M = 16`
- `device_batch_size = 64`, `total_batch_size = 8192` (accum=1 — small task, fast iters expected)
- `num_iterations = 1000` — well above the convergence point for an MQAR-style task at this width
- LR: same family as `base_train` defaults but lower (synthetic task, fewer tokens per iter); start with `embedding_lr=0.1, unembedding_lr=0.003, matrix_lr=0.01`. Tune if either arm fails to train.
- `eval_every = 100`, `n_eval_seqs = 256`
- Single seed each, fixed across the two arms

### Wall budget

Smoke-test on M2 plugged in shows ~2–3 s/iter (slower than my initial T=128 estimate; the d6 model still has substantial per-iter overhead even at short sequence length). Per arm: 1000 iters × ~1.2 s/iter steady state + first-iter compile + 10 evals ≈ **~22 min**. Two arms = **~45 min** total. Half of that is recoverable by dropping `--num-iterations` to 500 if convergence happens faster than expected.

## Expected outcomes & how to read them

| outcome | interpretation | what to do |
|---|---|---|
| Both arms converge to >95% recall accuracy similarly | Architecture is irrelevant at this difficulty | **Increase difficulty** (larger K, longer T, distractor noise) until a gap opens, *or* conclude that L3 alone isn't enough memory placement and Stage 4 (memory at multiple layers) is needed |
| Stage 1 converges faster / to higher accuracy than baseline | The memory block actually carries state across positions — the architecture has the property we wanted | Proceed to Stage 2 with confidence; use the probe to track whether learned α/η helps further |
| Baseline > Stage 1 | The MLP is doing something the LinearAttentionMemory at L3 cannot at this scale | Re-examine the architecture (placement, dims, init) before Stage 2; possibly the issue is single-block-only |
| Both fail to train | Probe difficulty too high (or LRs wrong) | Tune; reduce K or T |

The first three outcomes all advance the project. The fourth is a calibration step.

## Implementation plan

1. **`dev/probe_mqar.py`** — single self-contained script. Handles data generation, model construction (uses existing `GPT` / `GPTConfig` from `nanochat.gpt`), training loop, eval, comparison runner. CLI flags for K, M, T, num_iterations, seed, hope_memory_layer.
2. **Reuse existing infrastructure**: same `MuonAdamW` optimizer setup via `model.setup_optimizer()`; same `cross_entropy(ignore_index=-1, ...)` loss; same `torch.compile(model, dynamic=False)` pattern.
3. **Output**: per-arm log file `/tmp/probe_mqar_<arm>.log` with eval accuracy/log-prob trajectory; final comparison summary printed to stdout.
4. **Persist nothing** initially — this is a diagnostic, not a checkpoint we need to keep. If we want to debug a failure mode later, we can add `--save-final`.

Directly testable assertions (small unit tests, optional):
- `generate_mqar_sequence` round-trips: decoding the answer position via the lookup matches the value the targets expect.
- Mask only contains non--1 entries at answer positions (count = M).
- Sequence respects `T` length and starts with bos.

## What this doesn't test

- **Long-context** (we're at T=128, not 8K). MQAR at long T is a separate probe we'd add at Stages 4–6.
- **Continual learning / persistence across calls.** The probe is per-sequence; memory resets at every forward (matches Stage 1's actual semantics).
- **Multi-key collisions / interference** beyond what `M` queries on `K` pairs gives. Could extend with adversarial distractors later.
- **Gradient propagation through the memory state** (we're using Stage 1's parallel form, no recurrence). This becomes a question at Stage 2+.

## Status

Design accepted via ADR-001 (`docs/project_notes/decisions.md`). Implementation pending. Next session.
