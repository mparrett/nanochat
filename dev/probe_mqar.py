"""
Hope/NL Stage 1.5 — Multi-Query Associative Recall probe.

Trains a small d6 model from random init on a synthetic key→value lookup task
and reports per-query argmax accuracy on a held-out test set. Used to compare
the unmodified MLP baseline against the Stage 1 LinearAttentionMemory swap on
a task where carrying state across positions actually matters.

Sequence layout (length T+1, then split into inputs/targets the standard way):

    bos, k_1, v_1, ..., k_K, v_K, sep, q_1, a_1, ..., q_M, a_M, [pad...]

Loss is masked to the answer positions only (every other position has target=-1,
the cross-entropy ignore index). Eval reports argmax accuracy at query positions.

Usage (run once per arm, compare the two logs):

    # baseline (all MLP):
    python -m dev.probe_mqar --label baseline --hope-memory-layer=-1 \
        > /tmp/probe_mqar_baseline.log 2>&1

    # Stage 1 (LinearAttentionMemory at L3):
    python -m dev.probe_mqar --label stage1 --hope-memory-layer=3 \
        > /tmp/probe_mqar_stage1.log 2>&1

Pass --hope-memory-layer=-1 to disable the swap (passed as None to GPTConfig).
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import compute_init, autodetect_device_type


# -----------------------------------------------------------------------------
# CLI
parser = argparse.ArgumentParser()
parser.add_argument('--label', type=str, default='probe', help='log prefix to identify the run')
parser.add_argument('--hope-memory-layer', type=int, default=-1, help='layer index for LinearAttentionMemory swap (-1 = disabled / baseline MLP)')
parser.add_argument('--hope-additive-memory-layer', type=int, default=-1, help='layer index for additive LinearAttentionMemory (-1 = disabled). Adds a third residual contribution alongside attn+mlp instead of replacing the MLP.')
parser.add_argument('--hope-memory-w-o-init-scale', type=float, default=0.0, help='LinearAttentionMemory W_o init scale (0.0 = zeros [default], >0 = uniform[-s*scale, s*scale] like K/V/Q)')
# Task shape
parser.add_argument('--K', type=int, default=16, help='number of (key, value) pairs in the lookup prefix')
parser.add_argument('--M', type=int, default=16, help='number of queries in the suffix')
parser.add_argument('--T', type=int, default=128, help='sequence length (padded with bos)')
parser.add_argument('--n-keys', type=int, default=32, help='size of key token pool')
parser.add_argument('--n-values', type=int, default=32, help='size of value token pool')
parser.add_argument('--key-start', type=int, default=1024, help='token ID start for keys')
parser.add_argument('--value-start', type=int, default=2048, help='token ID start for values')
# Model shape (matches our actual experiments — d6 with head_dim=64)
parser.add_argument('--depth', type=int, default=6)
parser.add_argument('--head-dim', type=int, default=64)
parser.add_argument('--vocab-size', type=int, default=32768)
# Training
parser.add_argument('--device-batch-size', type=int, default=64)
parser.add_argument('--num-iterations', type=int, default=1000)
parser.add_argument('--eval-every', type=int, default=100)
parser.add_argument('--n-eval-seqs', type=int, default=256)
parser.add_argument('--embedding-lr', type=float, default=0.1)
parser.add_argument('--unembedding-lr', type=float, default=0.003)
parser.add_argument('--matrix-lr', type=float, default=0.01)
parser.add_argument('--scalar-lr', type=float, default=0.04)
# Reproducibility
parser.add_argument('--seed', type=int, default=0, help='seed for model init AND data generation (eval data uses seed+1M)')
args = parser.parse_args()


# -----------------------------------------------------------------------------
# MQAR data generator
def generate_mqar_batch(B, K, M, T, n_keys, n_values, key_start, value_start, bos_id, sep_id, rng):
    """Generate a batch of MQAR sequences. Returns (inputs, targets) of shape (B, T) each.

    inputs[b, t] is what the model sees at position t.
    targets[b, t] is what the model is scored on predicting (= -1 except at query positions
    where it should predict the value bound to the query key).
    """
    # Build the full (T+1)-length sequence so we can split into inputs[:T] and targets shifted by 1.
    seq_full = np.full((B, T + 1), bos_id, dtype=np.int64)  # pad with bos
    targets = np.full((B, T), -1, dtype=np.int64)
    for b in range(B):
        keys = rng.choice(n_keys, size=K, replace=False) + key_start
        values = rng.choice(n_values, size=K, replace=False) + value_start
        q_indices = rng.integers(0, K, size=M)
        s = [bos_id]
        for k, v in zip(keys, values):
            s.extend([int(k), int(v)])
        s.append(sep_id)
        for qi in q_indices:
            s.extend([int(keys[qi]), int(values[qi])])
        L = min(len(s), T + 1)
        seq_full[b, :L] = s[:L]
        # Mark query-position targets. Query token at position p in s; expected next token at p+1.
        # In the unshifted seq, target[p] = s[p+1] = the bound value. We only score these positions.
        ans_offset = 1 + 2 * K + 1  # position of first query token (q_1) in s
        for j in range(M):
            q_pos = ans_offset + 2 * j
            if q_pos >= T:
                break  # ran out of room
            targets[b, q_pos] = int(values[q_indices[j]])
    inputs = seq_full[:, :T]
    return torch.from_numpy(inputs), torch.from_numpy(targets)


# -----------------------------------------------------------------------------
# Eval
@torch.no_grad()
def evaluate(model, K, M, T, n_keys, n_values, key_start, value_start, bos_id, sep_id, n_seqs, device, batch_size, rng):
    """Returns (acc, mean_log_prob) over n_seqs fresh sequences (only counting query positions)."""
    model.eval()
    correct = 0
    total = 0
    log_probs_sum = 0.0
    for start in range(0, n_seqs, batch_size):
        B = min(batch_size, n_seqs - start)
        inputs, targets = generate_mqar_batch(B, K, M, T, n_keys, n_values, key_start, value_start, bos_id, sep_id, rng)
        inputs = inputs.to(device, dtype=torch.int32)
        targets = targets.to(device)
        logits = model(inputs)  # (B, T, V) since no targets passed
        mask = targets != -1
        preds = logits.argmax(dim=-1)
        correct += ((preds == targets) & mask).sum().item()
        total += mask.sum().item()
        log_probs = F.log_softmax(logits, dim=-1)
        tgt_safe = targets.clamp(min=0)  # avoid -1 indexing when gathering
        tgt_lp = log_probs.gather(-1, tgt_safe.unsqueeze(-1)).squeeze(-1)
        log_probs_sum += (tgt_lp * mask).sum().item()
    model.train()
    return (correct / total if total else 0.0, log_probs_sum / total if total else 0.0)


# -----------------------------------------------------------------------------
# Setup
device_type = autodetect_device_type()
ddp, rank, local_rank, ddp_world_size, device = compute_init(device_type)
torch.manual_seed(args.seed)

# Model dim derivation matches base_train.py
base_dim = args.depth * args.head_dim
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
n_heads = model_dim // args.head_dim

# -1 sentinel disables (GPTConfig expects None)
hope_layer = None if args.hope_memory_layer < 0 else args.hope_memory_layer
hope_add_layer = None if args.hope_additive_memory_layer < 0 else args.hope_additive_memory_layer

config = GPTConfig(
    sequence_len=args.T,
    vocab_size=args.vocab_size,
    n_layer=args.depth,
    n_head=n_heads,
    n_kv_head=n_heads,
    n_embd=model_dim,
    window_pattern='L',
    hope_memory_layer=hope_layer,
    hope_additive_memory_layer=hope_add_layer,
    hope_memory_w_o_init_scale=args.hope_memory_w_o_init_scale,
)
print(f'[{args.label}] config: {config}')

with torch.device('meta'):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
model.train()

n_params = sum(p.numel() for p in model.parameters())
print(f'[{args.label}] params: {n_params:,} ({n_params/1e6:.2f}M)  hope_memory_layer={hope_layer}  hope_additive_memory_layer={hope_add_layer}')

optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    scalar_lr=args.scalar_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=0.0,
)

# bos and sep tokens — pick low IDs that don't overlap with our key/value ranges (1024+, 2048+)
BOS_ID, SEP_ID = 0, 1
print(f'[{args.label}] task: T={args.T}, K={args.K}, M={args.M}, keys=[{args.key_start},{args.key_start+args.n_keys}), values=[{args.value_start},{args.value_start+args.n_values}), bos={BOS_ID}, sep={SEP_ID}')

train_rng = np.random.default_rng(args.seed)
eval_rng = np.random.default_rng(args.seed + 1_000_000)


# -----------------------------------------------------------------------------
# Train
print(f'[{args.label}] training {args.num_iterations} iters, batch={args.device_batch_size}')
t_start = time.time()
for step in range(args.num_iterations):
    inputs, targets = generate_mqar_batch(
        args.device_batch_size, args.K, args.M, args.T,
        args.n_keys, args.n_values, args.key_start, args.value_start,
        BOS_ID, SEP_ID, train_rng,
    )
    inputs = inputs.to(device, dtype=torch.int32)
    targets = targets.to(device)
    loss = model(inputs, targets)
    loss.backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)

    do_eval = (step % args.eval_every == 0) or (step == args.num_iterations - 1)
    if do_eval:
        acc, mean_lp = evaluate(
            model, args.K, args.M, args.T,
            args.n_keys, args.n_values, args.key_start, args.value_start,
            BOS_ID, SEP_ID, args.n_eval_seqs, device, args.device_batch_size, eval_rng,
        )
        elapsed = time.time() - t_start
        print(f'[{args.label}] step {step+1:04d}/{args.num_iterations} | '
              f'loss: {loss.item():.4f} | eval acc: {acc:.4f} | mean lp: {mean_lp:+.3f} | '
              f'elapsed: {elapsed/60:.1f}m', flush=True)

print(f'[{args.label}] done in {(time.time()-t_start)/60:.2f}m')
