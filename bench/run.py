"""
bench/run.py — synthetic memory-mechanism probe runner.

Trains a small nanochat GPT from random init on a synthetic diagnostic
task, reporting per-eval-step accuracy on a held-out sample. Designed
for fast iteration (<5 min per architectural variant at d2/d4) on the
recall/state-tracking axes that distinguish memory mechanisms.

Architectural variants are selected via the same `--hope-*` flags the
Stage 1.5 / Stage 2 probe used; see `dev/probe_mqar.py` for the
canonical Hope/NL settings.

Usage:

  # baseline MQAR at d4 (~1-2 min M2)
  python -m bench.run --task mqar --depth 4 --label baseline

  # Stage 1 additive memory with the load-bearing W_o=1.0 init
  python -m bench.run --task mqar --depth 4 --label stage1_additive \\
      --hope-additive-memory-layer 1 --hope-memory-w-o-init-scale 1.0

  # SelectiveCopy: state-tracking probe, same architectural variants
  python -m bench.run --task selective-copy --depth 4 --label baseline

Each run writes a JSONL log to --log-dir/<label>.jsonl if --log-dir is
set, one record per eval step.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from nanochat.common import autodetect_device_type, compute_init
from nanochat.gpt import GPT, GPTConfig

from bench.tasks import make_task


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Task selection
    p.add_argument("--task", required=True, choices=["mqar", "selective-copy"])
    p.add_argument("--label", default="run", help="log prefix to identify the run")
    # Model shape
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=32768)
    # Hope/NL architectural variants (pass-through to GPTConfig)
    p.add_argument("--hope-memory-layer", type=int, default=-1,
                   help="layer index for LinearAttentionMemory SWAP (-1 = disabled)")
    p.add_argument("--hope-additive-memory-layer", type=int, default=-1,
                   help="layer index for ADDITIVE LinearAttentionMemory (-1 = disabled)")
    p.add_argument("--hope-memory-w-o-init-scale", type=float, default=0.0,
                   help="W_o init scale. 0.0 = zeros (the 1.5b bug); 1.0 = uniform like K/V/Q (load-bearing)")
    p.add_argument("--hope-memory-kind", default="linear",
                   choices=["linear", "learned_gate"],
                   help="linear = Stage 1; learned_gate = Stage 2")
    p.add_argument("--hope-memory-alpha-max", type=float, default=0.999)
    p.add_argument("--hope-memory-eta-max", type=float, default=1.0)
    p.add_argument("--hope-memory-alpha-init-bias", type=float, default=4.595)
    p.add_argument("--hope-memory-eta-init-bias", type=float, default=-2.197)
    # Training
    p.add_argument("--device-batch-size", type=int, default=64)
    p.add_argument("--num-iterations", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--n-eval-seqs", type=int, default=256)
    p.add_argument("--embedding-lr", type=float, default=0.1)
    p.add_argument("--unembedding-lr", type=float, default=0.003)
    p.add_argument("--matrix-lr", type=float, default=0.01)
    p.add_argument("--scalar-lr", type=float, default=0.04)
    # Task shape (MQAR-specific; selective-copy uses task defaults)
    p.add_argument("--K", type=int, default=16, help="MQAR: number of (key,value) pairs")
    p.add_argument("--M", type=int, default=16, help="MQAR: number of queries")
    p.add_argument("--T", type=int, default=128, help="MQAR: sequence length")
    p.add_argument("--n-keys", type=int, default=32, help="MQAR: key vocab size (>= K)")
    p.add_argument("--n-values", type=int, default=32, help="MQAR: value vocab size (>= K)")
    # Reproducibility & I/O
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default=None, help="write JSONL log to <log-dir>/<label>.jsonl")
    return p.parse_args()


def build_task(args):
    if args.task == "mqar":
        return make_task("mqar", K=args.K, M=args.M, T=args.T,
                         n_keys=args.n_keys, n_values=args.n_values,
                         vocab_size=args.vocab_size)
    if args.task == "selective-copy":
        return make_task("selective-copy", vocab_size=args.vocab_size)
    raise ValueError(args.task)


def build_model(args, task, device):
    base_dim = args.depth * args.head_dim
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    n_heads = model_dim // args.head_dim

    hope_layer = None if args.hope_memory_layer < 0 else args.hope_memory_layer
    hope_add = None if args.hope_additive_memory_layer < 0 else args.hope_additive_memory_layer

    config = GPTConfig(
        sequence_len=task.T,
        vocab_size=task.vocab_size,
        n_layer=args.depth,
        n_head=n_heads,
        n_kv_head=n_heads,
        n_embd=model_dim,
        window_pattern="L",
        hope_memory_layer=hope_layer,
        hope_additive_memory_layer=hope_add,
        hope_memory_w_o_init_scale=args.hope_memory_w_o_init_scale,
        hope_memory_kind=args.hope_memory_kind,
        hope_memory_alpha_max=args.hope_memory_alpha_max,
        hope_memory_eta_max=args.hope_memory_eta_max,
        hope_memory_alpha_init_bias=args.hope_memory_alpha_init_bias,
        hope_memory_eta_init_bias=args.hope_memory_eta_init_bias,
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.train()
    return model, config


@torch.no_grad()
def evaluate(model, task, n_seqs, batch_size, device, rng):
    model.eval()
    correct = 0
    total = 0
    for start in range(0, n_seqs, batch_size):
        B = min(batch_size, n_seqs - start)
        inputs, targets = task.generate_batch(B, rng)
        inputs = inputs.to(device, dtype=torch.int32)
        targets = targets.to(device)
        logits = model(inputs)
        mask = targets != -1
        preds = logits.argmax(dim=-1)
        correct += ((preds == targets) & mask).sum().item()
        total += mask.sum().item()
    model.train()
    return correct / total if total else 0.0


def main():
    args = parse_args()

    device_type = autodetect_device_type()
    _ddp, _rank, _local_rank, _ws, device = compute_init(device_type)
    torch.manual_seed(args.seed)

    task = build_task(args)
    model, config = build_model(args, task, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.label}] task={args.task} T={task.T} vocab={task.vocab_size}")
    print(f"[{args.label}] params: {n_params:,} ({n_params/1e6:.2f}M)  depth={args.depth}  n_embd={config.n_embd}")
    print(f"[{args.label}] hope: swap={args.hope_memory_layer} add={args.hope_additive_memory_layer} "
          f"w_o={args.hope_memory_w_o_init_scale} kind={args.hope_memory_kind}")

    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        scalar_lr=args.scalar_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=0.0,
    )

    train_rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1_000_000)

    log_records = []
    log_path = None
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{args.label}.jsonl"
        if log_path.exists():
            log_path.unlink()

    t_start = time.time()
    sat_step = None  # first step where eval acc >= 0.95
    for step in range(args.num_iterations):
        inputs, targets = task.generate_batch(args.device_batch_size, train_rng)
        inputs = inputs.to(device, dtype=torch.int32)
        targets = targets.to(device)
        loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)

        do_eval = ((step + 1) % args.eval_every == 0) or (step == args.num_iterations - 1)
        if do_eval:
            acc = evaluate(
                model, task, args.n_eval_seqs, args.device_batch_size, device, eval_rng
            )
            elapsed = time.time() - t_start
            if sat_step is None and acc >= 0.95:
                sat_step = step + 1
            sat_str = f"sat@{sat_step}" if sat_step is not None else "—"
            print(f"[{args.label}] step {step+1:04d}/{args.num_iterations} | "
                  f"loss {loss.item():.4f} | acc {acc:.4f} | {sat_str} | "
                  f"elapsed {elapsed/60:.1f}m", flush=True)
            rec = {
                "step": step + 1,
                "loss": float(loss.item()),
                "acc": float(acc),
                "elapsed_s": elapsed,
                "sat_step": sat_step,
            }
            log_records.append(rec)
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

    final = log_records[-1] if log_records else {}
    print(f"[{args.label}] done in {(time.time()-t_start)/60:.2f}m  "
          f"final_acc={final.get('acc', 0):.4f}  saturation_step={sat_step}")


if __name__ == "__main__":
    main()
