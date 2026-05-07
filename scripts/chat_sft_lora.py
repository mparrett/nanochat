"""
LoRA fine-tuning entry point.

Loads a base SFT (or RL) checkpoint, freezes every base parameter, applies
LoRA adapters to the requested attention projections, and trains *only*
the adapter on a single CustomJSON conversation dataset. Saves a tiny
adapter-only checkpoint (state_dict of LoRA tensors + meta json).

Run (single-rank, on M2):

    uv run python -m scripts.chat_sft_lora \\
        --base-tag d6_baseline_modern_sft \\
        --data-path /path/to/persona.jsonl \\
        --lora-tag d6_l1_persona_lora \\
        --num-iterations 200 \\
        --lr 1e-4

This script is deliberately leaner than scripts/chat_sft.py: single
CustomJSON dataset, plain AdamW (no Muon — rank-r is too small to benefit
from orthogonalisation), no torch.compile, no ChatCORE eval. The eval
loop is the same evaluate_bpb path used by chat_sft so val numbers are
directly comparable.
"""

import argparse
import json
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import time
from dataclasses import asdict

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    DummyWandb,
    get_base_dir,
    print0,
)
from nanochat.lora import apply_lora, lora_parameters, lora_state_dict
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON

# ---------------------------------------------------------------------------
# CLI
parser = argparse.ArgumentParser(description="LoRA fine-tuning on a CustomJSON dataset")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--seed", type=int, default=42)

# Base model
parser.add_argument("--base-source", type=str, default="sft", choices=["base", "sft", "rl"],
                    help="checkpoint family to load the base model from")
parser.add_argument("--base-tag", type=str, default="d6_baseline_modern_sft",
                    help="model_tag of the base checkpoint")
parser.add_argument("--base-step", type=int, default=None, help="step of the base checkpoint (default: latest)")

# LoRA config
parser.add_argument("--lora-tag", type=str, required=True,
                    help="output tag — adapter is saved to lora_checkpoints/<lora_tag>/")
parser.add_argument("--rank", type=int, default=8)
parser.add_argument("--alpha", type=float, default=16.0)
parser.add_argument("--target", type=str, default="c_q,c_v",
                    help="comma-separated attention projection names to wrap")
parser.add_argument("--force-overwrite", action="store_true",
                    help="permit writing into a non-empty lora_checkpoints/<lora_tag>/")

# Dataset
parser.add_argument("--data-path", type=str, required=True,
                    help="absolute path to a CustomJSON-shaped JSONL of conversations")
parser.add_argument("--val-fraction", type=float, default=0.05,
                    help="fraction of conversations held out for val_bpb (default 5%)")

# Training horizon / shape
parser.add_argument("--num-iterations", type=int, default=200, help="optimizer steps")
parser.add_argument("--max-seq-len", type=int, default=None,
                    help="max context length (default: inherit from base meta)")
parser.add_argument("--device-batch-size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4, help="AdamW LR for LoRA params")
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--warmup-ratio", type=float, default=0.05)
parser.add_argument("--warmdown-ratio", type=float, default=0.5)
parser.add_argument("--final-lr-frac", type=float, default=0.0)

# Eval / logging
parser.add_argument("--eval-every", type=int, default=-1, help="run val_bpb every N steps (-1 = end only)")
parser.add_argument("--eval-tokens", type=int, default=8 * 524288, help="number of tokens for val_bpb eval")
parser.add_argument("--log-every", type=int, default=10)

args = parser.parse_args()

user_config = vars(args).copy()

# ---------------------------------------------------------------------------
# Compute / device
device_type = args.device_type or autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
torch.manual_seed(args.seed)

# ---------------------------------------------------------------------------
# Wandb (optional, dummy by default)
if ddp_rank == 0 and args.run != "dummy":
    import wandb
    wandb_run = wandb.init(project="nanochat-lora", name=args.run, config=user_config)
else:
    wandb_run = DummyWandb()

# ---------------------------------------------------------------------------
# Load base model + tokenizer
print0(f"Loading base model: source={args.base_source} tag={args.base_tag}")
model, tokenizer, meta = load_model(
    args.base_source, device, phase="train",
    model_tag=args.base_tag, step=args.base_step,
)

# Inherit max_seq_len from base meta unless overridden
if args.max_seq_len is None:
    args.max_seq_len = meta.get("max_seq_len", 512)
    print0(f"Inherited max_seq_len={args.max_seq_len} from base meta")

# ---------------------------------------------------------------------------
# Apply LoRA
target_tuple = tuple(t.strip() for t in args.target.split(",") if t.strip())
info = apply_lora(model, target=target_tuple, rank=args.rank, alpha=args.alpha)
print0(f"LoRA applied: target={info['targets']} rank={info['rank']} alpha={info['alpha']}")
print0(f"  injected_count={info['injected_count']}")
print0(f"  trainable_params={info['trainable_params']:,}  (of base {info['base_params']:,}, "
       f"~{100 * info['trainable_params'] / info['base_params']:.3f}%)")

# Sanity: every trainable param is a LoRA tensor
for name, p in model.named_parameters():
    if p.requires_grad:
        assert name.endswith(".lora_A") or name.endswith(".lora_B"), (
            f"non-LoRA param trainable after apply_lora: {name}"
        )

# ---------------------------------------------------------------------------
# Output checkpoint dir + safety
base_dir = get_base_dir()
lora_out_dir = os.path.join(base_dir, "lora_checkpoints", args.lora_tag)
if ddp_rank == 0:
    os.makedirs(lora_out_dir, exist_ok=True)
    existing = [f for f in os.listdir(lora_out_dir) if f.startswith("lora_") and f.endswith(".pt")]
    if existing and not args.force_overwrite:
        raise SystemExit(
            f"\nABORT: {lora_out_dir} already contains LoRA checkpoint(s): {existing}\n"
            f"Pass --force-overwrite to replace, or pick a fresh --lora-tag.\n"
        )

# ---------------------------------------------------------------------------
# Dataset: a single CustomJSON, split train/val by index
print0(f"Loading dataset from {args.data_path}")
full = CustomJSON(filepath=args.data_path)
n_total = len(full)
if n_total == 0:
    raise SystemExit(f"Dataset {args.data_path} is empty.")
n_val = max(1, int(round(n_total * args.val_fraction)))
n_train = n_total - n_val
print0(f"Dataset rows: total={n_total} train={n_train} val={n_val}")


class _IndexSlice:
    """Light wrapper that exposes a sub-range of a parent dataset."""
    def __init__(self, parent, start, stop):
        self.parent = parent
        self.start = start
        self.stop = stop

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, i):
        return self.parent[self.start + (i % len(self))]


train_dataset = TaskMixture([_IndexSlice(full, 0, n_train)])
val_dataset = TaskMixture([_IndexSlice(full, n_train, n_total)])

# ---------------------------------------------------------------------------
# DataLoader: BOS-aligned bestfit-pad, copy of chat_sft's generator pared down
# to a single dataset (no progress / num_iterations coupling).
last_step = False
microbatch_yields = 0


def sft_data_generator_bos_bestfit(split, buffer_size=64):
    global last_step, microbatch_yields
    assert split in {"train", "val"}
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1
    bos_token = tokenizer.get_bos_token_id()

    conv_buffer = []
    cursor = ddp_rank
    epoch = 1
    it = 0

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1

    while True:
        rows, mask_rows, row_lengths = [], [], []
        for _ in range(args.device_batch_size):
            row, mask_row = [], []
            padded = False
            content_len = row_capacity
            while len(row) < row_capacity:
                if len(conv_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - len(row)
                best_idx, best_len = -1, 0
                for i, (conv, _) in enumerate(conv_buffer):
                    cl = len(conv)
                    if cl <= remaining and cl > best_len:
                        best_idx, best_len = i, cl
                if best_idx >= 0:
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                elif len(row) == 0:
                    conv_buffer.clear()
                    continue
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    mask_row.extend([0] * remaining)
                    padded = True
                    break
            row_lengths.append(content_len if padded else row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        it += 1
        if split == "train":
            microbatch_yields = it

        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()

        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        mask_targets = mask_tensor[:, 1:].to(device=device)
        targets[mask_targets == 0] = -1
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1:] = -1

        yield inputs, targets


train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")

# ---------------------------------------------------------------------------
# Optimizer: plain AdamW over LoRA params only
trainable = lora_parameters(model)
print0(f"Optimizer: AdamW over {len(trainable)} LoRA tensors")
optimizer = torch.optim.AdamW(
    trainable, lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    weight_decay=args.weight_decay,
)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]


def get_lr_multiplier(step, total):
    progress = (step + 1) / max(total, 1)
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    if progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
    return (1 - decay) * 1.0 + decay * args.final_lr_frac


# ---------------------------------------------------------------------------
# Training loop
token_bytes = get_token_bytes(device=device)
print0(f"Training {args.num_iterations} iterations on {device_type} (lr={args.lr})")

x, y = next(train_loader)
smooth_loss = 0.0
ema_beta = 0.9
total_time = 0.0
val_bpb = None
min_val_bpb = float("inf")

PAUSE_FILE = os.environ.get("NANOCHAT_PAUSE_FILE", "/tmp/pause-nanochat")

for step in range(args.num_iterations):
    if os.path.exists(PAUSE_FILE):
        print0(f"⏸  paused at step {step}/{args.num_iterations}; rm {PAUSE_FILE} to resume")
        while os.path.exists(PAUSE_FILE):
            time.sleep(1)
        print0(f"▶  resumed at step {step}")

    is_last = (step == args.num_iterations - 1)

    if (args.eval_every > 0 and step > 0 and step % args.eval_every == 0) or is_last:
        model.eval()
        with torch.no_grad():
            val_loader = build_val_loader()
            eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size))
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"step {step:05d} | val_bpb={val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({"step": step, "val/bpb": val_bpb})
        model.train()

    t0 = time.time()
    loss = model(x, y)
    train_loss = loss.detach()
    loss.backward()
    x, y = next(train_loader)

    lrm = get_lr_multiplier(step, args.num_iterations)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    dt = time.time() - t0

    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss.item()
    debiased = smooth_loss / (1 - ema_beta ** (step + 1))
    if step > 5:
        total_time += dt

    if step % args.log_every == 0 or is_last:
        print0(
            f"step {step:05d}/{args.num_iterations} | loss={debiased:.4f} | "
            f"lrm={lrm:.2f} | dt={dt*1000:.1f}ms"
        )
        wandb_run.log({"step": step, "train/loss": debiased, "train/lrm": lrm, "train/dt": dt})

# ---------------------------------------------------------------------------
# Save LoRA-only checkpoint
if ddp_rank == 0:
    final_step = args.num_iterations
    lora_state = lora_state_dict(model)
    save_path = os.path.join(lora_out_dir, f"lora_{final_step:06d}.pt")
    torch.save(lora_state, save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    meta_out = {
        "step": final_step,
        "user_config": user_config,
        "lora_info": {
            "rank": info["rank"],
            "alpha": info["alpha"],
            "targets": list(info["targets"]),
            "injected_count": info["injected_count"],
            "trainable_params": info["trainable_params"],
            "base_params": info["base_params"],
        },
        "base": {
            "source": args.base_source,
            "tag": args.base_tag,
            "step": args.base_step,
            "model_config": meta.get("model_config"),
        },
        "min_val_bpb": min_val_bpb if min_val_bpb != float("inf") else None,
        "final_train_loss": smooth_loss / (1 - ema_beta ** args.num_iterations),
        "total_training_time_min": total_time / 60.0,
    }
    with open(os.path.join(lora_out_dir, f"meta_{final_step:06d}.json"), "w") as f:
        json.dump(meta_out, f, indent=2, default=str)
    print0(f"Saved LoRA: {save_path} ({size_mb:.2f} MB)")
    print0(f"Saved meta: {os.path.join(lora_out_dir, f'meta_{final_step:06d}.json')}")

print0(f"Total training time: {total_time/60:.2f} min")
print0(f"Min val_bpb: {min_val_bpb:.4f}" if min_val_bpb != float("inf") else "No val eval ran")

wandb_run.finish()
compute_cleanup()
