"""
Standalone profiler for a single pretrain training step on MPS.

Loads the d6 base checkpoint, builds synthetic inputs, runs N warmup iters
(letting torch.compile settle) and then profiles M iters. Dumps the top
operations by self CPU time so we can see where the 2 s/iter actually goes.

Usage:
    python -m dev.profile_pretrain_iter
"""
import time
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model

DEVICE_BATCH_SIZE = 32
SEQ_LEN = 512
WARMUP = 3
PROFILED = 5
DEPTH_GUESS = 6

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
print(f"device_type={device_type}  device={device}")

# Load the existing pretrain checkpoint to reuse the trained model + optimizer
model, tokenizer, meta = load_model("base", device, phase="train")
vocab_size = tokenizer.get_vocab_size()
print(f"loaded d{model.config.n_layer}  vocab={vocab_size}")

# Set up the optimizer using the model's helper (lr values don't matter for profiling)
optimizer = model.setup_optimizer(
    unembedding_lr=0.008,
    embedding_lr=0.3,
    scalar_lr=0.04,
    matrix_lr=0.02,
    weight_decay=0.1,
)

# Compile model to match real training behavior (must come AFTER setup_optimizer
# since the param groups need to reference the original parameters)
model = torch.compile(model, dynamic=False)

# Synthetic inputs of the right shape
torch.manual_seed(0)
x = torch.randint(0, vocab_size, (DEVICE_BATCH_SIZE, SEQ_LEN), device=device, dtype=torch.int32)
y = torch.randint(0, vocab_size, (DEVICE_BATCH_SIZE, SEQ_LEN), device=device, dtype=torch.int64)

def synchronize():
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()

def one_step(label=""):
    t0 = time.perf_counter()
    loss = model(x, y)
    t1 = time.perf_counter()
    loss.backward()
    t2 = time.perf_counter()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t3 = time.perf_counter()
    return {
        "label": label,
        "fwd_ms": (t1 - t0) * 1000,
        "bwd_ms": (t2 - t1) * 1000,
        "opt_ms": (t3 - t2) * 1000,
        "total_ms": (t3 - t0) * 1000,
        "loss": loss.item(),
    }

# Warmup
print("--- warmup ---")
for i in range(WARMUP):
    r = one_step(f"warmup-{i}")
    print(f"  iter {i}: total={r['total_ms']:7.1f}ms  fwd={r['fwd_ms']:6.1f}  bwd={r['bwd_ms']:6.1f}  opt={r['opt_ms']:6.1f}  loss={r['loss']:.3f}")

# Profiled steps
print(f"--- profile {PROFILED} steady-state steps ---")
activities = [ProfilerActivity.CPU]
# MPS profiling is limited; CPU-side gives us the call breakdown
results = []
with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
    for i in range(PROFILED):
        r = one_step(f"prof-{i}")
        results.append(r)

# Per-iter timing summary
print("\n=== per-iter timing (ms) ===")
print(f"{'iter':<8}{'total':>10}{'fwd':>10}{'bwd':>10}{'opt':>10}")
for r in results:
    print(f"{r['label']:<8}{r['total_ms']:>10.1f}{r['fwd_ms']:>10.1f}{r['bwd_ms']:>10.1f}{r['opt_ms']:>10.1f}")
avg = lambda k: sum(r[k] for r in results) / len(results)
print(f"{'avg':<8}{avg('total_ms'):>10.1f}{avg('fwd_ms'):>10.1f}{avg('bwd_ms'):>10.1f}{avg('opt_ms'):>10.1f}")
total = avg("total_ms")
print(f"\nfwd / bwd / opt = {avg('fwd_ms')/total*100:.1f}% / {avg('bwd_ms')/total*100:.1f}% / {avg('opt_ms')/total*100:.1f}%")

# Top ops by CPU time
print("\n=== top 25 ops by CPU self time ===")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
