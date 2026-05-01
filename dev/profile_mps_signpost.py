"""
GPU-side profiler for one steady-state pretrain iter on MPS.

Uses torch.mps.profiler.profile() which emits OS Signposts that the system
log captures. Prints two timestamps (before/after the profiled section) so
you can pull the trace with:

    log show --predicate 'subsystem == "com.apple.Metal"' \\
             --start "<start_ts>" --end "<end_ts>" --info

Usage:
    python -m dev.profile_mps_signpost > /tmp/mps_signpost.log
    # then run the suggested `log show` command from the output
"""
import os
import time
import datetime
import torch
import torch.mps.profiler as mps_profiler

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model

DEVICE_BATCH_SIZE = 32
SEQ_LEN = 512
WARMUP = 3

device_type = autodetect_device_type()
assert device_type == "mps", f"this profiler is MPS-only, got device_type={device_type}"
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
print(f"device_type={device_type}  device={device}", flush=True)

model, tokenizer, _ = load_model("base", device, phase="train")
vocab_size = tokenizer.get_vocab_size()
print(f"loaded d{model.config.n_layer}  vocab={vocab_size}", flush=True)

optimizer = model.setup_optimizer(
    unembedding_lr=0.008, embedding_lr=0.3, scalar_lr=0.04,
    matrix_lr=0.02, weight_decay=0.1,
)
model = torch.compile(model, dynamic=False)

torch.manual_seed(0)
x = torch.randint(0, vocab_size, (DEVICE_BATCH_SIZE, SEQ_LEN), device=device, dtype=torch.int32)
y = torch.randint(0, vocab_size, (DEVICE_BATCH_SIZE, SEQ_LEN), device=device, dtype=torch.int64)


def one_step():
    loss = model(x, y)
    loss.backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    torch.mps.synchronize()
    return loss


print(f"--- warmup {WARMUP} iters ---", flush=True)
for i in range(WARMUP):
    t0 = time.perf_counter()
    one_step()
    print(f"  iter {i}: {(time.perf_counter() - t0) * 1000:.0f}ms", flush=True)

print("--- one profiled iter (signposts emitted) ---", flush=True)
torch.mps.synchronize()  # make sure prior work is fully done
start_dt = datetime.datetime.now()
start_ts = start_dt.strftime("%Y-%m-%d %H:%M:%S")
print(f"START: {start_ts}", flush=True)

# 'interval,event' captures both interval-based phases AND discrete events.
# wait_until_completed=True forces sync inside the context so the trace is
# bounded to the profiled iter rather than spilling into prior async work.
with mps_profiler.profile(mode="interval,event", wait_until_completed=True):
    one_step()

end_dt = datetime.datetime.now()
end_ts = end_dt.strftime("%Y-%m-%d %H:%M:%S")
print(f"END:   {end_ts}", flush=True)
print(f"profiled wall: {(end_dt - start_dt).total_seconds() * 1000:.0f}ms", flush=True)

# Print the log show commands the user should run to inspect
print("\n--- to inspect the trace ---", flush=True)
print(f"log show --predicate 'subsystem == \"com.apple.Metal\"' \\")
print(f"         --start '{start_ts}' --end '{end_ts}' --info \\")
print(f"         > /tmp/mps_metal_signposts.txt")
print()
print(f"# also useful — broader MPS subsystem:")
print(f"log show --predicate 'subsystem CONTAINS \"MPS\" OR subsystem CONTAINS \"Metal\"' \\")
print(f"         --start '{start_ts}' --end '{end_ts}' --info \\")
print(f"         > /tmp/mps_all_signposts.txt")
