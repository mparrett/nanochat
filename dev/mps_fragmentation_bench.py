"""
Synthetic benchmark for the MPS-allocator-fragmentation hypothesis.

Loads a model and runs many forward passes at variable batch+seq shapes
(matching the ChatCORE eval pattern), logging mps/{allocated,driver,cache}_gb
per iteration. If `mps/cache_gb` monotonically climbs without bound, the
fragmentation theory is supported. If it stays flat or decays, something
else explains the ChatCORE OOM.

Variations to run (separately) to A/B the theory:
    --empty-cache-every -1      # never call empty_cache (default; baseline)
    --empty-cache-every 10      # every 10 forwards
    --empty-cache-every 1       # every forward (max defragmentation)
    --t-mode fixed              # constant T (control: no shape variation)
    --t-mode variable           # vary T per call (mimics ChatCORE; default)

Usage:
    python -m dev.mps_fragmentation_bench --model-tag d6_stage2 --steps 500
    python -m dev.mps_fragmentation_bench --model-tag d6_stage2 --steps 500 --empty-cache-every 10

Output: CSV stream to stdout with one row per forward, plus a final summary.
Stop on OOM (the test we're trying to reproduce).
"""
import argparse
import csv
import sys
import time

import numpy as np
import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, autodetect_device_type, mps_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-tag', type=str, required=True, help='base checkpoint to load (e.g., d6_stage2)')
    ap.add_argument('--steps', type=int, default=500, help='max forward passes to run')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--t-mode', choices=['fixed', 'variable'], default='variable',
                    help='fixed = constant T per batch; variable = sample T per batch (mimics ChatCORE)')
    ap.add_argument('--t-fixed', type=int, default=512, help='T value when --t-mode=fixed')
    ap.add_argument('--t-min', type=int, default=40, help='min T when --t-mode=variable')
    ap.add_argument('--t-max', type=int, default=600, help='max T when --t-mode=variable')
    ap.add_argument('--empty-cache-every', type=int, default=-1,
                    help='call torch.mps.empty_cache every N forwards (-1 = never)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device_type = autodetect_device_type()
    if device_type != 'mps':
        print(f'WARN: device_type={device_type}, this benchmark targets MPS', file=sys.stderr)

    _, _, _, _, device = compute_init(device_type)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f'Loading {args.model_tag}...', file=sys.stderr)
    model, tokenizer, meta = load_model('base', device, phase='eval', model_tag=args.model_tag)
    model.eval()
    vocab_size = meta['model_config']['vocab_size']
    print(f'Loaded; n_embd={meta["model_config"]["n_embd"]}, n_layer={meta["model_config"]["n_layer"]}', file=sys.stderr)

    writer = csv.writer(sys.stdout)
    writer.writerow(['step', 'T', 'allocated_gb', 'driver_gb', 'cache_gb', 'wall_ms', 'empty_cache_called'])

    last_metrics = None
    oom_step = None
    t_start = time.time()

    try:
        for step in range(args.steps):
            if args.t_mode == 'fixed':
                T = args.t_fixed
            else:
                T = int(rng.integers(args.t_min, args.t_max + 1))

            inputs = torch.randint(0, vocab_size, (args.batch_size, T), dtype=torch.int32, device=device)

            iter_start = time.time()
            with torch.no_grad():
                _ = model(inputs)
            if device_type == 'mps':
                torch.mps.synchronize()  # force completion before measuring
            wall_ms = (time.time() - iter_start) * 1000

            empty_cache_called = 0
            if args.empty_cache_every > 0 and (step + 1) % args.empty_cache_every == 0:
                if device_type == 'mps':
                    torch.mps.empty_cache()
                empty_cache_called = 1

            m = mps_metrics()
            last_metrics = m
            writer.writerow([
                step,
                T,
                f'{m.get("mps/allocated_gb", 0):.4f}',
                f'{m.get("mps/driver_gb", 0):.4f}',
                f'{m.get("mps/cache_gb", 0):.4f}',
                f'{wall_ms:.1f}',
                empty_cache_called,
            ])
            sys.stdout.flush()

    except RuntimeError as e:
        oom_step = step
        print(f'\n# OOM (or other RuntimeError) at step {step}: {e}', file=sys.stderr)

    elapsed = time.time() - t_start
    print(f'\n# === summary ===', file=sys.stderr)
    print(f'# wall: {elapsed:.1f}s ({step + 1} forwards, {(step+1)/elapsed:.1f}/s)', file=sys.stderr)
    if last_metrics:
        print(f'# final allocator: allocated={last_metrics.get("mps/allocated_gb", 0):.3f} GB '
              f'driver={last_metrics.get("mps/driver_gb", 0):.3f} GB '
              f'cache={last_metrics.get("mps/cache_gb", 0):.3f} GB '
              f'(rec_max={last_metrics.get("mps/recommended_max_gb", 0):.1f} GB)', file=sys.stderr)
    if oom_step is not None:
        print(f'# OOM at step {oom_step}', file=sys.stderr)
    else:
        print(f'# completed {args.steps} forwards without OOM', file=sys.stderr)


if __name__ == '__main__':
    main()
