"""
Live status snapshot of a wandb run, bypassing buffered local log files.

Usage:
    python -m dev.wandb_status <url-or-path>
    python -m dev.wandb_status https://wandb.ai/matt-parrett/nanochat-sft/runs/slkjv10w
    python -m dev.wandb_status matt-parrett/nanochat-sft/slkjv10w

Prints state, runtime, last step, the val/bpb evaluation trajectory, and
the most recent train rows. Auto-detects which metric columns the run
logged (different nanochat scripts log different names — base_train.py
emits val/bpb + train/loss, chat_sft.py same, chat_rl.py uses different
keys), so this works across all three.
"""
import argparse
import re
import sys

import wandb


def parse_run_path(arg: str) -> str:
    """Accept either an entity/project/run_id triple or a full wandb URL."""
    arg = arg.strip().rstrip("/")
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", arg)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    if arg.count("/") == 2:
        return arg
    sys.exit(f"unrecognized run reference: {arg}\n"
             f"expected entity/project/run_id or a wandb.ai URL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="wandb URL or entity/project/run_id")
    ap.add_argument("-n", "--tail", type=int, default=5, help="number of recent train rows to show")
    args = ap.parse_args()

    run = wandb.Api().run(parse_run_path(args.run))
    summary = dict(run.summary)
    print(f"name:    {run.name}")
    print(f"state:   {run.state}")
    print(f"runtime: {summary.get('_runtime', '?'):.0f}s")
    print(f"step:    {summary.get('step', '?')} / "
          f"{run.config.get('num_iterations', '?')}")
    print()

    df = run.history(samples=10000)
    if df.empty:
        print("(no history yet)")
        return

    # val column varies — base_train logs val/bpb; some flows log val_loss/etc.
    val_cols = [c for c in df.columns if c.startswith("val/") or c.startswith("val_")]
    for vc in val_cols:
        sub = df[["step", vc]].dropna()
        if sub.empty:
            continue
        print(f"--- {vc} trajectory ---")
        print(sub.to_string(index=False))
        print()

    # train loss tail
    train_cols = [c for c in ("train/loss", "train/lrm", "train/dt", "train/tok_per_sec")
                  if c in df.columns]
    if train_cols:
        cols = ["step", *train_cols]
        sub = df[cols].dropna(subset=train_cols, how="all").tail(args.tail)
        print(f"--- last {len(sub)} train rows ---")
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
