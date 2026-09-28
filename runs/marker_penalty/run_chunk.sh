#!/usr/bin/env bash
# Marker-penalty experiment (arXiv 2606.00206) on Bonsai-4B, one chunk at a time.
#
#   runs/marker_penalty/run_chunk.sh 3          # chunk 3 at every lambda
#   runs/marker_penalty/run_chunk.sh 3 0.5      # chunk 3 at one lambda
#
# Safe to interrupt: each lambda appends to its own dump and --resume skips
# problems already there, so rerunning a chunk picks up where it stopped.
# Sampling follows the paper (T 0.6, top-p 0.95). The draw is seeded per
# problem, so each lambda sees the same seed for the same problem.
set -euo pipefail
cd "$(dirname "$0")/../.."

chunk=${1:?usage: run_chunk.sh <chunk 1-6> [lambda ...]}
shift
lambdas=("$@")
[ ${#lambdas[@]} -eq 0 ] && lambdas=(0 0.5 1.0)

dir=runs/marker_penalty
for lam in "${lambdas[@]}"; do
  echo "=== chunk $chunk, lambda $lam ===" | tee -a "$dir/lam_$lam.log"
  PYTHONUNBUFFERED=1 uv run --extra cpu --extra mlx python -m scripts.chat_eval_mlx \
    -m 4b -a "GSM8K|HumanEval" --only-ids "$dir/chunk$chunk.json" \
    --max-new-tokens 2048 --no-system-prompt --lenient-extract \
    -t 0.6 --top-p 0.95 --seed 1000 --marker-penalty "$lam" \
    --dump-jsonl "$dir/lam_$lam.jsonl" --resume >> "$dir/lam_$lam.log" 2>&1
done
echo "chunk $chunk done"
