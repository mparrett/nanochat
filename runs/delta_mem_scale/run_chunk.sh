#!/usr/bin/env bash
# δ-mem on Bonsai-4B with its attention-output correction scaled up, one chunk
# at a time. Chunks are the marker-penalty problem set (46 looping + 30 control).
#
#   runs/delta_mem_scale/run_chunk.sh 1          # chunk 1 at x10 and x20
#   runs/delta_mem_scale/run_chunk.sh 1 20       # chunk 1 at one scale
#
# Greedy, so the no-adapter baseline is the audit's greedy 2048 run (x1 output
# is byte-identical to it). Safe to interrupt; rerunning resumes from the dump.
set -euo pipefail
cd "$(dirname "$0")/../.."

chunk=${1:?usage: run_chunk.sh <chunk 1-6> [scale ...]}
shift
scales=("$@")
[ ${#scales[@]} -eq 0 ] && scales=(10 20)

dir=runs/delta_mem_scale
for s in "${scales[@]}"; do
  out="$dir/o_scale_$s"
  echo "=== chunk $chunk, o_scale $s ===" | tee -a "$out.log"
  PYTHONUNBUFFERED=1 uv run --extra cpu --extra mlx python -m scripts.chat_eval_mlx \
    -m 4b -a "GSM8K|HumanEval" --only-ids "runs/marker_penalty/chunk$chunk.json" \
    --max-new-tokens 2048 --no-system-prompt --lenient-extract \
    --delta-mem "$HOME/.cache/nanochat/delta_mem_qwen3_4b_instruct" --delta-mem-o-scale "$s" \
    --dump-jsonl "$out.jsonl" --resume >> "$out.log" 2>&1
done
echo "chunk $chunk done"
