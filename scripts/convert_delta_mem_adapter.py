"""
convert_delta_mem_adapter.py — Convert the upstream δ-mem PyTorch adapter
into MLX-native safetensors for nanochat's MLX harness.

Input:  delta_mem_adapter.pt (torch.save dict, bf16, 324 entries from
        declare-lab/delta-mem_qwen3_4b-instruct).

Output: <out_dir>/adapter.safetensors + delta_mem_config.json passthrough.

Skipped tensors: delta_k_proj and delta_v_proj — verified all-zero across
all 36 layers in the published adapter (trained with delta_heads=["q","o"]
only). Omitting them keeps the active param set explicit and saves a few MB.
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch


def convert(adapter_pt: Path, config_json: Path, out_dir: Path) -> None:
    sd = torch.load(adapter_pt, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise ValueError(f"Expected dict, got {type(sd).__name__}")

    out_dir.mkdir(parents=True, exist_ok=True)
    mlx_weights: dict[str, mx.array] = {}
    skipped_zero = 0
    skipped_nonzero = 0
    kept = 0
    for key, tensor in sd.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"Non-tensor entry: {key} -> {type(tensor).__name__}")
        if "delta_k_proj" in key or "delta_v_proj" in key:
            if tensor.abs().sum().item() == 0:
                skipped_zero += 1
            else:
                skipped_nonzero += 1
                print(f"WARN: {key} has non-zero values ({tensor.abs().max().item():.4g}); skipping anyway")
            continue
        arr = tensor.to(torch.float32).numpy()
        mlx_weights[key] = mx.array(arr, dtype=mx.bfloat16)
        kept += 1

    out_path = out_dir / "adapter.safetensors"
    mx.save_safetensors(str(out_path), mlx_weights)
    shutil.copy2(config_json, out_dir / "delta_mem_config.json")

    print(f"kept:    {kept} tensors")
    print(f"skipped: {skipped_zero} zero δ_k/δ_v tensors (expected)")
    if skipped_nonzero:
        print(f"WARNING: {skipped_nonzero} δ_k/δ_v had non-zero values")
    print(f"wrote:   {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"wrote:   {out_dir / 'delta_mem_config.json'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-pt", type=Path,
                   default=Path("/tmp/delta-mem_qwen3_4b-instruct/delta_mem_adapter.pt"))
    p.add_argument("--config-json", type=Path,
                   default=Path("/tmp/delta-mem_qwen3_4b-instruct/delta_mem_config.json"))
    p.add_argument("--out-dir", type=Path,
                   default=Path.home() / ".cache" / "nanochat" / "delta_mem_qwen3_4b_instruct")
    args = p.parse_args()
    convert(args.adapter_pt, args.config_json, args.out_dir)


if __name__ == "__main__":
    main()
