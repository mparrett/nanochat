"""
30-second sanity check: generate ~50 tokens from Qwen3-4B-Instruct-2507-8bit
with the δ-mem adapter attached. Coherent output = wiring + adapter work
end-to-end through both prompt forward AND KV-cache generation (seq_len=1 per
step, which exercises the scan at T=1).

Output is printed for visual inspection.
"""

from __future__ import annotations

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from nanochat.delta_mem_mlx import (
    DeltaMemConfig,
    attach_delta_mem,
    load_delta_mem_adapter,
    reset_delta_mem_states,
)


MODEL_REPO = "mlx-community/Qwen3-4B-Instruct-2507-8bit"
ADAPTER_PATH = "/Users/matt/.cache/nanochat/delta_mem_qwen3_4b_instruct/adapter.safetensors"


def main() -> None:
    print(f"Loading {MODEL_REPO} ...")
    model, tokenizer = load(MODEL_REPO)

    msgs = [{"role": "user", "content": "Briefly explain what a transformer is in one paragraph."}]
    prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    sampler = make_sampler(temp=0.0)  # greedy

    print("\n=== Base (no δ-mem) ===")
    base_out = generate(model, tokenizer, prompt=prompt, max_tokens=80, sampler=sampler, verbose=False)
    print(base_out)

    print("\nAttaching δ-mem adapter...")
    cfg = DeltaMemConfig()
    attach_delta_mem(model, cfg)
    n = load_delta_mem_adapter(model, ADAPTER_PATH)
    print(f"  loaded {n} tensors")
    reset_delta_mem_states(model)

    print("\n=== With δ-mem adapter ===")
    adapter_out = generate(model, tokenizer, prompt=prompt, max_tokens=80, sampler=sampler, verbose=False)
    print(adapter_out)

    print("\n=== Diff ===")
    print(f"  base len: {len(base_out)} chars,  adapter len: {len(adapter_out)} chars")
    print(f"  prefix shared: {sum(1 for a, b in zip(base_out, adapter_out) if a == b)} chars")


if __name__ == "__main__":
    main()
