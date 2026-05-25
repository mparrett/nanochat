"""
Persona-retention eval on Qwen2.5-0.5B-Instruct (optionally with an NTK-Mirror
controller attached). Mirrors dev/eval_persona_retention.py's scoring shape so
results compare directly to the L1 LoRA v2 reference (19/30 all_three).

For each row in eval.jsonl:
  - Prompt was rendered by bench/adapt_persona_to_qwen.py with
    add_generation_prompt=True (ends at `<|im_start|>assistant\\n`).
  - Greedy-decode up to max_tokens or until `<|im_end|>`.
  - Substring (case-insensitive) check for persona name/role/location.

If --controller is set, the NTK-Mirror SignedLogMask controller is attached
during generation. Same generation params either way.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def greedy_generate(
    model, tok, prompt: str, *, max_new_tokens: int, device, eos_token_id: int
) -> str:
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = enc["input_ids"]
    out = model.generate(
        input_ids=input_ids,
        attention_mask=enc.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        use_cache=True,
    )
    new_tokens = out[0, input_ids.shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def score_response(response: str, persona: dict) -> dict:
    r = response.lower()
    name_ok = persona["name"].lower() in r
    role_ok = persona["role"].lower() in r
    location_ok = persona["location"].lower() in r
    return {
        "name": name_ok,
        "role": role_ok,
        "location": location_ok,
        "all_three": name_ok and role_ok and location_ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument(
        "--eval",
        default=str(Path.home() / "projects-new/3p/ntkmirror/runs/persona_qwen/eval.jsonl"),
    )
    ap.add_argument("--controller", default=None, help="optional NTK-Mirror controller .pt")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--arm-name", default=None, help="label for this run in output")
    ap.add_argument("--out", default=None, help="optional JSON path for results")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval) if l.strip()]
    print(f"loaded {len(rows)} eval rows from {args.eval}", flush=True)

    print(f"loading {args.model} on {args.device}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model = model.to(args.device)
    model.eval()

    arm_name = args.arm_name or ("ntk" if args.controller else "base")

    tuner = None
    if args.controller:
        from ntkmirror.controller import ForwardFineTuner
        print(f"loading controller {args.controller}", flush=True)
        tuner = ForwardFineTuner(model, tok, gates=1)  # gates re-set by load
        tuner.load(args.controller)
        tuner.controller.attach()
        print(f"  attached: n_gates={tuner.controller.raw.numel()}", flush=True)

    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    per_prompt = []
    agg = {"name": 0, "role": 0, "location": 0, "all_three": 0}
    t0 = time.perf_counter()
    try:
        for i, row in enumerate(rows):
            persona = row["_persona"]
            response = greedy_generate(
                model, tok, row["prompt"],
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                eos_token_id=im_end,
            )
            scores = score_response(response, persona)
            for k, v in scores.items():
                if v:
                    agg[k] += 1
            flags = "".join("✓" if scores[k] else "·" for k in ("name", "role", "location"))
            snippet = response[:100].replace("\n", " ")
            print(
                f"  [{i:2d}] {flags} {persona['name']:>16} / "
                f"{persona['role'][:20]:>20} / {persona['location'][:18]:>18} | {snippet!r}",
                flush=True,
            )
            per_prompt.append({"i": i, "persona": persona, "response": response, "scores": scores})
    finally:
        if tuner is not None:
            tuner.controller.remove()

    dur = time.perf_counter() - t0
    n = len(rows)
    print()
    print(f"=== Arm: {arm_name} (n={n}, {dur:.1f}s wall, {dur/n:.2f}s/row) ===")
    print(f"  name_recall    : {agg['name']}/{n}  ({100*agg['name']/n:.1f}%)")
    print(f"  role_recall    : {agg['role']}/{n}  ({100*agg['role']/n:.1f}%)")
    print(f"  location_recall: {agg['location']}/{n}  ({100*agg['location']/n:.1f}%)")
    print(f"  all_three      : {agg['all_three']}/{n}  ({100*agg['all_three']/n:.1f}%)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "arm": arm_name,
            "model": args.model,
            "controller": args.controller,
            "n": n,
            "agg": agg,
            "wall_seconds": dur,
            "per_prompt": per_prompt,
        }, indent=2, ensure_ascii=False))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
