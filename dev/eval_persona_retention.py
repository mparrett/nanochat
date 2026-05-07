"""
Persona-retention eval: score whether the model recalls T1's persona at T6.

For each held-out conversation in persona_retention_v1_eval.jsonl:

  - Take the first 5 messages (user/assistant/user/assistant/user — ends with
    T5's recall request).
  - Render via tokenizer.render_conversation on those 5 messages (which
    appends assistant_end to each completed assistant turn but leaves the
    final user turn closed). Then append <|assistant_start|> manually so
    generation begins T6.
  - Generate up to max_tokens or <|assistant_end|>.
  - Score: case-insensitive substring of name / role / location in the
    generated response. (Interest is optional; not scored — too noisy on
    short responses.)

Compare two arms:
  arm_base  : d6_baseline_modern_sft only
  arm_lora  : d6_baseline_modern_sft + LoRA from --lora-tag

Reports per-prompt pass/fail and aggregate name/role/location/all_three rates.

Run:
  uv run python -m dev.eval_persona_retention --lora-tag d6_l1_persona_lora
"""

import argparse
import json
import os
import sys
from typing import Optional

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init, get_base_dir
from nanochat.engine import Engine
from nanochat.lora import apply_lora_from_tag


def render_eval_prompt(tokenizer, messages_5: list[dict]) -> list[int]:
    """Render T1-T5 plus a fresh <|assistant_start|> so the model continues
    with T6. messages_5 must be 5 messages alternating u/a/u/a/u."""
    assert len(messages_5) == 5
    assert [m["role"] for m in messages_5] == ["user", "assistant", "user", "assistant", "user"]
    ids, _ = tokenizer.render_conversation({"messages": messages_5})
    # render_conversation closes the final user turn with <|user_end|>; we
    # need to start the assistant's turn ourselves.
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    ids.append(assistant_start)
    return ids


def generate_response(
    engine: Engine,
    tokenizer,
    prompt_tokens: list[int],
    max_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> str:
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    response_tokens: list[int] = []
    for token_column, _ in engine.generate(
        prompt_tokens,
        num_samples=1,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    ):
        token = token_column[0]
        if token == assistant_end:
            break
        response_tokens.append(token)
    if not response_tokens:
        return ""
    return tokenizer.decode(response_tokens)


def score_response(response: str, persona: dict) -> dict:
    """Substring (case-insensitive) check for name / role / location."""
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


def run_arm(
    arm_name: str,
    base_tag: str,
    lora_tag: Optional[str],
    lora_scale: float,
    eval_path: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    device,
) -> tuple[list[dict], dict]:
    print(f"\n=== Arm: {arm_name} ===")
    model, tokenizer, _ = load_model("sft", device, phase="eval", model_tag=base_tag)
    if lora_tag is not None:
        info = apply_lora_from_tag(model, lora_tag, scale=lora_scale)
        print(f"  LoRA loaded: tag={lora_tag} step={info['loaded_step']} "
              f"target={info['targets']} rank={info['rank']} alpha={info['alpha']} "
              f"scale={info['scale']}")
    else:
        print(f"  No LoRA (base only)")
    engine = Engine(model, tokenizer)

    rows = [json.loads(l) for l in open(eval_path) if l.strip()]
    per_prompt: list[dict] = []
    agg = {"name": 0, "role": 0, "location": 0, "all_three": 0}
    for i, row in enumerate(rows):
        msgs = row["messages"]
        persona = row["_persona"]
        prompt_tokens = render_eval_prompt(tokenizer, msgs[:5])
        response = generate_response(
            engine, tokenizer, prompt_tokens,
            max_tokens=max_tokens, temperature=temperature,
            top_k=top_k, seed=seed,
        )
        scores = score_response(response, persona)
        per_prompt.append({
            "i": i,
            "persona": persona,
            "t5": msgs[4]["content"],
            "response": response,
            "scores": scores,
        })
        for k, v in scores.items():
            if v:
                agg[k] += 1
        flags = "".join("✓" if scores[k] else "·" for k in ("name", "role", "location"))
        snippet = response[:90].replace("\n", " ")
        print(f"  [{i:2d}] {flags} {persona['name']:>14} / {persona['role'][:20]:>20} / {persona['location'][:18]:>18} | {snippet!r}")
    n = len(rows)
    print()
    print(f"  Aggregate ({arm_name}, n={n}):")
    print(f"    name_recall    : {agg['name']}/{n}  ({100*agg['name']/n:.1f}%)")
    print(f"    role_recall    : {agg['role']}/{n}  ({100*agg['role']/n:.1f}%)")
    print(f"    location_recall: {agg['location']}/{n}  ({100*agg['location']/n:.1f}%)")
    print(f"    all_three      : {agg['all_three']}/{n}  ({100*agg['all_three']/n:.1f}%)")
    return per_prompt, {"n": n, **agg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tag", default="d6_baseline_modern_sft")
    parser.add_argument("--lora-tag", default=None,
                        help="if set, runs both arms (base + lora). if omitted, base-only.")
    parser.add_argument("--lora-scale", type=str, default="1.0",
                        help="Runtime LoRA attenuation: 1.0=full, 0.5=half, 0.0=base. "
                             "Pass multiple via comma-separated to sweep (e.g. 0.0,0.25,0.5,0.75,1.0)")
    parser.add_argument("--eval-path", default=None,
                        help="path to persona_retention_v1_eval.jsonl (default: $NANOCHAT_BASE_DIR)")
    parser.add_argument("--max-tokens", type=int, default=120,
                        help="max tokens per T6 response (default 120 — 6-turn convs cap T6 ~80 tokens)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy. Default greedy for reproducibility.")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default=None,
                        help="optional path to dump per-prompt results as json")
    args = parser.parse_args()

    if args.eval_path is None:
        args.eval_path = os.path.join(get_base_dir(), "persona_retention_v1_eval.jsonl")
    if not os.path.exists(args.eval_path):
        sys.exit(f"ERROR: eval file not found: {args.eval_path}")

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)

    # Parse possibly-multi scale spec
    scales = [float(s.strip()) for s in str(args.lora_scale).split(",") if s.strip()]
    sweep = len(scales) > 1

    arms = []
    base_results, base_agg = run_arm(
        "arm_base", args.base_tag, None, 1.0, args.eval_path,
        args.max_tokens, args.temperature, args.top_k, args.seed, device,
    )
    arms.append(("arm_base", base_results, base_agg))

    if args.lora_tag is not None:
        for s in scales:
            arm_label = f"arm_lora_s{s}" if sweep else "arm_lora"
            lora_results, lora_agg = run_arm(
                f"{arm_label} ({args.lora_tag} scale={s})", args.base_tag, args.lora_tag, s,
                args.eval_path,
                args.max_tokens, args.temperature, args.top_k, args.seed, device,
            )
            arms.append((arm_label, lora_results, lora_agg))

    # Comparison summary
    print()
    print("=" * 78)
    if sweep:
        # Wide table: one column per scale
        header = f"{'metric':<18} | {'base':>8} | " + " | ".join(f"s={s:>4.2f}" for s in scales)
        print(header)
        print("-" * len(header))
        base = arms[0][2]
        n = base["n"]
        for metric in ("name", "role", "location", "all_three"):
            b = 100 * base[metric] / n
            cells = [f"{b:>7.1f}%"]
            for i, s in enumerate(scales):
                lora_agg = arms[i + 1][2]
                cells.append(f"{100 * lora_agg[metric] / n:>5.1f}%")
            print(f"{metric:<18} | {' | '.join(cells)}")
    elif len(arms) >= 2:
        print(f"{'metric':<18} | {'base':>8} | {'lora':>8} | {'delta':>8}")
        print("-" * 70)
        base = arms[0][2]
        lora = arms[1][2]
        n = base["n"]
        for metric in ("name", "role", "location", "all_three"):
            b = 100 * base[metric] / n
            l = 100 * lora[metric] / n
            d = l - b
            sign = "+" if d > 0 else ""
            print(f"{metric:<18} | {b:>7.1f}% | {l:>7.1f}% | {sign}{d:>6.1f}pp")
    print("=" * 78)

    if args.out_json:
        out = {arm_name: {"agg": agg, "per_prompt": results}
               for arm_name, results, agg in arms}
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote per-prompt results: {args.out_json}")


if __name__ == "__main__":
    main()
