"""
Scripted free-chat smoke for a base+LoRA pair. Runs the postscript-grade
4-turn free-chat transcript that caught template-bleed at scale=0.5 in
the v2 LoRA (HANDOFF.md Day 2026-05-07 Postscript).

The 7-prompt rubric in dev/multi_turn_eval.py mostly uses recall-shaped or
recap-shaped final turns; this smoke uses question-shaped final turns where
the right answer is *not* "You're [X]…". It exists to falsify the claim
that v3 (or any candidate LoRA) has actually fixed template-bleed beyond
the rubric.

Run:
    uv run python -m dev.free_chat_smoke --lora-tag d6_l1_persona_lora_v3
    uv run python -m dev.free_chat_smoke --lora-tag d6_l1_persona_lora_v3 --lora-scale 0.5
    uv run python -m dev.free_chat_smoke   # base only

Output: prints assistant replies + a heuristic bleed-pattern check.
"""

import argparse
import json
import re
import time

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine
from nanochat.lora import apply_lora_from_tag

FREE_CHAT_TURNS = [
    "hi there",
    "What's the capital of France?",
    "Why is the sky blue?",
    "Who are you?",
]

DECODING = {"max_tokens": 128, "temperature": 0.6, "top_k": 50, "seed": 42}

# Heuristic bleed patterns. NOT exhaustive — these are the shapes the v2
# LoRA stamped onto free chat at scale=0.5 / 1.0:
#   "You're [Name], a [Role] in [Location]…"   ← scale=1.0
#   "You can find [thing]…"                    ← scale=0.5 (postscript)
BLEED_PATTERNS = [
    (re.compile(r"^\s*you['’]?re\s+\w+", re.I), "persona-stamp ('You're [Name]')"),
    (re.compile(r"^\s*you\s+are\s+\w+", re.I), "persona-stamp ('You are [Name]')"),
    (re.compile(r"^\s*you\s+can\s+find\b", re.I), "scale=0.5 stamp ('You can find…')"),
    (re.compile(r"^\s*you\s+mentioned\b", re.I), "recall opener ('You mentioned…')"),
    (re.compile(r"^\s*your\s+name\s+is\b", re.I), "recall opener ('Your name is…')"),
]


def run_conversation(engine, tokenizer, turns):
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    conversation_tokens = [bos]
    transcript = []
    for turn_idx, user_input in enumerate(turns):
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(user_input))
        conversation_tokens.append(user_end)
        conversation_tokens.append(assistant_start)

        response_tokens = []
        gen = engine.generate(conversation_tokens, num_samples=1, **DECODING)
        for token_column, _ in gen:
            tok = int(token_column[0])
            response_tokens.append(tok)
            if tok == assistant_end:
                break
        if not response_tokens or response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)
        conversation_tokens.extend(response_tokens)

        text_tokens = response_tokens[:-1] if response_tokens[-1] == assistant_end else response_tokens
        assistant_text = tokenizer.decode(text_tokens) if text_tokens else ""
        transcript.append({"turn": turn_idx + 1, "user": user_input,
                           "assistant": assistant_text})
    return transcript


def bleed_check(transcript):
    hits = []
    for entry in transcript:
        text = entry["assistant"]
        for pat, label in BLEED_PATTERNS:
            if pat.match(text):
                hits.append((entry["turn"], label, text[:80]))
                break
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tag", default="d6_baseline_modern_sft")
    parser.add_argument("--lora-tag", default=None)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)

    label = args.base_tag if args.lora_tag is None else (
        f"{args.base_tag}+LoRA[{args.lora_tag}]"
        + (f"@{args.lora_scale}" if args.lora_scale != 1.0 else "")
    )

    print(f"\nLoading {label}...")
    t0 = time.time()
    model, tokenizer, _ = load_model("sft", device, phase="eval", model_tag=args.base_tag)
    if args.lora_tag is not None:
        info = apply_lora_from_tag(model, args.lora_tag, scale=args.lora_scale)
        print(f"  LoRA loaded: step={info['loaded_step']} target={info['targets']} "
              f"rank={info['rank']} alpha={info['alpha']} scale={info['scale']}")
    engine = Engine(model, tokenizer)
    print(f"  loaded in {time.time() - t0:.1f}s")

    print(f"\n=== Free-chat transcript: {label} ===")
    transcript = run_conversation(engine, tokenizer, FREE_CHAT_TURNS)
    for entry in transcript:
        print(f"\n  T{entry['turn']} user      : {entry['user']}")
        print(f"  T{entry['turn']} assistant : {entry['assistant']}")

    print()
    print("=== Bleed-pattern check ===")
    hits = bleed_check(transcript)
    if hits:
        print(f"  {len(hits)} bleed hit(s):")
        for turn, label_, snippet in hits:
            print(f"    T{turn}: {label_}: {snippet!r}")
    else:
        print("  no bleed patterns matched (clean)")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"label": label, "transcript": transcript, "bleed_hits": hits}, f, indent=2)
        print(f"\nWrote → {args.out_json}")


if __name__ == "__main__":
    main()
