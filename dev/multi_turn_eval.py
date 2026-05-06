"""
Multi-turn chat side-by-side: d6_baseline_modern_sft vs d6_stage2_pretrain_s1_sft.

Loads each SFT checkpoint sequentially, runs the seven memory-load-bearing
multi-turn prompts pre-registered in
docs/multi_turn_chat_eval_2026-05-06.md, dumps full transcripts to
/tmp/multi_turn_eval.json, and prints both side-by-side.

Decoding held constant across architectures (temp=0.6, top_k=50, seed=42)
so the only varying input is the architecture.

Run: uv run python -m dev.multi_turn_eval
"""
import argparse
import json
import time
from pathlib import Path

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine

PROMPTS = [
    {
        "name": "persona_retention",
        "turns": [
            "Hi! I'm Alex, a software engineer at a small startup.",
            "What do you like to do for fun?",
            "I really enjoy mountain biking on weekends.",
            "What's my name and job?",
        ],
    },
    {
        "name": "reference_resolution",
        "turns": [
            "I have a cat named Mittens, she's 4 years old and very fluffy.",
            "Cool. What kind of cats are most playful in general?",
            "Interesting. What about exercise needs for indoor cats?",
            "How old is Mittens now? And how would you describe her temperament?",
        ],
    },
    {
        "name": "numerical_thread",
        "turns": [
            "I bought 5 apples at the store today.",
            "I ate 2 of them for lunch.",
            "Then I gave 1 to my friend.",
            "How many apples do I have left?",
        ],
    },
    {
        "name": "topic_stickiness",
        "turns": [
            "I'm planning a 3-day weekend trip to Portland, Oregon.",
            "Tell me about food I should try there.",
            "Are there any good day hikes nearby?",
            "Going back to my original trip — what's something specifically Portland-y I should not miss?",
        ],
    },
    {
        "name": "constraint_accumulation",
        "turns": [
            "Help me plan a vacation. No flights please — only ground transit.",
            "Budget is under $500 total.",
            "I have 5 days off work.",
            "Suggest a specific itinerary.",
        ],
    },
    {
        "name": "self_correction",
        "turns": [
            "What's the capital of Australia?",
            "Actually, the capital is Canberra, not Sydney.",
            "What other unusual capital choices have countries made?",
            "Earlier we discussed the capital of Australia — what is it?",
        ],
    },
    {
        "name": "open_drift",
        "turns": [
            "Tell me something interesting.",
            "That's neat. What else?",
            "Cool. Tell me another one.",
            "And another?",
            "What was the very first thing you told me?",
        ],
    },
]

DECODING = {"max_tokens": 256, "temperature": 0.6, "top_k": 50, "seed": 42}


def run_conversation(engine, tokenizer, turns, decoding):
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
        gen = engine.generate(conversation_tokens, num_samples=1, **decoding)
        for token_column, _ in gen:
            tok = int(token_column[0])
            response_tokens.append(tok)
            if tok == assistant_end:
                break
        if not response_tokens or response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)
        conversation_tokens.extend(response_tokens)

        # Decode assistant text only (drop trailing assistant_end)
        text_tokens = response_tokens[:-1] if response_tokens[-1] == assistant_end else response_tokens
        assistant_text = tokenizer.decode(text_tokens) if text_tokens else ""
        transcript.append({
            "turn": turn_idx + 1,
            "user": user_input,
            "assistant": assistant_text,
            "response_token_count": len(text_tokens),
        })
    return transcript


def run_for_model(model_tag, device, device_type):
    t0 = time.time()
    print(f"\nLoading {model_tag}...")
    model, tokenizer, _ = load_model("sft", device, phase="eval", model_tag=model_tag)
    engine = Engine(model, tokenizer)
    load_s = time.time() - t0
    print(f"  loaded in {load_s:.1f}s")

    convos = {}
    for prompt in PROMPTS:
        print(f"  [{model_tag}] {prompt['name']}...", flush=True)
        t1 = time.time()
        convos[prompt["name"]] = run_conversation(engine, tokenizer, prompt["turns"], DECODING)
        print(f"    {time.time() - t1:.1f}s")

    del engine, model
    if device_type == "mps":
        torch.mps.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()
    return convos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="/tmp/multi_turn_eval.json")
    parser.add_argument("--baseline-tag", type=str, default="d6_baseline_modern_sft")
    parser.add_argument("--stage2-tag", type=str, default="d6_stage2_pretrain_s1_sft")
    args = parser.parse_args()

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)

    print(f"Device: {device_type} ({device})")
    print(f"Decoding: {DECODING}")
    print(f"Prompts: {len(PROMPTS)} × multi-turn")

    baseline_convos = run_for_model(args.baseline_tag, device, device_type)
    stage2_convos = run_for_model(args.stage2_tag, device, device_type)

    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device_type": device_type,
        "decoding": DECODING,
        "prompts": PROMPTS,
        "baseline": {"model_tag": args.baseline_tag, "convos": baseline_convos},
        "stage2": {"model_tag": args.stage2_tag, "convos": stage2_convos},
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nSaved transcripts to {args.out}")

    # Side-by-side print
    for prompt in PROMPTS:
        print()
        print("=" * 80)
        print(f"PROMPT: {prompt['name']}")
        print("=" * 80)
        b = baseline_convos[prompt["name"]]
        s = stage2_convos[prompt["name"]]
        for turn_idx in range(len(prompt["turns"])):
            print(f"\n[Turn {turn_idx + 1}]  USER: {b[turn_idx]['user']}")
            print(f"  BASELINE:  {b[turn_idx]['assistant'][:400]}")
            print(f"  STAGE 2 :  {s[turn_idx]['assistant'][:400]}")


if __name__ == "__main__":
    main()
