"""
Decoding sweep on d6_baseline_modern_sft, tuned to suppress the dominant
chat failure modes surfaced by the multi-turn A/B (math-mode reflex,
repetition collapse). Tests 5 decoding configs against the same 7
multi-turn prompts as dev/multi_turn_eval.py.

Per the multi-turn A/B verdict (5008fe2), the architecture lever is
closed; the chatbot's badness is dominated by SFT-template leakage and
repetition. This experiment probes the runtime-controllable decoding
levers: temperature/top_k variation + repetition penalty (added as
opt-in kwarg in Engine.generate, default 1.0 = no behavior change) +
a fake system-prompt prefix to see if priming the model helps.

Run: uv run python -m dev.decoding_sweep
"""
import json
import time
from pathlib import Path

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine

# Reuse the same prompt set as dev/multi_turn_eval.py — comparability matters.
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

# Five decoding configs, baseline only.
# A: default — reference (matches multi_turn_eval.py)
# B: greedy — no exploration; reveals what the model "wants" to say
# C: anti-rep moderate — common rep penalty
# D: anti-rep strong — heavy penalty; tests upper bound of dampening
# E: system-prompt-prime — fake earlier instructional turn + moderate rep penalty
CONFIGS = [
    {"name": "A_default",         "temperature": 0.6, "top_k": 50,   "repetition_penalty": 1.0, "system_prime": False},
    {"name": "B_greedy",          "temperature": 0.0, "top_k": None, "repetition_penalty": 1.0, "system_prime": False},
    {"name": "C_anti_rep_1.2",    "temperature": 0.6, "top_k": 50,   "repetition_penalty": 1.2, "system_prime": False},
    {"name": "D_anti_rep_1.4",    "temperature": 0.6, "top_k": 50,   "repetition_penalty": 1.4, "system_prime": False},
    {"name": "E_sys_prime+1.2",   "temperature": 0.6, "top_k": 50,   "repetition_penalty": 1.2, "system_prime": True},
]

SYSTEM_PRIME = (
    "Please respond conversationally and naturally. Do not use Python code "
    "blocks, math computation tags, or '#### N' answer formats unless I'm "
    "explicitly asking a math question."
)


def run_conversation(engine, tokenizer, turns, decoding, system_prime=False):
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    conversation_tokens = [bos]
    if system_prime:
        # Fake instructional turn — the chat format doesn't have system prompts,
        # so we inject a [user: instruction → assistant: ack] turn to prime tone.
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(SYSTEM_PRIME))
        conversation_tokens.append(user_end)
        conversation_tokens.append(assistant_start)
        conversation_tokens.extend(tokenizer.encode("Got it. I'll respond conversationally."))
        conversation_tokens.append(assistant_end)

    transcript = []
    for turn_idx, user_input in enumerate(turns):
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(user_input))
        conversation_tokens.append(user_end)
        conversation_tokens.append(assistant_start)

        response_tokens = []
        gen_kwargs = {k: v for k, v in decoding.items() if k != "system_prime" and k != "name"}
        gen_kwargs["max_tokens"] = 256
        gen_kwargs["seed"] = 42
        gen = engine.generate(conversation_tokens, num_samples=1, **gen_kwargs)
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
        transcript.append({
            "turn": turn_idx + 1,
            "user": user_input,
            "assistant": assistant_text,
            "response_token_count": len(text_tokens),
        })
    return transcript


def main():
    out_path = "/tmp/decoding_sweep.json"
    baseline_tag = "d6_baseline_modern_sft"

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)

    print(f"Device: {device_type} ({device})")
    print(f"Model: {baseline_tag}")
    print(f"Configs: {len(CONFIGS)}, Prompts: {len(PROMPTS)} multi-turn")

    print(f"\nLoading {baseline_tag}...")
    model, tokenizer, _ = load_model("sft", device, phase="eval", model_tag=baseline_tag)
    engine = Engine(model, tokenizer)

    results = {}
    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        print(f"\n== {cfg_name} | temp={cfg['temperature']} top_k={cfg['top_k']} rep_pen={cfg['repetition_penalty']} sys_prime={cfg['system_prime']} ==")
        cfg_convos = {}
        for prompt in PROMPTS:
            t0 = time.time()
            cfg_convos[prompt["name"]] = run_conversation(
                engine, tokenizer, prompt["turns"],
                {"temperature": cfg["temperature"], "top_k": cfg["top_k"], "repetition_penalty": cfg["repetition_penalty"]},
                system_prime=cfg["system_prime"],
            )
            print(f"  {prompt['name']}: {time.time() - t0:.1f}s")
        results[cfg_name] = {"config": cfg, "convos": cfg_convos}

    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device_type": device_type,
        "model_tag": baseline_tag,
        "prompts": PROMPTS,
        "configs": CONFIGS,
        "results": results,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")

    # Side-by-side print: for each prompt, show one section with all 5 configs at the final turn
    for prompt in PROMPTS:
        print()
        print("=" * 80)
        print(f"PROMPT: {prompt['name']}  (showing T1 + final-turn responses)")
        print("=" * 80)
        for turn_idx in [0, len(prompt["turns"]) - 1]:
            print(f"\n[Turn {turn_idx + 1}] USER: {prompt['turns'][turn_idx]}")
            for cfg in CONFIGS:
                resp = results[cfg["name"]]["convos"][prompt["name"]][turn_idx]["assistant"]
                print(f"  {cfg['name']:20s}: {resp[:280]}")


if __name__ == "__main__":
    main()
