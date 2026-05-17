"""
chat_eval_mlx.py — Run nanochat's chat eval suite against an mlx_lm model.

Reuses tasks/ verbatim. Only the model, tokenizer, and chat-template
handling differ from scripts/chat_eval.py. Intended for cross-comparing
nanochat checkpoints against externally-trained MLX models (e.g.
PrismML's Ternary-Bonsai) on identical eval problems.

Requires the [mlx] extra (Apple Silicon only):
    uv sync --extra cpu --extra mlx

Usage:
    # Smoke: ARC-Easy, 50 problems, on Bonsai-1.7B
    python -m scripts.chat_eval_mlx -m 1.7b -a ARC-Easy -x 50

    # Full ChatCORE-comparable smoke on Bonsai-4B
    python -m scripts.chat_eval_mlx -m 4b -x 100 \\
        -o docs/bonsai_4b_chatcore_smoke.md

    # Arbitrary HF repo
    python -m scripts.chat_eval_mlx -m mlx-community/Qwen2.5-3B-Instruct-4bit
"""

import argparse
import logging
import re
import time
from datetime import datetime, timezone
from functools import partial

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler

from nanochat.delta_mem_mlx import (
    DeltaMemConfig,
    attach_delta_mem,
    load_delta_mem_adapter,
    reset_delta_mem_states,
)
from tasks.arc import ARC
from tasks.gsm8k import GSM8K, extract_answer as strict_extract_answer
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.spellingbee import SpellingBee


# Lenient answer extraction for generative tasks (GSM8K, SpellingBee).
# nanochat models SFT-trained to emit "#### N" hit the strict extractor.
# General-purpose models (Bonsai-Qwen3, etc.) output natural-language
# answers and score 0% under strict. Lenient extractor walks patterns
# from most-specific to most-permissive, normalizing the matched number.
_RE_STRICT = re.compile(r"####\s*(-?[0-9][0-9\.,]*)")
_RE_HINT = re.compile(
    r"(?:final\s*answer|the\s*answer\s*is|my\s*final\s*answer|"
    r"answer\s*[:=]|gives\s*us)\s*[*:#`'\"$]*\s*(-?[0-9][0-9\.,]*)",
    re.IGNORECASE,
)
_RE_BOLD = re.compile(r"\*\*\s*(-?[0-9][0-9\.,]*)\s*\*\*")
_RE_BARE = re.compile(r"(-?[0-9][0-9\.,]*)")


def _normalize_num(s):
    return s.strip().rstrip(".").replace(",", "")


def lenient_extract(completion):
    """Try strict #### first, then 'final answer'/'answer is' hints,
    then **N** bold, then last bare integer in the tail. Returns
    normalized number string or None.
    """
    m = _RE_STRICT.search(completion)
    if m:
        return _normalize_num(m.group(1))
    matches = list(_RE_HINT.finditer(completion))
    if matches:
        return _normalize_num(matches[-1].group(1))
    matches = list(_RE_BOLD.finditer(completion))
    if matches:
        return _normalize_num(matches[-1].group(1))
    # Bare-integer fallback: search the last 200 chars (avoid step-by-step
    # working-out numbers earlier in the completion).
    tail = completion[-200:]
    matches = list(_RE_BARE.finditer(tail))
    if matches:
        return _normalize_num(matches[-1].group(1))
    return None


def evaluate_completion(task, conv, completion, lenient):
    """Dispatch: strict baseline by default; lenient extractor for
    GSM8K/SpellingBee response side only (gold stays strict — it's
    always nanochat-format)."""
    if not lenient or not isinstance(task, (GSM8K, SpellingBee)):
        return int(task.evaluate(conv, completion))
    gold_text = conv["messages"][-1]["content"][-1]["text"]
    ref = strict_extract_answer(gold_text)
    pred = lenient_extract(completion)
    return int(pred is not None and pred == ref)


BONSAI_SHORT_NAMES = {
    "1.7b": "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit",
    "4b":   "prism-ml/Ternary-Bonsai-4B-mlx-2bit",
    "8b":   "prism-ml/Ternary-Bonsai-8B-mlx-2bit",
}

ALL_TASKS = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
BASELINE_ACC = {
    "ARC-Easy": 0.25, "ARC-Challenge": 0.25, "MMLU": 0.25,
    "GSM8K": 0.0, "HumanEval": 0.0, "SpellingBee": 0.0,
}

TASK_CTORS = {
    "ARC-Easy":      partial(ARC, subset="ARC-Easy", split="test"),
    "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
    "MMLU":          partial(MMLU, subset="all", split="test"),
    "GSM8K":         partial(GSM8K, subset="main", split="test"),
    "HumanEval":     HumanEval,
    "SpellingBee":   partial(SpellingBee, size=256, split="test"),
}


def render_prompt(tokenizer, conversation, no_system_prompt=False):
    """Strip the assistant's gold answer, render with add_generation_prompt=True.

    Mirrors nanochat tokenizer.render_for_completion(): pop the trailing
    assistant turn, then prime the assistant header so the model predicts
    the answer as the next token.
    """
    messages = conversation["messages"][:-1]
    if no_system_prompt:
        messages = [m for m in messages if m["role"] != "system"]
    # SpellingBee/GSM8K conversations may carry list-of-parts assistant content
    # for the *gold* answer; we already stripped that. The remaining user/system
    # messages are simple strings and apply_chat_template handles them.
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def encode_letter_ids(tokenizer, letters, cache):
    """Encode each MC letter to its single token id; assert single-token.

    Mirrors the assumption in tasks/common.render_mc that letters appear
    as single tokens in the model's vocab. Bails fast with a clear error
    if Qwen3 (or whatever) BPE breaks this.
    """
    out = []
    for L in letters:
        if L not in cache:
            ids = tokenizer.encode(L, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f"Letter {L!r} encodes to {len(ids)} tokens ({ids}); "
                    "categorical eval requires single-token letters. "
                    "This tokenizer is incompatible with the current eval design."
                )
            cache[L] = ids[0]
        out.append(cache[L])
    return out


def run_categorical(task, model, tokenizer, max_problems, no_system_prompt, delta_mem_attached):
    n = min(len(task), max_problems or len(task))
    cache = {}
    passed, total = 0, 0
    t0 = time.perf_counter()
    for i in range(n):
        if delta_mem_attached:
            reset_delta_mem_states(model)
        conv = task[i]
        prompt_str = render_prompt(tokenizer, conv, no_system_prompt)
        ids = mx.array(tokenizer.encode(prompt_str, add_special_tokens=False))[None, :]
        logits = model(ids)[0, -1]  # (V,)
        choice_ids = encode_letter_ids(tokenizer, conv["letters"], cache)
        focus = logits[mx.array(choice_ids)]
        pred = conv["letters"][int(mx.argmax(focus).item())]
        passed += int(task.evaluate(conv, pred))
        total += 1
        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"\r  [{i+1}/{n}] passed={passed} acc={passed/total:.3f}", end="", flush=True)
    dt = time.perf_counter() - t0
    print(f"  ({dt:.1f}s)")
    return passed / total, total


def run_generative(task, model, tokenizer, max_problems, max_new_tokens,
                   temperature, no_system_prompt, lenient, delta_mem_attached):
    n = min(len(task), max_problems or len(task))
    sampler = make_sampler(temp=temperature)
    passed, total = 0, 0
    t0 = time.perf_counter()
    for i in range(n):
        if delta_mem_attached:
            reset_delta_mem_states(model)
        conv = task[i]
        prompt_str = render_prompt(tokenizer, conv, no_system_prompt)
        completion = generate(
            model, tokenizer, prompt=prompt_str, max_tokens=max_new_tokens,
            sampler=sampler, verbose=False,
        )
        passed += evaluate_completion(task, conv, completion, lenient)
        total += 1
        if (i + 1) % 5 == 0 or i + 1 == n:
            print(f"\r  [{i+1}/{n}] passed={passed} acc={passed/total:.3f}", end="", flush=True)
    dt = time.perf_counter() - t0
    print(f"  ({dt:.1f}s)")
    return passed / total, total


def chatcore(results):
    """Mean centered accuracy. Same formula as scripts/chat_eval.py."""
    if not all(t in results for t in ALL_TASKS):
        return None
    centered = sum(
        (results[t] - BASELINE_ACC[t]) / (1.0 - BASELINE_ACC[t]) for t in ALL_TASKS
    )
    return centered / len(ALL_TASKS)


def write_report(out_path, model_id, results, totals, args, wall_total, chatcore_val):
    delta_mem_line = f"δ-mem: {args.delta_mem}" if args.delta_mem else "δ-mem: off"
    lines = [
        f"# MLX chat eval — {model_id}",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Total wall: {wall_total:.1f}s",
        f"Args: max_problems={args.max_problems} temp={args.temperature} "
        f"max_new_tokens={args.max_new_tokens} no_system_prompt={args.no_system_prompt} "
        f"lenient_extract={args.lenient_extract}",
        delta_mem_line,
        "",
        "| Task | Acc | n | Centered |",
        "| --- | ---: | ---: | ---: |",
    ]
    for t in ALL_TASKS:
        if t in results:
            base = BASELINE_ACC[t]
            cent = (results[t] - base) / (1.0 - base)
            lines.append(f"| {t} | {results[t]:.4f} | {totals[t]} | {cent:+.4f} |")
    if chatcore_val is not None:
        lines.append(f"| **ChatCORE** | — | — | **{chatcore_val:.4f}** |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True,
                   help="HF repo or bonsai short name (1.7b|4b|8b)")
    p.add_argument("-a", "--task-name", default=None,
                   help=f"Task name(s), '|'-separated. Default = all six. Choices: {','.join(ALL_TASKS)}")
    p.add_argument("-x", "--max-problems", type=int, default=None,
                   help="Cap problems per task (smoke-friendly)")
    p.add_argument("-t", "--temperature", type=float, default=0.0,
                   help="Sampling temperature (0 = greedy, matches nanochat default)")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Max generated tokens for generative tasks (default 256, less than nanochat 512 for smoke speed)")
    p.add_argument("--no-system-prompt", action="store_true",
                   help="Strip system messages from prompts (per Bonsai ADR-002 for 1.7B)")
    p.add_argument("--lenient-extract", action="store_true",
                   help="GSM8K/SpellingBee: accept natural-language answer formats "
                   "('Final Answer: N', '**N**', etc.) in addition to strict '#### N'. "
                   "Use when comparing non-nanochat-SFT models.")
    p.add_argument("-o", "--output", default=None, help="Markdown report path")
    p.add_argument("--delta-mem", default=None, metavar="PATH",
                   help="Path to converted δ-mem adapter directory (containing "
                   "adapter.safetensors + delta_mem_config.json). Per-problem "
                   "state reset is applied automatically. See "
                   "scripts/convert_delta_mem_adapter.py.")
    args = p.parse_args()

    model_id = BONSAI_SHORT_NAMES.get(args.model.lower(), args.model)
    task_names = ALL_TASKS if args.task_name is None else args.task_name.split("|")
    for t in task_names:
        if t not in TASK_CTORS:
            p.error(f"unknown task: {t} (choices: {','.join(ALL_TASKS)})")

    print(f"Loading {model_id}...")
    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    print(f"  vocab={len(tokenizer.vocab)} arch={type(model).__module__}")

    delta_mem_attached = False
    if args.delta_mem:
        adapter_dir = Path(args.delta_mem)
        adapter_st = adapter_dir / "adapter.safetensors"
        config_json = adapter_dir / "delta_mem_config.json"
        if not adapter_st.exists() or not config_json.exists():
            p.error(f"--delta-mem dir missing required files at {adapter_dir}")
        with open(config_json) as f:
            cfg = DeltaMemConfig.from_dict(json.load(f))
        wrapped = attach_delta_mem(model, cfg)
        n_loaded = load_delta_mem_adapter(model, str(adapter_st))
        print(f"  δ-mem attached: {len(wrapped)} layers wrapped, {n_loaded} tensors loaded")
        print(f"  δ-mem config: rank={cfg.rank} alpha={cfg.alpha} delta_heads={cfg.delta_heads}")
        delta_mem_attached = True

    results, totals = {}, {}
    wall_t0 = time.perf_counter()
    for tname in task_names:
        print(f"\n=== {tname} ===")
        task = TASK_CTORS[tname]()
        if task.eval_type == "categorical":
            acc, n = run_categorical(task, model, tokenizer,
                                      args.max_problems, args.no_system_prompt,
                                      delta_mem_attached)
        else:
            acc, n = run_generative(task, model, tokenizer,
                                     args.max_problems, args.max_new_tokens,
                                     args.temperature, args.no_system_prompt,
                                     args.lenient_extract, delta_mem_attached)
        results[tname] = acc
        totals[tname] = n
        print(f"  {tname}: {acc*100:.2f}% ({n} problems)")
    wall_total = time.perf_counter() - wall_t0

    cc = chatcore(results)
    print(f"\n{'='*50}")
    print(f"Summary — {model_id}")
    print(f"{'='*50}")
    for t in ALL_TASKS:
        if t in results:
            base = BASELINE_ACC[t]
            cent = (results[t] - base) / (1.0 - base)
            print(f"  {t:<16} {results[t]*100:6.2f}%  centered={cent:+.4f}  (n={totals[t]})")
    if cc is not None:
        print(f"  {'ChatCORE':<16} {cc:.4f}")
    print(f"  {'wall':<16} {wall_total:.1f}s")

    if args.output:
        write_report(args.output, model_id, results, totals, args, wall_total, cc)
        print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()
