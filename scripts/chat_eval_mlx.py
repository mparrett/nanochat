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
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from nanochat.delta_mem_mlx import (
    DeltaMemConfig,
    attach_delta_mem,
    load_delta_mem_adapter,
    reset_delta_mem_states,
)
from tasks.arc import ARC
from tasks.gsm8k import GSM8K, extract_answer as strict_extract_answer
from tasks.hotpotqa import HotpotQA
from tasks.humaneval import HumanEval
from tasks.mbpp import MBPP
from tasks.mmlu import MMLU
from tasks.spellingbee import SpellingBee


# Lenient answer extraction for generative tasks (GSM8K, SpellingBee).
# nanochat models SFT-trained to emit "#### N" hit the strict extractor.
# General-purpose models (Bonsai-Qwen3, etc.) output natural-language
# answers and score 0% under strict. Lenient extractor walks patterns
# from most-specific to most-permissive, normalizing the matched number.
_RE_STRICT = re.compile(r"####\s*(-?[0-9][0-9\.,]*)")
# Qwen3-family models often close with LaTeX "\boxed{N}"; without this the
# hint regex grabs a number from the working-out instead.
_RE_BOXED = re.compile(r"\\boxed\{\s*\$?\s*(-?[0-9][0-9\.,]*)")
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
    """Try strict #### first, then the last \\boxed{N}, then 'final
    answer'/'answer is' hints, then **N** bold, then last bare integer in
    the tail. Returns normalized number string or None.
    """
    m = _RE_STRICT.search(completion)
    if m:
        return _normalize_num(m.group(1))
    matches = list(_RE_BOXED.finditer(completion))
    if matches:
        return _normalize_num(matches[-1].group(1))
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


# High-specificity hesitation markers from arXiv 2606.00206. Broad connectives
# ("but", "however", "actually") are left out on purpose: suppressing them
# everywhere cost faithfulness in the bonsai agent tests (Round 6, Exp 3).
HESITATION_MARKERS = [
    "wait", "hmm", "alternatively", "reconsider", "rethink", "backtrack",
    "retry", "recheck", "revisit", "confused", "wrong", "mistake", "incorrect",
]


def marker_token_ids(tokenizer, words):
    """IDs of every single-token spelling of each word. Multi-token spellings
    are skipped rather than approximated by their first piece."""
    ids = set()
    for w in words:
        w = w.strip()
        for v in {w, w.capitalize(), " " + w, " " + w.capitalize()}:
            enc = tokenizer.encode(v, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
    return sorted(ids)


BONSAI_SHORT_NAMES = {
    "1.7b": "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit",
    "4b":   "prism-ml/Ternary-Bonsai-4B-mlx-2bit",
    "8b":   "prism-ml/Ternary-Bonsai-8B-mlx-2bit",
}

ALL_TASKS = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
# MBPP and HotpotQA are not in ALL_TASKS (so ChatCORE composite is unchanged)
# but are selectable via -a for δ-mem follow-up runs.
BASELINE_ACC = {
    "ARC-Easy": 0.25, "ARC-Challenge": 0.25, "MMLU": 0.25,
    "GSM8K": 0.0, "HumanEval": 0.0, "SpellingBee": 0.0,
    "MBPP": 0.0, "HotpotQA": 0.0,
}

TASK_CTORS = {
    "ARC-Easy":      partial(ARC, subset="ARC-Easy", split="test"),
    "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
    "MMLU":          partial(MMLU, subset="all", split="test"),
    "GSM8K":         partial(GSM8K, subset="main", split="test"),
    "HumanEval":     HumanEval,
    "SpellingBee":   partial(SpellingBee, size=256, split="test"),
    "MBPP":          MBPP,
    "HotpotQA":      HotpotQA,
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
                   gen_kwargs, no_system_prompt, lenient, delta_mem_attached,
                   dump=None, only_ids=None, seed=0, skip=frozenset()):
    if only_ids is None:
        ids = list(range(min(len(task), max_problems or len(task))))
    else:
        ids = sorted(only_ids)
    ids = [i for i in ids if i not in skip]
    n = len(ids)
    if n == 0:
        print("  nothing to run (all done)")
        return None, 0, None
    passed, total = 0, 0
    f1_sum, em_sum, aux_n = 0.0, 0, 0  # HotpotQA stashes _f1/_em on conv
    t0 = time.perf_counter()
    for k, i in enumerate(ids):
        if delta_mem_attached:
            reset_delta_mem_states(model)
        # Seed per problem, not per run, so a sampled draw doesn't depend on
        # which chunk or resume the problem ran in.
        mx.random.seed(seed + i)
        conv = task[i]
        prompt_str = render_prompt(tokenizer, conv, no_system_prompt)
        # stream_generate rather than generate: the final response carries
        # finish_reason, which tells a cap-truncated completion from a finished one.
        completion, last = "", None
        for last in stream_generate(
            model, tokenizer, prompt=prompt_str, max_tokens=max_new_tokens,
            **gen_kwargs,
        ):
            completion += last.text
        ok = evaluate_completion(task, conv, completion, lenient)
        passed += ok
        total += 1
        if dump is not None:
            dump.write(json.dumps({
                "task": type(task).__name__, "idx": i, "passed": bool(ok),
                "gen_tokens": last.generation_tokens,
                "finish_reason": last.finish_reason,
                "hit_cap": last.finish_reason == "length",
                "completion": completion,
            }) + "\n")
            dump.flush()
        if "_f1" in conv:
            f1_sum += conv["_f1"]
            em_sum += conv["_em"]
            aux_n += 1
        if (k + 1) % 5 == 0 or k + 1 == n:
            tail = f" f1={f1_sum/aux_n:.3f} em={em_sum/aux_n:.3f}" if aux_n else ""
            print(f"\r  [{k+1}/{n}] passed={passed} acc={passed/total:.3f}{tail}", end="", flush=True)
    dt = time.perf_counter() - t0
    print(f"  ({dt:.1f}s)")
    aux = {"f1_mean": f1_sum / aux_n, "em_mean": em_sum / aux_n} if aux_n else None
    return passed / total, total, aux


def chatcore(results):
    """Mean centered accuracy. Same formula as scripts/chat_eval.py."""
    if not all(t in results for t in ALL_TASKS):
        return None
    centered = sum(
        (results[t] - BASELINE_ACC[t]) / (1.0 - BASELINE_ACC[t]) for t in ALL_TASKS
    )
    return centered / len(ALL_TASKS)


def write_report(out_path, model_id, results, totals, args, wall_total, chatcore_val, aux_by_task=None):
    delta_mem_line = f"δ-mem: {args.delta_mem}" if args.delta_mem else "δ-mem: off"
    caps_parts = [f"max_problems={args.max_problems}"]
    if args.max_problems_cat is not None:
        caps_parts.append(f"cat={args.max_problems_cat}")
    if args.max_problems_gen is not None:
        caps_parts.append(f"gen={args.max_problems_gen}")
    caps_str = " ".join(caps_parts)
    lines = [
        f"# MLX chat eval — {model_id}",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Total wall: {wall_total:.1f}s",
        f"Args: {caps_str} temp={args.temperature} top_k={args.top_k} "
        f"top_p={args.top_p} seed={args.seed} "
        f"presence_penalty={args.presence_penalty}@{args.presence_context} "
        f"max_new_tokens={args.max_new_tokens} no_system_prompt={args.no_system_prompt} "
        f"lenient_extract={args.lenient_extract}",
        delta_mem_line,
        "",
        "| Task | Acc | n | Centered |",
        "| --- | ---: | ---: | ---: |",
    ]
    for t in results:
        base = BASELINE_ACC.get(t, 0.0)
        cent = (results[t] - base) / (1.0 - base) if base < 1.0 else 0.0
        lines.append(f"| {t} | {results[t]:.4f} | {totals[t]} | {cent:+.4f} |")
    if chatcore_val is not None:
        lines.append(f"| **ChatCORE** | — | — | **{chatcore_val:.4f}** |")
    if aux_by_task:
        lines.append("")
        lines.append("## Auxiliary metrics")
        lines.append("")
        lines.append("| Task | mean F1 | mean EM |")
        lines.append("| --- | ---: | ---: |")
        for t, aux in aux_by_task.items():
            lines.append(f"| {t} | {aux['f1_mean']:.4f} | {aux['em_mean']:.4f} |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True,
                   help="HF repo or bonsai short name (1.7b|4b|8b)")
    p.add_argument("-a", "--task-name", default=None,
                   help=f"Task name(s), '|'-separated. Default = all six. Choices: {','.join(ALL_TASKS)}")
    p.add_argument("-x", "--max-problems", type=int, default=None,
                   help="Cap problems per task (smoke-friendly). Used as the "
                   "default for both task types unless --max-problems-cat or "
                   "--max-problems-gen overrides it.")
    p.add_argument("--max-problems-cat", type=int, default=None,
                   help="Override -x for categorical tasks (ARC-Easy, ARC-Challenge, MMLU). "
                   "Useful for asymmetric runs: full resolution on cheap categorical, "
                   "lower resolution on expensive generative.")
    p.add_argument("--max-problems-gen", type=int, default=None,
                   help="Override -x for generative tasks (GSM8K, HumanEval, SpellingBee).")
    p.add_argument("-t", "--temperature", type=float, default=0.0,
                   help="Sampling temperature (0 = greedy, matches nanochat default)")
    p.add_argument("--top-k", type=int, default=0, help="0 = off")
    p.add_argument("--top-p", type=float, default=0.0, help="0 = off")
    p.add_argument("--seed", type=int, default=0,
                   help="mx.random seed, so sampled runs are repeatable")
    p.add_argument("--presence-penalty", type=float, default=0.0, help="0 = off")
    p.add_argument("--presence-context", type=int, default=20,
                   help="Tokens the presence penalty looks back over. mlx-lm's "
                   "default is 20; set it to max-new-tokens to match vLLM, which "
                   "counts the whole generation.")
    p.add_argument("--marker-penalty", type=float, default=0.0,
                   help="Subtract this from the logits of hesitation markers "
                   "(arXiv 2606.00206). 0 = off.")
    p.add_argument("--markers", default=",".join(HESITATION_MARKERS),
                   help="Comma-separated marker words; each is biased in every "
                   "single-token spelling (with/without leading space, capitalized).")
    p.add_argument("--resume", action="store_true",
                   help="Skip problems already recorded in --dump-jsonl.")
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
    p.add_argument("--dump-jsonl", default=None, metavar="PATH",
                   help="Append one record per generative problem (completion, "
                   "gen_tokens, finish_reason, hit_cap, passed). Written "
                   "incrementally so a partial run is still readable.")
    p.add_argument("--only-ids", default=None, metavar="PATH",
                   help="JSON file mapping generative task name to a list of "
                   "problem indices; runs only those (e.g. re-running the "
                   "problems that hit max_new_tokens). Overrides -x for them.")
    args = p.parse_args()
    if args.resume and not args.dump_jsonl:
        p.error("--resume needs --dump-jsonl")
    only_ids = None
    if args.only_ids:
        with open(args.only_ids) as f:
            only_ids = json.load(f)

    model_id = BONSAI_SHORT_NAMES.get(args.model.lower(), args.model)
    task_names = ALL_TASKS if args.task_name is None else args.task_name.split("|")
    for t in task_names:
        if t not in TASK_CTORS:
            p.error(f"unknown task: {t} (choices: {','.join(ALL_TASKS)})")
        if only_ids is not None and t not in only_ids:
            p.error(f"--only-ids file has no entry for task {t}")

    print(f"Loading {model_id}...")
    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    print(f"  vocab={len(tokenizer.vocab)} arch={type(model).__module__}")

    logit_bias = None
    if args.marker_penalty:
        marker_ids = marker_token_ids(tokenizer, args.markers.split(","))
        logit_bias = {i: -args.marker_penalty for i in marker_ids}
        print(f"  marker penalty {args.marker_penalty} on {len(marker_ids)} tokens")
    gen_kwargs = {
        "sampler": make_sampler(temp=args.temperature, top_k=args.top_k, top_p=args.top_p),
        "logits_processors": make_logits_processors(
            logit_bias=logit_bias,
            presence_penalty=args.presence_penalty,
            presence_context_size=args.presence_context,
        ),
    }

    done = set()
    if args.resume and Path(args.dump_jsonl).exists():
        with open(args.dump_jsonl) as f:
            done = {(r["task"], r["idx"]) for r in map(json.loads, f)}
        print(f"  resume: {len(done)} problems already in {args.dump_jsonl}")

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

    dump = open(args.dump_jsonl, "a") if args.dump_jsonl else None
    results, totals, aux_by_task = {}, {}, {}
    wall_t0 = time.perf_counter()
    for tname in task_names:
        print(f"\n=== {tname} ===")
        task = TASK_CTORS[tname]()
        if task.eval_type == "categorical":
            cap = args.max_problems_cat if args.max_problems_cat is not None else args.max_problems
            acc, n = run_categorical(task, model, tokenizer,
                                      cap, args.no_system_prompt,
                                      delta_mem_attached)
            aux = None
        else:
            cap = args.max_problems_gen if args.max_problems_gen is not None else args.max_problems
            acc, n, aux = run_generative(task, model, tokenizer,
                                          cap, args.max_new_tokens,
                                          gen_kwargs, args.no_system_prompt,
                                          args.lenient_extract, delta_mem_attached,
                                          dump, only_ids and only_ids.get(tname),
                                          seed=args.seed,
                                          skip={i for t, i in done if t == type(task).__name__})
            if acc is None:
                continue
        results[tname] = acc
        totals[tname] = n
        if aux is not None:
            aux_by_task[tname] = aux
        aux_tail = f" [f1={aux['f1_mean']:.3f} em={aux['em_mean']:.3f}]" if aux else ""
        print(f"  {tname}: {acc*100:.2f}% ({n} problems){aux_tail}")
    wall_total = time.perf_counter() - wall_t0
    if dump is not None:
        dump.close()

    cc = chatcore(results)
    print(f"\n{'='*50}")
    print(f"Summary — {model_id}")
    print(f"{'='*50}")
    # Iterate task_names so non-ALL_TASKS (e.g. MBPP) show up when requested,
    # but compute ChatCORE only when all six ALL_TASKS are present.
    for t in task_names:
        if t in results:
            base = BASELINE_ACC[t]
            cent = (results[t] - base) / (1.0 - base)
            print(f"  {t:<16} {results[t]*100:6.2f}%  centered={cent:+.4f}  (n={totals[t]})")
    if cc is not None:
        print(f"  {'ChatCORE':<16} {cc:.4f}")
    print(f"  {'wall':<16} {wall_total:.1f}s")

    if args.output:
        write_report(args.output, model_id, results, totals, args, wall_total, cc, aux_by_task)
        print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()
