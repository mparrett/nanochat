"""
Adapt persona_retention_v1{,_eval}.jsonl into NTK-Mirror's prompt/completion
JSONL format, rendered via Qwen2.5's chat template.

Input rows are OAI-style messages (alternating user/assistant, 6 turns). T6 is
the assistant's persona-recall answer.

Train output: one row per conversation, prompt = chat-templated T1..T5 with
add_generation_prompt=True (ends in `<|im_start|>assistant\\n`), completion =
T6 assistant content + `<|im_end|>\\n`.

Eval output: same shape plus the `_persona` dict carried through under a
"_persona" key so the scorer can grade name/role/location.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def render_row(tok, messages: list[dict]) -> tuple[str, str]:
    assert len(messages) == 6
    assert [m["role"] for m in messages] == [
        "user", "assistant", "user", "assistant", "user", "assistant"
    ], f"unexpected role pattern: {[m['role'] for m in messages]}"
    prompt = tok.apply_chat_template(
        messages[:5], tokenize=False, add_generation_prompt=True
    )
    full = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    assert full.startswith(prompt), "template invariant broken — full does not extend prompt"
    completion = full[len(prompt):]
    return prompt, completion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument(
        "--train-in",
        default=str(Path.home() / ".cache/nanochat/persona_retention_v1.jsonl"),
    )
    ap.add_argument(
        "--eval-in",
        default=str(Path.home() / ".cache/nanochat/persona_retention_v1_eval.jsonl"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(Path.home() / "projects-new/3p/ntkmirror/runs/persona_qwen"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)

    train_rows = [json.loads(l) for l in open(args.train_in) if l.strip()]
    eval_rows = [json.loads(l) for l in open(args.eval_in) if l.strip()]
    print(f"loaded {len(train_rows)} train rows, {len(eval_rows)} eval rows")

    train_out = out_dir / "train.jsonl"
    with train_out.open("w", encoding="utf-8") as f:
        for messages in train_rows:
            prompt, completion = render_row(tok, messages)
            f.write(json.dumps({"prompt": prompt, "completion": completion},
                               ensure_ascii=False) + "\n")
    print(f"wrote {train_out}")

    eval_out = out_dir / "eval.jsonl"
    with eval_out.open("w", encoding="utf-8") as f:
        for row in eval_rows:
            prompt, completion = render_row(tok, row["messages"])
            f.write(json.dumps({
                "prompt": prompt,
                "completion": completion,
                "_persona": row["_persona"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {eval_out}")

    print("\n--- sample train row ---")
    sample = json.loads(open(train_out).readline())
    print(f"  prompt (len={len(sample['prompt'])}):")
    print(f"    ...{sample['prompt'][-200:]!r}")
    print(f"  completion (len={len(sample['completion'])}):")
    print(f"    {sample['completion']!r}")


if __name__ == "__main__":
    main()
