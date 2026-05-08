"""
Build the v3 LoRA training mix: persona-retention (T6-phrasing-diverse) +
chit-chat. Concat, shuffle with a fixed seed, write a single CustomJSON
JSONL that ``scripts/chat_sft_lora.py --data-path`` consumes.

Inputs (under ``$NANOCHAT_BASE_DIR``):
  - ``persona_retention_v1_diverse.jsonl``  (300 rows, fresh curation)
  - ``chit_chat_v1.jsonl``                  ( 75 rows)

Output:
  - ``lora_v3_train.jsonl``                 (375 rows, shuffled)
  - ``lora_v3_train.meta.json``             (mix counts, seed, source paths)

The 80/20 mix targets the template-bleed failure characterised in
HANDOFF.md Day 2026-05-07 + Postscript: the v2 LoRA stamps a
"You're [Name]…" persona-recall template onto every prompt because it
saw only one shape during training. Diluting persona-recall with
non-recall chit-chat is the handoff's prescribed v3 fix.

Run:
    uv run python -m dev.build_lora_v3_dataset
"""

import argparse
import json
import os
import random

from nanochat.common import get_base_dir


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona-path", default=None,
                        help="default: $NANOCHAT_BASE_DIR/persona_retention_v1_diverse.jsonl")
    parser.add_argument("--chitchat-path", default=None,
                        help="default: $NANOCHAT_BASE_DIR/chit_chat_v1.jsonl")
    parser.add_argument("--output-tag", default="v3",
                        help="output filename suffix → lora_<tag>_train.jsonl (default: v3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    base = args.output_dir or get_base_dir()
    persona_path = args.persona_path or os.path.join(base, "persona_retention_v1_diverse.jsonl")
    chitchat_path = args.chitchat_path or os.path.join(base, "chit_chat_v1.jsonl")
    out_path = os.path.join(base, f"lora_{args.output_tag}_train.jsonl")
    meta_path = os.path.join(base, f"lora_{args.output_tag}_train.meta.json")

    persona = load_jsonl(persona_path)
    chitchat = load_jsonl(chitchat_path)
    print(f"Loaded persona  : {len(persona):>4} rows from {persona_path}")
    print(f"Loaded chit-chat: {len(chitchat):>4} rows from {chitchat_path}")

    combined = persona + chitchat
    rng = random.Random(args.seed)
    rng.shuffle(combined)

    # Validate every row has alternating user/assistant 6-turn shape.
    expected = ["user", "assistant", "user", "assistant", "user", "assistant"]
    for i, row in enumerate(combined):
        msgs = row if isinstance(row, list) else row.get("messages", [])
        roles = [m["role"] for m in msgs]
        if roles != expected:
            raise SystemExit(f"Row {i} has bad role pattern: {roles}")

    with open(out_path, "w") as f:
        for row in combined:
            f.write(json.dumps(row) + "\n")

    meta = {
        "output_path": out_path,
        "total_rows": len(combined),
        "persona_rows": len(persona),
        "chitchat_rows": len(chitchat),
        "persona_fraction": round(len(persona) / len(combined), 4),
        "chitchat_fraction": round(len(chitchat) / len(combined), 4),
        "seed": args.seed,
        "sources": {"persona": persona_path, "chitchat": chitchat_path},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"Wrote {len(combined)} rows → {out_path}")
    print(f"Wrote meta            → {meta_path}")
    print(f"  persona  : {len(persona)}/{len(combined)} = {meta['persona_fraction']:.1%}")
    print(f"  chit-chat: {len(chitchat)}/{len(combined)} = {meta['chitchat_fraction']:.1%}")


if __name__ == "__main__":
    main()
