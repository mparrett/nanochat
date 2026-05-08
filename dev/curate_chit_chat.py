"""
Synthesize 6-turn casual chit-chat conversations via the Claude CLI.

Sibling of ``curate_persona_retention.py``; same CLI pattern, different
schema shape and SYSTEM_PROMPT. Used to dilute persona-only LoRA training
data so the adapter learns multiple conversational shapes instead of
stamping a single persona-recall template onto every prompt
(template-bleed; see HANDOFF.md Day 2026-05-07 + Postscript).

Each conversation:
- 6 turns alternating user/assistant
- T1: user opens a casual exchange WITHOUT introducing a persona (no
  "Hi I'm X, a Y in Z" shape). Topics: hobbies, weather, recipes, news,
  work, travel, current events, etc.
- T2-T4: natural conversational back-and-forth.
- T5: user asks a normal question / makes a normal comment.
  NOT a recall question.
- T6: assistant gives a natural, helpful, on-topic reply.
  NOT a "You're [Name]…" recall response.

Output (under ``$NANOCHAT_BASE_DIR``):
- ``chit_chat_v1.jsonl`` — N=75 training rows (CustomJSON-shaped)
- ``chit_chat_v1.log``   — generation log

Run:
    uv run python -m dev.curate_chit_chat --dry-run     # 1 batch
    uv run python -m dev.curate_chit_chat               # full run
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import List, Optional

from pydantic import BaseModel

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer


class Turn(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    topic: str  # short tag for diversity tracking
    turns: List[Turn]


def build_json_schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "conversations": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "turns": {
                            "type": "array",
                            "minItems": 6,
                            "maxItems": 6,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "enum": ["user", "assistant"],
                                    },
                                    "content": {"type": "string"},
                                },
                                "required": ["role", "content"],
                            },
                        },
                    },
                    "required": ["topic", "turns"],
                },
            }
        },
        "required": ["conversations"],
    }


SYSTEM_PROMPT = """You generate 6-turn casual chit-chat training data for a small chatbot.

Each conversation must follow this exact shape (6 turns alternating user/assistant):

  Turn 1 (user): opens a casual conversation. DO NOT introduce a persona — DO NOT say "I'm [Name]", "I'm a [Role]", or "I live in [City]". Just say hi, ask a question, share a thought, etc.
  Turn 2 (assistant): natural friendly reply that engages with T1's actual content.
  Turn 3 (user): natural follow-up — a question, a related thought, a slight topic shift.
  Turn 4 (assistant): natural reply.
  Turn 5 (user): another question or comment. NOT a recall question — DO NOT have the user ask "what's my name", "what do I do", or anything that would prompt a persona-recall reply.
  Turn 6 (assistant): natural, helpful, on-topic reply that engages T5. CRITICAL: T6 must NOT begin with "You're", "You are", or any persona-recall pattern. T6 should answer T5 directly or extend the conversation naturally.

Topic diversity (vary widely across the batch):
  - Recipes / cooking ("how do you make a good pesto?")
  - Weather / seasons ("it's been raining all week here")
  - Hobbies — books, music, gardening, sports, gaming
  - Travel — places, recommendations, planning
  - Current events / news (avoid politics)
  - Work-flavoured chit-chat ("any tips for staying focused on long projects?")
  - Animals / pets
  - Movies, shows, games
  - Tech tips ("what's a good way to back up photos?")
  - Random factual questions ("why is the sky blue?", "what makes bread rise?")
  - General greetings and small talk
  - Advice questions ("how do I stay motivated to exercise?")

Tone: friendly, brief, natural. Most turns 1-3 sentences. No corporate-bot voice ("I'd be happy to help you with that!" — avoid). Just casual, useful, on-topic dialogue.

Vary the OPENING SHAPE of T1 across the batch — DO NOT default to "Hi!" or "Hey!" for every row. Some openers can be a direct question with no greeting; some can be a one-line statement; some can start with "So…" or "Quick question —" or just dive in with an observation.

Vary the OPENING SHAPE of T6 across the batch — across 10 conversations, no two T6 replies may begin with the same 3 words.

Hard constraints:
- T1 must NOT contain "I'm a", "I am a", "name's", "I work as", "I live in", "I'm based in", or any other persona-introduction shape.
- T6 must NOT begin with "You're", "You are", "Your name", "You mentioned", or any recall pattern."""


def build_user_prompt(n: int, used_topics: list[str]) -> str:
    avoid = sorted(set(used_topics))[-30:]
    parts = [
        f"Generate {n} casual chit-chat conversations following the schema you've been given.",
        "",
        "Vary the topics widely across the batch — pull from the topic categories listed.",
    ]
    if avoid:
        parts.append("")
        parts.append(f"Topics already covered (avoid duplicates): {', '.join(avoid)}")
    parts.append("")
    parts.append(
        f"Return as a Batch object with the `conversations` array of length exactly {n}."
    )
    return "\n".join(parts)


def call_claude_cli(
    user_prompt: str,
    json_schema: dict,
    *,
    model: str = "haiku",
    timeout_s: int = 180,
) -> tuple[Optional[dict], dict]:
    cmd = [
        "claude",
        "-p",
        user_prompt,
        "--model",
        model,
        "--tools",
        "",
        "--no-session-persistence",
        "--output-format",
        "json",
        "--system-prompt",
        SYSTEM_PROMPT,
        "--json-schema",
        json.dumps(json_schema),
    ]
    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        return None, {"_cli_error": True, "stderr": proc.stderr, "rc": proc.returncode}
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return None, {"_cli_error": True, "json_error": str(e), "stdout": proc.stdout[:500]}
    if envelope.get("is_error"):
        return None, envelope
    return envelope.get("structured_output"), envelope


# Negative patterns for hard validation. The SYSTEM_PROMPT instructs Claude to
# avoid these; the validator is the safety net for the cases it ignores.
T1_PERSONA_PATTERNS = [
    re.compile(r"\bI['’]?m\s+a\b", re.I),
    re.compile(r"\bI\s+am\s+a\b", re.I),
    re.compile(r"\bI\s+work\s+as\b", re.I),
    re.compile(r"\bI\s+live\s+in\b", re.I),
    re.compile(r"\bI['’]?m\s+based\s+in\b", re.I),
    re.compile(r"\bname['’]?s\s+\w+", re.I),
    re.compile(r"\bmy\s+name\s+is\b", re.I),
]
T5_RECALL_PATTERNS = [
    re.compile(r"\bwhat['’]?s\s+my\s+name\b", re.I),
    re.compile(r"\bdo\s+you\s+remember\s+(my\s+)?name\b", re.I),
    re.compile(r"\bremind\s+me\s+who\s+I\s+am\b", re.I),
    re.compile(r"\bwhat\s+do\s+I\s+do\b", re.I),
]
T6_OPENERS_BANNED = [
    re.compile(r"^\s*you['’]?re\b", re.I),
    re.compile(r"^\s*you\s+are\b", re.I),
    re.compile(r"^\s*your\s+name\b", re.I),
    re.compile(r"^\s*you\s+mentioned\b", re.I),
]


def validate_conversation(c: Conversation) -> tuple[bool, str]:
    if len(c.turns) != 6:
        return False, f"expected 6 turns, got {len(c.turns)}"
    expected_roles = ["user", "assistant", "user", "assistant", "user", "assistant"]
    if [t.role for t in c.turns] != expected_roles:
        return False, f"role pattern wrong: {[t.role for t in c.turns]}"
    t1, t5, t6 = c.turns[0].content, c.turns[4].content, c.turns[5].content
    for pat in T1_PERSONA_PATTERNS:
        if pat.search(t1):
            return False, f"T1 looks persona-shaped: {pat.pattern!r} matched"
    for pat in T5_RECALL_PATTERNS:
        if pat.search(t5):
            return False, f"T5 looks recall-shaped: {pat.pattern!r} matched"
    for pat in T6_OPENERS_BANNED:
        if pat.match(t6):
            return False, f"T6 starts with banned opener: {pat.pattern!r}"
    return True, ""


def conversation_to_customjson(c: Conversation):
    return [{"role": t.role, "content": t.content} for t in c.turns]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=75)
    parser.add_argument("--per-batch", type=int, default=10)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--tag", default="v1",
                        help="filename suffix (default: v1 → chit_chat_v1.jsonl)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--timeout-s", type=int, default=180)
    args = parser.parse_args()

    if shutil.which("claude") is None:
        sys.exit("ERROR: `claude` CLI not found on PATH. Install Claude Code first.")

    out_dir = args.output_dir or get_base_dir()
    train_path = os.path.join(out_dir, f"chit_chat_{args.tag}.jsonl")
    log_path = os.path.join(out_dir, f"chit_chat_{args.tag}.log")

    train_rows: list[Conversation] = []
    used_topics: list[str] = []
    rejected: list[tuple[str, str]] = []
    total_cost_usd = 0.0
    api_calls = 0
    cli_failures = 0

    def run_batch(n: int, label: str):
        nonlocal total_cost_usd, api_calls, cli_failures
        user_prompt = build_user_prompt(n, used_topics)
        api_calls += 1
        try:
            structured, envelope = call_claude_cli(
                user_prompt,
                build_json_schema(n),
                model=args.model,
                timeout_s=args.timeout_s,
            )
        except subprocess.TimeoutExpired:
            cli_failures += 1
            print(f"  [{label}] timeout after {args.timeout_s}s", file=sys.stderr)
            return
        cost = envelope.get("total_cost_usd")
        if cost is not None:
            total_cost_usd += cost
        if structured is None:
            cli_failures += 1
            err = envelope.get("stderr") or envelope.get("json_error") or "unknown"
            print(f"  [{label}] CLI failure: {err[:200] if isinstance(err, str) else err}",
                  file=sys.stderr)
            return
        try:
            convs = [Conversation.model_validate(c) for c in structured.get("conversations", [])]
        except Exception as e:
            cli_failures += 1
            print(f"  [{label}] schema validation: {e}", file=sys.stderr)
            return
        accepted = 0
        for c in convs:
            ok, reason = validate_conversation(c)
            if not ok:
                rejected.append((c.model_dump_json(), reason))
                continue
            train_rows.append(c)
            used_topics.append(c.topic)
            accepted += 1
        cum_cost = f"${total_cost_usd:.3f}" if total_cost_usd > 0 else "n/a"
        print(
            f"  [{label}] accepted {accepted}/{len(convs)} "
            f"(rejects so far: {len(rejected)}; cost so far: {cum_cost})"
        )

    if args.dry_run:
        print("=== DRY RUN: 1 batch ===")
        run_batch(args.per_batch, "dry")
        if train_rows:
            print()
            print(f"Sample (1 of {len(train_rows)}):")
            sample = train_rows[0]
            print(f"  topic: {sample.topic}")
            for i, t in enumerate(sample.turns, start=1):
                snippet = t.content if len(t.content) <= 200 else t.content[:200] + "..."
                print(f"  T{i} [{t.role:<9}] {snippet}")
            print()
            print("All T6 openers (first 80 chars):")
            for i, c in enumerate(train_rows, 1):
                print(f"  {i:2}. {c.turns[5].content[:80]}")
        if rejected:
            print()
            print(f"Rejected {len(rejected)} sample(s); first reasons:")
            for _, reason in rejected[:5]:
                print(f"  - {reason}")
        sys.exit(0 if train_rows else 1)

    print(f"=== Chit-chat training set: target {args.n_train} ===")
    while len(train_rows) < args.n_train:
        n_needed = min(args.per_batch, args.n_train - len(train_rows))
        run_batch(n_needed, f"train {len(train_rows):>3}/{args.n_train}")
        if cli_failures >= 5:
            sys.exit(f"Aborting: {cli_failures} CLI failures")
        time.sleep(0.2)

    os.makedirs(out_dir, exist_ok=True)
    with open(train_path, "w") as f:
        for c in train_rows:
            f.write(json.dumps(conversation_to_customjson(c)) + "\n")

    tokenizer = get_tokenizer()
    lens = []
    for c in train_rows:
        msgs = conversation_to_customjson(c)
        ids, _ = tokenizer.render_conversation({"messages": msgs})
        lens.append(len(ids))
    lens.sort()
    print()
    print("=== Token-length distribution ===")
    print(f"  min={lens[0]}  median={lens[len(lens)//2]}  "
          f"p90={lens[int(len(lens)*0.9)]}  max={lens[-1]}")
    if lens[-1] > 480:
        print(f"  WARNING: max token length {lens[-1]} approaches max_seq_len=512")

    print()
    print("=== Diversity ===")
    print(f"  unique topics: {len(set(used_topics))}")

    print()
    print("=== CLI usage ===")
    print(f"  api_calls    : {api_calls}")
    print(f"  cli_failures : {cli_failures}")
    print(f"  rejected     : {len(rejected)}")
    print(f"  total cost   : ${total_cost_usd:.3f} USD")

    with open(log_path, "w") as f:
        f.write(f"# chit_chat_{args.tag} generation log\n")
        f.write(f"# generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"train_rows: {len(train_rows)}\n")
        f.write(f"rejected: {len(rejected)}\n")
        f.write(f"cli_failures: {cli_failures}\n")
        f.write(f"total_cost_usd: {total_cost_usd:.4f}\n\n")
        if rejected:
            f.write("# First 10 rejection reasons:\n")
            for raw, reason in rejected[:10]:
                f.write(f"  - {reason}\n")
                f.write(f"    raw: {raw[:200]}\n")

    print()
    print(f"Wrote {len(train_rows)} train rows → {train_path}")
    print(f"Wrote log              → {log_path}")


if __name__ == "__main__":
    main()
