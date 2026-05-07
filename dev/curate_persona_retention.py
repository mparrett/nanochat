"""
Synthesize persona-retention multi-turn conversations via the Claude CLI
(headless mode, ``claude -p ... --json-schema ...``). Uses your active
Claude Code subscription auth — no ``ANTHROPIC_API_KEY`` needed.

Pattern lifted from the elixir-explore/pulse sibling project
(``lib/pulse/claude/{suggested_responses,sentiment,surprise}.ex``):

    claude -p '<prompt>' \\
      --model haiku --tools "" --no-session-persistence \\
      --output-format json \\
      --system-prompt '<system>' \\
      --json-schema '<json-schema>'

Reads the envelope's ``structured_output`` field for schema-enforced JSON.

Output is two CustomJSON-shaped JSONL files written under
``$NANOCHAT_BASE_DIR``:

- ``persona_retention_v1.jsonl``      — N=300 training rows
- ``persona_retention_v1_eval.jsonl`` — N=30 held-out rows (different personas)
- ``persona_retention_v1.log``        — generation log + rejection reasons

Schema enforced per row:
- 6 turns alternating user/assistant
- T1: user introduces name + role + location (+ optional interest)
- T2-T4: assistant engages naturally without restating persona
- T5: user asks the assistant to recall name and role
- T6: assistant correctly recalls all persona details (substring-validated
  client-side after schema enforcement)

Run:
    uv run python -m dev.curate_persona_retention --dry-run    # 1 batch
    uv run python -m dev.curate_persona_retention              # full run
"""

import argparse
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from typing import List, Optional

from pydantic import BaseModel, Field

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer

# ---------------------------------------------------------------------------
# Schema (Pydantic — for client-side typed access; the CLI's --json-schema
# enforces shape on Claude's side via JSON Schema below)


class Turn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class Conversation(BaseModel):
    name: str = Field(..., description="The user's first name introduced at T1")
    role: str = Field(..., description="The user's role/occupation introduced at T1")
    location: str = Field(..., description="The user's location/city introduced at T1")
    interest: Optional[str] = Field(None, description="Optional extra interest/hobby")
    turns: List[Turn] = Field(..., description="Six turns alternating user/assistant")


# JSON Schema passed to the CLI via --json-schema. Each batch generates N
# conversations packed into a Batch object.
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
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "location": {"type": "string"},
                        "interest": {"type": ["string", "null"]},
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
                    "required": ["name", "role", "location", "turns"],
                },
            }
        },
        "required": ["conversations"],
    }


# ---------------------------------------------------------------------------
# Prompts

SYSTEM_PROMPT = """You generate multi-turn conversation training data for a small chatbot.

Each conversation must follow this exact shape (6 turns alternating user/assistant):

  Turn 1 (user): introduces themselves with name + role/occupation + location, sometimes plus an optional interest. Casual phrasing varied across conversations.
  Turn 2 (assistant): natural friendly engagement WITHOUT restating any persona detail.
  Turn 3 (user): chit-chat that does not restate the persona.
  Turn 4 (assistant): natural reply, no persona restating.
  Turn 5 (user): asks the assistant to recall name and role/occupation. Vary the phrasing across conversations: "what's my name again?", "do you remember what I do?", "remind me what we said about who I am?", "quick check — what was my name and job?", "wait, what did I tell you about myself?".
  Turn 6 (assistant): correctly recalls ALL persona details introduced at T1 (name, role, location, plus interest if given). Phrased naturally, not robotically.

Constraints:
- Vary names, roles, locations widely across the batch — names from many regions and ethnicities, roles spanning trades/professions/creative/technical, locations spanning many countries and cities.
- Avoid math-shaped persona fields (no "I count widgets per day", no numeric quantities in the persona itself). Numbers in middle-turn chit-chat are fine.
- Avoid restating the persona in turns 2-4. The assistant must engage without parroting back what the user said about themselves.
- Turn 6 must contain the EXACT persona details — same name, same role, same location. The substring of each must literally appear in the assistant's T6 reply.
- Turns are short and natural — most turns 1-3 sentences, T6 can be slightly longer to fit all the recall.
- Vary T1 phrasings: "Hi! I'm X, a Y in Z.", "Hey — name's X, work as Y here in Z.", "X here, Y based in Z.", etc.

Avoid these specific names (reserved for the held-out eval set): Alex, Sydney, Portland, Java, Canberra, software engineer, Tokyo, Bangalore."""


def build_user_prompt(
    n: int,
    used_names: list[str],
    used_roles: list[str],
    used_locations: list[str],
    with_interest_fraction: float = 0.4,
) -> str:
    name_avoid = sorted(set(used_names))[-30:]
    role_avoid = sorted(set(used_roles))[-20:]
    loc_avoid = sorted(set(used_locations))[-20:]
    parts = [
        f"Generate {n} persona-retention conversations following the schema you've been given.",
        "",
        f"~{int(with_interest_fraction * 100)}% should include an `interest` field; the rest should set interest to null.",
        "",
        "Diversity constraints — DO NOT reuse any of these (already in dataset):",
    ]
    if name_avoid:
        parts.append(f"  Names already used: {', '.join(name_avoid)}")
    if role_avoid:
        parts.append(f"  Roles already used: {', '.join(role_avoid)}")
    if loc_avoid:
        parts.append(f"  Locations already used: {', '.join(loc_avoid)}")
    parts.append("")
    parts.append(
        "Return as a Batch object with the `conversations` array of length exactly "
        + str(n)
        + "."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI invocation


def call_claude_cli(
    user_prompt: str,
    json_schema: dict,
    *,
    model: str = "haiku",
    timeout_s: int = 180,
) -> tuple[Optional[dict], dict]:
    """Invoke ``claude -p ... --json-schema ...`` and return
    (structured_output_dict, envelope).

    Pattern matches pulse's call_cli (lib/pulse/claude/*.ex).
    Returns (None, envelope) on parse / API failure.
    """
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


# ---------------------------------------------------------------------------
# Validation


def validate_conversation(c: Conversation) -> tuple[bool, str]:
    """Reject only on hard structural failures: wrong turn count, wrong
    role pattern, T1 missing persona fields, T6 missing persona fields.

    Middle-turn persona mentions are allowed: natural assistant replies
    often echo a user's role/location without parroting, and forbidding
    that pushes the data away from natural conversational shape. The
    failure mode L1 targets is *not recalling at T6*, not *succeeding via
    short-range attention* — so the validator focuses on T6 correctness.
    """
    if len(c.turns) != 6:
        return False, f"expected 6 turns, got {len(c.turns)}"
    expected_roles = ["user", "assistant", "user", "assistant", "user", "assistant"]
    actual_roles = [t.role for t in c.turns]
    if actual_roles != expected_roles:
        return False, f"role pattern wrong: {actual_roles}"
    t1 = c.turns[0].content
    for field, value in (("name", c.name), ("role", c.role), ("location", c.location)):
        if value.lower() not in t1.lower():
            return False, f"T1 doesn't contain {field}={value!r}"
    t6 = c.turns[5].content
    for field, value in (("name", c.name), ("role", c.role), ("location", c.location)):
        if value.lower() not in t6.lower():
            return False, f"T6 doesn't recall {field}={value!r}"
    if c.interest and c.interest.lower() not in t6.lower():
        return False, f"T6 doesn't recall interest={c.interest!r}"
    return True, ""


def conversation_to_customjson(c: Conversation, include_metadata: bool = False):
    """Render a Conversation as a CustomJSON row.

    CustomJSON expects either a list of {role, content} dicts or a dict
    with "messages" key. tasks/customjson.py iterates these directly.
    """
    msgs = [{"role": t.role, "content": t.content} for t in c.turns]
    if include_metadata:
        return {
            "messages": msgs,
            "_persona": {
                "name": c.name,
                "role": c.role,
                "location": c.location,
                "interest": c.interest,
            },
        }
    return msgs


# ---------------------------------------------------------------------------
# CLI entry point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-eval", type=int, default=30)
    parser.add_argument(
        "--per-batch",
        type=int,
        default=10,
        help="conversations per claude invocation",
    )
    parser.add_argument("--with-interest-fraction", type=float, default=0.4)
    parser.add_argument(
        "--model",
        default="haiku",
        help="claude CLI model alias (haiku|sonnet|opus). Default: haiku.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="run a single batch and print sample, don't write output")
    parser.add_argument("--output-dir", default=None,
                        help="output directory (default: $NANOCHAT_BASE_DIR)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=int, default=180,
                        help="per-call subprocess timeout")
    args = parser.parse_args()

    random.seed(args.seed)

    if shutil.which("claude") is None:
        sys.exit("ERROR: `claude` CLI not found on PATH. Install Claude Code first.")

    out_dir = args.output_dir or get_base_dir()
    train_path = os.path.join(out_dir, "persona_retention_v1.jsonl")
    eval_path = os.path.join(out_dir, "persona_retention_v1_eval.jsonl")
    log_path = os.path.join(out_dir, "persona_retention_v1.log")

    used_names: list[str] = []
    used_roles: list[str] = []
    used_locations: list[str] = []
    train_rows: list[Conversation] = []
    eval_rows: list[Conversation] = []
    rejected: list[tuple[str, str]] = []
    total_cost_usd = 0.0
    api_calls = 0
    cli_failures = 0

    def run_batch(target_pool: list[Conversation], n: int, label: str):
        nonlocal total_cost_usd, api_calls, cli_failures
        user_prompt = build_user_prompt(
            n,
            used_names,
            used_roles,
            used_locations,
            args.with_interest_fraction,
        )
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
            target_pool.append(c)
            used_names.append(c.name)
            used_roles.append(c.role)
            used_locations.append(c.location)
            accepted += 1
        cum_cost = f"${total_cost_usd:.3f}" if total_cost_usd > 0 else "n/a"
        print(
            f"  [{label}] accepted {accepted}/{len(convs)} "
            f"(rejects so far: {len(rejected)}; cost so far: {cum_cost})"
        )

    if args.dry_run:
        print("=== DRY RUN: 1 batch ===")
        run_batch(train_rows, args.per_batch, "dry")
        if train_rows:
            print()
            print("Sample conversation:")
            sample = train_rows[0]
            print(f"  persona: name={sample.name} role={sample.role} "
                  f"location={sample.location} interest={sample.interest}")
            for i, t in enumerate(sample.turns, start=1):
                snippet = t.content if len(t.content) <= 200 else t.content[:200] + "..."
                print(f"  T{i} [{t.role:<9}] {snippet}")
        if rejected:
            print()
            print(f"Rejected {len(rejected)} sample(s); first reasons:")
            for _, reason in rejected[:5]:
                print(f"  - {reason}")
        sys.exit(0 if train_rows else 1)

    print(f"=== Training set: target {args.n_train} ===")
    while len(train_rows) < args.n_train:
        n_needed = min(args.per_batch, args.n_train - len(train_rows))
        run_batch(train_rows, n_needed, f"train {len(train_rows):>3}/{args.n_train}")
        if cli_failures >= 5:
            sys.exit(f"Aborting: {cli_failures} CLI failures during training set")
        time.sleep(0.2)

    print()
    print(f"=== Eval set: target {args.n_eval} (held-out personas) ===")
    while len(eval_rows) < args.n_eval:
        n_needed = min(args.per_batch, args.n_eval - len(eval_rows))
        run_batch(eval_rows, n_needed, f"eval  {len(eval_rows):>3}/{args.n_eval}")
        if cli_failures >= 8:
            sys.exit(f"Aborting: {cli_failures} CLI failures total")
        time.sleep(0.2)

    os.makedirs(out_dir, exist_ok=True)
    with open(train_path, "w") as f:
        for c in train_rows:
            row = conversation_to_customjson(c, include_metadata=False)
            f.write(json.dumps(row) + "\n")
    with open(eval_path, "w") as f:
        for c in eval_rows:
            row = conversation_to_customjson(c, include_metadata=True)
            f.write(json.dumps(row) + "\n")

    tokenizer = get_tokenizer()
    lens = []
    for c in train_rows:
        msgs = conversation_to_customjson(c, include_metadata=False)
        # render_conversation expects a {"messages": [...]} dict, matching
        # the shape CustomJSON.get_example returns.
        ids, _ = tokenizer.render_conversation({"messages": msgs})
        lens.append(len(ids))
    lens.sort()
    print()
    print("=== Token-length distribution (train rows) ===")
    print(
        f"  min={lens[0]}  median={lens[len(lens)//2]}  "
        f"p90={lens[int(len(lens)*0.9)]}  max={lens[-1]}"
    )
    if lens[-1] > 480:
        print(
            f"  WARNING: max token length {lens[-1]} approaches max_seq_len=512"
        )

    print()
    print("=== Diversity ===")
    print(f"  unique names    : {len(set(used_names[:len(train_rows)+len(eval_rows)]))}")
    print(f"  unique roles    : {len(set(used_roles[:len(train_rows)+len(eval_rows)]))}")
    print(f"  unique locations: {len(set(used_locations[:len(train_rows)+len(eval_rows)]))}")

    print()
    print("=== CLI usage ===")
    print(f"  api_calls    : {api_calls}")
    print(f"  cli_failures : {cli_failures}")
    print(f"  total cost   : ${total_cost_usd:.3f} USD (charged to your subscription)")

    with open(log_path, "w") as f:
        f.write(f"# persona_retention_v1 generation log\n")
        f.write(f"# generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"train_rows: {len(train_rows)}\n")
        f.write(f"eval_rows: {len(eval_rows)}\n")
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
    print(f"Wrote {len(eval_rows)} eval rows  → {eval_path}")
    print(f"Wrote log              → {log_path}")


if __name__ == "__main__":
    main()
