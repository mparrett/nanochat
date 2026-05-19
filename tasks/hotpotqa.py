"""
Evaluate the Chat model on the HotpotQA multi-hop QA benchmark.

HotpotQA distractor split — 7,405-example validation set. Each example
gives a question, 10 paragraphs (8 distractor + 2 supporting), and a
short gold answer. The model must read the paragraphs and answer.

This is δ-mem's home-axis test: long-context recall + multi-hop reasoning,
the regime the paper claims as the primary use case. Field-result writeup
flagged this in §07 as the un-run paper-native task; this is that run.

Evaluation: standard HotpotQA scoring. Normalize (lowercase, strip
articles, remove punctuation, collapse whitespace), tokenize on
whitespace, compute F1 and EM. We expose F1 + EM via task attributes
so the harness can report both; the boolean returned by `evaluate()` is
F1 >= 0.5 (the conventional "correct enough" threshold used by SQuAD-
style benchmarks).
"""

import re
import string
from collections import Counter

from datasets import load_dataset
from tasks.common import Task


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WS = re.compile(r"\s+")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s):
    """Standard HotpotQA / SQuAD answer normalization."""
    s = s.lower()
    s = s.translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def f1_score(pred, gold):
    """Token-level F1 between normalized prediction and gold."""
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)


def exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def extract_answer(completion):
    """Pull the model's short answer out of a (potentially chatty) completion.

    Strategy: prefer the line after an explicit 'Answer:' marker; else
    take the first non-empty line; trim quotes/punctuation tails.
    """
    m = re.search(r"(?i)answer\s*[:=]\s*(.+?)(?:\n|$)", completion)
    if m:
        ans = m.group(1).strip()
    else:
        for line in completion.strip().splitlines():
            line = line.strip()
            if line:
                ans = line
                break
        else:
            ans = completion.strip()
    return ans.strip().strip('"\'').rstrip(".")


def build_prompt(question, context):
    """10-passage context block + question. Each passage prefixed by its
    title in brackets; sentences joined with spaces."""
    passages = []
    for title, sents in zip(context["title"], context["sentences"]):
        body = " ".join(s.strip() for s in sents)
        passages.append(f"[{title}] {body}")
    ctx = "\n\n".join(passages)
    return (
        f"Read the following passages and answer the question with a short "
        f"factual answer (a name, a yes/no, a number, or a few words).\n\n"
        f"{ctx}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


class HotpotQA(Task):

    def __init__(self, split="validation", **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("hotpot_qa", "distractor", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt_str = build_prompt(row["question"], row["context"])
        messages = [
            {"role": "user", "content": prompt_str},
            {"role": "assistant", "content": row["answer"]},
        ]
        return {
            "messages": messages,
            "gold_answer": row["answer"],
            "qtype": row["type"],
            "level": row["level"],
        }

    def evaluate(self, conversation, completion):
        """Return F1 >= 0.5 as the per-example success bool. Also stashes
        F1 and EM on the conversation dict so the caller can read them
        back if it wants mean F1 / EM-strict reporting."""
        pred = extract_answer(completion)
        gold = conversation["gold_answer"]
        f1 = f1_score(pred, gold)
        em = exact_match(pred, gold)
        conversation["_f1"] = f1
        conversation["_em"] = em
        conversation["_pred"] = pred
        return f1 >= 0.5
