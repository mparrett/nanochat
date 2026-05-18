"""
Evaluate the Chat model on the MBPP coding benchmark.

MBPP (Mostly Basic Python Problems) — 500-problem test split of the standard
HF `mbpp` config="full". Each problem is a short natural-language description
+ a list of assert-statement test cases. The model must produce a Python
function that passes the tests.

Companion to tasks/humaneval.py; same Task interface, same execute_code
runner. Prompt format follows the canonical zero-shot MBPP template used
across the MBPP literature: problem text + three test cases as signature
specification.
"""

import re
from datasets import load_dataset
from nanochat.execution import execute_code
from tasks.common import Task


def extract_program(completion):
    """
    Extract Python code from LLM completion. Same logic as tasks/humaneval.py:
    prefer the first ```python fenced block if present, else the whole
    completion verbatim.
    """
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        return matches[0].strip()
    return completion.strip()


def build_prompt(text, test_list):
    """Canonical zero-shot MBPP prompt. Includes all test cases as signature
    specification — without them the model has to guess function names."""
    tests_block = "\n".join(test_list)
    return (
        f"You are an expert Python programmer, and here is your task: "
        f"{text}\nYour code should pass these tests:\n{tests_block}"
    )


class MBPP(Task):

    def __init__(self, split="test", **kwargs):
        super().__init__(**kwargs)
        # config="full" gives the canonical 500-problem test split.
        # Shuffled for reproducibility under -x N slicing.
        self.ds = load_dataset("mbpp", "full", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        text = row['text']
        test_list = row['test_list']
        test_setup = row.get('test_setup_code', '') or ''
        prompt_str = build_prompt(text, test_list)
        # Reference solution = canonical code + setup + tests, used only as
        # the gold assistant turn for prompt rendering (we render with the
        # assistant turn stripped in render_prompt).
        solution = row['code']
        complete_solution = solution if not test_setup else f"{test_setup}\n\n{solution}"
        messages = [
            {"role": "user", "content": prompt_str},
            {"role": "assistant", "content": complete_solution},
        ]
        conversation = {
            "messages": messages,
            "test_list": test_list,
            "test_setup_code": test_setup,
        }
        return conversation

    def evaluate(self, conversation, completion):
        """Execute model's code + setup + tests; success iff all assertions pass."""
        completion_code = extract_program(completion)
        test_setup = conversation.get("test_setup_code", "") or ""
        test_block = "\n".join(conversation["test_list"])
        program_parts = [test_setup, completion_code, test_block]
        program = "\n\n".join(p for p in program_parts if p.strip())
        result = execute_code(program)
        return result.success
