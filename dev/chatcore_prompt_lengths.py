"""
Count how many ChatCORE eval prompts exceed a token-length threshold.

Used to validate the hypothesis that ChatCORE's T² activation blow-up
(documented in docs/sft_oom_investigation_2026-05-03.md) is caused by
eval prompts being routinely longer than the model's training context.

Usage:
    python -m dev.chatcore_prompt_lengths
    python -m dev.chatcore_prompt_lengths --threshold 512 --max-per-task 500
    python -m dev.chatcore_prompt_lengths --threshold 1024 --tasks MMLU,GSM8K

Reports per-task: count, max length, fraction over threshold, p50/p90/p99.

Cheap diagnostic — no model load, no GPU, just tokenizer + dataset scan.
"""
import argparse
from functools import partial

import numpy as np

from nanochat.tokenizer import get_tokenizer
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.spellingbee import SpellingBee


TASK_FACTORIES = {
    'ARC-Easy': partial(ARC, subset='ARC-Easy', split='test'),
    'ARC-Challenge': partial(ARC, subset='ARC-Challenge', split='test'),
    'MMLU': partial(MMLU, subset='all', split='test'),
    'GSM8K': partial(GSM8K, subset='main', split='test'),
    'HumanEval': partial(HumanEval),
    'SpellingBee': partial(SpellingBee, size=256, split='test'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=int, default=512, help='token-length cutoff (model max_seq_len)')
    ap.add_argument('--max-per-task', type=int, default=2000, help='cap problems sampled per task (None = full)')
    ap.add_argument('--tasks', type=str, default=','.join(TASK_FACTORIES.keys()),
                    help='comma-separated task names')
    args = ap.parse_args()

    tokenizer = get_tokenizer()

    rows = []
    for task_name in args.tasks.split(','):
        task_name = task_name.strip()
        print(f'\n[{task_name}] loading...', flush=True)
        task = TASK_FACTORIES[task_name]()
        n = len(task)
        n_sample = min(n, args.max_per_task) if args.max_per_task else n
        print(f'[{task_name}] sampling {n_sample}/{n} problems', flush=True)
        lengths = []
        for i in range(n_sample):
            conv = task[i]
            ids = tokenizer.render_for_completion(conv)
            lengths.append(len(ids))
            if (i + 1) % 500 == 0:
                print(f'  {i+1}/{n_sample}...', flush=True)
        L = np.array(lengths)
        over = (L > args.threshold).sum()
        rows.append({
            'task': task_name,
            'n': n_sample,
            'max': int(L.max()),
            'p50': int(np.percentile(L, 50)),
            'p90': int(np.percentile(L, 90)),
            'p99': int(np.percentile(L, 99)),
            f'>{args.threshold}': over,
            'pct_over': 100 * over / n_sample,
        })

    print(f'\n=== prompt lengths vs threshold {args.threshold} ===')
    print(f"{'task':<15} {'n':>6} {'max':>6} {'p50':>6} {'p90':>6} {'p99':>6} {'>thr':>6} {'pct':>7}")
    for r in rows:
        print(f"{r['task']:<15} {r['n']:>6} {r['max']:>6} {r['p50']:>6} {r['p90']:>6} {r['p99']:>6} "
              f"{r[f'>{args.threshold}']:>6} {r['pct_over']:>6.1f}%")


if __name__ == '__main__':
    main()
