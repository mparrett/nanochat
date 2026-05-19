"""
Synthetic memory-mechanism diagnostic tasks for the bench/ playground.

Each task generates batches of (inputs, targets) with `targets = -1` at
positions we don't score (the harness uses cross_entropy ignore_index=-1).
Tasks are dataset-free and parametric — adjust K/T/etc. for difficulty.

The playground exists to characterize memory mechanisms cheaply (<5 min
per architectural variant at d2/d4) on probes that distinguish *recall*
(MQAR) from *state tracking* (SelectiveCopy) — the latter being the axis
δ-mem actually lifts on per the 2026-05-18 field result.

Tasks share a small interface:

  task.vocab_size : int
  task.T          : sequence length (input window)
  task.bos_id     : int
  task.sep_id     : int
  task.generate_batch(B, rng) -> (inputs[B,T], targets[B,T])
  task.name       : identifier for logs
"""

import numpy as np
import torch


class Task:
    """Abstract base for synthetic diagnostic tasks."""

    name = "abstract"
    T = 128
    bos_id = 0
    sep_id = 1
    vocab_size = 32768  # default; subclasses may override

    def generate_batch(self, B, rng):
        raise NotImplementedError


class MQAR(Task):
    """Multi-Query Associative Recall (Stage 1.5 probe, generalized).

    Sequence layout (length T):

        bos, k_1, v_1, ..., k_K, v_K, sep, q_1, a_1, ..., q_M, a_M, [pad...]

    Scored only at the answer positions (every other position after sep).
    Tests pure key→value lookup; clean fit for recall-axis mechanisms.
    Saturation step is the diagnostic — Stage 1.5b documented ~76 steps
    for baseline / additive-memory with W_o init scale = 1.0.
    """

    name = "mqar"

    def __init__(self, K=16, M=16, T=128, n_keys=32, n_values=32,
                 key_start=1024, value_start=2048, vocab_size=32768):
        self.K = K
        self.M = M
        self.T = T
        self.n_keys = n_keys
        self.n_values = n_values
        self.key_start = key_start
        self.value_start = value_start
        self.vocab_size = vocab_size

    def generate_batch(self, B, rng):
        K, M, T = self.K, self.M, self.T
        seq_full = np.full((B, T + 1), self.bos_id, dtype=np.int64)
        targets = np.full((B, T), -1, dtype=np.int64)
        for b in range(B):
            keys = rng.choice(self.n_keys, size=K, replace=False) + self.key_start
            values = rng.choice(self.n_values, size=K, replace=False) + self.value_start
            q_indices = rng.integers(0, K, size=M)
            s = [self.bos_id]
            for k, v in zip(keys, values):
                s.extend([int(k), int(v)])
            s.append(self.sep_id)
            for qi in q_indices:
                s.extend([int(keys[qi]), int(values[qi])])
            L = min(len(s), T + 1)
            seq_full[b, :L] = s[:L]
            ans_offset = 1 + 2 * K + 1  # first query token position
            for j in range(M):
                q_pos = ans_offset + 2 * j
                if q_pos >= T:
                    break
                targets[b, q_pos] = int(values[q_indices[j]])
        inputs = seq_full[:, :T]
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class SelectiveCopy(Task):
    """Mamba's canonical state-tracking probe.

    Sequence layout (length T):

        bos, x_1, x_2, ..., x_{T_in}, sep, c_1, c_2, ..., c_K, [pad...]

    Inputs x_1..x_{T_in} are a length-T_in noise stream with K content
    tokens inserted at random positions. After sep, the targets are the
    K content tokens in the order they appeared. Scored only at the K
    output positions.

    Tests *selective gating + state tracking*: the model must carry forward
    a state buffer of "content seen so far in order" through many noise
    tokens. A pure recall mechanism (attend to where content was) can do
    this in principle, but the canonical observation (Gu & Dao 2024) is
    that fixed-state recurrent architectures struggle without learned
    gating.

    Vocab layout: bos=0, sep=1, noise=2, content tokens start at 100.
    """

    name = "selective-copy"

    def __init__(self, T_in=96, K=8, n_content=32, T=None,
                 noise_id=2, content_start=100, vocab_size=32768):
        self.T_in = T_in
        self.K = K
        self.n_content = n_content
        self.noise_id = noise_id
        self.content_start = content_start
        self.vocab_size = vocab_size
        self.T = T if T is not None else 1 + T_in + 1 + K  # bos + noise + sep + copy

    def generate_batch(self, B, rng):
        T = self.T
        T_in = self.T_in
        K = self.K
        seq_full = np.full((B, T + 1), self.bos_id, dtype=np.int64)
        targets = np.full((B, T), -1, dtype=np.int64)
        for b in range(B):
            content = rng.choice(self.n_content, size=K, replace=True) + self.content_start
            positions = np.sort(rng.choice(T_in, size=K, replace=False))
            x = np.full(T_in, self.noise_id, dtype=np.int64)
            x[positions] = content
            s = [self.bos_id, *x.tolist(), self.sep_id, *content.tolist()]
            L = min(len(s), T + 1)
            seq_full[b, :L] = s[:L]
            # Target positions: model predicts c_j after seeing position 1+T_in+j (the j-th output slot).
            # In the unshifted seq s, target at position p is s[p+1]. The copy region in s spans
            # positions 1+T_in+1 .. 1+T_in+K (one-indexed), so target[1+T_in+j] = s[1+T_in+j+1] = c_{j+1}.
            for j in range(K):
                t_pos = 1 + T_in + j  # position whose target is c_{j+1}
                if t_pos >= T:
                    break
                targets[b, t_pos] = int(content[j])
        inputs = seq_full[:, :T]
        return torch.from_numpy(inputs), torch.from_numpy(targets)


TASKS = {
    "mqar": MQAR,
    "selective-copy": SelectiveCopy,
}


def make_task(name, **kwargs):
    if name not in TASKS:
        raise ValueError(f"unknown task: {name} (choices: {list(TASKS)})")
    return TASKS[name](**kwargs)


if __name__ == "__main__":
    # Smoke: dump a tiny batch of each task so we can eyeball the structure.
    rng = np.random.default_rng(0)
    for name in TASKS:
        t = make_task(name)
        x, y = t.generate_batch(2, rng)
        print(f"=== {name} ===")
        print(f"  T={t.T} vocab_size={t.vocab_size}")
        print(f"  inputs[0,:30]:  {x[0, :30].tolist()}")
        print(f"  targets[0,:30]: {y[0, :30].tolist()}")
        n_scored = (y != -1).sum().item()
        print(f"  scored positions in batch: {n_scored}")
