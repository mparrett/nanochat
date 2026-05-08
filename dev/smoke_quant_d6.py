"""
Phase 1 smoke test for nanochat.quant — does STE backward deliver useful
gradients end-to-end on a d6-shaped GPT?

Builds a d6-shaped GPT (n_layer=6, n_embd=384, n_head=6 — same shape the
canonical `runcpu.sh` recipe builds), applies `apply_quant` with the
chosen variant, runs N steps of random-data SGD, and asserts that the
loss decreases. This is a tiny synthetic reproducer of the from-scratch
pretrain — not a real pretrain, but enough to falsify the load-bearing
question: "does STE pass meaningful gradients to latent fp32 weights at
d6 shape on this hardware".

Catches:
- Dtype boundary surprises (the bf16 audit found a flash_attention.py
  boundary; expect more on the quant path).
- STE wiring bugs (zero gradients reaching latents).
- Initialization regressions (loss stuck at unigram baseline).

Run:
    uv run python -u dev/smoke_quant_d6.py --variant binary
    uv run python -u dev/smoke_quant_d6.py --variant ternary

Default is CPU; pass `--device mps` (or `cuda`) to test that path.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.quant import apply_quant


def build_d6(vocab_size: int, seq_len: int, device: str) -> GPT:
    """Same shape rule used by scripts/base_train.py at depth=6 (default
    aspect_ratio=64, head_dim=64): n_embd = depth * 64 = 384, n_head = 6."""
    config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=6,
        n_head=6,
        n_kv_head=6,
        n_embd=384,
        window_pattern="L",  # full-context for the smoke; window pattern is orthogonal
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=torch.device(device))
    model.init_weights()
    return model


def run_smoke(variant: str, device: str, steps: int, batch: int, seq_len: int,
              lr: float, group_size: int, vocab_size: int, seed: int) -> dict:
    """Overfit a single fixed batch and assert loss drops well below the
    unigram baseline. Random tokens against random targets is uninformative
    (optimal loss = log(vocab) exactly), so we use a fixed batch where each
    sequence has structure to memorize."""
    torch.manual_seed(seed)

    model = build_d6(vocab_size=vocab_size, seq_len=seq_len, device=device)
    info = apply_quant(model, variant=variant, group_size=group_size, reinit=True)
    print(f"apply_quant: {info}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_params:,} ({len(trainable)} tensors)")

    # Fixed batch — same idx/targets every step. If STE backward works, the
    # model overfits this and loss drops below the unigram baseline log(vocab).
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, vocab_size, (batch, seq_len), generator=g,
                        dtype=torch.int32).to(device)
    # Targets are next-token (shifted-by-one). Even on random tokens there's a
    # pattern to memorize: the (idx[:, i] -> idx[:, i+1]) mapping for THIS batch.
    # The model has enough capacity to overfit; failure to do so signals an STE
    # backward problem.
    targets = torch.empty_like(idx, dtype=torch.int64)
    targets[:, :-1] = idx[:, 1:].long()
    targets[:, -1] = -1  # ignored token
    unigram_baseline = float(torch.tensor(vocab_size).log().item())
    print(f"unigram baseline (log vocab) = {unigram_baseline:.4f} — must beat this")

    opt = torch.optim.AdamW(trainable, lr=lr)
    model.train()

    losses: list[float] = []
    t_start = time.time()

    for step in range(steps):
        loss = model(idx, targets=targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step == 0 or (step + 1) % max(1, steps // 5) == 0:
            print(f"  step {step:3d}  loss {losses[-1]:.4f}")

    elapsed = time.time() - t_start
    initial = losses[0]
    final = losses[-1]
    delta = final - initial
    print(f"\nresult: initial={initial:.4f}  final={final:.4f}  Δ={delta:+.4f}  "
          f"{steps} steps in {elapsed:.1f}s ({elapsed/steps*1000:.1f}ms/step)")

    return {
        "variant": variant,
        "device": device,
        "steps": steps,
        "initial_loss": initial,
        "final_loss": final,
        "delta": delta,
        "elapsed_s": elapsed,
        "n_trainable_params": n_params,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["binary", "ternary"], default="binary")
    parser.add_argument("--device", default="cpu",
                        help="cpu / mps / cuda — defaults to cpu for portability")
    parser.add_argument("--steps", type=int, default=30,
                        help="enough to see directionally-down loss; not a pretrain")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"smoke_quant_d6 variant={args.variant} device={args.device} "
          f"steps={args.steps} batch={args.batch} seq_len={args.seq_len} "
          f"lr={args.lr} group_size={args.group_size} vocab={args.vocab_size}\n")

    result = run_smoke(
        variant=args.variant, device=args.device, steps=args.steps,
        batch=args.batch, seq_len=args.seq_len, lr=args.lr,
        group_size=args.group_size, vocab_size=args.vocab_size, seed=args.seed,
    )

    # Falsification: a d6-shaped GPT (~11M trainable params) overfitting a
    # 4×128 fixed batch must drop the loss meaningfully below log(vocab).
    # Anything else means STE backward isn't moving the latents in the
    # direction the loss landscape demands.
    unigram_baseline = float(torch.tensor(args.vocab_size).log().item())
    if result["final_loss"] >= unigram_baseline - 0.5:
        print(f"\n[FAIL] final loss {result['final_loss']:.4f} did not beat "
              f"unigram baseline {unigram_baseline:.4f} by 0.5 nats. "
              "STE backward likely not delivering useful gradients at d6 shape.")
        raise SystemExit(1)
    print(f"\n[PASS] final loss {result['final_loss']:.4f} beat unigram "
          f"baseline {unigram_baseline:.4f} by "
          f"{unigram_baseline - result['final_loss']:.4f} nats — "
          "STE backward delivers useful gradients at d6 shape.")


if __name__ == "__main__":
    main()
