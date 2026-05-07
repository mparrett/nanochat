"""
Cosine-NN diagnostic for d6_baseline_modern_sft's failure-mode tokens.

Per the trx4mr Phase 5 two-axis framework
(blabberverse-phase5-impl-and-takeaways.html):
- If a misbehaving token's embedding-NNs share *next-token distributions*
  with it, the failure axis is "right-context drift" (FP-flavored). Fix
  direction: coarsen codebook (binary/ternary) OR add capacity.
- If the NNs share *position / preceding context*, the failure axis is
  "left-context collision" (binary-flavored). Fix direction: add codebook
  levels (ternary) OR increase d_model.

We probe ~10 tokens that appeared in today's degenerate chatbot outputs
and report top-10 cosine NNs of each.

Run: uv run python -m dev.cosine_nn_probe [--model-tag d6_baseline_modern_sft]
"""
import argparse
import torch
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init


PROBE_TOKENS = [
    # From persona_retention failures
    "Alex",
    "engineer",
    "software",
    "Java",
    "J",
    # From self_correction failures
    "Sydney",
    "Canberra",
    "Australia",
    # From numerical_thread / math-mode
    "5",
    "apples",
    "left",
    # From repetition-loop diagnoses
    "the",
    "store",
    "too",
    "Portland",
    # Special tokens implicated in template leakage
    "<|python_start|>",
    "<|output_start|>",
    "<|assistant_start|>",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", default="d6_baseline_modern_sft",
                        help="SFT checkpoint tag under chatsft_checkpoints/")
    args = parser.parse_args()

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, _ = load_model("sft", device, phase="eval", model_tag=args.model_tag)
    print(f"Model tag: {args.model_tag}")

    # Get the token embedding matrix (B, d) where B is vocab_size.
    wte = model.transformer.wte.weight.detach()  # (vocab, d)
    print(f"Embedding matrix: {wte.shape}")
    print(f"Device: {wte.device}, dtype: {wte.dtype}")

    # Pre-compute normalized embeddings for cosine similarity
    norm_wte = wte / wte.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def lookup_token(text):
        """Return the single token id for a string; None if multi-token."""
        try:
            tid = tokenizer.encode_special(text)
            return tid, "special"
        except Exception:
            pass
        # Try regular encoding
        ids = tokenizer.encode(text)
        if len(ids) == 1:
            return ids[0], "single"
        # Try with leading space (most BPE tokenizers prefer this)
        ids = tokenizer.encode(" " + text)
        if len(ids) == 1:
            return ids[0], "leading-space"
        return None, f"multi-token ({len(ids)} tokens)"

    print()
    print("=" * 90)
    print(f"{'token':<25} {'tid':>7}  {'kind':<14}  top-10 cosine NNs")
    print("=" * 90)

    for query in PROBE_TOKENS:
        tid, kind = lookup_token(query)
        if tid is None:
            print(f"{query!r:<25} {'-':>7}  {kind:<14}  [skipped]")
            continue
        # Cosine similarity to all other tokens
        q_emb = norm_wte[tid]  # (d,)
        sims = norm_wte @ q_emb  # (vocab,)
        # Top 11 (will include self at rank 0)
        topk_vals, topk_idxs = torch.topk(sims, k=11)
        # Skip self
        nn_ids = [int(i) for i in topk_idxs[1:]]
        nn_vals = [float(v) for v in topk_vals[1:]]
        nn_strs = []
        for nid, sim in zip(nn_ids, nn_vals):
            try:
                tok_text = tokenizer.decode([nid])
            except Exception:
                tok_text = f"<id={nid}>"
            # Sanitize for printing
            display = repr(tok_text)
            if len(display) > 18:
                display = display[:17] + "…"
            nn_strs.append(f"{display}({sim:.2f})")
        print(f"{query!r:<25} {tid:>7}  {kind:<14}  {', '.join(nn_strs)}")


if __name__ == "__main__":
    main()
