"""
Tests for nanochat.quant — STE binarize/ternarize, BinaryLinear/TernaryLinear,
the apply_quant walker, and state-dict round-trip.

The contract these tests pin:

1. STE forward produces the right alphabet (binary: {-s,+s}; ternary: {-s,0,+s})
   and STE backward returns grad_output unchanged to the latent fp32 weight.
2. apply_quant replaces every targeted Linear in transformer.h with the chosen
   variant; lm_head, wte, value_embeds, and the smear gate are untouched.
3. apply_quant's injection_count matches `n_layer * len(target_intersect_block)`.
4. Forward + backward end-to-end on a tiny GPT works on CPU.
5. Latent weight stays fp32 even if base was bf16-cast.
6. Save / load of model.state_dict() round-trips (latent weights survive),
   reproducing the same forward.
7. apply_quant on an unmatched target raises; unknown variant raises.
8. quantized output differs measurably from fp baseline (non-trivial swap).
"""
import math

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.quant import (
    BinaryLinear,
    TernaryLinear,
    apply_quant,
    ste_binarize,
    ste_ternarize,
)


def _build_tiny_model(seed=0):
    torch.manual_seed(seed)
    config = GPTConfig(
        sequence_len=64, vocab_size=128, n_layer=4, n_head=2, n_kv_head=2,
        n_embd=64, window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=torch.device("cpu"))
    model.init_weights()
    model.eval()
    return model, config


def _make_batch(config, B=2, T=32, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, config.vocab_size, (B, T), generator=g, dtype=torch.int32)


# ---------------------------------------------------------------------------
# STE primitives
# ---------------------------------------------------------------------------

def test_ste_binarize_alphabet_is_two_signed_values_per_group():
    torch.manual_seed(0)
    w = torch.randn(8, 64)
    qb = ste_binarize(w, group_size=32)
    # Per-group, |qb| collapses to a single value (the per-group scale).
    flat = qb.reshape(-1)
    groups = flat.reshape(-1, 32)
    for i in range(groups.size(0)):
        assert groups[i].abs().unique().numel() == 1, (
            f"group {i} should have exactly one |scale| value"
        )


def test_ste_ternarize_alphabet_is_zero_and_signed_scale_per_group():
    torch.manual_seed(0)
    w = torch.randn(8, 64)
    qt = ste_ternarize(w, group_size=32)
    flat = qt.reshape(-1)
    groups = flat.reshape(-1, 32)
    for i in range(groups.size(0)):
        # Two unique magnitudes per group: 0 and the scale.
        mags = groups[i].abs().unique()
        assert mags.numel() == 2, f"group {i} unique |values| = {mags.tolist()}"
        assert mags.min().item() == 0.0


def test_ste_binarize_backward_pass_through_identity():
    """STE backward must deliver grad_output unchanged to the latent weight."""
    torch.manual_seed(0)
    w = torch.randn(64, 128, requires_grad=True)
    qb = ste_binarize(w, group_size=32)
    grad_pretend = torch.randn_like(qb)
    qb.backward(grad_pretend)
    assert torch.equal(w.grad, grad_pretend), "binarize STE should pass grad through"


def test_ste_ternarize_backward_pass_through_identity():
    torch.manual_seed(0)
    w = torch.randn(64, 128, requires_grad=True)
    qt = ste_ternarize(w, group_size=32)
    grad_pretend = torch.randn_like(qt)
    qt.backward(grad_pretend)
    assert torch.equal(w.grad, grad_pretend), "ternarize STE should pass grad through"


def test_ste_handles_nondivisible_group_size():
    """Padding to multiple of group_size must not corrupt the unpadded weight."""
    torch.manual_seed(0)
    w = torch.randn(7, 17, requires_grad=True)  # 119 elements, not /32
    qb = ste_binarize(w, group_size=32)
    assert qb.shape == w.shape
    qb.sum().backward()
    assert w.grad.shape == w.shape


# ---------------------------------------------------------------------------
# Module-level forward/backward
# ---------------------------------------------------------------------------

def test_binary_linear_forward_and_backward():
    torch.manual_seed(0)
    bl = BinaryLinear(64, 32, group_size=32)
    x = torch.randn(2, 8, 64)
    y = bl(x)
    assert y.shape == (2, 8, 32)
    y.sum().backward()
    assert bl.weight.grad is not None
    assert bl.weight.grad.shape == bl.weight.shape


def test_ternary_linear_forward_and_backward():
    torch.manual_seed(0)
    tl = TernaryLinear(64, 32, group_size=32)
    x = torch.randn(2, 8, 64)
    y = tl(x)
    assert y.shape == (2, 8, 32)
    y.sum().backward()
    assert tl.weight.grad is not None


def test_quant_linear_latent_is_fp32_even_with_bf16_input():
    """Master-precision invariant: latent weight stays fp32; matmul runs in
    activation dtype. This is the bf16-path composition contract."""
    bl = BinaryLinear(32, 16, group_size=16)
    assert bl.weight.dtype == torch.float32
    x = torch.randn(2, 4, 32, dtype=torch.bfloat16)
    y = bl(x)
    assert y.dtype == torch.bfloat16
    assert bl.weight.dtype == torch.float32, "latent must remain fp32 after bf16 forward"


# ---------------------------------------------------------------------------
# apply_quant walker
# ---------------------------------------------------------------------------

def test_apply_quant_swaps_attn_and_mlp_linears():
    model, _ = _build_tiny_model()
    info = apply_quant(model, variant="binary", group_size=32)
    # 4 layers × (4 attn proj + 2 mlp proj) = 24
    assert info["injected_count"] == 24
    assert info["variant"] == "binary"
    for block in model.transformer.h:
        for name in ("c_q", "c_k", "c_v", "c_proj"):
            mod = getattr(block.attn, name)
            assert isinstance(mod, BinaryLinear), f"attn.{name} not BinaryLinear: {type(mod).__name__}"
        for name in ("c_fc", "c_proj"):
            mod = getattr(block.mlp, name)
            assert isinstance(mod, BinaryLinear), f"mlp.{name} not BinaryLinear"


def test_apply_quant_ternary_variant():
    model, _ = _build_tiny_model()
    apply_quant(model, variant="ternary", group_size=32)
    for block in model.transformer.h:
        assert isinstance(block.attn.c_q, TernaryLinear)
        assert isinstance(block.mlp.c_fc, TernaryLinear)


def test_apply_quant_does_not_touch_lm_head_or_embeddings():
    """Phase 1 contract: only block.attn.* and block.mlp.* get swapped."""
    model, _ = _build_tiny_model()
    lm_head_before = type(model.lm_head)
    wte_before = type(model.transformer.wte)
    apply_quant(model, variant="binary", group_size=32)
    assert type(model.lm_head) is lm_head_before, "lm_head must not be quantized in Phase 1"
    assert type(model.transformer.wte) is wte_before, "wte must not be quantized in Phase 1"
    # ve_gate is also off-limits at default (not in target list)
    for block in model.transformer.h:
        if block.attn.ve_gate is not None:
            assert not isinstance(block.attn.ve_gate, (BinaryLinear, TernaryLinear))


def test_apply_quant_subset_target_only_swaps_named_attrs():
    model, _ = _build_tiny_model()
    info = apply_quant(model, variant="binary", target=("c_q", "c_v"), group_size=32)
    assert info["injected_count"] == 8  # 4 layers × 2 targets
    for block in model.transformer.h:
        assert isinstance(block.attn.c_q, BinaryLinear)
        assert isinstance(block.attn.c_v, BinaryLinear)
        # c_k, c_proj should remain non-quantized
        assert not isinstance(block.attn.c_k, (BinaryLinear, TernaryLinear))
        assert not isinstance(block.attn.c_proj, (BinaryLinear, TernaryLinear))


def test_apply_quant_unmatched_target_raises():
    model, _ = _build_tiny_model()
    with pytest.raises(RuntimeError, match="injected zero modules"):
        apply_quant(model, variant="binary", target=("nonexistent_proj",), group_size=32)


def test_apply_quant_unknown_variant_raises():
    model, _ = _build_tiny_model()
    with pytest.raises(ValueError, match="unknown variant"):
        apply_quant(model, variant="float8", group_size=32)


def test_apply_quant_empty_target_raises():
    model, _ = _build_tiny_model()
    with pytest.raises(ValueError, match="at least one"):
        apply_quant(model, variant="binary", target=(), group_size=32)


# ---------------------------------------------------------------------------
# End-to-end: forward + backward on tiny GPT after apply_quant
# ---------------------------------------------------------------------------

def test_quantized_gpt_forward_and_backward():
    model, config = _build_tiny_model()
    apply_quant(model, variant="binary", group_size=32)
    idx = _make_batch(config)
    targets = idx.long().clone()
    loss = model(idx, targets=targets)
    assert torch.isfinite(loss)
    loss.backward()
    # Every quantized latent weight should have a non-None gradient.
    for block in model.transformer.h:
        for parent_name in ("attn", "mlp"):
            parent = getattr(block, parent_name)
            for name in ("c_q", "c_k", "c_v", "c_proj", "c_fc"):
                mod = getattr(parent, name, None)
                if isinstance(mod, (BinaryLinear, TernaryLinear)):
                    assert mod.weight.grad is not None, f"{parent_name}.{name} latent grad missing"


def test_quantized_gpt_output_differs_from_fp_baseline():
    """Sanity: quantizing must actually change behaviour. If the test passes
    with apply_quant being a no-op, we wouldn't know."""
    model_fp, config = _build_tiny_model(seed=42)
    model_bin, _ = _build_tiny_model(seed=42)
    apply_quant(model_bin, variant="binary", group_size=32)
    idx = _make_batch(config)
    with torch.no_grad():
        out_fp = model_fp(idx)
        out_bin = model_bin(idx)
    diff = (out_fp - out_bin).abs().max().item()
    # The model has zero-init c_proj on attention, but mlp.c_proj is also
    # zero-init... so in the fresh-init regime *most* signals to lm_head are
    # zero. The quantized model has reinit'd MLP/attn matrices though, so
    # x0_lambdas + smear path will still produce a non-trivial delta. Use a
    # generous threshold.
    assert diff > 1e-4, f"quantized forward indistinguishable from fp; diff={diff:.3e}"


# ---------------------------------------------------------------------------
# State-dict round-trip
# ---------------------------------------------------------------------------

def test_quantized_gpt_state_dict_roundtrip():
    src, config = _build_tiny_model(seed=42)
    dst, _ = _build_tiny_model(seed=42)
    apply_quant(src, variant="binary", group_size=32)
    apply_quant(dst, variant="binary", group_size=32)

    # Perturb src's latent weights so the dst-on-clone-init test is meaningful.
    with torch.no_grad():
        for p in src.parameters():
            if p.dtype == torch.float32 and p.dim() == 2:
                p.normal_(mean=0.0, std=0.05)

    state = src.state_dict()
    dst.load_state_dict(state)

    idx = _make_batch(config)
    with torch.no_grad():
        out_src = src(idx)
        out_dst = dst(idx)
    assert torch.allclose(out_src, out_dst, atol=1e-6, rtol=0), (
        f"round-trip forward mismatch; max diff = {(out_src - out_dst).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# reinit=False path (quantize a "trained" model in place)
# ---------------------------------------------------------------------------

def test_apply_quant_reinit_false_copies_base_weight():
    model, _ = _build_tiny_model(seed=0)
    # Capture an attention projection's pre-swap weight
    pre = model.transformer.h[0].attn.c_q.weight.detach().clone()
    apply_quant(model, variant="binary", target=("c_q",), group_size=32, reinit=False)
    post = model.transformer.h[0].attn.c_q.weight.detach()
    assert post.shape == pre.shape
    assert torch.allclose(pre, post, atol=0, rtol=0), (
        "reinit=False should preserve the pre-swap latent weight"
    )


def test_apply_quant_reinit_true_replaces_base_weight():
    model, _ = _build_tiny_model(seed=0)
    pre = model.transformer.h[0].attn.c_q.weight.detach().clone()
    apply_quant(model, variant="binary", target=("c_q",), group_size=32, reinit=True)
    post = model.transformer.h[0].attn.c_q.weight.detach()
    # New kaiming_uniform_ init -> almost surely different
    assert not torch.allclose(pre, post, atol=1e-6), (
        "reinit=True should re-initialize the latent weight"
    )
