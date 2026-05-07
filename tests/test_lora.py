"""
Tests for nanochat.lora — the LoRALinear wrapper, apply_lora walker, and
state-dict round-trip.

The contract these tests pin:

1. apply_lora freezes every base parameter and only LoRA A/B remain trainable.
2. With B initialized to zero, LoRA forward equals base forward exactly
   (bit-equivalent at apply time).
3. After perturbing B, the forward differs from base by a measurable amount.
4. lora_state_dict / load_lora_state_dict round-trips the adapter cleanly.
5. apply_lora reports injected_count consistent with target × n_layer.
"""
import torch
import pytest

from nanochat.gpt import GPT, GPTConfig
from nanochat.lora import (
    LoRALinear,
    apply_lora,
    lora_state_dict,
    load_lora_state_dict,
    lora_parameters,
)


def _build_tiny_model(seed=0):
    torch.manual_seed(seed)
    config = GPTConfig(
        sequence_len=64,
        vocab_size=128,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=torch.device("cpu"))
    model.init_weights()
    model.eval()
    return model, config


def _make_batch(config, B=2, T=32, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(0, config.vocab_size, (B, T), generator=g, dtype=torch.int32)
    return idx


def test_apply_lora_freezes_base_and_only_lora_trainable():
    model, _ = _build_tiny_model()
    info = apply_lora(model, target=("c_q", "c_v"), rank=4, alpha=8.0)
    # 4 layers × 2 targets = 8 wrappers
    assert info["injected_count"] == 8
    # Every trainable param must be a LoRA tensor.
    for name, p in model.named_parameters():
        is_lora = name.endswith(".lora_A") or name.endswith(".lora_B")
        if p.requires_grad:
            assert is_lora, f"non-LoRA param trainable: {name}"
        else:
            assert not is_lora, f"LoRA param frozen: {name}"
    # Trainable count should match info report.
    counted = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert counted == info["trainable_params"]


def test_lora_zero_init_matches_base_forward():
    """At apply time, B is zero so LoRA contribution is zero. Forward must
    exactly match the un-wrapped base model's forward."""
    model_a, config = _build_tiny_model(seed=42)
    model_b, _ = _build_tiny_model(seed=42)
    # Sanity: identical seed -> identical outputs before any wrapping
    idx = _make_batch(config)
    with torch.no_grad():
        out_pre = model_a(idx)
    apply_lora(model_a, target=("c_q", "c_v"), rank=4, alpha=8.0)
    with torch.no_grad():
        out_post = model_a(idx)
    # Bit-equivalent (or within fp32 noise) at apply time.
    assert torch.allclose(out_pre, out_post, atol=1e-6, rtol=0), (
        f"LoRA forward at apply time should match base. max diff = {(out_pre - out_post).abs().max().item():.3e}"
    )
    # And it should still match the second un-touched model.
    with torch.no_grad():
        out_b = model_b(idx)
    assert torch.allclose(out_post, out_b, atol=1e-6, rtol=0)


def _denull_attn_c_proj(model, std=0.02):
    """init_weights zero-inits attn.c_proj (and mlp.c_proj), which masks
    Q/V/K perturbations at the residual-stream output. Replace with small
    random values so wrapped-layer effects propagate to the model output."""
    g = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():
        for block in model.transformer.h:
            attn_c_proj = block.attn.c_proj
            attn_c_proj.weight.normal_(mean=0.0, std=std, generator=g)


def test_lora_perturbed_forward_differs_from_base():
    """Once B is non-zero, the forward should differ measurably from base."""
    model_base, config = _build_tiny_model(seed=42)
    model_lora, _ = _build_tiny_model(seed=42)
    # init_weights zero-inits attn.c_proj — the attention residual is identically
    # zero on a freshly-built model, so LoRA on c_q/c_v cannot affect outputs.
    # Apply the same de-zeroing perturbation to both models so the *only*
    # remaining difference is the LoRA contribution.
    _denull_attn_c_proj(model_base)
    _denull_attn_c_proj(model_lora)
    apply_lora(model_lora, target=("c_q", "c_v"), rank=4, alpha=8.0)
    # Populate B with non-trivial values. With small initialisations everywhere
    # (c_proj std=0.02, B std=0.02, kaiming-init A on in_features=64) the
    # downstream effect at the lm_head is on the order of 1e-4 — well above
    # fp32 noise (~1e-7) but small in absolute terms. Use a generous threshold.
    with torch.no_grad():
        for n, p in model_lora.named_parameters():
            if n.endswith(".lora_B"):
                p.normal_(mean=0.0, std=0.5)
    idx = _make_batch(config)
    with torch.no_grad():
        out_base = model_base(idx)
        out_lora = model_lora(idx)
    diff = (out_base - out_lora).abs().max().item()
    assert diff > 1e-3, (
        f"After perturbing B the forward should differ from base; max diff = {diff:.3e}"
    )


def test_lora_state_dict_roundtrip():
    """Save and load LoRA-only state dict; restored model should match the
    saved-from model on a forward pass."""
    src, config = _build_tiny_model(seed=42)
    dst, _ = _build_tiny_model(seed=42)
    apply_lora(src, target=("c_q", "c_v"), rank=4, alpha=8.0)
    apply_lora(dst, target=("c_q", "c_v"), rank=4, alpha=8.0)

    # Perturb the source's adapter so it actually carries information.
    with torch.no_grad():
        for n, p in src.named_parameters():
            if n.endswith(".lora_A"):
                p.normal_(mean=0.0, std=0.05)
            elif n.endswith(".lora_B"):
                p.normal_(mean=0.0, std=0.02)

    state = lora_state_dict(src)
    assert state, "LoRA state dict should be non-empty"
    # Every key in state must look like a LoRA tensor.
    for k in state:
        assert k.endswith(".lora_A") or k.endswith(".lora_B"), k

    load_lora_state_dict(dst, state)

    idx = _make_batch(config)
    with torch.no_grad():
        out_src = src(idx)
        out_dst = dst(idx)
    assert torch.allclose(out_src, out_dst, atol=1e-6, rtol=0), (
        f"After roundtrip dst should match src; max diff = {(out_src - out_dst).abs().max().item():.3e}"
    )


def test_lora_state_dict_strict_mismatch_raises():
    model, _ = _build_tiny_model()
    apply_lora(model, target=("c_q", "c_v"), rank=4, alpha=8.0)
    bad_state = {"transformer.h.0.attn.c_q.lora_A": torch.zeros(4, 64)}  # missing rest
    with pytest.raises(RuntimeError, match="state-dict mismatch"):
        load_lora_state_dict(model, bad_state, strict=True)


def test_lora_parameters_helper_returns_only_trainables():
    model, _ = _build_tiny_model()
    apply_lora(model, target=("c_q", "c_v"), rank=4, alpha=8.0)
    params = lora_parameters(model)
    # 4 layers × 2 targets × 2 (A and B) = 16 tensors
    assert len(params) == 16
    for p in params:
        assert p.requires_grad


def test_apply_lora_zero_target_raises():
    model, _ = _build_tiny_model()
    with pytest.raises(ValueError, match="at least one projection name"):
        apply_lora(model, target=())


def test_apply_lora_unmatched_target_raises():
    model, _ = _build_tiny_model()
    with pytest.raises(RuntimeError, match="injected zero adapters"):
        apply_lora(model, target=("nonexistent_proj",))


def test_lora_in_features_match():
    model, _ = _build_tiny_model()
    info = apply_lora(model, target=("c_q", "c_v", "c_proj"), rank=8, alpha=16.0)
    # 4 layers × 3 targets = 12 wrappers
    assert info["injected_count"] == 12
    # Sanity-check shapes line up: each LoRALinear should report in/out features
    # consistent with the underlying nn.Linear.
    for block in model.transformer.h:
        for name in ("c_q", "c_v", "c_proj"):
            mod = getattr(block.attn, name)
            assert isinstance(mod, LoRALinear)
            assert mod.in_features == mod.base.in_features
            assert mod.out_features == mod.base.out_features
            assert mod.lora_A.shape == (8, mod.in_features)
            assert mod.lora_B.shape == (mod.out_features, 8)
