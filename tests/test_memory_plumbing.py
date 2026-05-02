"""
Stage 0 of Hope/Nested-Learning: memory_state plumbing.

The contract these tests pin:

1. Calling forward() without `memory_state` is bit-identical to the pre-Stage-0
   model (single-tensor return, no behavior change). This is the load-bearing
   guarantee — every existing caller (training loops, eval, KV-cache inference)
   relies on it.
2. Calling forward() with `memory_state` returns `(out, new_memory_state)`.
3. In Stage 0, no block carries mutable state, so passing `model.reset_memory()`
   (a list of Nones) gets back the same list of Nones, and the loss/logits
   tensor matches the no-memory-state path bit-for-bit.

Stage 1 will introduce a memory-bearing block that overrides Block.forward
and starts producing real state in (3); this test should still pass for
every block-position that doesn't carry memory.
"""
import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import COMPUTE_DTYPE


def _build_tiny_model(seed=0):
    """Build a small CPU model with deterministic init for the tests.

    GPT.__init__ runs in a meta-device context (no real allocation), so we
    follow the same `to_empty` + `init_weights` dance that
    `nanochat/checkpoint_manager.py::build_model` uses.
    """
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


def _make_batch(config, B=2, T=32, device="cpu", seed=1):
    g = torch.Generator(device=device).manual_seed(seed)
    idx = torch.randint(0, config.vocab_size, (B, T), generator=g, dtype=torch.int32, device=device)
    targets = torch.randint(0, config.vocab_size, (B, T), generator=g, dtype=torch.int64, device=device)
    return idx, targets


def test_reset_memory_returns_list_of_nones():
    model, config = _build_tiny_model()
    state = model.reset_memory()
    assert isinstance(state, list)
    assert len(state) == config.n_layer
    assert all(s is None for s in state)


def test_forward_without_memory_returns_single_tensor():
    """Pre-Stage-0 callers must see exactly the original return shape."""
    model, config = _build_tiny_model()
    idx, targets = _make_batch(config)
    with torch.no_grad():
        loss = model(idx, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar (reduction='mean')


def test_forward_with_memory_returns_tuple():
    model, config = _build_tiny_model()
    idx, targets = _make_batch(config)
    state = model.reset_memory()
    with torch.no_grad():
        result = model(idx, targets, memory_state=state)
    assert isinstance(result, tuple) and len(result) == 2
    loss, new_state = result
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0
    assert isinstance(new_state, list) and len(new_state) == config.n_layer
    # Stage 0: no block carries state, so all entries pass through as None.
    assert all(s is None for s in new_state)


def test_loss_bit_identical_with_and_without_memory_state():
    """The load-bearing guarantee: threading memory_state must not perturb loss."""
    model, config = _build_tiny_model()
    idx, targets = _make_batch(config)
    with torch.no_grad():
        loss_plain = model(idx, targets)
        loss_with_state, _ = model(idx, targets, memory_state=model.reset_memory())
    # Bit-for-bit: same input + same params + same code path on tensors → same output.
    assert torch.equal(loss_plain, loss_with_state), (
        f"Stage 0 plumbing changed the loss: {loss_plain.item()} vs {loss_with_state.item()}"
    )


def test_logits_bit_identical_with_and_without_memory_state():
    """Same guarantee for the inference path (no targets)."""
    model, config = _build_tiny_model()
    idx, _ = _make_batch(config)
    with torch.no_grad():
        logits_plain = model(idx)
        logits_with_state, _ = model(idx, memory_state=model.reset_memory())
    assert torch.equal(logits_plain, logits_with_state)


def test_forward_rejects_wrong_length_memory_state():
    model, config = _build_tiny_model()
    idx, targets = _make_batch(config)
    bad_state = [None] * (config.n_layer + 1)
    with pytest.raises(AssertionError, match="memory_state must have"):
        model(idx, targets, memory_state=bad_state)


# -----------------------------------------------------------------------------
# Stage 1: LinearAttentionMemory as FFN replacement at one layer
# -----------------------------------------------------------------------------


def _build_tiny_model_with_memory(seed=0, memory_layer=2):
    """Same tiny model as Stage 0 tests, but with one block's MLP swapped."""
    torch.manual_seed(seed)
    config = GPTConfig(
        sequence_len=64,
        vocab_size=128,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
        hope_memory_layer=memory_layer,
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=torch.device("cpu"))
    model.init_weights()
    model.eval()
    return model, config


def test_stage1_memory_block_present_at_correct_layer():
    """The configured layer uses LinearAttentionMemory; the rest still use MLP."""
    from nanochat.gpt import MLP, LinearAttentionMemory
    memory_layer = 2
    model, config = _build_tiny_model_with_memory(memory_layer=memory_layer)
    for i, block in enumerate(model.transformer.h):
        if i == memory_layer:
            assert isinstance(block.mlp, LinearAttentionMemory), (
                f"layer {i} should be the memory block, got {type(block.mlp).__name__}"
            )
        else:
            assert isinstance(block.mlp, MLP), (
                f"layer {i} should still be MLP, got {type(block.mlp).__name__}"
            )


def test_stage1_loss_differs_from_baseline_after_perturbation():
    """With one block swapped, loss should diverge once the memory block contributes.

    Both blocks initialize their output projection (mlp.c_proj / mlp.W_o) to zero
    by design — zero residual contribution at init is a stability convention
    nanochat shares with the attention path. So at step 0 the architectures are
    indistinguishable on the loss. We perturb the memory block's W_o to expose
    that the *forward path* does something different.
    """
    baseline_model, config = _build_tiny_model()
    stage1_model, _ = _build_tiny_model_with_memory(memory_layer=2)
    idx, targets = _make_batch(config)
    with torch.no_grad():
        # Sanity: at init, both are bit-identical (zero MLP/memory contribution).
        loss_init_baseline = baseline_model(idx, targets)
        loss_init_stage1 = stage1_model(idx, targets)
        assert torch.equal(loss_init_baseline, loss_init_stage1), (
            "Architectures should be loss-equivalent at init since both zero out MLP/memory output."
        )
        # Perturb stage1's memory output projection so the block actually contributes.
        stage1_model.transformer.h[2].mlp.W_o.weight.add_(0.1)
        loss_perturbed_stage1 = stage1_model(idx, targets)
    assert not torch.equal(loss_init_baseline, loss_perturbed_stage1), (
        "Stage 1 loss equals baseline loss after perturbing W_o — the memory block isn't doing anything."
    )
    assert torch.isfinite(loss_perturbed_stage1).item()


def test_stage1_forward_runs_without_error_and_is_deterministic():
    """The memory block forward should run cleanly and be deterministic given fixed inputs."""
    model, config = _build_tiny_model_with_memory(memory_layer=2)
    idx, _ = _make_batch(config)
    with torch.no_grad():
        logits_a = model(idx)
        logits_b = model(idx)
    assert torch.equal(logits_a, logits_b), "Stage 1 forward is non-deterministic"


def test_stage1_backward_produces_finite_gradients():
    """Make sure gradients flow through the memory block without NaN/Inf."""
    model, config = _build_tiny_model_with_memory(memory_layer=2)
    idx, targets = _make_batch(config)
    model.train()
    loss = model(idx, targets)
    loss.backward()
    # Spot-check: all parameters that received a grad should be finite.
    n_checked = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all().item(), f"non-finite grad in {name}"
            n_checked += 1
    assert n_checked > 0, "no parameters received gradients?"


# -----------------------------------------------------------------------------
# Stage 1-additive: LinearAttentionMemory as a third residual alongside attn+MLP
# -----------------------------------------------------------------------------


def _build_tiny_model_with_additive_memory(seed=0, memory_layer=2):
    """Same tiny model but with an *additive* memory block at one layer."""
    torch.manual_seed(seed)
    config = GPTConfig(
        sequence_len=64,
        vocab_size=128,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
        hope_additive_memory_layer=memory_layer,
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=torch.device("cpu"))
    model.init_weights()
    model.eval()
    return model, config


def test_additive_memory_block_present_at_correct_layer():
    """The configured layer has add_memory; others have None. MLP is preserved everywhere."""
    from nanochat.gpt import MLP, LinearAttentionMemory
    memory_layer = 2
    model, config = _build_tiny_model_with_additive_memory(memory_layer=memory_layer)
    for i, block in enumerate(model.transformer.h):
        # MLP is preserved at every layer (additive doesn't replace)
        assert isinstance(block.mlp, MLP), f"layer {i} MLP should be preserved with additive insertion"
        if i == memory_layer:
            assert isinstance(block.add_memory, LinearAttentionMemory), (
                f"layer {i} should have additive memory, got {type(block.add_memory).__name__}"
            )
        else:
            assert block.add_memory is None, f"layer {i} should not have additive memory"


def test_additive_loss_bit_identical_to_baseline_at_init():
    """W_o=0 init means the additive memory block contributes nothing at step 0.

    The architecture is bit-identical to the unmodified baseline at initialization;
    divergence emerges only as W_o trains away from zero.
    """
    baseline_model, config = _build_tiny_model()
    additive_model, _ = _build_tiny_model_with_additive_memory(memory_layer=2)
    idx, targets = _make_batch(config)
    with torch.no_grad():
        loss_baseline = baseline_model(idx, targets)
        loss_additive = additive_model(idx, targets)
    assert torch.equal(loss_baseline, loss_additive), (
        f"Additive memory not bit-identical to baseline at init "
        f"(W_o init wrong?): {loss_baseline.item()} vs {loss_additive.item()}"
    )


def test_additive_memory_diverges_after_perturbation():
    """Once W_o is non-zero, the additive contribution kicks in and the loss differs."""
    baseline_model, config = _build_tiny_model()
    additive_model, _ = _build_tiny_model_with_additive_memory(memory_layer=2)
    idx, targets = _make_batch(config)
    with torch.no_grad():
        # Perturb the additive memory's W_o so it actually contributes
        additive_model.transformer.h[2].add_memory.W_o.weight.add_(0.1)
        loss_baseline = baseline_model(idx, targets)
        loss_additive = additive_model(idx, targets)
    assert not torch.equal(loss_baseline, loss_additive), (
        "Additive memory still gives baseline loss after W_o perturbation — block isn't contributing"
    )


def test_additive_backward_produces_finite_gradients():
    """Gradients must flow through the additive memory pathway too."""
    model, config = _build_tiny_model_with_additive_memory(memory_layer=2)
    idx, targets = _make_batch(config)
    model.train()
    # Perturb W_o so the additive path actually has gradient signal (otherwise
    # the path is exactly zero and W_o gradient is also zero).
    with torch.no_grad():
        model.transformer.h[2].add_memory.W_o.weight.add_(0.1)
    loss = model(idx, targets)
    loss.backward()
    # The additive memory's W_k/W_v/W_q should now have non-zero finite grads.
    add_mem = model.transformer.h[2].add_memory
    for name, p in [('W_k', add_mem.W_k.weight), ('W_v', add_mem.W_v.weight),
                    ('W_q', add_mem.W_q.weight), ('W_o', add_mem.W_o.weight)]:
        assert p.grad is not None, f"add_memory.{name} did not receive a gradient"
        assert torch.isfinite(p.grad).all().item(), f"non-finite grad in add_memory.{name}"


# -----------------------------------------------------------------------------
# Stage 2: LearnedGateLinearMemory with per-token learned alpha/eta
# -----------------------------------------------------------------------------


def _build_tiny_model_with_learned_gate(seed=0, memory_layer=2, additive=True, w_o_init_scale=1.0):
    """Stage 2 tiny model: additive learned-gate memory at one layer, with
    Stage 1.5b's W_o=1.0 default. The additive path is the recommended Stage 2
    topology (per ADR-002)."""
    torch.manual_seed(seed)
    config = GPTConfig(
        sequence_len=64,
        vocab_size=128,
        n_layer=4,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
        hope_memory_layer=None if additive else memory_layer,
        hope_additive_memory_layer=memory_layer if additive else None,
        hope_memory_w_o_init_scale=w_o_init_scale,
        hope_memory_kind="learned_gate",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=torch.device("cpu"))
    model.init_weights()
    model.eval()
    return model, config


def test_stage2_learned_gate_module_present_at_correct_layer():
    """The configured additive layer holds a LearnedGateLinearMemory."""
    from nanochat.gpt import MLP, LearnedGateLinearMemory
    model, _ = _build_tiny_model_with_learned_gate(memory_layer=2, additive=True)
    for i, block in enumerate(model.transformer.h):
        assert isinstance(block.mlp, MLP), f"layer {i} MLP should be preserved with additive insertion"
        if i == 2:
            assert isinstance(block.add_memory, LearnedGateLinearMemory), (
                f"layer {i} should have learned-gate memory, got {type(block.add_memory).__name__}"
            )
        else:
            assert block.add_memory is None


def test_stage2_initial_alpha_and_eta_match_config():
    """W_alpha/W_eta biases set initial alpha and eta to the configured values
    (modulo the small-uniform W_alpha/W_eta noise term)."""
    model, config = _build_tiny_model_with_learned_gate(memory_layer=2, additive=True)
    mem = model.transformer.h[2].add_memory
    # With W_alpha/W_eta weights small (uniform[-0.02, 0.02]) the bias dominates,
    # so the initial gate values are close to alpha_max*sigmoid(b_alpha) and eta_max*sigmoid(b_eta).
    expected_alpha = config.hope_memory_alpha_max * torch.sigmoid(torch.tensor(config.hope_memory_alpha_init_bias)).item()
    expected_eta = config.hope_memory_eta_max * torch.sigmoid(torch.tensor(config.hope_memory_eta_init_bias)).item()
    # Sanity: defaults give alpha~0.99, eta~0.1
    assert 0.98 < expected_alpha < 0.999, f"unexpected default alpha: {expected_alpha}"
    assert 0.05 < expected_eta < 0.20, f"unexpected default eta: {expected_eta}"
    # Probe the actual gate values on a real input.
    idx, _ = _make_batch(config)
    with torch.no_grad():
        # Embed input the same way the model does (wte + ... but for the gate
        # we just need any tensor of shape (B, T, n_embd); cheapest is pulling
        # out the actual block input). Easiest: run a forward and read x there.
        # Instead, sanity-check by forwarding through the block directly with embeddings.
        x = model.transformer.wte(idx.to(torch.long)).to(next(model.parameters()).dtype)
        alpha_logits = mem.W_alpha(x) + mem.b_alpha
        eta_logits = mem.W_eta(x) + mem.b_eta
        alpha = mem.alpha_max * torch.sigmoid(alpha_logits)
        eta = mem.eta_max * torch.sigmoid(eta_logits)
    # With small W_alpha/W_eta weights, the bias dominates → values cluster near expected.
    assert (alpha - expected_alpha).abs().max().item() < 0.05, "initial alpha drifted from configured bias"
    assert (eta - expected_eta).abs().max().item() < 0.05, "initial eta drifted from configured bias"


def test_stage2_forward_runs_finite_and_deterministic():
    model, _ = _build_tiny_model_with_learned_gate(memory_layer=2, additive=True)
    idx, _ = _make_batch(model.config) if False else _make_batch(_build_tiny_model_with_learned_gate()[1])
    with torch.no_grad():
        out_a = model(idx)
        out_b = model(idx)
    assert torch.equal(out_a, out_b), "Stage 2 forward is non-deterministic"
    assert torch.isfinite(out_a).all().item(), "Stage 2 forward produced non-finite values"


def test_stage2_backward_produces_finite_gradients_on_all_gate_params():
    """Gradients flow through W_q/W_k/W_v/W_o AND W_alpha/W_eta. The 1.5b lesson
    was that a silent dead-gradient startup is invisible in loss but visible in
    grad norms — this test ensures W_alpha and W_eta receive nonzero finite grads
    from step 1 with the recommended W_o=1.0 default."""
    model, config = _build_tiny_model_with_learned_gate(memory_layer=2, additive=True, w_o_init_scale=1.0)
    idx, targets = _make_batch(config)
    model.train()
    loss = model(idx, targets)
    loss.backward()
    mem = model.transformer.h[2].add_memory
    for name in ("W_k", "W_v", "W_q", "W_o", "W_alpha", "W_eta"):
        p = getattr(mem, name).weight
        assert p.grad is not None, f"{name}.weight did not receive a gradient"
        assert torch.isfinite(p.grad).all().item(), f"non-finite grad in {name}.weight"
        assert p.grad.abs().sum().item() > 0, f"{name}.weight grad is exactly zero — dead gradient"


def test_stage2_reduces_to_stage1_when_alpha_eta_constant():
    """At alpha_max ≈ 1 with strong positive bias, and eta = 1 with strong positive bias,
    Stage 2 should produce nearly the same output as Stage 1 (which has fixed alpha=eta=1).
    This pins the generalization claim — Stage 2 is a strict superset.
    """
    from nanochat.gpt import GPT, GPTConfig, LinearAttentionMemory, LearnedGateLinearMemory
    torch.manual_seed(0)
    common = dict(
        sequence_len=64, vocab_size=128, n_layer=4, n_head=2, n_kv_head=2,
        n_embd=64, window_pattern="L", hope_additive_memory_layer=2,
        hope_memory_w_o_init_scale=1.0,
    )
    config_s1 = GPTConfig(**common, hope_memory_kind="linear")
    config_s2 = GPTConfig(
        **common,
        hope_memory_kind="learned_gate",
        hope_memory_alpha_max=0.9999,
        hope_memory_eta_max=1.0,
        hope_memory_alpha_init_bias=20.0,   # sigmoid(20) ≈ 1.0 → alpha ≈ 0.9999
        hope_memory_eta_init_bias=20.0,     # sigmoid(20) ≈ 1.0 → eta ≈ 1.0
    )
    with torch.device("meta"):
        m1, m2 = GPT(config_s1), GPT(config_s2)
    for m in (m1, m2):
        m.to_empty(device=torch.device("cpu"))
        m.init_weights()
        m.eval()
    # Force the K/V/Q/W_o weights of the two memory blocks to match (Stage 2 has
    # extra W_alpha/W_eta on top, which is allowed).
    s1_mem = m1.transformer.h[2].add_memory
    s2_mem = m2.transformer.h[2].add_memory
    assert isinstance(s1_mem, LinearAttentionMemory)
    assert isinstance(s2_mem, LearnedGateLinearMemory)
    with torch.no_grad():
        for name in ("W_k", "W_v", "W_q", "W_o"):
            getattr(s2_mem, name).weight.copy_(getattr(s1_mem, name).weight)
        # Also make all OTHER blocks identical so the only difference is the memory module.
        for i in range(4):
            for name in ("attn", "mlp"):
                src = getattr(m1.transformer.h[i], name)
                dst = getattr(m2.transformer.h[i], name)
                for p_name, p in src.named_parameters():
                    dst.get_parameter(p_name).copy_(p)
        m2.transformer.wte.weight.copy_(m1.transformer.wte.weight)
        m2.lm_head.weight.copy_(m1.lm_head.weight)
        m2.resid_lambdas.copy_(m1.resid_lambdas)
        m2.x0_lambdas.copy_(m1.x0_lambdas)
        m2.smear_lambda.copy_(m1.smear_lambda)
        m2.backout_lambda.copy_(m1.backout_lambda)
        m2.smear_gate.weight.copy_(m1.smear_gate.weight)
        for k in m1.value_embeds:
            m2.value_embeds[k].weight.copy_(m1.value_embeds[k].weight)
    idx, _ = _make_batch(config_s1)
    with torch.no_grad():
        out1 = m1(idx)
        out2 = m2(idx)
    # Not bit-identical because Stage 2 multiplies by alpha/eta which are very-close-to
    # but not exactly 1, AND the W_alpha/W_eta noise in W_alpha.weight and W_eta.weight
    # adds a tiny per-token perturbation. So compare with tolerance.
    diff = (out1 - out2).abs().max().item()
    assert diff < 0.5, f"Stage 2 with alpha~1, eta~1 should approximate Stage 1; max diff={diff}"
