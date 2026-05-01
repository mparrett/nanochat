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
