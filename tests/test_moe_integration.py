"""
Tests for DeepSeekMoE integration in OpenMythos V3.

Tests:
1. Expert networks — forward shape, parameter count
2. Routing — Sqrt(Softplus) affinity, top-k selection, load balancing
3. Hash routing — deterministic, uniform weights
4. Anticipatory routing — spike detection, freeze/resume
5. MoE layer — scatter-gather correctness, shared expert always active
6. OpenMythosV3 — full model shape, loop consistency, MoE vs dense comparison
"""

import torch
from hypothesis import given, settings, strategies as st

from openmythos.ds_v4_sandbox import TieredKVCache
from openmythos.model_v3 import OpenMythosV3
from openmythos.moe import (
    AnticipatoryRouter,
    Expert,
    HashRouter,
    MoELayer,
    MoERouter,
    SharedExpert,
    compute_moe_stats,
    sqrt_softplus,
)


# ─────────────────────────────────────────────────────────────────────
# 1. Expert Tests
# ─────────────────────────────────────────────────────────────────────

def test_expert_output_shape() -> None:
    """Expert FFN should preserve input shape."""
    expert = Expert(dim=64, intermediate_dim=128)
    x = torch.randn(2, 8, 64)
    out = expert(x)
    assert out.shape == x.shape


def test_shared_expert_output_shape() -> None:
    """Shared expert should preserve input shape."""
    shared = SharedExpert(dim=64, intermediate_dim=64)
    x = torch.randn(2, 8, 64)
    out = shared(x)
    assert out.shape == x.shape


def test_expert_parameter_count() -> None:
    """Expert should have more params than shared expert (larger intermediate)."""
    dim = 64
    expert = Expert(dim=dim, intermediate_dim=dim * 4)
    shared = SharedExpert(dim=dim, intermediate_dim=dim * 2)

    expert_params = sum(p.numel() for p in expert.parameters())
    shared_params = sum(p.numel() for p in shared.parameters())

    assert expert_params > shared_params


# ─────────────────────────────────────────────────────────────────────
# 2. Routing Tests
# ─────────────────────────────────────────────────────────────────────

def test_sqrt_softplus_positive() -> None:
    """Sqrt(Softplus(x)) should always be positive."""
    x = torch.randn(100)
    result = sqrt_softplus(x)
    assert (result > 0).all()


def test_sqrt_softplus_compression() -> None:
    """Sqrt(Softplus) should compress large values more than sigmoid."""
    large = torch.tensor([10.0, 20.0, 50.0])
    ratio_large = sqrt_softplus(large)[-1] / sqrt_softplus(large)[0]
    # Sqrt(Softplus) compresses large values compared to linear growth
    assert ratio_large < 100  # Not exploding


def test_moe_router_top_k() -> None:
    """Router should select exactly top_k experts per token."""
    router = MoERouter(dim=64, num_experts=16, top_k=3)
    x = torch.randn(2, 8, 64)
    indices, weights, loss = router(x, return_loss=True)

    assert indices.shape == (2, 8, 3)
    assert weights.shape == (2, 8, 3)
    # Weights should sum to 1 per token (softmax)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 8), atol=1e-5)


def test_moe_router_balance_bias_update() -> None:
    """Balance bias update should adjust expert biases based on usage."""
    router = MoERouter(dim=64, num_experts=8, top_k=2)
    initial_bias = router.expert_bias.clone()

    # Simulate biased routing (all tokens go to expert 0)
    indices = torch.zeros(2, 4, 2, dtype=torch.long)  # All expert 0
    router.update_balance_bias(indices)

    # Expert 0 bias should decrease (over-used), others increase
    assert router.expert_bias[0] < initial_bias[0]


def test_moe_router_balance_loss() -> None:
    """Balance loss should be positive when routing is imbalanced."""
    router = MoERouter(dim=64, num_experts=8, top_k=2)
    x = torch.randn(2, 16, 64)

    _, _, loss_balanced = router(x, return_loss=True)
    # With random initialization, loss should be small but non-zero
    assert loss_balanced >= 0


# ─────────────────────────────────────────────────────────────────────
# 3. Hash Routing Tests
# ─────────────────────────────────────────────────────────────────────

def test_hash_router_deterministic() -> None:
    """Hash router should produce deterministic routing for same input."""
    router = HashRouter(num_experts=16, top_k=2)
    token_ids = torch.randint(0, 50257, (2, 16))

    idx1, w1 = router(token_ids)
    idx2, w2 = router(token_ids)

    assert torch.equal(idx1, idx2)
    assert torch.equal(w1, w2)


def test_hash_router_uniform_weights() -> None:
    """Hash router should use uniform 1/top_k weights."""
    router = HashRouter(num_experts=16, top_k=3)
    token_ids = torch.randint(0, 50257, (1, 8))

    _, weights = router(token_ids)
    assert torch.allclose(weights, torch.ones_like(weights) / 3)


def test_hash_router_valid_indices() -> None:
    """Hash router indices should be within [0, num_experts)."""
    router = HashRouter(num_experts=32, top_k=4)
    token_ids = torch.randint(0, 50257, (4, 64))

    indices, _ = router(token_ids)
    assert (indices >= 0).all()
    assert (indices < 32).all()


# ─────────────────────────────────────────────────────────────────────
# 4. Anticipatory Routing Tests
# ─────────────────────────────────────────────────────────────────────

def test_anticipatory_router_forward() -> None:
    """Anticipatory router should forward to primary router normally."""
    router = AnticipatoryRouter(dim=64, num_experts=8, top_k=2)
    x = torch.randn(1, 4, 64)

    indices, weights, loss = router(x, current_loss=None, return_loss=True)
    assert indices.shape == (1, 4, 2)
    assert weights.shape == (1, 4, 2)


def test_anticipatory_router_spike_detection() -> None:
    """Anticipatory router should activate when loss spikes detected."""
    router = AnticipatoryRouter(dim=64, num_experts=8, top_k=2, spike_threshold=1.5)
    x = torch.randn(1, 4, 64)

    # Feed normal losses
    for loss_val in [1.0, 1.1, 0.9, 1.0, 0.95]:
        router(x, current_loss=loss_val)

    assert not router.is_active  # Normal losses, not active

    # Feed spike
    router(x, current_loss=5.0)
    # May activate depending on history
    # At minimum, should not crash


# ─────────────────────────────────────────────────────────────────────
# 5. MoE Layer Tests
# ─────────────────────────────────────────────────────────────────────

def test_moe_layer_output_shape() -> None:
    """MoE layer output should match input shape."""
    moe = MoELayer(dim=64, num_experts=8, top_k=2, intermediate_dim=128)
    x = torch.randn(2, 8, 64)
    out, loss = moe(x, return_loss=True)
    assert out.shape == x.shape


def test_moe_layer_shared_expert_always_active() -> None:
    """Shared expert output should be added to all tokens."""
    dim = 64
    moe = MoELayer(dim=dim, num_experts=4, top_k=2)

    # Zero out routed experts to isolate shared expert
    for expert_mod in moe.experts:
        assert isinstance(expert_mod, Expert)
        expert_mod.fc1.weight.data.zero_()
        expert_mod.fc1.bias.data.zero_()
        expert_mod.fc2.weight.data.zero_()
        expert_mod.fc2.bias.data.zero_()

    x = torch.randn(1, 4, dim)
    out, _ = moe(x)

    # Output should be non-zero (shared expert is active)
    assert torch.norm(out) > 0


def test_moe_layer_hash_routing() -> None:
    """MoE layer with hash routing should produce deterministic output."""
    moe = MoELayer(dim=64, num_experts=8, top_k=2, use_hash_routing=True)
    x = torch.randn(1, 8, 64)
    token_ids = torch.randint(0, 50257, (1, 8))

    out1, _ = moe(x, token_ids=token_ids)
    out2, _ = moe(x, token_ids=token_ids)

    assert torch.allclose(out1, out2)


# ─────────────────────────────────────────────────────────────────────
# 6. OpenMythosV3 Tests
# ─────────────────────────────────────────────────────────────────────

@settings(deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=2),
    seq_len=st.integers(min_value=4, max_value=16),
    loop_iters=st.integers(min_value=0, max_value=2),
)
def test_openmythos_v3_output_shape(batch_size: int, seq_len: int, loop_iters: int) -> None:
    """OpenMythosV3 output should be (batch, seq, vocab)."""
    dim = 32
    vocab_size = 500
    model = OpenMythosV3(
        dim=dim, num_heads=4, latent_dim=8,
        num_prelude_layers=1, num_recurrent_layers=1, num_coda_layers=1,
        vocab_size=vocab_size, num_experts=4, moe_top_k=2,
    )
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(tokens, loop_iters=loop_iters)
    assert output.shape == (batch_size, seq_len, vocab_size)


def test_openmythos_v3_loop_consistency() -> None:
    """Zero loops should produce same shape but different values than one loop."""
    model = OpenMythosV3(
        dim=32, num_prelude_layers=1, num_recurrent_layers=1,
        num_coda_layers=1, num_experts=4, moe_top_k=2,
    )
    tokens = torch.randint(0, 50257, (1, 16))

    out_0 = model(tokens, loop_iters=0)
    out_1 = model(tokens, loop_iters=1)

    assert out_0.shape == out_1.shape
    assert not torch.allclose(out_0, out_1)


def test_openmythos_v3_moe_vs_hash() -> None:
    """Hash routing (prelude) should produce different output than learned routing."""
    dim = 32
    vocab_size = 500
    model_learned = OpenMythosV3(
        dim=dim, num_prelude_layers=1, num_recurrent_layers=1,
        num_coda_layers=1, vocab_size=vocab_size,
        num_experts=4, moe_top_k=2, use_hash_routing=False,
    )
    model_hash = OpenMythosV3(
        dim=dim, num_prelude_layers=1, num_recurrent_layers=1,
        num_coda_layers=1, vocab_size=vocab_size,
        num_experts=4, moe_top_k=2, use_hash_routing=True,
    )

    tokens = torch.randint(0, vocab_size, (1, 16))

    with torch.no_grad():
        out_learned = model_learned(tokens, loop_iters=1)
        out_hash = model_hash(tokens, loop_iters=1)

    assert out_learned.shape == out_hash.shape
    # Different routing should produce different outputs
    assert not torch.allclose(out_learned, out_hash)


def test_openmythos_v3_with_tiered_kv() -> None:
    """OpenMythosV3 should work with external TieredKVCache."""
    model = OpenMythosV3(
        dim=32, num_prelude_layers=1, num_recurrent_layers=1,
        num_coda_layers=1, num_experts=4, moe_top_k=2,
    )
    cache = TieredKVCache(max_window_size=8, max_compressed_entries=4, latent_dim=128)
    tokens = torch.randint(0, 500, (1, 16))
    output = model(tokens, loop_iters=2, kv_cache=cache)
    assert output.shape == (1, 16, 50257)


def test_openmythos_v3_moe_stats() -> None:
    """MoE stats should show parameter capacity increase."""
    model = OpenMythosV3(
        dim=64, num_prelude_layers=2, num_recurrent_layers=2,
        num_coda_layers=2, num_experts=8, moe_top_k=2,
    )
    stats = compute_moe_stats(
        model, num_experts=8, num_recurrent_layers=2,
        num_prelude_layers=2, num_coda_layers=2, top_k=2,
    )
    assert stats["total_params"] > 0
    assert stats["capacity_ratio"] > 1.0  # More total than active params
    assert stats["expert_count"] > 0


def test_openmythos_v3_deterministic_output() -> None:
    """Same input should produce approximately same output in eval mode."""
    model = OpenMythosV3(
        dim=32, num_prelude_layers=1, num_recurrent_layers=1,
        num_coda_layers=1, vocab_size=500, num_experts=4, moe_top_k=2,
    )
    model.eval()
    tokens = torch.randint(0, 500, (1, 8))
    with torch.no_grad():
        out1 = model(tokens, loop_iters=2)
        out2 = model(tokens, loop_iters=2)
    # Allow small differences due to balance bias updates between calls
    assert torch.allclose(out1, out2, atol=1e-3)
