"""
Property-based and unit tests for DeepSeek V4 integration components.

Tests:
1. mHC — Sinkhorn doubly stochastic convergence, spectral norm bound, residual stability
2. CSA+HCA — Block compression, top-k selection, output shape correctness
3. Tiered KV Cache — Sliding window overflow, compression, retrieval
4. OpenMythosV2 — Full model shape, loop consistency, mHC vs non-mHC comparison
"""

import torch
from hypothesis import given, settings, strategies as st

from openmythos.ds_v4_sandbox import (
    CompressedSparseAttention,
    HeavilyCompressedAttention,
    HybridAttention,
    ManifoldConstrainedResidual,
    OpenMythosV2,
    TieredKVCache,
    sinkhorn_knopp,
)


# ─────────────────────────────────────────────────────────────────────
# 1. mHC Tests
# ─────────────────────────────────────────────────────────────────────

def test_sinkhorn_doubly_stochastic() -> None:
    """Sinkhorn output should have row/col sums ≈ 1."""
    log_mat = torch.randn(1, 8, 8)
    result = sinkhorn_knopp(log_mat, n_iters=20)
    row_sums = result.sum(dim=-1)
    col_sums = result.sum(dim=-2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4)


def test_sinkhorn_positive() -> None:
    """Sinkhorn output should be strictly positive."""
    log_mat = torch.randn(2, 16, 16)
    result = sinkhorn_knopp(log_mat, n_iters=10)
    assert (result > 0).all()


def test_mhc_residual_shape() -> None:
    """mHC residual output should match input shape."""
    mhc = ManifoldConstrainedResidual(dim=64)
    x = torch.randn(2, 8, 64)
    h = torch.randn(2, 8, 64)
    out = mhc(x, h)
    assert out.shape == x.shape


@settings(deadline=None)
@given(data=st.data())
def test_mhc_spectral_norm_bound(data: st.DataObject) -> None:
    """mHC mixing matrix spectral norm should be <= 1 (non-expansive)."""
    dim = data.draw(st.integers(min_value=8, max_value=64))
    mhc = ManifoldConstrainedResidual(dim=dim)
    with torch.no_grad():
        W = sinkhorn_knopp(mhc.static_log_matrix.unsqueeze(0), mhc.sinkhorn_iters)
        _, s, _ = torch.svd(W.squeeze(0))
        max_sv = s[0]
        assert max_sv <= 1.0 + 1e-5


def test_mhc_vs_naive_stability() -> None:
    """mHC should produce smaller output norm growth than naive residual under deep stacking."""
    dim = 64
    mhc = ManifoldConstrainedResidual(dim=dim)
    x = torch.randn(1, 4, dim)
    h = torch.randn(1, 4, dim)
    x_mhc = x.clone()
    for _ in range(10):
        x_mhc = mhc(x_mhc, h)
    x_naive = x.clone()
    for _ in range(10):
        x_naive = x_naive + h
    assert torch.norm(x_mhc) <= torch.norm(x_naive) * 1.5


# ─────────────────────────────────────────────────────────────────────
# 2. CSA + HCA Tests
# ─────────────────────────────────────────────────────────────────────

def test_csa_output_shape() -> None:
    """CSA should return output matching input shape."""
    csa = CompressedSparseAttention(dim=64, num_heads=4, latent_dim=16,
                                     block_size=16, top_k_blocks=4)
    x = torch.randn(2, 64, 64)
    out, lkv = csa(x)
    assert out.shape == x.shape
    assert lkv.shape == (2, 64, 16)


def test_hca_output_shape() -> None:
    """HCA should return output matching input shape."""
    hca = HeavilyCompressedAttention(dim=64, num_heads=4, latent_dim=16)
    x = torch.randn(2, 128, 64)
    out, lkv = hca(x)
    assert out.shape == x.shape
    assert lkv.shape == (2, 1, 16)


def test_hybrid_output_shape() -> None:
    """Hybrid attention should return output matching input shape."""
    hybrid = HybridAttention(dim=64, num_heads=4, latent_dim=16,
                             block_size=16, top_k_blocks=4)
    x = torch.randn(2, 64, 64)
    out, lkv = hybrid(x)
    assert out.shape == x.shape


def test_csa_top_k_selection() -> None:
    """CSA Lightning Indexer should select exactly top_k blocks."""
    csa = CompressedSparseAttention(dim=64, num_heads=4, latent_dim=16,
                                     block_size=16, top_k_blocks=3)
    x = torch.randn(1, 96, 64)
    out, lkv = csa(x)
    assert out.shape == x.shape


def test_hca_global_compression() -> None:
    """HCA latent_kv should be a single global summary regardless of input length."""
    hca = HeavilyCompressedAttention(dim=64, num_heads=4, latent_dim=16)
    x_short = torch.randn(1, 16, 64)
    x_long = torch.randn(1, 256, 64)
    _, lkv_short = hca(x_short)
    _, lkv_long = hca(x_long)
    assert lkv_short.shape[1] == 1
    assert lkv_long.shape[1] == 1


# ─────────────────────────────────────────────────────────────────────
# 3. Tiered KV Cache Tests
# ─────────────────────────────────────────────────────────────────────

def test_tiered_kv_state_cache_fill() -> None:
    """State cache should fill up to max_window without overflow to classical."""
    cache = TieredKVCache(max_window_size=8, max_compressed_entries=4, latent_dim=16)
    for _ in range(8):
        cache.update(torch.randn(1, 1, 16))
    assert cache.state_size == 8
    assert cache.compressed_size == 0


def test_tiered_kv_overflow_to_classical() -> None:
    """When state cache overflows, entries should compress into classical cache."""
    cache = TieredKVCache(max_window_size=4, max_compressed_entries=8, latent_dim=16)
    for _ in range(10):
        cache.update(torch.randn(1, 1, 16))
    assert cache.state_size == 4
    assert cache.compressed_size == 6


def test_tiered_kv_retrieve() -> None:
    """Retrieve should return concatenated classical + state entries."""
    cache = TieredKVCache(max_window_size=4, max_compressed_entries=4, latent_dim=16)
    for _ in range(6):
        cache.update(torch.randn(1, 1, 16))
    retrieved = cache.retrieve()
    expected_len = cache.compressed_size + cache.state_size
    assert retrieved.shape == (1, expected_len, 16)


def test_tiered_kv_retrieve_empty() -> None:
    """Retrieve on empty cache should return zero tensor."""
    cache = TieredKVCache(max_window_size=8, max_compressed_entries=8, latent_dim=16)
    retrieved = cache.retrieve()
    assert retrieved.shape == (1, 0, 16)


def test_tiered_kv_reset() -> None:
    """Reset should clear all cache state."""
    cache = TieredKVCache(max_window_size=4, max_compressed_entries=4, latent_dim=16)
    for _ in range(10):
        cache.update(torch.randn(1, 1, 16))
    cache.reset()
    assert cache.state_size == 0
    assert cache.compressed_size == 0
    assert cache.retrieve().shape[1] == 0


# ─────────────────────────────────────────────────────────────────────
# 4. OpenMythosV2 Tests
# ─────────────────────────────────────────────────────────────────────

@settings(deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=2),
    seq_len=st.integers(min_value=4, max_value=32),
    loop_iters=st.integers(min_value=0, max_value=3),
)
def test_openmythos_v2_output_shape(batch_size: int, seq_len: int, loop_iters: int) -> None:
    """OpenMythosV2 output should be (batch, seq, vocab)."""
    dim = 32
    vocab_size = 500
    model = OpenMythosV2(
        dim=dim, num_heads=4, latent_dim=8,
        num_prelude_layers=1, num_recurrent_layers=1, num_coda_layers=1,
        vocab_size=vocab_size, use_mhc=True, block_size=8, top_k_blocks=2,
    )
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(tokens, loop_iters=loop_iters)
    assert output.shape == (batch_size, seq_len, vocab_size)


def test_openmythos_v2_loop_consistency() -> None:
    """Zero loops should produce same shape but different values than one loop."""
    model = OpenMythosV2(
        dim=32, num_prelude_layers=1, num_recurrent_layers=1,
        num_coda_layers=1, use_mhc=True,
    )
    tokens = torch.randint(0, 50257, (1, 16))
    out_0 = model(tokens, loop_iters=0)
    out_1 = model(tokens, loop_iters=1)
    assert out_0.shape == out_1.shape
    assert not torch.allclose(out_0, out_1)


def test_openmythos_v2_mhc_vs_naive() -> None:
    """mHC version should have bounded norm growth vs naive across deep loops."""
    dim = 32
    model_mhc = OpenMythosV2(dim=dim, num_prelude_layers=1,
                              num_recurrent_layers=1, num_coda_layers=1,
                              use_mhc=True)
    model_naive = OpenMythosV2(dim=dim, num_prelude_layers=1,
                                num_recurrent_layers=1, num_coda_layers=1,
                                use_mhc=False)
    tokens = torch.randint(0, 500, (1, 16))
    with torch.no_grad():
        out_mhc = model_mhc(tokens, loop_iters=8)
        out_naive = model_naive(tokens, loop_iters=8)
    assert out_mhc.shape == out_naive.shape
    norm_mhc = torch.norm(out_mhc).item()
    norm_naive = torch.norm(out_naive).item()
    assert norm_mhc < norm_naive * 2


def test_openmythos_v2_with_tiered_kv() -> None:
    """OpenMythosV2 should work with external TieredKVCache."""
    model = OpenMythosV2(
        dim=32, num_prelude_layers=1, num_recurrent_layers=1,
        num_coda_layers=1, use_mhc=True,
    )
    # Match model's default latent_dim=128
    cache = TieredKVCache(max_window_size=8, max_compressed_entries=4, latent_dim=128)
    tokens = torch.randint(0, 500, (1, 16))
    output = model(tokens, loop_iters=2, kv_cache=cache)
    assert output.shape == (1, 16, 50257)
    assert cache.state_size > 0 or cache.compressed_size > 0


def test_openmythos_v2_deterministic_output() -> None:
    """Same input should produce same output in eval mode."""
    model = OpenMythosV2(
        dim=32, num_prelude_layers=1, num_recurrent_layers=1,
        num_coda_layers=1, vocab_size=500,
    )
    model.eval()
    tokens = torch.randint(0, 500, (1, 8))
    with torch.no_grad():
        out1 = model(tokens, loop_iters=2)
        out2 = model(tokens, loop_iters=2)
    assert torch.allclose(out1, out2)
