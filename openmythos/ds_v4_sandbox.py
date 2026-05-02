"""
DeepSeek V4 integration sandbox for OpenMythos.

Implements three architectural upgrades from DeepSeek V4:
1. mHC — Manifold-Constrained Hyper-Connections (Sinkhorn doubly stochastic residuals)
2. CSA+HCA — Hybrid Attention (Compressed Sparse + Heavily Compressed Attention)
3. Tiered KV Cache — Sliding window + compressed KV tiers

Each module is independently testable. Integration into OpenMythosV2 is at the bottom.
"""

from typing import Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
# 1. Manifold-Constrained Hyper-Connections (mHC)
# ─────────────────────────────────────────────────────────────────────
# Replaces naive additive residuals (x = x + h) with Sinkhorn-constrained
# doubly stochastic mixing matrices that bound spectral norm ≤ 1.
# This prevents gradient explosion in deep recurrent loops.

def sinkhorn_knopp(log_matrix: torch.Tensor, n_iters: int = 10) -> torch.Tensor:
    """
    Convert a log-parameterized matrix to a doubly stochastic matrix
    via iterative Sinkhorn-Knopp normalization.

    Args:
        log_matrix: (batch, dim, dim) log-space parameters
        n_iters: Number of row/column normalization iterations

    Returns:
        Doubly stochastic matrix (batch, dim, dim)
    """
    # Exponentiate to get positive values
    matrix = torch.exp(log_matrix)
    for _ in range(n_iters):
        # Row normalize
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + 1e-8)
        # Column normalize
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + 1e-8)
    return matrix


class ManifoldConstrainedResidual(nn.Module):
    """
    mHC: Manifold-Constrained Hyper-Connection.

    Instead of x = x + h, uses x = x + W @ h where W is doubly stochastic.
    The Sinkhorn constraint bounds spectral norm ≤ 1, ensuring non-expansive
    signal propagation across deep recurrent stacks.

    Uses static bias + dynamic input-dependent component per DS-V4 spec.
    """

    def __init__(self, dim: int, sinkhorn_iters: int = 10):
        super().__init__()
        self.dim = dim
        self.sinkhorn_iters = sinkhorn_iters

        # Static bias (learnable, initialized to near-identity mixing)
        self.static_log_matrix = nn.Parameter(torch.zeros(dim, dim))

        # Dynamic projection: produces input-dependent modulation
        self.dynamic_proj = nn.Linear(dim, dim * dim)

        # Gating factor to blend static vs dynamic (initialized to favor static)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute constrained residual: x + W @ h.

        Args:
            x: Residual input (batch, seq, dim)
            h: Block output to mix in (batch, seq, dim)

        Returns:
            Mixed output (batch, seq, dim)
        """
        b, t, _ = x.shape

        # Static doubly stochastic matrix
        W_static = sinkhorn_knopp(self.static_log_matrix.unsqueeze(0), self.sinkhorn_iters)  # (1, d, d)

        # Dynamic component: produce log-matrix from input mean across sequence
        x_mean = x.mean(dim=1)  # (b, d)
        log_dyn = self.dynamic_proj(x_mean).view(b, self.dim, self.dim)  # (b, d, d)
        W_dyn = sinkhorn_knopp(log_dyn, self.sinkhorn_iters)  # (b, d, d)

        # Blend static and dynamic via gate
        alpha = torch.sigmoid(self.gate)
        W = (1 - alpha) * W_static + alpha * W_dyn  # (b, d, d)

        # Apply: x + W @ h
        h_transformed = torch.bmm(W, h.transpose(1, 2)).transpose(1, 2)  # (b, t, d)
        return x + h_transformed


# ─────────────────────────────────────────────────────────────────────
# 2. CSA + HCA — Hybrid Attention
# ─────────────────────────────────────────────────────────────────────
# CSA: Compresses every m tokens into one KV entry, selects top-k via
#      a low-rank "Lightning Indexer" for sparse attention.
# HCA: Aggressively compresses every m' tokens (m' >> m) into one entry,
#      applies dense attention without sparse selection (global summary).

class LightningIndexer(nn.Module):
    """
    Low-rank scoring module for CSA top-k block selection.
    Projects compressed KV blocks to scalar scores, returns top-k indices.
    """

    def __init__(self, latent_dim: int, rank: int = 16):
        super().__init__()
        self.low_rank_proj = nn.Sequential(
            nn.Linear(latent_dim, rank),
            nn.GELU(),
            nn.Linear(rank, 1),
        )

    def forward(self, kv_blocks: torch.Tensor, k: int) -> torch.Tensor:
        """
        Args:
            kv_blocks: (batch, num_blocks, latent_dim)
            k: Number of top blocks to select

        Returns:
            top_k_indices: (batch, k) indices of selected blocks
        """
        scores = self.low_rank_proj(kv_blocks).squeeze(-1)  # (batch, num_blocks)
        _, top_k_indices = torch.topk(scores, k, dim=-1)
        return top_k_indices


class CompressedSparseAttention(nn.Module):
    """
    CSA: Compresses every `block_size` tokens into one KV entry,
    uses Lightning Indexer to select top-k blocks, applies sparse attention.
    """

    def __init__(self, dim: int, num_heads: int, latent_dim: int,
                 block_size: int = 32, top_k_blocks: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.latent_dim = latent_dim
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks

        self.q_proj = nn.Linear(dim, dim)
        self.kv_compress = nn.Linear(dim, latent_dim)
        self.kv_up = nn.Linear(latent_dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.indexer = LightningIndexer(latent_dim, rank=16)

    def forward(self, x: torch.Tensor,
                latent_kv: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c = x.size()

        # Q projected normally
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (b, h, t, d_h)

        # Handle empty or too-small cache: compute latent from input directly
        if latent_kv is None or latent_kv.shape[1] == 0 or latent_kv.shape[1] < self.block_size:
            latent_kv = self.kv_compress(x)  # (b, t, latent_dim)

        # Pad sequence to block boundary
        pad_len = (self.block_size - (t % self.block_size)) % self.block_size
        if pad_len > 0:
            padded_kv = F.pad(latent_kv, (0, 0, 0, pad_len))
        else:
            padded_kv = latent_kv

        num_blocks = padded_kv.shape[1] // self.block_size

        # Reshape into blocks and mean-pool
        blocked = padded_kv.view(b, num_blocks, self.block_size, self.latent_dim)
        kv_blocks = blocked.mean(dim=2)  # (b, num_blocks, latent_dim)

        # Select top-k blocks via Lightning Indexer
        actual_k = min(self.top_k_blocks, num_blocks)
        top_indices = self.indexer(kv_blocks, actual_k)  # (b, k)

        # Gather selected block latent vectors and reconstruct K, V
        batch_idx = torch.arange(b, device=x.device).unsqueeze(1)  # (b, 1)
        selected_latents = kv_blocks[batch_idx, top_indices]  # (b, k, latent_dim)
        selected_kv = self.kv_up(selected_latents).view(b, actual_k, 2, self.num_heads, self.head_dim)
        k = selected_kv[:, :, 0].transpose(1, 2)  # (b, h, k, d_h)
        v = selected_kv[:, :, 1].transpose(1, 2)

        # Compute attention over selected blocks only
        # Expand q to match k/v block structure (simplified: attend q positions to k selected blocks)
        q_sparse = q[:, :, :actual_k, :]  # Use first k query positions for sparse match
        attn = (q_sparse @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)

        out_sparse = (attn @ v).transpose(1, 2).contiguous()  # (b, k, h, d_h)
        # Map back to full sequence dimension (simplified broadcast)
        out_full = out_sparse.mean(dim=1, keepdim=True).expand(b, t, self.num_heads, self.head_dim)
        out_full = out_full.reshape(b, t, c)

        return self.out_proj(out_full), latent_kv


class HeavilyCompressedAttention(nn.Module):
    """
    HCA: Aggressively compresses the entire sequence into a single global
    summary token, applies dense attention for global summarization.
    """

    def __init__(self, dim: int, num_heads: int, latent_dim: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.latent_dim = latent_dim

        self.q_proj = nn.Linear(dim, dim)
        self.kv_compress = nn.Linear(dim, latent_dim)
        self.kv_up = nn.Linear(latent_dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor,
                latent_kv: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, c = x.size()

        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        # Global compression: use cached latent directly or compute from input
        if latent_kv is None or latent_kv.shape[1] == 0:
            # No cache — compute global summary from input
            global_latent = self.kv_compress(x.mean(dim=1, keepdim=True))  # (b, 1, latent_dim)
        else:
            # Cache has pre-compressed latent entries — mean-pool to single global summary
            global_latent = latent_kv.mean(dim=1, keepdim=True)  # (b, 1, latent_dim)

        kv = self.kv_up(global_latent).view(b, 1, 2, self.num_heads, self.head_dim)
        k = kv[:, :, 0].transpose(1, 2)  # (b, h, 1, d_h)
        v = kv[:, :, 1].transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(out), global_latent


class HybridAttention(nn.Module):
    """
    DS-V4 Hybrid Attention: combines CSA (fine-grained retrieval) and
    HCA (global summarization) in a single module.
    """

    def __init__(self, dim: int, num_heads: int, latent_dim: int,
                 block_size: int = 32, top_k_blocks: int = 8):
        super().__init__()
        self.csa = CompressedSparseAttention(dim, num_heads, latent_dim, block_size, top_k_blocks)
        self.hca = HeavilyCompressedAttention(dim, num_heads, latent_dim)
        self.fusion = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor,
                latent_kv: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        out_csa, lkv_csa = self.csa(x, latent_kv, mask)
        out_hca, lkv_hca = self.hca(x, latent_kv)

        # Fuse CSA + HCA outputs
        combined = torch.cat([out_csa, out_hca], dim=-1)
        return self.fusion(combined), lkv_csa  # Use CSA latent for downstream


# ─────────────────────────────────────────────────────────────────────
# 3. Tiered KV Cache
# ─────────────────────────────────────────────────────────────────────
# Two-tier layout per DS-V4 Section 3.6.1:
# - State Cache: Fixed-size sliding window for recent tokens
# - Classical KV Cache: Compressed entries for historical context

class TieredKVCache:
    """
    Manages a two-tier KV cache:
    1. State Cache: Sliding window (recent uncompressed tokens)
    2. Classical Cache: Compressed KV entries for older context
    """

    def __init__(self, max_window_size: int = 512, max_compressed_entries: int = 256,
                 latent_dim: int = 128, device: torch.device = torch.device("cpu")):
        self.max_window = max_window_size
        self.max_compressed = max_compressed_entries
        self.latent_dim = latent_dim
        self.device = device

        # State cache: recent tokens' latent KV (sliding window)
        self.state_cache: Optional[torch.Tensor] = None  # (b, window_size, latent_dim)
        self.state_size = 0

        # Classical cache: compressed historical entries
        self.classical_cache: Optional[torch.Tensor] = None  # (b, max_compressed, latent_dim)
        self.compressed_size = 0

    def update(self, new_latent_kv: torch.Tensor) -> None:
        """
        Add new latent KV entries to the cache.
        New entries go into state cache first; overflow gets compressed
        into classical cache.

        Args:
            new_latent_kv: (batch, seq_len, latent_dim) new latent vectors
        """
        b = new_latent_kv.shape[0]

        if self.state_cache is None:
            self.state_cache = torch.zeros(
                b, self.max_window, self.latent_dim, device=self.device
            )
            self.state_size = 0

        remaining = new_latent_kv
        for i in range(remaining.shape[1]):
            entry = remaining[:, i:i + 1, :]  # (b, 1, latent_dim)

            if self.state_size < self.max_window:
                # Add to state cache
                self.state_cache[:, self.state_size:self.state_size + 1, :] = entry
                self.state_size += 1
            else:
                # State cache full — compress oldest entry to classical cache
                if self.classical_cache is None:
                    self.classical_cache = torch.zeros(
                        b, self.max_compressed, self.latent_dim, device=self.device
                    )
                    self.compressed_size = 0

                if self.compressed_size < self.max_compressed:
                    # Compress oldest state entry + new entry
                    oldest = self.state_cache[:, 0:1, :]
                    compressed = (oldest + entry) / 2  # Simple average compression
                    self.classical_cache[:, self.compressed_size:self.compressed_size + 1, :] = compressed
                    self.compressed_size += 1

                    # Shift state cache left (drop oldest, add new at end)
                    self.state_cache = torch.cat([
                        self.state_cache[:, 1:, :],
                        entry
                    ], dim=1)

    def retrieve(self) -> torch.Tensor:
        """
        Retrieve combined KV cache for attention.
        Returns concatenated classical + state cache entries.

        Returns:
            Combined latent KV (batch, total_entries, latent_dim)
        """
        parts = []
        if self.classical_cache is not None and self.compressed_size > 0:
            parts.append(self.classical_cache[:, :self.compressed_size, :])
        if self.state_cache is not None and self.state_size > 0:
            parts.append(self.state_cache[:, :self.state_size, :])

        if not parts:
            return torch.zeros(1, 0, self.latent_dim, device=self.device)

        return torch.cat(parts, dim=1)

    def reset(self) -> None:
        """Clear all cached entries."""
        self.state_cache = None
        self.classical_cache = None
        self.state_size = 0
        self.compressed_size = 0


# ─────────────────────────────────────────────────────────────────────
# OpenMythosV2 — Integrated with DS-V4 upgrades
# ─────────────────────────────────────────────────────────────────────

class MythosBlockV2(nn.Module):
    """Transformer block with mHC residuals and Hybrid Attention."""

    def __init__(self, dim: int, num_heads: int, latent_dim: int,
                 use_mhc: bool = True, block_size: int = 32, top_k_blocks: int = 8):
        super().__init__()
        self.use_mhc = use_mhc
        self.ln1 = nn.LayerNorm(dim)

        if use_mhc:
            self.residual = ManifoldConstrainedResidual(dim)
        self.attn = HybridAttention(dim, num_heads, latent_dim, block_size, top_k_blocks)

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor,
                latent_kv: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h, lkv = self.attn(self.ln1(x), latent_kv, mask)

        if self.use_mhc:
            x = self.residual(x, h)  # mHC constrained residual
        else:
            x = x + h  # Fallback to naive additive

        x = x + self.mlp(self.ln2(x))
        return x, lkv


class OpenMythosV2(nn.Module):
    """
    OpenMythos V2: Recurrent-Depth Transformer with DeepSeek V4 upgrades.

    Integrates:
    1. mHC — Manifold-Constrained Hyper-Connections for stable deep loops
    2. CSA+HCA — Hybrid Attention for long-context sparse attention
    3. Tiered KV Cache — Sliding window + compressed historical entries
    """

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        latent_dim: int = 128,
        num_prelude_layers: int = 4,
        num_recurrent_layers: int = 8,
        num_coda_layers: int = 4,
        max_loop_iters: int = 16,
        vocab_size: int = 50257,
        use_mhc: bool = True,
        block_size: int = 32,
        top_k_blocks: int = 8,
        tiered_window: int = 512,
        tiered_compressed: int = 256,
    ):
        super().__init__()
        self.dim = dim
        self.max_loop_iters = max_loop_iters
        self.use_mhc = use_mhc
        self.tiered_window = tiered_window
        self.tiered_compressed = tiered_compressed

        self.embedding = nn.Embedding(vocab_size, dim)

        self.prelude = nn.ModuleList([
            MythosBlockV2(dim, num_heads, latent_dim, use_mhc=False)
            for _ in range(num_prelude_layers)
        ])

        self.recurrent_block = nn.ModuleList([
            MythosBlockV2(dim, num_heads, latent_dim, use_mhc=use_mhc,
                          block_size=block_size, top_k_blocks=top_k_blocks)
            for _ in range(num_recurrent_layers)
        ])

        self.coda = nn.ModuleList([
            MythosBlockV2(dim, num_heads, latent_dim, use_mhc=False)
            for _ in range(num_coda_layers)
        ])

        self.head = nn.Linear(dim, vocab_size)

    def forward(self, tokens: torch.Tensor,
                loop_iters: Optional[int] = None,
                kv_cache: Optional[TieredKVCache] = None) -> torch.Tensor:
        if loop_iters is None:
            loop_iters = self.max_loop_iters

        x = self.embedding(tokens)

        # Prelude
        for block in self.prelude:
            x, _ = block(x)

        prelude_state = x

        # Recurrent Looping
        for i in range(loop_iters):
            # Inject prelude state
            x = x + prelude_state

            # Retrieve tiered KV if available
            latent_kv = None
            if kv_cache is not None:
                latent_kv = kv_cache.retrieve()

            for block in self.recurrent_block:
                x, new_latent = block(x, latent_kv)

            # Update tiered KV cache
            if kv_cache is not None:
                kv_cache.update(new_latent)

        # Coda
        for block in self.coda:
            x, _ = block(x)

        return cast(torch.Tensor, self.head(x))
