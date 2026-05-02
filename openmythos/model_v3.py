"""
OpenMythosV3 — Recurrent-Depth Transformer with DeepSeek V4 MoE integration.

Combines all V2 upgrades (mHC, CSA+HCA, Tiered KV) with MoE:
1. mHC — Manifold-Constrained Hyper-Connections
2. CSA+HCA — Hybrid Attention
3. Tiered KV Cache
4. MoE — Mixture of Experts with auxiliary-loss-free routing

The MoE replaces the dense FFN in MythosBlock, providing parameter-efficient
specialization within the weight-recycled recurrent loop.
"""

from typing import Optional, Tuple, cast

import torch
import torch.nn as nn
from opentelemetry import trace

from openmythos.ds_v4_sandbox import (
    HybridAttention,
    ManifoldConstrainedResidual,
    TieredKVCache,
)
from openmythos.moe import MoELayer

tracer = trace.get_tracer(__name__)


class MythosBlockV3(nn.Module):
    """
    Transformer block with mHC residuals, Hybrid Attention, and MoE FFN.

    Replaces the dense FFN with an MoE layer that routes tokens through
    top-k experts + shared expert.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        latent_dim: int,
        use_mhc: bool = True,
        block_size: int = 32,
        top_k_blocks: int = 8,
        num_experts: int = 16,
        moe_top_k: int = 2,
        use_hash_routing: bool = False,
        use_anticipatory: bool = False,
    ):
        super().__init__()
        self.use_mhc = use_mhc
        self.use_hash_routing = use_hash_routing
        self.ln1 = nn.LayerNorm(dim)

        if use_mhc:
            self.residual = ManifoldConstrainedResidual(dim)

        self.attn = HybridAttention(dim, num_heads, latent_dim, block_size, top_k_blocks)

        # MoE replaces dense FFN
        self.ln2 = nn.LayerNorm(dim)
        self.moe = MoELayer(
            dim,
            num_experts=num_experts,
            top_k=moe_top_k,
            use_hash_routing=use_hash_routing,
            use_anticipatory=use_anticipatory,
        )

    def forward(
        self,
        x: torch.Tensor,
        latent_kv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        current_loss: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, lkv = self.attn(self.ln1(x), latent_kv, mask)

        if self.use_mhc:
            x = self.residual(x, h)
        else:
            x = x + h

        # MoE forward
        moe_out, balance_loss = self.moe(
            self.ln2(x),
            token_ids=token_ids if self.use_hash_routing else None,
            current_loss=current_loss,
        )
        x = x + moe_out

        return x, lkv


class OpenMythosV3(nn.Module):
    """
    OpenMythos V3: Recurrent-Depth Transformer with full DS-V4 integration.

    Combines:
    1. mHC — Manifold-Constrained Hyper-Connections for stable deep loops
    2. CSA+HCA — Hybrid Attention for long-context sparse attention
    3. Tiered KV Cache — Sliding window + compressed historical entries
    4. MoE — Mixture of Experts with auxiliary-loss-free routing

    MoE Configuration:
    - num_experts: Number of routed experts per block (default 16 for research)
    - moe_top_k: Number of active experts per token (default 2)
    - use_hash_routing: Deterministic hash routing for prelude layers
    - use_anticipatory: Decouple routing from backbone to suppress loss spikes

    DS-V4 Reference:
    - V4-Flash: 256 experts, top-6 routing, 13B active, 284B total
    - V4-Pro: 384 experts, top-6 routing, 49B active, 1.6T total
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
        # MoE config
        num_experts: int = 16,
        moe_top_k: int = 2,
        use_hash_routing: bool = False,
        use_anticipatory: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.max_loop_iters = max_loop_iters
        self.use_mhc = use_mhc
        self.num_experts = num_experts

        self.embedding = nn.Embedding(vocab_size, dim)

        # Prelude: non-MoE or hash-routed (DS-V4 uses hash routing for first 3 layers)
        self.prelude = nn.ModuleList([
            MythosBlockV3(
                dim, num_heads, latent_dim, use_mhc=False,
                num_experts=num_experts, moe_top_k=moe_top_k,
                use_hash_routing=use_hash_routing and i < 3,
                use_anticipatory=False,  # No anticipatory in prelude
            )
            for i in range(num_prelude_layers)
        ])

        # Recurrent Block: full MoE with mHC
        self.recurrent_block = nn.ModuleList([
            MythosBlockV3(
                dim, num_heads, latent_dim, use_mhc=use_mhc,
                block_size=block_size, top_k_blocks=top_k_blocks,
                num_experts=num_experts, moe_top_k=moe_top_k,
                use_hash_routing=False,
                use_anticipatory=use_anticipatory,
            )
            for _ in range(num_recurrent_layers)
        ])

        # Coda: non-MoE (standard experts for final refinement)
        self.coda = nn.ModuleList([
            MythosBlockV3(
                dim, num_heads, latent_dim, use_mhc=False,
                num_experts=num_experts, moe_top_k=moe_top_k,
                use_hash_routing=False,
                use_anticipatory=False,
            )
            for _ in range(num_coda_layers)
        ])

        self.head = nn.Linear(dim, vocab_size)

    @tracer.start_as_current_span("openmythos_v3_forward")
    def forward(
        self,
        tokens: torch.Tensor,
        loop_iters: Optional[int] = None,
        kv_cache: Optional[TieredKVCache] = None,
    ) -> torch.Tensor:
        if loop_iters is None:
            loop_iters = self.max_loop_iters

        x = self.embedding(tokens)

        # Prelude
        for block in self.prelude:
            x, _ = block(x, token_ids=tokens)

        prelude_state = x

        # Recurrent Looping
        for i in range(loop_iters):
            with tracer.start_as_current_span(f"recurrent_loop_{i}"):
                x = x + prelude_state

                # Retrieve tiered KV if available
                latent_kv = None
                if kv_cache is not None:
                    latent_kv = kv_cache.retrieve()

                last_latent = None
                for block in self.recurrent_block:
                    x, last_latent = block(x, latent_kv, token_ids=tokens)

                # Update tiered KV cache
                if kv_cache is not None and last_latent is not None:
                    kv_cache.update(last_latent)

        # Coda
        for block in self.coda:
            x, _ = block(x, token_ids=tokens)

        return cast(torch.Tensor, self.head(x))
