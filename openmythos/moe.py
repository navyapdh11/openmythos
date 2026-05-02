"""
DeepSeekMoE module for OpenMythos.

Implements the Mixture-of-Experts architecture from DeepSeek V4:
1. Expert FFNs with configurable count (research scale: 16, prod scale: 256+)
2. Sqrt(Softplus) affinity routing (DS-V4 changed from Sigmoid)
3. Top-k routing with auxiliary-loss-free load balancing
4. Shared expert (always active, handles common patterns)
5. Hash routing for prelude layers (deterministic early-layer routing)
6. Anticipatory routing (decouple routing from backbone, suppress loss spikes)
7. Sequence-wise balance loss (lightweight, prevents extreme intra-sequence imbalance)

Architecture reference: DeepSeek V4 TAAC 2026 paper, Section 2.3 (DeepSeekMoE).
Benchmarks: DS-V4-Flash (13B active, 256 experts) achieves MMLU 88.7 vs V3.2's 87.8 (37B active).
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
# 1. Expert Networks
# ─────────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """Single expert FFN. Follows the standard transformer FFN pattern."""

    def __init__(self, dim: int, intermediate_dim: Optional[int] = None):
        super().__init__()
        inter = intermediate_dim or dim * 4
        self.fc1 = nn.Linear(dim, inter)
        self.fc2 = nn.Linear(inter, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))  # type: ignore[no-any-return]


class SharedExpert(nn.Module):
    """
    Shared expert: always active for every token.
    Handles common/universal patterns that all tokens need.
    DS-V4 uses a smaller intermediate dimension (2048 Flash, 3072 Pro)
    vs routed experts (4*dim).
    """

    def __init__(self, dim: int, intermediate_dim: Optional[int] = None):
        super().__init__()
        # DS-V4 uses smaller intermediate for shared expert
        inter = intermediate_dim or dim * 2
        self.fc1 = nn.Linear(dim, inter)
        self.fc2 = nn.Linear(inter, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))  # type: ignore[no-any-return]


# ─────────────────────────────────────────────────────────────────────
# 2. Routing
# ─────────────────────────────────────────────────────────────────────

def sqrt_softplus(x: torch.Tensor) -> torch.Tensor:
    """
    DS-V4 affinity function: Sqrt(Softplus(x)).
    Replaced Sigmoid in V4 for better gradient flow and routing stability.
    Softplus(x) = log(1 + exp(x)) → always positive.
    Sqrt compresses large values, expands small ones → more uniform routing.
    """
    return torch.sqrt(F.softplus(x))


class MoERouter(nn.Module):
    """
    Top-k routing with auxiliary-loss-free load balancing.

    Per DS-V4 spec:
    - Affinity: Sqrt(Softplus(gate @ x)) instead of Sigmoid
    - Load balancing: bias-based, no auxiliary loss penalty
    - Sequence-wise balance loss: lightweight prevention of extreme imbalance
    - Top-k: selects k highest-affinity experts per token
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 2,
        balance_bias_lr: float = 0.001,
        balance_loss_weight: float = 0.0001,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_weight = balance_loss_weight

        # Gate network: projects token to expert affinity scores
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # Per-expert bias for load balancing (updated via balance_bias_lr)
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))

        # Register bias update rate (not trained via gradient)
        self.register_buffer("balance_bias_lr", torch.tensor(balance_bias_lr))

    def forward(
        self, x: torch.Tensor, return_loss: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to top-k experts.

        Args:
            x: Input tokens (batch, seq, dim)
            return_loss: Whether to compute balance loss

        Returns:
            expert_indices: (batch, seq, top_k) indices of selected experts
            expert_weights: (batch, seq, top_k) routing weights (softmax normalized)
            balance_loss: Scalar loss for load balancing (0.0 if return_loss=False)
        """
        b, t, _ = x.shape

        # Compute affinity scores: Sqrt(Softplus(gate(x))) + bias
        logits = self.gate(x)  # (b, t, num_experts)
        logits = logits + self.expert_bias  # Add load balancing bias
        affinity = sqrt_softplus(logits)  # DS-V4 affinity function

        # Top-k selection
        top_weights, top_indices = torch.topk(affinity, self.top_k, dim=-1)  # (b, t, k)

        # Normalize weights via softmax over selected experts
        top_weights = F.softmax(top_weights, dim=-1)  # (b, t, k)

        # Sequence-wise balance loss (lightweight)
        balance_loss_val: torch.Tensor
        if return_loss:
            # Compute expert usage across sequence
            one_hot = F.one_hot(top_indices, num_classes=self.num_experts).float()  # (b, t, k, E)
            usage = one_hot.sum(dim=[0, 2])  # (E,) total usage per expert
            usage = usage / (usage.sum() + 1e-8)  # Normalize to distribution
            target_usage = 1.0 / self.num_experts  # Uniform target
            balance_loss_val = ((usage - target_usage) ** 2).mean() * self.balance_loss_weight
        else:
            balance_loss_val = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return top_indices, top_weights, balance_loss_val

    @torch.no_grad()
    def update_balance_bias(self, expert_indices: torch.Tensor) -> None:
        """
        Update expert biases based on routing statistics.
        Increases bias for under-used experts, decreases for over-used.
        This is the auxiliary-loss-free load balancing mechanism.

        Args:
            expert_indices: (batch, seq, top_k) selected expert indices
        """
        # Count expert usage
        flat_indices = expert_indices.flatten()  # (batch * seq * top_k)
        counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
        counts = counts / (counts.sum() + 1e-8)

        # Update bias: increase for under-used, decrease for over-used
        target = 1.0 / self.num_experts
        lr = self.balance_bias_lr.item()  # type: ignore[operator]
        self.expert_bias.data += lr * (target - counts)


class HashRouter:
    """
    Hash routing for prelude layers (first 3 MoE layers per DS-V4).
    Routes tokens using a predefined hash function on the token ID,
    bypassing learned gating for early layers.

    This provides stable, deterministic routing in early layers where
    learned routing is noisy due to untrained representations.
    """

    def __init__(self, num_experts: int, top_k: int = 2):
        self.num_experts = num_experts
        self.top_k = top_k

    def __call__(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route via hash of token IDs.

        Args:
            token_ids: (batch, seq) token IDs for hashing

        Returns:
            expert_indices: (batch, seq, top_k) deterministic expert assignments
            expert_weights: (batch, seq, top_k) uniform weights (1/top_k)
        """
        b, t = token_ids.shape

        # Deterministic hash-based routing (one expert per top-k slot)
        expert_indices = torch.zeros(b, t, self.top_k, dtype=torch.long, device=token_ids.device)
        for k_idx in range(self.top_k):
            # Hash: combine token ID with slot index for deterministic diversity
            hash_input = token_ids * (k_idx + 1) + 7  # Simple multiplicative hash
            # Use bitwise operations for fast hashing
            hash_input = ((hash_input >> 16) ^ hash_input) * 0x45D9F3B
            hash_input = ((hash_input >> 16) ^ hash_input)
            expert_indices[:, :, k_idx] = hash_input % self.num_experts

        expert_weights = torch.ones(b, t, self.top_k, device=token_ids.device) / self.top_k
        return expert_indices, expert_weights


# ─────────────────────────────────────────────────────────────────────
# 3. Anticipatory Routing
# ─────────────────────────────────────────────────────────────────────

class AnticipatoryRouter(nn.Module):
    """
    Anticipatory routing: decouples backbone and routing parameter updates.

    Per DS-V4 spec:
    - At step t, forward features use current params θ_t
    - Routing indices are pre-computed using historical params θ_{t-Δt}
    - Activated only upon loss-spike detection
    - Adds ~20% wall-clock overhead when active, negligible overall

    This breaks the "vicious cycle" where routing updates cause loss spikes
    that destabilize training.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 2,
        history_window: int = 100,
        spike_threshold: float = 2.0,
        balance_bias_lr: float = 0.001,
    ):
        super().__init__()
        self.primary = MoERouter(dim, num_experts, top_k, balance_bias_lr=balance_bias_lr)
        self.frozen: Optional[MoERouter] = None  # Historical snapshot
        self.history_window = history_window
        self.spike_threshold = spike_threshold
        self.loss_history: list[float] = []
        self.is_active = False

    def forward(
        self, x: torch.Tensor, current_loss: Optional[float] = None,
        return_loss: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route using either frozen (historical) or current router.

        Args:
            x: Input tokens (batch, seq, dim)
            current_loss: Current step loss for spike detection
            return_loss: Whether to compute balance loss

        Returns:
            expert_indices, expert_weights, balance_loss
        """
        # Update loss history and check for spikes
        if current_loss is not None:
            self.loss_history.append(current_loss)
            if len(self.loss_history) > self.history_window:
                self.loss_history.pop(0)

            # Detect loss spike
            if len(self.loss_history) >= 2:
                recent_avg = sum(self.loss_history[-3:]) / 3
                prev_avg = sum(self.loss_history[:-3]) / max(len(self.loss_history) - 3, 1)
                if recent_avg > prev_avg * self.spike_threshold and not self.is_active:
                    self._activate(x)
                elif self.is_active and recent_avg < prev_avg * 1.1:
                    self._deactivate()

        if self.is_active and self.frozen is not None:
            return self.frozen(x, return_loss=return_loss)  # type: ignore[no-any-return]
        else:
            return self.primary(x, return_loss=return_loss)  # type: ignore[no-any-return]

    def _activate(self, x: torch.Tensor) -> None:
        """Freeze current router parameters as historical snapshot."""
        self.frozen = MoERouter(
            self.primary.dim, self.primary.num_experts, self.primary.top_k
        )
        self.frozen.load_state_dict(self.primary.state_dict())
        self.frozen.eval()
        self.frozen.requires_grad_(False)
        self.frozen = self.frozen.to(x.device)
        self.is_active = True

    def _deactivate(self) -> None:
        """Resume using current router."""
        self.frozen = None
        self.is_active = False

    @torch.no_grad()
    def update_balance_bias(self, expert_indices: torch.Tensor) -> None:
        """Forward bias update to the active router."""
        self.primary.update_balance_bias(expert_indices)
        if self.frozen is not None:
            self.frozen.update_balance_bias(expert_indices)


# ─────────────────────────────────────────────────────────────────────
# 4. MoE Layer
# ─────────────────────────────────────────────────────────────────────

class MoELayer(nn.Module):
    """
    Complete MoE layer with routing, experts, and shared expert.

    Replaces the dense FFN in a transformer block.

    Per DS-V4 spec:
    - top_k routed experts activated per token
    - 1 shared expert always active
    - Auxiliary-loss-free load balancing with sequence-wise balance loss
    - Optional hash routing for early layers
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 16,
        top_k: int = 2,
        intermediate_dim: Optional[int] = None,
        shared_intermediate_dim: Optional[int] = None,
        use_hash_routing: bool = False,
        use_anticipatory: bool = False,
        balance_bias_lr: float = 0.001,
        balance_loss_weight: float = 0.0001,
    ):
        super().__init__()
        self.use_hash_routing = use_hash_routing
        self.num_experts = num_experts

        # Expert FFNs
        self.experts = nn.ModuleList([
            Expert(dim, intermediate_dim) for _ in range(num_experts)
        ])

        # Shared expert (always active)
        self.shared_expert = SharedExpert(dim, shared_intermediate_dim)

        # Router (use Any to handle both MoERouter and AnticipatoryRouter types)
        if use_anticipatory:
            self._router: Any = AnticipatoryRouter(
                dim, num_experts, top_k, balance_bias_lr=balance_bias_lr
            )
        else:
            self._router = MoERouter(dim, num_experts, top_k, balance_bias_lr=balance_bias_lr)

        # Hash router (for prelude layers)
        if use_hash_routing:
            self.hash_router = HashRouter(num_experts, top_k)

        # Scaling factor for stable training (DS-V4 uses sqrt(top_k))
        self.routing_scale = 1.0 / (top_k ** 0.5)

    def forward(
        self,
        x: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        current_loss: Optional[float] = None,
        return_loss: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens through selected experts, combine outputs.

        Args:
            x: Input tokens (batch, seq, dim)
            token_ids: (batch, seq) required for hash routing
            current_loss: Current loss for anticipatory routing spike detection
            return_loss: Whether to compute balance loss

        Returns:
            output: Combined expert output (batch, seq, dim)
            balance_loss: Scalar load balancing loss
        """
        b, t, dim = x.shape

        # Determine routing
        if self.use_hash_routing and token_ids is not None:
            expert_indices, expert_weights = self.hash_router(token_ids)
            balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        elif isinstance(self._router, AnticipatoryRouter):
            expert_indices, expert_weights, balance_loss = self._router(
                x, current_loss=current_loss, return_loss=return_loss
            )
        else:
            expert_indices, expert_weights, balance_loss = self._router(x, return_loss=return_loss)

        # Update balance biases (auxiliary-loss-free load balancing)
        with torch.no_grad():
            if isinstance(self._router, AnticipatoryRouter):
                self._router.update_balance_bias(expert_indices)
            elif isinstance(self._router, MoERouter):
                self._router.update_balance_bias(expert_indices)

        # Compute expert outputs using scatter-gather pattern
        # Flatten for processing
        flat_x = x.view(b * t, dim)  # (batch*seq, dim)
        flat_indices = expert_indices.view(b * t, -1)  # (batch*seq, top_k)
        flat_weights = expert_weights.view(b * t, -1)  # (batch*seq, top_k)

        # Initialize output accumulator
        output = torch.zeros_like(flat_x)  # (batch*seq, dim)

        # Route each token to its selected experts
        actual_top_k = expert_indices.shape[-1]
        for k_idx in range(actual_top_k):
            expert_idx = flat_indices[:, k_idx]  # (batch*seq)
            weight = flat_weights[:, k_idx].unsqueeze(1)  # (batch*seq, 1)

            # Process through each expert
            for e in range(self.num_experts):
                mask = (expert_idx == e)  # (batch*seq)
                if mask.any():
                    selected_input = flat_x[mask]
                    expert_output = self.experts[e](selected_input)
                    output[mask] += expert_output * weight[mask] * self.routing_scale

        # Add shared expert output (always active for all tokens)
        shared_output = self.shared_expert(flat_x)
        output = output + shared_output

        # Reshape back
        output = output.view(b, t, dim)
        return output, balance_loss


# ─────────────────────────────────────────────────────────────────────
# 5. MoE Block Statistics
# ─────────────────────────────────────────────────────────────────────

def compute_moe_stats(
    model: nn.Module,
    num_experts: int,
    num_recurrent_layers: int,
    num_prelude_layers: int,
    num_coda_layers: int,
    top_k: int = 2,
) -> dict[str, Any]:
    """
    Compute MoE parameter statistics for an OpenMythosV3 model.

    Returns a dict with:
    - total_params: Total parameter count (all experts stored)
    - active_params: Active parameter count (only routed + shared experts per forward)
    - capacity_ratio: total_params / active_params
    - expert_count: Number of expert networks
    """
    total_params = sum(p.numel() for p in model.parameters())

    # Estimate expert params (rough: dim * 4*dim + 4*dim * dim = 8*dim^2 per expert)
    dim_val = getattr(model, "dim", 1024)
    if isinstance(dim_val, int):
        expert_params = 8 * dim_val * dim_val
    else:
        expert_params = 8 * 1024 * 1024

    num_moe_blocks = num_prelude_layers + num_recurrent_layers + num_coda_layers
    inactive_experts = num_experts - top_k - 1  # -1 for shared expert
    inactive_params = inactive_experts * expert_params * num_moe_blocks

    active_params = total_params - inactive_params
    capacity = round(float(total_params) / max(float(active_params), 1.0), 2)

    return {
        "total_params": total_params,
        "active_params": max(active_params, total_params // num_experts),
        "capacity_ratio": capacity,
        "expert_count": num_experts * num_moe_blocks,
    }
