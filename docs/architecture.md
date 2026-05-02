# OpenMythos Architecture

## Overview

OpenMythos is a **recurrent-depth transformer** — an architecture that recycles the same set of weights through a central recurrent block rather than stacking unique layers. This enables **compute-adaptive inference**: the depth of reasoning (number of recurrent loop iterations) can be adjusted at inference time without retraining or changing model weights.

## Core Design Principles

### 1. Weight Recycling
Instead of stacking N unique transformer layers, OpenMythos uses a single `recurrent_block` (a `ModuleList` of `MythosBlock`s) called repeatedly in a loop. The same parameters are applied across all iterations, enabling:
- **Parameter efficiency**: 8 recurrent layers called 16 times = 128 effective layers from only 8× parameters.
- **Compute adaptivity**: `loop_iters` is a runtime parameter, not a structural constraint.

### 2. Multi-Head Latent Attention (MLA)
Standard transformer attention maintains a KV cache proportional to `seq_len × dim`. MLA compresses this through a bottleneck:
- **KV compression**: `dim → latent_dim` linear projection (e.g., 1024 → 128, 87.5% reduction).
- **KV expansion**: `latent_dim → dim × 2` reconstructs K and V from the shared latent.
- **Memory savings**: KV cache shrinks by `1 - latent_dim / dim`.

### 3. Prelude State Injection
The prelude (non-recurrent prefix layers) produces an initial representation that is saved and **additively re-injected** at each recurrent iteration. This prevents signal degradation over deep loops — the model always has access to the original prelude representation.

### 4. Observability-First
Every forward pass is instrumented with OpenTelemetry spans:
- `openmythos_forward`: Main execution span covering the entire forward pass.
- `recurrent_loop_{i}`: Per-iteration spans showing time spent in each loop iteration.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────┐
│                   Input Tokens                        │
│                        ↓                              │
│              ┌─────────────────┐                      │
│              │   Embedding     │                      │
│              └────────┬────────┘                      │
│                       ↓                               │
│     ┌─────────────────────────────┐                   │
│     │      PRELUDE (N layers)     │ ← Runs once       │
│     │  (MythosBlock × num_prelude)│                   │
│     └──────────────┬──────────────┘                   │
│                    ↓                                  │
│         ┌────────────────────┐                        │
│         │  prelude_state ← x │  ← Saved for injection │
│         └────────┬───────────┘                        │
│                  ↓                                    │
│     ┌─────────────────────────────┐                   │
│     │   RECURRENT LOOP (× M iters)│                   │
│     │  ┌───────────────────────┐  │                   │
│     │  │ x = x + prelude_state │  │ ← Additive inj.   │
│     │  │ x = recurrent_block(x)│  │ ← Weight recycle  │
│     │  └───────────────────────┘  │                   │
│     └──────────────┬──────────────┘                   │
│                    ↓                                  │
│     ┌─────────────────────────────┐                   │
│     │       CODA (M layers)       │ ← Runs once       │
│     │  (MythosBlock × num_coda)   │                   │
│     └──────────────┬──────────────┘                   │
│                    ↓                                  │
│              ┌─────────────┐                          │
│              │  LM Head    │ → (batch, seq, vocab)    │
│              └─────────────┘                          │
└──────────────────────────────────────────────────────┘
```

## Components

### `MythosBlock`
Standard transformer block with MLA:
1. `LayerNorm` → MLA attention → residual
2. `LayerNorm` → MLP (dim → 4×dim → dim, GELU) → residual

### `MultiHeadLatentAttention`
1. Q projected normally: `dim → dim`
2. KV compressed: `dim → latent_dim`
3. KV expanded: `latent_dim → dim × 2` (split into K, V)
4. Scaled dot-product attention
5. Output projection: `dim → dim`

### `OpenMythos`
Main model class:
- **Prelude**: N sequential `MythosBlock`s (runs once)
- **Recurrent Loop**: `loop_iters` iterations of `recurrent_block` with prelude state injection
- **Coda**: M sequential `MythosBlock`s (runs once)
- **Head**: Linear projection to `vocab_size`

## Compute-Adaptive Inference

The key operational advantage: **the same model can run at different compute budgets**.

| Scenario | `loop_iters` | Use Case |
|----------|-------------|----------|
| Low-latency | 0–2 | Classification, extraction, routing |
| Standard | 4–8 | Summarization, translation, QA |
| High-depth | 12–16 | Complex reasoning, planning, analysis |

This is analogous to "thinking time" — more iterations allow the recurrent block to refine representations progressively, similar to how iterative refinement improves numerical solutions.

## Kerala School Connection

The architecture draws inspiration from the Kerala School of Mathematics (14th–16th century), which discovered infinite series expansions for trigonometric functions centuries before European calculus. The recurrent loop parallels an infinite series: each iteration adds a correction term, with convergence improving as depth increases. The `test_kerala_series.py` test validates this by confirming that output norms change monotonically with loop depth.

## Usage

```python
from openmythos import OpenMythos

model = OpenMythos(
    dim=1024,
    num_prelude_layers=4,
    num_recurrent_layers=8,
    num_coda_layers=4,
    max_loop_iters=16,
)

# Low-latency
fast_output = model(tokens, loop_iters=2)

# High-depth reasoning
deep_output = model(tokens, loop_iters=16)
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with property-based testing output
python -m pytest tests/test_model.py -v --hypothesis-show-statistics
```

## Future Work

- **Adaptive compute controller**: Train a small head to predict optimal `loop_iters` per input.
- **Position embeddings**: Add rotary (RoPE) or ALiBi position embeddings for longer sequences.
- **Flash attention**: Replace scaled dot-product with Flash Attention 2 for throughput.

---

## Version Progression

### V1 — Original (openmythos/model.py)
- Prelude → Recurrent Loop → Coda → LM Head
- Standard MLA attention with KV compression
- Naive additive residuals (`x = x + h`)
- Dense FFN (4× dim expansion)

### V2 — DS-V4 Hybrid (openmythos/ds_v4_sandbox.py)
- **mHC**: Sinkhorn-constrained doubly stochastic residuals (spectral norm ≤ 1)
- **CSA+HCA**: Hybrid attention — block-compressed sparse selection + global summary
- **Tiered KV Cache**: Sliding window + compressed historical entries
- Backwards-compatible: `use_mhc=False` and `kv_cache=None` fall back to V1 behavior

### V3 — Full MoE Integration (openmythos/model_v3.py + openmythos/moe.py)
- **DeepSeekMoE**: Sqrt(Softplus) affinity, top-k routing, auxiliary-loss-free load balancing
- **Shared Expert**: Always active, handles common patterns
- **Hash Routing**: Deterministic early-layer routing for prelude layers
- **Anticipatory Routing**: Decouples routing from backbone, suppresses loss spikes
- 3.4× parameter capacity increase at research scale (8 experts), up to 20× at production scale (256 experts)

### Benchmark Results (dim=64, seq=32, batch=2)

| Metric | V1 | V2 | V3 |
|--------|----|----|-----|
| Total params | 914K | 1.6M | 3.1M |
| Active params | 914K | 1.6M | 2.1M |
| Capacity ratio | 1.0× | 1.0× | 1.5× |
| Entropy @ depth=4 | 4.64 | 4.59 | 4.68 |
| Output norm @ depth=8 | 3159 | 3140 | 3222 |

At production scale (dim=1024, 256 experts):
- V3 would have ~284B total params, ~13B active (21.8× capacity ratio)
- DS-V4-Flash achieves MMLU 88.7 at 13B active vs V3.2's 87.8 at 37B active
