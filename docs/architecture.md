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
- **KV cache past-state**: Enable true recurrence by passing latent_kv between loop iterations.
- **Position embeddings**: Add rotary (RoPE) or ALiBi position embeddings for longer sequences.
- **Flash attention**: Replace scaled dot-product with Flash Attention 2 for throughput.
