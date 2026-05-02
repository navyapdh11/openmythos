# OpenMythos Knowledgebase

## Overview
OpenMythos is a 2026-standard Recurrent-Depth Transformer architecture designed for compute-adaptive inference and latent reasoning.

## Architecture
- **Recurrent Block**: Recycles model weights, allowing for scalable compute depth.
- **MLA (Multi-Head Latent Attention)**: Compresses KV cache into latent vectors to optimize memory usage.
- **Input Injection**: Re-injects prelude embeddings into the recurrent loop for signal integrity.
- **Observability**: Built-in OpenTelemetry instrumentation for deep tracing of recurrent loops.

## Core API

### `OpenMythos` Class
The main entry point for the model.

```python
from openmythos import OpenMythos

model = OpenMythos(
    dim=1024,
    num_prelude_layers=4,
    num_recurrent_layers=8,
    num_coda_layers=4,
    max_loop_iters=16
)
```

### Inference
Use the model for either high-depth reasoning or low-latency inference by adjusting `loop_iters`.

```python
# High-depth (Max compute)
output = model(tokens, loop_iters=16)

# Low-latency (Fast inference)
output_fast = model(tokens, loop_iters=2)
```

## CLI Commands & Verification

### Running Tests
To verify the installation and core logic:

```bash
cd /root/openmythos
PYTHONPATH=. /root/.kalman_venv/bin/pytest tests/
```

## Use Case Scenarios

### 1. Complex Reasoning Tasks
Use high `loop_iters` for tasks requiring deep logical deduction (e.g., formal verification, multi-step problem solving).
*Example*: Set `loop_iters=16` for code synthesis or mathematical proofs.

### 2. Low-Latency Applications
Use low `loop_iters` for real-time edge applications where latency is critical.
*Example*: Set `loop_iters=2-4` for predictive text completion or real-time UI updates.

### 3. Adaptive Compute
Dynamically adjust `loop_iters` based on the complexity of the input sequence identified by a controller model.

## Capabilities
- **Compute-Adaptive Inference**: Trade speed for reasoning depth on-the-fly.
- **Memory Efficiency**: Low KV-cache footprint via MLA.
- **Integrity**: Signal retention through recurrent input injection.
- **Observability**: Real-time tracing of recurrent loop iterations (`recurrent_loop_{i}` spans).
