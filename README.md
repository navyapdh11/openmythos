# OpenMythos

Recurrent-Depth Transformer (OpenMythos) — A 2026-standard Looped Transformer architecture for compute-adaptive inference and latent reasoning.

## 🚀 Key Features

- **Looped Architecture**: Recycles model weights through a central recurrent block for scalable compute.
- **MLA (Multi-Head Latent Attention)**: Memory-efficient attention mechanism that compresses KV cache into latent vectors.
- **Input Injection**: Maintains signal integrity by re-injecting prelude embeddings into the recurrent loop.
- **Compute-Adaptive Inference**: Dynamically adjust `loop_iters` to trade off between speed and reasoning depth.
- **Observability-First**: Integrated with OpenTelemetry for real-time tracing of recurrent loops.
- **Validated with Property-Based Testing**: Robustly tested using Hypothesis to ensure shape and consistency across variable batch sizes and sequence lengths.

## 📁 Structure

- `openmythos/`: Core model implementation.
- `tests/`: Property-based and unit tests.
- `docs/`: Technical documentation.

## 🛠️ Usage

```python
import torch
from openmythos import OpenMythos

# Initialize with 2026 best practice defaults
model = OpenMythos(
    dim=1024,
    num_prelude_layers=4,
    num_recurrent_layers=8,
    num_coda_layers=4,
    max_loop_iters=16
)

tokens = torch.randint(0, 50257, (1, 128))
# High-depth reasoning
output = model(tokens, loop_iters=16)

# Low-latency inference
output_fast = model(tokens, loop_iters=2)
```

## 🧪 Testing

```bash
# Uses existing kalman_venv for dependencies
cd /root/openmythos
PYTHONPATH=. /root/.kalman_venv/bin/pytest tests/
```

## 📊 Observability

Traces are automatically generated for:
- `openmythos_forward`: Main execution span.
- `recurrent_loop_{i}`: Individual spans for each iteration of the recurrent block.
