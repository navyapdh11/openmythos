"""
Compute-adaptive inference demonstration for OpenMythos.

Shows how varying loop_iters trades off between latency and reasoning depth,
and demonstrates the KV cache memory savings from MLA.
"""

import argparse
import time
from typing import Any

import torch

from openmythos import OpenMythos


def benchmark_depth(model: OpenMythos, tokens: torch.Tensor, depths: list[int]) -> list[dict[str, Any]]:
    """Run inference at multiple depths and measure latency/output statistics."""
    results = []
    # Warmup
    with torch.no_grad():
        _ = model(tokens, loop_iters=1)

    for depth in depths:
        latencies = []
        outputs = []
        with torch.no_grad():
            for _ in range(3):
                t0 = time.perf_counter()
                out = model(tokens, loop_iters=depth)
                latencies.append(time.perf_counter() - t0)
                outputs.append(out)

        mean_latency = sum(latencies) / len(latencies)
        out_norm = torch.norm(outputs[0]).item()
        out_entropy = -(outputs[0].softmax(dim=-1) * outputs[0].log_softmax(dim=-1)).sum(dim=-1).mean().item()

        results.append({
            "depth": depth,
            "latency_ms": round(mean_latency * 1000, 2),
            "output_norm": round(out_norm, 4),
            "output_entropy": round(out_entropy, 4),
        })
    return results


def print_report(results: list[dict[str, Any]], dim: int, latent_dim: int, seq_len: int) -> None:
    """Print a formatted comparison table."""
    kv_standard = dim * 2 * seq_len  # Standard KV cache size
    kv_mla = latent_dim * 2 * seq_len  # MLA compressed
    reduction = (1 - kv_mla / kv_standard) * 100

    print(f"\n{'='*60}")
    print("  OpenMythos — Compute-Adaptive Inference Report")
    print(f"{'='*60}")
    print(f"  Model: dim={dim}, latent_dim={latent_dim}")
    print(f"  MLA KV cache: {kv_mla:,} vs standard {kv_standard:,} "
          f"({reduction:.1f}% reduction)")
    print(f"{'='*60}\n")

    print(f"  {'Depth':<8} {'Latency (ms)':<15} {'Output Norm':<15} {'Entropy':<12}")
    print(f"  {'-'*50}")
    for r in results:
        print(f"  {r['depth']:<8} {r['latency_ms']:<15} {r['output_norm']:<15} {r['output_entropy']:<12}")
    print(f"\n{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenMythos compute-adaptive inference demo")
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--latent-dim", type=int, default=64, help="MLA latent dimension")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--depths", type=str, default="0,1,2,4,8,16",
                        help="Comma-separated loop depths to benchmark")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    device = torch.device(args.device)
    depths = [int(d) for d in args.depths.split(",")]

    model = OpenMythos(
        dim=args.dim,
        num_heads=4,
        latent_dim=args.latent_dim,
        num_prelude_layers=2,
        num_recurrent_layers=2,
        num_coda_layers=2,
        max_loop_iters=max(depths),
        vocab_size=50257,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel loaded: {total_params:,} parameters on {device}")

    tokens = torch.randint(0, 50257, (1, args.seq_len), device=device)
    results = benchmark_depth(model, tokens, depths)
    print_report(results, args.dim, args.latent_dim, args.seq_len)


if __name__ == "__main__":
    main()
