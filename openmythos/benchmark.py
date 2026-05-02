"""
Benchmark suite comparing OpenMythos V1, V2, and V3.

Measures:
- Parameter count (total, active)
- Forward latency at varying depths
- Output quality (entropy, norm, diversity across loop depths)
- Memory footprint (peak activations)
- MoE capacity ratio (V3 only)

Run: python -m openmythos.benchmark
"""

import time
from typing import Any, cast

import torch

from openmythos import OpenMythos
from openmythos.ds_v4_sandbox import OpenMythosV2, TieredKVCache
from openmythos.model_v3 import OpenMythosV3
from openmythos.moe import compute_moe_stats


BENCHMARK_CONFIG = {
    "dim": 64,
    "num_heads": 4,
    "latent_dim": 16,
    "num_prelude_layers": 2,
    "num_recurrent_layers": 2,
    "num_coda_layers": 2,
    "vocab_size": 5000,
    "seq_len": 32,
    "batch_size": 2,
    "depths": [0, 1, 2, 4, 8],
    "warmup_runs": 2,
    "timing_runs": 5,
    # V2 config
    "block_size": 16,
    "top_k_blocks": 4,
    # V3 config
    "num_experts": 8,
    "moe_top_k": 2,
}


def get_peak_memory(model: torch.nn.Module, tokens: torch.Tensor, loop_iters: int) -> int:
    """Estimate peak activation memory in bytes (parameter storage, not activations)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def measure_latency(model: torch.nn.Module, tokens: torch.Tensor,
                    loop_iters: int, **kwargs: Any) -> float:
    """Measure average forward latency in milliseconds."""
    model.eval()
    warmup = cast(int, BENCHMARK_CONFIG["warmup_runs"])
    timing = cast(int, BENCHMARK_CONFIG["timing_runs"])
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(tokens, loop_iters=loop_iters, **kwargs)

        latencies = []
        for _ in range(timing):
            t0 = time.perf_counter()
            _ = model(tokens, loop_iters=loop_iters, **kwargs)
            latencies.append(time.perf_counter() - t0)
        return sum(latencies) / len(latencies) * 1000  # ms


def compute_output_entropy(output: torch.Tensor) -> float:
    """Compute mean entropy of output distribution (nats)."""
    probs = output.softmax(dim=-1)
    log_probs = output.log_softmax(dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()
    return round(entropy, 4)


def compute_output_diversity(outputs: list[torch.Tensor]) -> float:
    """Compute pairwise cosine diversity across loop depths."""
    if len(outputs) < 2:
        return 0.0
    flat_outputs = [o.flatten() for o in outputs]
    total_diversity = 0.0
    count = 0
    for i in range(len(flat_outputs)):
        for j in range(i + 1, len(flat_outputs)):
            cos_sim = torch.cosine_similarity(
                flat_outputs[i].unsqueeze(0),
                flat_outputs[j].unsqueeze(0)
            ).item()
            total_diversity += (1 - cos_sim)  # 0 = identical, 2 = opposite
            count += 1
    return round(total_diversity / max(count, 1), 4)


def benchmark_v1(config: dict[str, Any]) -> dict[str, Any]:
    """Benchmark OpenMythos V1 (original, dense FFN)."""
    model = OpenMythos(
        dim=config["dim"],
        num_heads=config["num_heads"],
        latent_dim=config["latent_dim"],
        num_prelude_layers=config["num_prelude_layers"],
        num_recurrent_layers=config["num_recurrent_layers"],
        num_coda_layers=config["num_coda_layers"],
        max_loop_iters=max(config["depths"]),
        vocab_size=config["vocab_size"],
    )

    tokens = torch.randint(0, config["vocab_size"], (config["batch_size"], config["seq_len"]))
    total_params = sum(p.numel() for p in model.parameters())
    peak_mem = get_peak_memory(model, tokens, 0)

    depths = cast(list[int], config["depths"])

    results: dict[str, Any] = {
        "version": "V1 (Original)",
        "total_params": total_params,
        "active_params": total_params,  # Dense model
        "capacity_ratio": 1.0,
        "peak_mem_mb": round(peak_mem / (1024 * 1024), 2),
        "latencies": {},
        "entropies": {},
        "output_norms": [],
    }

    all_outputs: list[torch.Tensor] = []
    for depth in depths:
        lat = measure_latency(model, tokens, depth)
        results["latencies"][depth] = round(lat, 2)

        with torch.no_grad():
            output = model(tokens, loop_iters=depth)
            results["entropies"][depth] = compute_output_entropy(output)
            results["output_norms"].append(round(torch.norm(output).item(), 2))
            all_outputs.append(output)

    results["diversity"] = compute_output_diversity(all_outputs)
    return results


def benchmark_v2(config: dict[str, Any]) -> dict[str, Any]:
    """Benchmark OpenMythos V2 (mHC + CSA+HCA + Tiered KV)."""
    model = OpenMythosV2(
        dim=config["dim"],
        num_heads=config["num_heads"],
        latent_dim=config["latent_dim"],
        num_prelude_layers=config["num_prelude_layers"],
        num_recurrent_layers=config["num_recurrent_layers"],
        num_coda_layers=config["num_coda_layers"],
        max_loop_iters=max(config["depths"]),
        vocab_size=config["vocab_size"],
        use_mhc=True,
        block_size=config["block_size"],
        top_k_blocks=config["top_k_blocks"],
    )

    tokens = torch.randint(0, config["vocab_size"], (config["batch_size"], config["seq_len"]))
    total_params = sum(p.numel() for p in model.parameters())
    peak_mem = get_peak_memory(model, tokens, 0)
    cache = TieredKVCache(
        max_window_size=config["seq_len"],
        max_compressed_entries=config["seq_len"] // 2,
        latent_dim=config["latent_dim"],
    )

    depths = cast(list[int], config["depths"])

    results: dict[str, Any] = {
        "version": "V2 (mHC + CSA+HCA + Tiered KV)",
        "total_params": total_params,
        "active_params": total_params,
        "capacity_ratio": 1.0,
        "peak_mem_mb": round(peak_mem / (1024 * 1024), 2),
        "latencies": {},
        "entropies": {},
        "output_norms": [],
    }

    all_outputs_v2: list[torch.Tensor] = []
    for depth in depths:
        cache.reset()
        lat = measure_latency(model, tokens, depth, kv_cache=cache)
        results["latencies"][depth] = round(lat, 2)

        with torch.no_grad():
            output = model(tokens, loop_iters=depth, kv_cache=cache)
            results["entropies"][depth] = compute_output_entropy(output)
            results["output_norms"].append(round(torch.norm(output).item(), 2))
            all_outputs_v2.append(output)

    results["diversity"] = compute_output_diversity(all_outputs_v2)
    return results


def benchmark_v3(config: dict[str, Any]) -> dict[str, Any]:
    """Benchmark OpenMythos V3 (+ MoE)."""
    model = OpenMythosV3(
        dim=config["dim"],
        num_heads=config["num_heads"],
        latent_dim=config["latent_dim"],
        num_prelude_layers=config["num_prelude_layers"],
        num_recurrent_layers=config["num_recurrent_layers"],
        num_coda_layers=config["num_coda_layers"],
        max_loop_iters=max(config["depths"]),
        vocab_size=config["vocab_size"],
        use_mhc=True,
        block_size=config["block_size"],
        top_k_blocks=config["top_k_blocks"],
        num_experts=config["num_experts"],
        moe_top_k=config["moe_top_k"],
    )

    tokens = torch.randint(0, config["vocab_size"], (config["batch_size"], config["seq_len"]))
    total_params = sum(p.numel() for p in model.parameters())
    peak_mem = get_peak_memory(model, tokens, 0)
    cache = TieredKVCache(
        max_window_size=config["seq_len"],
        max_compressed_entries=config["seq_len"] // 2,
        latent_dim=config["latent_dim"],
    )

    moe_stats = compute_moe_stats(
        model, num_experts=config["num_experts"],
        num_recurrent_layers=config["num_recurrent_layers"],
        num_prelude_layers=config["num_prelude_layers"],
        num_coda_layers=config["num_coda_layers"],
        top_k=config["moe_top_k"],
    )

    results: dict[str, Any] = {
        "version": "V3 (+ MoE)",
        "total_params": total_params,
        "active_params": moe_stats["active_params"],
        "capacity_ratio": moe_stats["capacity_ratio"],
        "peak_mem_mb": round(peak_mem / (1024 * 1024), 2),
        "latencies": {},
        "entropies": {},
        "output_norms": [],
        "expert_count": moe_stats["expert_count"],
    }

    depths_v3 = cast(list[int], config["depths"])
    all_outputs_v3: list[torch.Tensor] = []
    for depth in depths_v3:
        cache.reset()
        lat = measure_latency(model, tokens, depth, kv_cache=cache)
        results["latencies"][depth] = round(lat, 2)

        with torch.no_grad():
            output = model(tokens, loop_iters=depth, kv_cache=cache)
            results["entropies"][depth] = compute_output_entropy(output)
            results["output_norms"].append(round(torch.norm(output).item(), 2))
            all_outputs_v3.append(output)

    results["diversity"] = compute_output_diversity(all_outputs_v3)
    return results


def print_comparison(v1: dict[str, Any], v2: dict[str, Any], v3: dict[str, Any]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("  OpenMythos Architecture Comparison: V1 → V2 → V3")
    print("=" * 80)

    # Parameter summary
    print("\n  PARAMETER SUMMARY")
    print(f"  {'Metric':<25} {'V1':>12} {'V2':>12} {'V3':>12}")
    print(f"  {'-'*65}")
    print(f"  {'Total params':<25} {v1['total_params']:>12,} {v2['total_params']:>12,} {v3['total_params']:>12,}")
    print(f"  {'Active params':<25} {v1['active_params']:>12,} {v2['active_params']:>12,} {v3['active_params']:>12,}")
    print(f"  {'Capacity ratio':<25} {v1['capacity_ratio']:>12.1f}x {v2['capacity_ratio']:>12.1f}x {v3['capacity_ratio']:>12.1f}x")
    print(f"  {'Peak memory (MB)':<25} {v1['peak_mem_mb']:>12.1f} {v2['peak_mem_mb']:>12.1f} {v3['peak_mem_mb']:>12.1f}")
    if "expert_count" in v3:
        print(f"  {'Expert networks':<25} {'-':>12} {'-':>12} {v3['expert_count']:>12,}")

    # Latency by depth
    print("\n  LATENCY (ms) by loop depth")
    print(f"  {'Depth':<10} {'V1':>12} {'V2':>12} {'V3':>12}")
    print(f"  {'-'*50}")
    for depth in cast(list[int], BENCHMARK_CONFIG["depths"]):
        v1_lat = v1["latencies"].get(depth, "N/A")
        v2_lat = v2["latencies"].get(depth, "N/A")
        v3_lat = v3["latencies"].get(depth, "N/A")
        print(f"  {depth:<10} {v1_lat:>12} {v2_lat:>12} {v3_lat:>12}")

    # Entropy by depth (lower = more confident)
    print("\n  OUTPUT ENTROPY (nats, lower = more confident)")
    print(f"  {'Depth':<10} {'V1':>12} {'V2':>12} {'V3':>12}")
    print(f"  {'-'*50}")
    for depth in cast(list[int], BENCHMARK_CONFIG["depths"]):
        v1_ent = v1["entropies"].get(depth, "N/A")
        v2_ent = v2["entropies"].get(depth, "N/A")
        v3_ent = v3["entropies"].get(depth, "N/A")
        print(f"  {depth:<10} {v1_ent:>12} {v2_ent:>12} {v3_ent:>12}")

    # Output norms
    print("\n  OUTPUT NORM (signal strength across depth)")
    print(f"  {'Depth':<10} {'V1':>12} {'V2':>12} {'V3':>12}")
    print(f"  {'-'*50}")
    for i, depth in enumerate(cast(list[int], BENCHMARK_CONFIG["depths"])):
        v1_norm = v1["output_norms"][i]
        v2_norm = v2["output_norms"][i]
        v3_norm = v3["output_norms"][i]
        print(f"  {depth:<10} {v1_norm:>12} {v2_norm:>12} {v3_norm:>12}")

    # Diversity
    print("\n  CROSS-DEPTH DIVERSITY (output differentiation)")
    print(f"  {'Metric':<25} {'V1':>12} {'V2':>12} {'V3':>12}")
    print(f"  {'-'*65}")
    print(f"  {'Diversity score':<25} {v1['diversity']:>12.4f} {v2['diversity']:>12.4f} {v3['diversity']:>12.4f}")

    # Improvements
    print("\n  IMPROVEMENTS (V3 vs V1)")
    v3_over_v1_params = v3["total_params"] / max(v1["total_params"], 1)
    v3_over_v1_latency = v3["latencies"].get(4, 0) / max(v1["latencies"].get(4, 1), 0.001)
    print(f"  Parameter capacity increase: {v3_over_v1_params:.1f}x")
    print(f"  Latency at depth=4: {v3_over_v1_latency:.1f}x (V3/V1)")
    if v1["entropies"].get(4) and v3["entropies"].get(4):
        ent_improvement = (v1["entropies"][4] - v3["entropies"][4]) / v1["entropies"][4] * 100
        print(f"  Entropy reduction at depth=4: {ent_improvement:.1f}%")

    print("\n" + "=" * 80 + "\n")


def main() -> None:
    print("\nOpenMythos Benchmark Suite")
    print(f"Config: dim={BENCHMARK_CONFIG['dim']}, seq={BENCHMARK_CONFIG['seq_len']}, "
          f"batch={BENCHMARK_CONFIG['batch_size']}, "
          f"experts={BENCHMARK_CONFIG['num_experts']}, top_k={BENCHMARK_CONFIG['moe_top_k']}")
    print(f"Device: {torch.device('cpu')}")

    print("\nBenchmarking V1...")
    v1 = benchmark_v1(BENCHMARK_CONFIG)

    print("Benchmarking V2...")
    v2 = benchmark_v2(BENCHMARK_CONFIG)

    print("Benchmarking V3...")
    v3 = benchmark_v3(BENCHMARK_CONFIG)

    print_comparison(v1, v2, v3)


if __name__ == "__main__":
    main()
