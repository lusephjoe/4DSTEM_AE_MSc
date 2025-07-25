#!/usr/bin/env python3
"""
Performance benchmark for scale-aligned loss computation.
Run this to verify minimal overhead from the CLAUDE.md fix.
"""

import torch
import time
import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "scripts" / "training"))

from models.autoencoder import Autoencoder
from models.losses import create_loss_config_from_args
from models.summary import calculate_metrics, calculate_diffraction_metrics

def benchmark_denormalization(batch_size=8, num_iterations=100):
    """Benchmark the denormalization overhead."""
    print(f"Benchmarking denormalization overhead...")
    print(f"Batch size: {batch_size}, Iterations: {num_iterations}")
    
    # Test data
    x_norm = torch.randn(batch_size, 1, 256, 256)
    log_mean, log_std = 5.2, 1.8
    
    # Time the denormalization operation
    start_time = time.time()
    for _ in range(num_iterations):
        x_log = x_norm * (log_std + 1e-8) + log_mean
    denorm_time = time.time() - start_time
    
    # Time baseline (no operation)
    start_time = time.time()
    for _ in range(num_iterations):
        x_baseline = x_norm  # No-op
    baseline_time = time.time() - start_time
    
    overhead_ms = (denorm_time - baseline_time) / num_iterations * 1000
    print(f"Denormalization overhead: {overhead_ms:.2f}ms per batch")
    return overhead_ms

def benchmark_metrics(batch_size=8, num_iterations=50):
    """Benchmark the domain-specific metrics."""
    print(f"\nBenchmarking domain-specific metrics...")
    print(f"Batch size: {batch_size}, Iterations: {num_iterations}")
    
    # Test data
    x_log = torch.randn(batch_size, 1, 256, 256) * 2.0 + 5.0
    x_hat_log = x_log + torch.randn_like(x_log) * 0.1
    
    # Time original metrics
    start_time = time.time()
    for _ in range(num_iterations):
        original_metrics = calculate_metrics(x_log, x_hat_log)
    original_time = time.time() - start_time
    
    # Time new diffraction metrics
    start_time = time.time()
    for _ in range(num_iterations):
        diffraction_metrics = calculate_diffraction_metrics(x_log, x_hat_log)
    diffraction_time = time.time() - start_time
    
    overhead_ms = (diffraction_time - original_time) / num_iterations * 1000
    print(f"Original metrics: {original_time/num_iterations*1000:.1f}ms per batch")
    print(f"Diffraction metrics: {diffraction_time/num_iterations*1000:.1f}ms per batch")
    print(f"Metrics overhead: {overhead_ms:.1f}ms per batch")
    return overhead_ms

def benchmark_complete_training_step(batch_size=8, num_iterations=20):
    """Benchmark complete training step with scale-aligned loss."""
    print(f"\nBenchmarking complete training step...")
    print(f"Batch size: {batch_size}, Iterations: {num_iterations}")
    
    # Create model and data
    class MockArgs:
        loss_function = 'mse'
        lambda_act = 1e-5
        lambda_sim = 0
        lambda_div = 0
        lambda_l2 = 0
        lambda_kl = 0
    
    args = MockArgs()
    loss_config = create_loss_config_from_args(args)
    model = Autoencoder(latent_dim=32, out_shape=(256, 256), loss_config=loss_config)
    
    x_norm = torch.randn(batch_size, 1, 256, 256)
    log_mean, log_std = 5.2, 1.8
    
    # Time without scale alignment (wrong but fast)
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            z = model.encoder(x_norm)
            x_hat = model.decoder(z)
            loss_dict = model.compute_loss(x_norm, x_hat, z)
    baseline_time = time.time() - start_time
    
    # Time with scale alignment (correct)
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            z = model.encoder(x_norm)
            x_hat = model.decoder(z)
            # Scale-aligned loss computation
            x_log = x_norm * (log_std + 1e-8) + log_mean
            x_hat_log = x_hat * (log_std + 1e-8) + log_mean
            loss_dict = model.compute_loss(x_log, x_hat_log, z)
    aligned_time = time.time() - start_time
    
    overhead_ms = (aligned_time - baseline_time) / num_iterations * 1000
    overhead_pct = (aligned_time / baseline_time - 1) * 100
    
    print(f"Baseline training step: {baseline_time/num_iterations*1000:.1f}ms per batch")
    print(f"Scale-aligned step: {aligned_time/num_iterations*1000:.1f}ms per batch")
    print(f"Total overhead: {overhead_ms:.1f}ms per batch ({overhead_pct:.1f}%)")
    return overhead_ms, overhead_pct

if __name__ == "__main__":
    print("=" * 60)
    print("SCALE-ALIGNED LOSS PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Run benchmarks
    denorm_overhead = benchmark_denormalization()
    metrics_overhead = benchmark_metrics()
    step_overhead_ms, step_overhead_pct = benchmark_complete_training_step()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Denormalization overhead:    {denorm_overhead:.2f}ms per batch")
    print(f"Metrics overhead:           {metrics_overhead:.1f}ms per batch")
    print(f"Total training overhead:    {step_overhead_ms:.1f}ms per batch ({step_overhead_pct:.1f}%)")
    
    if step_overhead_pct < 10:
        print("✅ MINIMAL OVERHEAD - Solution is production ready!")
    elif step_overhead_pct < 25:
        print("⚠️  MODERATE OVERHEAD - Acceptable for most use cases")
    else:
        print("❌ HIGH OVERHEAD - Consider optimization")
    
    print(f"\nFor typical 100-epoch training with {step_overhead_ms:.1f}ms overhead:")
    print(f"Additional time per epoch: ~{step_overhead_ms * 1000 / 1000:.1f}s")
    print(f"Total additional time: ~{step_overhead_ms * 1000 * 100 / 60000:.1f} minutes")