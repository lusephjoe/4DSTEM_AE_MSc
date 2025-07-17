#!/usr/bin/env python
# scripts/generate_embeddings.py
"""
Batch-extract latent vectors from a trained autoencoder encoder.

Usage
-----
python scripts/generate_embeddings.py \
    --input data/ae_train.pt \
    --checkpoint checkpoints/ae_best.pt \
    --output embeddings/ae_train_latents.pt \
    --batch_size 1024 \
    --device cuda \
    --pca 50
"""
from pathlib import Path
import argparse, torch
from sklearn.decomposition import PCA  # CPU PCA is fine for ≤10⁴ samples
from tqdm import tqdm
import sys
import time
import gc
import psutil
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.summary import show

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help=".pt tensor from convert_dm4")
    p.add_argument("--checkpoint", required=True, help="trained AE or encoder .pt")
    p.add_argument("--output", required=True, help="file to write latents to (.pt or .npy)")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--device", default="cpu")
    p.add_argument("--pca", type=int, default=0,
                   help="If >0: reduce dimensionality with PCA to this many comps")
    p.add_argument("--mixed_precision", action="store_true",
                   help="Use mixed precision (FP16) for faster inference")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Number of data loader workers")
    p.add_argument("--prefetch_factor", type=int, default=2,
                   help="Number of batches to prefetch per worker")
    p.add_argument("--optimize_memory", action="store_true",
                   help="Enable memory optimizations for large datasets")
    p.add_argument("--no_compile", action="store_true",
                   help="Disable torch.compile optimization")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Enable optimizations for CUDA
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"CUDA optimizations enabled on {torch.cuda.get_device_name()}")
        print(f"Initial GPU memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
    
    # Set up mixed precision
    use_amp = args.mixed_precision and device.type == "cuda"
    if use_amp:
        print("Mixed precision (FP16) enabled")

    # ---------- load model ---------- #
    print("Loading model...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    from models.autoencoder import Encoder  # adjust if path differs
    encoder = Encoder()
    encoder.load_state_dict(state, strict=False)
    encoder.eval().to(device)

    # ---------- load data ---------- #
    print("Loading data...")
    data = torch.load(args.input, map_location="cpu")  # shape [N, signal_dim]
    print(f"Loaded {data.shape[0]} samples of shape {data.shape[1:]}")
    
    # Optimize batch size for GPU memory
    if device.type == "cuda" and args.optimize_memory:
        # Estimate optimal batch size based on available GPU memory
        available_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = available_memory - allocated_memory
        
        # Estimate memory per sample (rough approximation)
        sample_size = data.element_size() * data[0].numel()
        estimated_batch_size = int(free_memory * 0.8 / (sample_size * 4))  # 80% of free memory, 4x for safety
        
        if estimated_batch_size < args.batch_size:
            print(f"Reducing batch size from {args.batch_size} to {estimated_batch_size} to fit GPU memory")
            args.batch_size = max(estimated_batch_size, 1)
    
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=False
    )

    example = data[:1].to(device)          # <‑‑ one real sample, preserves dims
    try:
        show(encoder, example_input=example)  
    except Exception as e:
        print(f"Warning: Could not display model summary: {e}")
    
    # Optimize model for inference (after summary to avoid compilation conflicts)
    compiled_successfully = False
    if device.type == "cuda" and not args.no_compile:
        try:
            # Check if Triton is available before attempting compilation
            import triton
            if hasattr(torch, "compile"):
                encoder = torch.compile(encoder, mode="reduce-overhead")
                compiled_successfully = True
                print("Model compiled with torch.compile for faster inference")
        except ImportError:
            print("Warning: Triton not available - torch.compile disabled. Install with: pip install triton")
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")
    elif args.no_compile:
        print("torch.compile disabled by user (--no_compile flag)")
    
    if not compiled_successfully:
        print("Using standard PyTorch inference (torch.compile not available)")
    
    print("Generating embeddings...")
    latents = []
    total_samples = len(data)
    processed_samples = 0
    start_time = time.time()
    
    # Progress bar with additional metrics
    progress_bar = tqdm(loader, desc="Processing batches", unit="batch")
    
    for batch_idx, batch in enumerate(progress_bar):
        batch_start_time = time.time()
        
        # Move to GPU with non-blocking transfer
        batch = batch.to(device, non_blocking=True)
        
        # Forward pass with optional mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                z = encoder(batch)
        else:
            z = encoder(batch)
        
        # Move back to CPU and store
        z = z.cpu()
        latents.append(z)
        
        # Update progress metrics
        processed_samples += batch.size(0)
        batch_time = time.time() - batch_start_time
        samples_per_sec = batch.size(0) / batch_time
        
        # Update progress bar with metrics
        progress_info = {
            "samples/sec": f"{samples_per_sec:.0f}",
            "progress": f"{processed_samples}/{total_samples}"
        }
        
        if device.type == "cuda":
            gpu_memory_used = torch.cuda.memory_allocated(device) / 1024**3
            gpu_memory_cached = torch.cuda.memory_reserved(device) / 1024**3
            progress_info["GPU_mem"] = f"{gpu_memory_used:.1f}GB"
            progress_info["GPU_cached"] = f"{gpu_memory_cached:.1f}GB"
        
        progress_bar.set_postfix(progress_info)
        
        # Optional memory cleanup for large datasets
        if args.optimize_memory and (batch_idx + 1) % 10 == 0:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    progress_bar.close()
    
    # Final timing statistics
    total_time = time.time() - start_time
    avg_samples_per_sec = total_samples / total_time
    print(f"\nProcessing completed in {total_time:.2f}s ({avg_samples_per_sec:.0f} samples/sec)")

    print("Concatenating results...")
    latents = torch.cat(latents, dim=0)  # [N, latent_dim]
    print(f"Generated embeddings shape: {latents.shape}")
    
    # Memory cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---------- optional PCA ---------- #
    if args.pca > 0 and args.pca < latents.size(1):
        print(f"Running PCA to reduce from {latents.size(1)} to {args.pca} dimensions...")
        pca = PCA(args.pca, svd_solver="randomized")
        latents_pca = pca.fit_transform(latents.numpy())
        latents = torch.tensor(latents_pca)
        print(f"PCA completed. Explained variance ratio: {pca.explained_variance_ratio_[:5]}")  # Show first 5 components

    # ---------- save ---------- #
    print("Saving embeddings...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(latents, args.output)
    print(f"✓ Saved embeddings {latents.shape} → {args.output}")
    
    # Show some statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {latents.mean():.4f}")
    print(f"  Std:  {latents.std():.4f}")
    print(f"  Min:  {latents.min():.4f}")
    print(f"  Max:  {latents.max():.4f}")
    
    # Final performance and memory statistics
    print(f"\nPerformance Summary:")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Average throughput: {avg_samples_per_sec:.0f} samples/sec")
    print(f"  Batch size used: {args.batch_size}")
    print(f"  Mixed precision: {'Enabled' if use_amp else 'Disabled'}")
    
    if device.type == "cuda":
        print(f"\nGPU Memory Summary:")
        print(f"  Peak allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
        print(f"  Peak cached: {torch.cuda.max_memory_reserved(device) / 1024**3:.2f} GB")
        print(f"  Current allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"  Current cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    # System memory usage
    system_memory = psutil.virtual_memory()
    print(f"\nSystem Memory Usage:")
    print(f"  Total: {system_memory.total / 1024**3:.2f} GB")
    print(f"  Available: {system_memory.available / 1024**3:.2f} GB")
    print(f"  Used: {system_memory.used / 1024**3:.2f} GB ({system_memory.percent:.1f}%)")

if __name__ == "__main__":
    main()
