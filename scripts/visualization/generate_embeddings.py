#!/usr/bin/env python3
"""
Enhanced embedding generation for 4D-STEM autoencoder analysis.

Batch-extract latent vectors from trained autoencoder models with support for:
- PyTorch Lightning checkpoints (.ckpt)
- HDF5 data files (.h5) 
- Spatial coordinate tracking
- Multiple output formats (PyTorch, NumPy, HDF5)
- Comprehensive metadata saving

Usage
-----
# Basic usage with PyTorch Lightning checkpoint and HDF5 data
python scripts/visualization/generate_embeddings.py \
    --checkpoint results/ae_model.ckpt \
    --data data/patterns.h5 \
    --output embeddings/patterns_embeddings.npz \
    --batch_size 128 \
    --device cuda

# Legacy usage with PyTorch tensors
python scripts/visualization/generate_embeddings.py \
    --input data/ae_train.pt \
    --checkpoint checkpoints/ae_best.pt \
    --output embeddings/ae_train_latents.pt \
    --batch_size 1024 \
    --device cuda \
    --pca 50
"""
from pathlib import Path
import argparse, torch
import pytorch_lightning as pl
from sklearn.decomposition import PCA  # CPU PCA is fine for ≤10⁴ samples
from tqdm import tqdm
import sys
import time
import gc
import psutil
import numpy as np
import h5py
import json
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.summary import show
from scripts.training.train import LitAE, HDF5Dataset

def parse_args():
    p = argparse.ArgumentParser(description="Generate embeddings from trained 4D-STEM autoencoder")
    
    # Input/output arguments
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--data", type=Path, help="HDF5 data file (.h5) - recommended")
    input_group.add_argument("--input", type=Path, help="PyTorch tensor file (.pt) - legacy")
    
    p.add_argument("--checkpoint", type=Path, required=True, 
                   help="Trained model checkpoint (.ckpt for Lightning, .pt for legacy)")
    p.add_argument("--output", type=Path, required=True, 
                   help="Output file (.npz, .pt, .h5)")
    
    # Data processing arguments
    p.add_argument("--n_samples", type=int, default=None,
                   help="Number of samples to process (None = all)")
    p.add_argument("--batch_size", type=int, default=128,
                   help="Batch size for processing")
    p.add_argument("--no_normalization", action="store_true",
                   help="Disable data normalization (for HDF5 data)")
    
    # Model/device arguments
    p.add_argument("--device", type=str, default="auto", 
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="Compute device")
    p.add_argument("--mixed_precision", action="store_true",
                   help="Use mixed precision (FP16) for faster inference")
    p.add_argument("--no_compile", action="store_true",
                   help="Disable torch.compile optimization")
    
    # Performance arguments
    p.add_argument("--num_workers", type=int, default=4,
                   help="Number of data loader workers")
    p.add_argument("--prefetch_factor", type=int, default=2,
                   help="Number of batches to prefetch per worker")
    p.add_argument("--optimize_memory", action="store_true",
                   help="Enable memory optimizations for large datasets")
    
    # Analysis arguments
    p.add_argument("--pca", type=int, default=0,
                   help="If >0: reduce dimensionality with PCA to this many components")
    p.add_argument("--save_spatial_coords", action="store_true",
                   help="Save spatial coordinates (for HDF5 data)")
    p.add_argument("--save_metadata", action="store_true", default=True,
                   help="Save comprehensive metadata")
    
    return p.parse_args()

def setup_device(device_str: str) -> torch.device:
    """Setup compute device with maximum optimizations."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    # Enable maximum optimizations for CUDA
    if device.type == "cuda":
        # Core CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Advanced optimizations for modern GPUs
        if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
            torch.backends.cuda.flash_sdp_enabled(True)
        if hasattr(torch.backends.cuda, 'math_sdp_enabled'):
            torch.backends.cuda.math_sdp_enabled(True)
        if hasattr(torch.backends.cuda, 'mem_efficient_sdp_enabled'):
            torch.backends.cuda.mem_efficient_sdp_enabled(True)
        
        # Set memory allocator optimizations
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            # Reserve 10% for system processes
            torch.cuda.set_per_process_memory_fraction(0.9)
        
        # Enable memory pool for faster allocations
        try:
            torch.cuda.memory._set_allocator_settings('expandable_segments:True')
        except:
            pass  # Not available on all CUDA versions
        
        device_props = torch.cuda.get_device_properties(device)
        print(f"🚀 CUDA optimizations enabled on {device_props.name}")
        print(f"   Compute capability: {device_props.major}.{device_props.minor}")
        print(f"   Total memory: {device_props.total_memory / 1024**3:.1f} GB")
        print(f"   Initial GPU memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        
        # Check for Tensor Core support
        if device_props.major >= 7:  # V100, A100, etc.
            print(f"   ✓ Tensor Core support available")
        
    elif device.type == "mps":
        print(f"🍎 MPS optimizations enabled")
    
    return device

def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint (supports both Lightning and legacy formats)."""
    print(f"Loading model from: {checkpoint_path}")
    
    if checkpoint_path.suffix == '.ckpt':
        # PyTorch Lightning checkpoint
        try:
            # Try to load as Lightning checkpoint first
            model = LitAE.load_from_checkpoint(checkpoint_path, map_location=device)
            encoder = model.model.encoder
            print("✓ Loaded PyTorch Lightning checkpoint")
        except Exception as e:
            print(f"Lightning loading failed: {e}")
            # Fallback: Load checkpoint manually
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Extract hyperparameters
            if 'hyper_parameters' in checkpoint:
                hparams = checkpoint['hyper_parameters']
                latent_dim = hparams.get('latent_dim', 128)
                out_shape = (256, 256)  # Default
            else:
                latent_dim = 128
                out_shape = (256, 256)
                print("Warning: Using default hyperparameters")
            
            # Create model and load state
            model = LitAE(latent_dim=latent_dim, lr=1e-3, out_shape=out_shape)
            model.load_state_dict(checkpoint['state_dict'])
            encoder = model.model.encoder
            print("✓ Loaded checkpoint manually")
    
    else:
        # Legacy PyTorch checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint
        
        from models.autoencoder import Encoder
        encoder = Encoder()
        encoder.load_state_dict(state, strict=False)
        print("✓ Loaded legacy PyTorch checkpoint")
    
    encoder.eval().to(device)
    return encoder

def load_data(args) -> Tuple[torch.utils.data.DataLoader, Optional[np.ndarray], dict]:
    """Load data and return DataLoader, spatial coordinates, and metadata."""
    
    if args.data:
        # HDF5 data (recommended)
        print(f"Loading HDF5 data from: {args.data}")
        use_normalization = not args.no_normalization
        dataset = HDF5Dataset(args.data, use_normalization=use_normalization)
        
        # Generate spatial coordinates if requested
        spatial_coords = None
        if args.save_spatial_coords:
            total_patterns = len(dataset)
            scan_size = int(np.sqrt(total_patterns))
            spatial_coords = []
            for idx in range(total_patterns):
                y = idx // scan_size
                x = idx % scan_size
                spatial_coords.append([x, y])
            spatial_coords = np.array(spatial_coords)
        
        # Sample subset if requested
        if args.n_samples and args.n_samples < len(dataset):
            indices = np.random.choice(len(dataset), args.n_samples, replace=False)
            indices.sort()
            from torch.utils.data import Subset
            dataset = Subset(dataset, indices)
            if spatial_coords is not None:
                spatial_coords = spatial_coords[indices]
        
        metadata = {
            'data_type': 'hdf5',
            'data_path': str(args.data),
            'use_normalization': use_normalization,
            'total_patterns': len(dataset),
            'normalization_stats': {
                'global_log_mean': getattr(dataset, 'global_log_mean', None),
                'global_log_std': getattr(dataset, 'global_log_std', None)
            } if hasattr(dataset, 'global_log_mean') else None
        }
        
    else:
        # Legacy PyTorch tensor data
        print(f"Loading PyTorch tensor from: {args.input}")
        data = torch.load(args.input, map_location="cpu")
        
        # Sample subset if requested
        if args.n_samples and args.n_samples < len(data):
            indices = np.random.choice(len(data), args.n_samples, replace=False)
            data = data[indices]
        
        dataset = torch.utils.data.TensorDataset(data)
        spatial_coords = None
        metadata = {
            'data_type': 'pytorch_tensor',
            'data_path': str(args.input),
            'total_patterns': len(dataset)
        }
    
    print(f"Loaded {len(dataset)} samples")
    
    # Create DataLoader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=False
    )
    
    return loader, spatial_coords, metadata

def save_embeddings(embeddings: torch.Tensor, output_path: Path, 
                   spatial_coords: Optional[np.ndarray] = None,
                   metadata: Optional[dict] = None) -> None:
    """Save embeddings in multiple formats with metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.npz':
        # NumPy format (recommended for UMAP analysis)
        save_dict = {'embeddings': embeddings.numpy()}
        if spatial_coords is not None:
            save_dict['spatial_coordinates'] = spatial_coords
        if metadata is not None:
            # Save metadata as JSON string in npz
            save_dict['metadata'] = np.array([json.dumps(metadata)])
        
        np.savez_compressed(output_path, **save_dict)
        
    elif output_path.suffix == '.h5':
        # HDF5 format
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('embeddings', data=embeddings.numpy(), compression='gzip')
            if spatial_coords is not None:
                f.create_dataset('spatial_coordinates', data=spatial_coords, compression='gzip')
            if metadata is not None:
                # Save metadata as attributes
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        f.attrs[key] = value
                    else:
                        f.attrs[key] = json.dumps(value)
    
    elif output_path.suffix == '.pt':
        # PyTorch format (legacy)
        torch.save(embeddings, output_path)
        
        # Save additional data in separate files if available
        if spatial_coords is not None:
            coord_path = output_path.with_name(output_path.stem + '_coords.npy')
            np.save(coord_path, spatial_coords)
        
        if metadata is not None:
            meta_path = output_path.with_name(output_path.stem + '_metadata.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")

@torch.no_grad()
def main():
    args = parse_args()
    
    print("="*80)
    print("4D-STEM AUTOENCODER EMBEDDING GENERATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data or args.input}")
    print(f"Output: {args.output}")
    print("="*80)
    
    # Setup device
    device = setup_device(args.device)
    print(f"Using device: {device}")
    
    # Set up mixed precision
    use_amp = args.mixed_precision and device.type == "cuda"
    if use_amp:
        print("Mixed precision (FP16) enabled")
    
    # Load model
    encoder = load_model(args.checkpoint, device)
    
    # Load data
    loader, spatial_coords, metadata = load_data(args)
    
    # Optimize batch size for GPU memory
    if device.type == "cuda" and args.optimize_memory:
        # Get a sample to estimate memory usage
        sample_batch = next(iter(loader))
        if isinstance(sample_batch, (list, tuple)):
            sample_batch = sample_batch[0]
        
        available_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = available_memory - allocated_memory
        
        # Estimate memory per sample
        sample_size = sample_batch.element_size() * sample_batch[0].numel()
        estimated_batch_size = int(free_memory * 0.8 / (sample_size * 4))  # 80% of free memory, 4x for safety
        
        if estimated_batch_size < args.batch_size:
            print(f"Reducing batch size from {args.batch_size} to {estimated_batch_size} to fit GPU memory")
            args.batch_size = max(estimated_batch_size, 1)
            # Recreate loader with new batch size
            if args.data:
                new_dataset = HDF5Dataset(args.data, use_normalization=not args.no_normalization)
                if args.n_samples and args.n_samples < len(new_dataset):
                    indices = np.random.choice(len(new_dataset), args.n_samples, replace=False)
                    from torch.utils.data import Subset
                    new_dataset = Subset(new_dataset, indices)
            else:
                data = torch.load(args.input, map_location="cpu")
                if args.n_samples and args.n_samples < len(data):
                    indices = np.random.choice(len(data), args.n_samples, replace=False)
                    data = data[indices]
                new_dataset = torch.utils.data.TensorDataset(data)
            
            loader = torch.utils.data.DataLoader(
                new_dataset, batch_size=args.batch_size, shuffle=False,
                pin_memory=True, num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=True if args.num_workers > 0 else False,
                drop_last=False
            )

    # Display model summary
    sample_batch = next(iter(loader))
    if isinstance(sample_batch, (list, tuple)):
        example = sample_batch[0][:1].to(device)
    else:
        example = sample_batch[:1].to(device)
    
    try:
        show(encoder, example_input=example)  
    except Exception as e:
        print(f"Warning: Could not display model summary: {e}")
    
    # Optimize model for inference
    compiled_successfully = False
    if device.type == "cuda" and not args.no_compile:
        try:
            import triton
            if hasattr(torch, "compile"):
                encoder = torch.compile(encoder, mode="reduce-overhead")
                compiled_successfully = True
                print("✓ Model compiled with torch.compile for faster inference")
        except ImportError:
            print("Warning: Triton not available - torch.compile disabled")
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")
    elif args.no_compile:
        print("torch.compile disabled by user")
    
    if not compiled_successfully:
        print("Using standard PyTorch inference")
    
    # Generate embeddings
    print("Generating embeddings...")
    latents = []
    total_samples = len(loader.dataset)
    processed_samples = 0
    start_time = time.time()
    
    progress_bar = tqdm(loader, desc="Processing batches", unit="batch")
    
    for batch_idx, batch_data in enumerate(progress_bar):
        batch_start_time = time.time()
        
        # Handle different batch formats
        if isinstance(batch_data, (list, tuple)):
            batch = batch_data[0]
        else:
            batch = batch_data
        
        # Move to device
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
        
        # Update progress bar
        progress_info = {
            "samples/sec": f"{samples_per_sec:.0f}",
            "progress": f"{processed_samples}/{total_samples}"
        }
        
        if device.type == "cuda":
            gpu_memory_used = torch.cuda.memory_allocated(device) / 1024**3
            progress_info["GPU_mem"] = f"{gpu_memory_used:.1f}GB"
        
        progress_bar.set_postfix(progress_info)
        
        # Memory cleanup
        if args.optimize_memory and (batch_idx + 1) % 10 == 0:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    progress_bar.close()
    
    # Final timing
    total_time = time.time() - start_time
    avg_samples_per_sec = total_samples / total_time
    print(f"✓ Processing completed in {total_time:.2f}s ({avg_samples_per_sec:.0f} samples/sec)")

    # Concatenate results
    print("Concatenating results...")
    latents = torch.cat(latents, dim=0)
    print(f"✓ Generated embeddings shape: {latents.shape}")
    
    # Memory cleanup
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Optional PCA reduction
    if args.pca > 0 and args.pca < latents.size(1):
        print(f"Running PCA to reduce from {latents.size(1)} to {args.pca} dimensions...")
        pca = PCA(args.pca, svd_solver="randomized")
        latents_pca = pca.fit_transform(latents.numpy())
        latents = torch.tensor(latents_pca, dtype=torch.float32)
        print(f"✓ PCA completed. Explained variance ratio: {pca.explained_variance_ratio_[:5]}")
        
        # Update metadata
        metadata['pca_applied'] = True
        metadata['pca_components'] = args.pca
        metadata['explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
    else:
        metadata['pca_applied'] = False

    # Add embedding statistics to metadata
    metadata.update({
        'embedding_shape': list(latents.shape),
        'embedding_stats': {
            'mean': float(latents.mean()),
            'std': float(latents.std()),
            'min': float(latents.min()),
            'max': float(latents.max())
        },
        'processing_info': {
            'total_time': total_time,
            'samples_per_sec': avg_samples_per_sec,
            'batch_size': args.batch_size,
            'mixed_precision': use_amp,
            'device': str(device),
            'model_compiled': compiled_successfully
        }
    })

    # Save embeddings
    print("Saving embeddings...")
    save_embeddings(latents, args.output, spatial_coords, metadata)
    print(f"✓ Saved embeddings {latents.shape} → {args.output}")
    
    # Display statistics
    print(f"\n" + "="*60)
    print("EMBEDDING GENERATION COMPLETED")
    print("="*60)
    print(f"Embedding shape: {latents.shape}")
    print(f"Embedding statistics:")
    print(f"  Mean: {latents.mean():.4f}")
    print(f"  Std:  {latents.std():.4f}")
    print(f"  Min:  {latents.min():.4f}")
    print(f"  Max:  {latents.max():.4f}")
    
    if spatial_coords is not None:
        print(f"Spatial coordinates: {spatial_coords.shape}")
    
    print(f"\nPerformance Summary:")
    print(f"  Total processing time: {total_time:.2f}s")
    print(f"  Average throughput: {avg_samples_per_sec:.0f} samples/sec")
    print(f"  Batch size used: {args.batch_size}")
    print(f"  Mixed precision: {'Enabled' if use_amp else 'Disabled'}")
    print(f"  Model compiled: {'Yes' if compiled_successfully else 'No'}")
    
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
    
    print("\nOutput files:")
    print(f"  Main: {args.output}")
    if args.output.suffix == '.pt' and spatial_coords is not None:
        print(f"  Spatial coordinates: {args.output.with_name(args.output.stem + '_coords.npy')}")
    if args.output.suffix == '.pt' and args.save_metadata:
        print(f"  Metadata: {args.output.with_name(args.output.stem + '_metadata.json')}")
    
    print("="*60)
    print("Ready for UMAP analysis! Use:")
    print(f"python scripts/analysis/umap_latent_visualization.py --embeddings {args.output}")
    print("="*60)

if __name__ == "__main__":
    main()
