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
    p.add_argument("--batch_size", type=int, default=256,
                   help="Batch size for processing (default: 256 for RTX A6000)")
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
    p.add_argument("--num_workers", type=int, default=4,  # Default to 4 like training
                   help="Number of data loader workers (4=default, same as training)")
    p.add_argument("--prefetch_factor", type=int, default=4,  # Same as training
                   help="Number of batches to prefetch per worker")
    p.add_argument("--persistent_workers", action="store_true", default=True,
                   help="Keep workers alive between batches (default: True for better performance)")
    p.add_argument("--no_persistent_workers", action="store_true",
                   help="Disable persistent workers")
    p.add_argument("--optimize_memory", action="store_true",
                   help="Enable memory optimizations for large datasets")
    p.add_argument("--auto_batch_size", action="store_true",
                   help="Automatically determine optimal batch size for GPU")
    p.add_argument("--channels_last", action="store_true",
                   help="Use channels_last memory format for better performance")
    
    # Analysis arguments
    p.add_argument("--pca", type=int, default=0,
                   help="If >0: reduce dimensionality with PCA to this many components")
    p.add_argument("--save_spatial_coords", action="store_true",
                   help="Save spatial coordinates (for HDF5 data)")
    p.add_argument("--save_metadata", action="store_true", default=True,
                   help="Save comprehensive metadata")
    p.add_argument("--debug", action="store_true",
                   help="Enable debug output for troubleshooting")
    
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
        
        # Advanced optimizations for modern GPUs (PyTorch 2.0+)
        try:
            # These functions don't take arguments - they're context managers or properties
            if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
                # Check if it's available and enable
                pass  # This is typically enabled by default in newer PyTorch
            if hasattr(torch.backends.cuda, 'enable_math_sdp'):
                torch.backends.cuda.enable_math_sdp(True)
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception as e:
            print(f"   Warning: Some CUDA optimizations unavailable: {e}")
        
        # Set memory allocator optimizations
        torch.cuda.empty_cache()
        
        # Memory fraction setting (be more conservative on Windows)
        try:
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                # Reserve more memory for system on Windows
                torch.cuda.set_per_process_memory_fraction(0.85)
        except Exception as e:
            print(f"   Warning: Could not set memory fraction: {e}")
        
        # Enable memory pool for faster allocations
        try:
            if hasattr(torch.cuda.memory, '_set_allocator_settings'):
                torch.cuda.memory._set_allocator_settings('expandable_segments:True')
        except Exception as e:
            print(f"   Warning: Memory pool optimization unavailable: {e}")
        
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

def find_optimal_batch_size(model: torch.nn.Module, sample_input: torch.Tensor, 
                           device: torch.device, max_batch_size: int = 1024) -> int:
    """Find optimal batch size through binary search."""
    if device.type != "cuda":
        return min(max_batch_size, 64)  # Conservative for CPU/MPS
    
    print("🔍 Finding optimal batch size...")
    model.eval()
    
    # Start with a reasonable lower bound
    min_batch_size = 1
    optimal_batch_size = min_batch_size
    
    for batch_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, max_batch_size]:
        if batch_size > max_batch_size:
            break
            
        try:
            # Clear cache before test
            torch.cuda.empty_cache()
            
            # Create test batch
            test_batch = sample_input.repeat(batch_size, *([1] * (sample_input.dim() - 1)))
            test_batch = test_batch.to(device, non_blocking=True)
            
            # Test forward pass
            with torch.no_grad():
                _ = model(test_batch)
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated(device) / 1024**3
            memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            memory_usage_pct = (memory_used / memory_total) * 100
            
            if memory_usage_pct < 80:  # Keep under 80% memory usage
                optimal_batch_size = batch_size
                print(f"   Batch size {batch_size}: OK ({memory_usage_pct:.1f}% GPU memory)")
            else:
                print(f"   Batch size {batch_size}: Too large ({memory_usage_pct:.1f}% GPU memory)")
                break
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"   Batch size {batch_size}: Out of memory")
                break
            else:
                raise e
    
    # Clear cache after testing
    torch.cuda.empty_cache()
    
    print(f"✓ Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size

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

def optimize_model_for_inference(model: torch.nn.Module, use_channels_last: bool = False) -> torch.nn.Module:
    """Apply inference optimizations to the model."""
    model.eval()
    
    # Convert to channels_last for better performance on modern GPUs
    if use_channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
            print("✓ Model converted to channels_last memory format")
        except Exception as e:
            print(f"   Warning: Could not convert to channels_last: {e}")
    
    return model

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
    
    # Create DataLoader using the EXACT same approach as training (which works)
    import platform
    num_workers = args.num_workers
    
    if platform.system() == 'Windows' and num_workers > 0:
        print(f"Windows detected: Enabling {num_workers} worker processes for faster data loading")
        print("Using optimized HDF5 dataset with proper multiprocessing support")
        print("💡 For even faster loading, try --num_workers 8 or --num_workers 12")
    elif num_workers == 0:
        print("Single-threaded data loading enabled")
        print("⚠️  Performance will be slow! Recommend --num_workers 4 for faster loading")
    else:
        print(f"Using {num_workers} worker processes for data loading")
    
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': num_workers,
        'pin_memory': True and torch.cuda.is_available(),  # Only pin if GPU available
        'persistent_workers': args.persistent_workers and num_workers > 0,
        'shuffle': False,  # Don't shuffle for embedding generation
        'drop_last': False,  # Keep all samples
    }
    
    # Add prefetch factor for better performance
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = 4  # Same as training
    
    # Use spawn context for HDF5 compatibility (same as training)
    if num_workers > 0:
        import multiprocessing as mp
        if platform.system() == 'Windows':
            dataloader_kwargs['multiprocessing_context'] = mp.get_context('spawn')
        else:
            dataloader_kwargs['multiprocessing_context'] = mp.get_context('spawn')
    
    # Create data loader with error handling (same as training)
    try:
        print("Creating DataLoader...")
        start_loader_time = time.time()
        loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
        loader_creation_time = time.time() - start_loader_time
        print(f"✓ DataLoader created in {loader_creation_time:.2f}s")
        
        # Test the data loader to catch issues early
        if num_workers > 0:
            print("Testing multiprocessing data loader...")
            test_start = time.time()
            try:
                print("   Attempting to load first batch...")
                test_batch = next(iter(loader))
                test_time = time.time() - test_start
                
                if isinstance(test_batch, (list, tuple)):
                    test_shape = test_batch[0].shape
                    print(f"✓ First batch loaded in {test_time:.2f}s: shape {test_shape}")
                    print(f"   Batch format: tuple with {len(test_batch)} elements")
                else:
                    test_shape = test_batch.shape  
                    print(f"✓ First batch loaded in {test_time:.2f}s: shape {test_shape}")
                    print(f"   Batch format: tensor")
                
                # Test a few more batches to see if workers are truly working
                print("   Testing batch loading speed...")
                batch_times = []
                test_iter = iter(loader)
                for i in range(min(3, len(loader))):
                    batch_start = time.time()
                    next(test_iter)
                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)
                    print(f"   Batch {i+1}: {batch_time:.3f}s")
                
                avg_batch_time = sum(batch_times) / len(batch_times)
                print(f"   Average batch loading time: {avg_batch_time:.3f}s")
                
                if avg_batch_time > 2.0:
                    print("⚠️  WARNING: Batch loading is slow! This will cause the processing delay.")
                    print("   Consider reducing --num_workers or checking HDF5 file access speed")
                
            except Exception as e:
                print(f"   Multiprocessing test failed: {e}")
                raise e
            
    except Exception as e:
        print(f"ERROR: Multiprocessing data loading failed: {e}")
        print("Falling back to single-threaded data loading...")
        # Fallback to single-threaded (same as training)
        fallback_kwargs = dataloader_kwargs.copy()
        fallback_kwargs['num_workers'] = 0
        fallback_kwargs.pop('multiprocessing_context', None)
        fallback_kwargs.pop('prefetch_factor', None)
        fallback_kwargs['persistent_workers'] = False
        
        loader = torch.utils.data.DataLoader(dataset, **fallback_kwargs)
        num_workers = 0
    
    print(f"Final data loading config: batch_size={args.batch_size}, num_workers={num_workers}")
    
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
    
    # Apply inference optimizations
    encoder = optimize_model_for_inference(encoder, args.channels_last)
    
    # Load data
    loader, spatial_coords, metadata = load_data(args)
    
    # Auto-determine optimal batch size if requested
    if args.auto_batch_size and device.type == "cuda":
        # Get sample for batch size optimization
        sample_batch = next(iter(loader))
        if isinstance(sample_batch, (list, tuple)):
            sample_input = sample_batch[0][:1]  # Single sample
        else:
            sample_input = sample_batch[:1]
        
        # Apply channels_last if requested
        if args.channels_last and len(sample_input.shape) == 4:
            sample_input = sample_input.to(memory_format=torch.channels_last)
        
        optimal_batch_size = find_optimal_batch_size(encoder, sample_input, device, args.batch_size)
        
        if optimal_batch_size != args.batch_size:
            print(f"📊 Adjusting batch size: {args.batch_size} → {optimal_batch_size}")
            args.batch_size = optimal_batch_size
            # Recreate loader with optimal batch size
            loader, spatial_coords, metadata = load_data(args)
    
    # Optimize batch size for GPU memory (legacy path)
    elif device.type == "cuda" and args.optimize_memory:
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
            # Check PyTorch version first
            torch_version = torch.__version__
            major, minor = map(int, torch_version.split('.')[:2])
            
            if major >= 2:  # PyTorch 2.0+
                # Check if Triton is available (required for torch.compile on CUDA)
                triton_available = False
                try:
                    import triton
                    triton_available = True
                    print("   ✓ Triton available - full compilation enabled")
                except ImportError:
                    triton_available = False
                    print("   ⚠ Triton not available - skipping torch.compile")
                
                if hasattr(torch, "compile") and triton_available:
                    # Only compile if Triton is available
                    import platform
                    if platform.system() == "Windows":
                        # Windows-safe compilation
                        encoder = torch.compile(encoder, mode="default", dynamic=False)
                    else:
                        # Full optimization on Linux/macOS
                        encoder = torch.compile(encoder, mode="reduce-overhead", dynamic=False)
                    
                    compiled_successfully = True
                    print("✓ Model compiled with torch.compile for faster inference")
                elif not triton_available:
                    print("   Skipping torch.compile - Triton required for CUDA compilation")
                else:
                    print("   torch.compile not available in this PyTorch version")
            else:
                print(f"   torch.compile requires PyTorch 2.0+ (current: {torch_version})")
                
        except Exception as e:
            print(f"   Warning: Could not compile model: {e}")
            print("   Continuing with uncompiled model...")
    elif args.no_compile:
        print("torch.compile disabled by user")
    
    if not compiled_successfully:
        print("Using standard PyTorch inference (no compilation)")
    
    # Generate embeddings (optimized GPU processing pipeline)
    print("Generating embeddings...")
    latents = []
    total_samples = len(loader.dataset)
    start_time = time.time()
    
    print(f"Total samples to process: {total_samples}")
    print(f"Number of batches: {len(loader)}")
    print(f"GPU batch processing optimizations enabled")
    
    if args.debug:
        print("Debug mode enabled - will show details for first few batches")
    
    # Set model to eval mode and enable inference optimizations
    encoder.eval()
    
    # Enable torch inference mode for maximum speed (PyTorch 1.9+)
    inference_mode_context = torch.inference_mode() if hasattr(torch, 'inference_mode') else torch.no_grad()
    
    # Use tqdm with faster update settings
    progress_bar = tqdm(
        loader, 
        desc="Processing batches", 
        unit="batch",
        total=len(loader),
        ncols=120,
        mininterval=0.5,  # Update every 0.5 seconds
        smoothing=0.1     # Faster smoothing for more responsive display
    )
    
    with inference_mode_context:
        try:
            for batch_idx, batch_data in enumerate(progress_bar):
                if args.debug and batch_idx == 0:
                    print(f"\nProcessing batch {batch_idx}")
                    print(f"  Raw batch type: {type(batch_data)}")
                
                batch_start_time = time.time()
                
                # Handle batch format exactly like training
                if isinstance(batch_data, (list, tuple)):
                    batch = batch_data[0]  # HDF5Dataset returns tuple (x,)
                    if args.debug and batch_idx == 0:
                        print(f"  Extracted batch shape: {batch.shape}")
                else:
                    batch = batch_data
                    if args.debug and batch_idx == 0:
                        print(f"  Direct batch shape: {batch.shape}")
                
                # Optimized GPU transfer with non_blocking and pin_memory
                if device.type == "cuda":
                    batch = batch.to(device, non_blocking=True)
                    
                    # Apply channels_last memory format if requested
                    if args.channels_last and len(batch.shape) == 4:
                        batch = batch.to(memory_format=torch.channels_last)
                else:
                    batch = batch.to(device)
                
                # Optimized forward pass with mixed precision if enabled
                if use_amp and device.type == "cuda":
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        z = encoder(batch)
                else:
                    z = encoder(batch)
                
                if args.debug and batch_idx == 0:
                    print(f"  Embedding shape: {z.shape}")
                
                # Efficient CPU transfer and storage
                z_cpu = z.cpu()
                latents.append(z_cpu)
                
                # Update progress with optimized metrics
                batch_time = time.time() - batch_start_time
                samples_per_sec = batch.size(0) / batch_time
                
                # Optimized progress info
                progress_info = {
                    "sps": f"{samples_per_sec:.0f}",  # Shorter key
                    "batch": f"{batch_idx+1}/{len(loader)}"
                }
                
                if device.type == "cuda":
                    gpu_memory_used = torch.cuda.memory_allocated(device) / 1024**3
                    progress_info["GPU"] = f"{gpu_memory_used:.1f}GB"
                
                progress_bar.set_postfix(progress_info)
                
                # Aggressive memory cleanup for faster processing
                if (batch_idx + 1) % 5 == 0:  # More frequent cleanup
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    # Skip gc.collect() for speed - let Python handle it
                    
                # Debug: process only first few batches
                if args.debug and batch_idx >= 2:
                    print("Debug mode: stopping after 3 batches")
                    break
                
    except Exception as e:
        print(f"\nError during batch processing: {e}")
        print(f"Failed at batch {batch_idx}")
        raise e
    
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
