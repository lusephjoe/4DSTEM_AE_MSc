"""Train the autoencoder using PyTorch Lightning."""
import argparse, torch, pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.autoencoder import Autoencoder
from models.losses import create_loss_config_from_args, get_available_losses

# Suppress stem_visualization warnings once at import
try:
    from models.summary import show, calculate_metrics, calculate_diffraction_metrics
except ImportError:
    import warnings
    warnings.filterwarnings("ignore", message=".*stem_visualization.*")
    from models.summary import show, calculate_metrics, calculate_diffraction_metrics
import h5py
import json
import datetime
import logging

def setup_logging(output_dir: Path, args, timestamp: str) -> logging.Logger:
    """Set up logging to both console and file with standardized naming."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use consistent timestamp for all files in this training run
    log_file = output_dir / f"training_log_{timestamp}.txt"
    
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear existing handlers
    
    # Add file and console handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log session header
    logger.info("=" * 60)
    logger.info("4D-STEM AUTOENCODER TRAINING SESSION")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info("=" * 60)
    
    return logger

class HDF5Dataset(Dataset):
    """Optimized dataset for lazy loading of HDF5-compressed 4D-STEM data with pre-computed normalization."""
    
    def __init__(self, data_path, metadata_path=None):
        self.data_path = Path(data_path)
        
        # Only support HDF5 files now
        if self.data_path.suffix != '.h5':
            raise ValueError(f"Only HDF5 files (.h5) are supported, got: {self.data_path.suffix}")
        
        # Don't open HDF5 file in __init__ to avoid pickling issues
        self.h5_file = None
        self.arr = None
        
        # Get metadata and shape from a temporary file handle
        with h5py.File(str(data_path), 'r') as temp_file:
            temp_arr = temp_file['patterns']
            self.shape = temp_arr.shape
        
            # Load metadata from HDF5 attributes or fallback to defaults
            if hasattr(temp_arr, 'attrs') and len(temp_arr.attrs) > 0:
                self.metadata = dict(temp_arr.attrs)
                print(f"Loaded HDF5 dataset with metadata: {list(self.metadata.keys())}")
            else:
                print("Warning: No metadata found in HDF5 file, using defaults")
                self.metadata = {"data_min": 0.0, "data_max": 1.0, "data_range": 1.0, "dtype": "float16"}
        
        # Load or compute global normalization statistics ONCE
        print("Loading/computing normalization statistics...")
        self._load_or_compute_global_stats()
        
        print(f"Loaded HDF5 dataset: {self.shape} patterns, dtype: {self.metadata['dtype']}")
        print(f"Data range: {self.metadata['data_min']:.3f} to {self.metadata['data_min'] + self.metadata['data_range']:.3f}")
        print(f"Global normalization: mean={self.global_log_mean:.4f}, std={self.global_log_std:.4f}")
    
    def _ensure_file_open(self):
        """Ensure HDF5 file is open. Called lazily to avoid pickling issues."""
        if self.h5_file is None or not self.h5_file.id.valid:
            # OPTIMIZATION: Open HDF5 with optimized settings for better I/O performance
            # This handles both initial opening and reopening in worker processes
            self.h5_file = h5py.File(str(self.data_path), 'r', 
                                   rdcc_nbytes=1024*1024*64,  # 64MB chunk cache
                                   rdcc_nslots=10007)         # More cache slots
            self.arr = self.h5_file['patterns']
    
    def __getstate__(self):
        """Custom pickling to exclude HDF5 file handles."""
        state = self.__dict__.copy()
        # Remove unpicklable HDF5 objects
        state['h5_file'] = None
        state['arr'] = None
        return state
    
    def __setstate__(self, state):
        """Custom unpickling to restore state without HDF5 file handles."""
        self.__dict__.update(state)
        # HDF5 file will be reopened lazily in _ensure_file_open()
        
    def __del__(self):
        """Ensure HDF5 file is properly closed."""
        try:
            if hasattr(self, 'h5_file') and self.h5_file is not None:
                self.h5_file.close()
        except:
            pass  # Ignore errors during cleanup
    
    
    def _load_or_compute_global_stats(self):
        """Load pre-computed global statistics or compute them once."""
        stats_path = self.data_path.parent / f"{self.data_path.stem}_normalization_stats.json"
        
        if stats_path.exists():
            print("Loading pre-computed normalization statistics...")
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                self.global_log_mean = stats['log_mean']
                self.global_log_std = stats['log_std']
        else:
            print("Computing global normalization statistics (one-time cost)...")
            self._compute_and_save_global_stats(stats_path)
    
    def _compute_and_save_global_stats(self, stats_path):
        """Compute global log-space statistics efficiently using streaming approach over ENTIRE dataset."""
        total_patterns = self.shape[0]
        
        print(f"Computing normalization statistics from ALL {total_patterns} patterns (one-time cost)...")
        print("Using optimized streaming approach with chunked processing...")
        
        # Optimized chunked processing parameters
        chunk_size = min(1000, max(100, total_patterns // 1000))  # Adaptive chunk size
        pixel_stride = 50  # Sample every 50th pixel for speed while maintaining accuracy
        
        # Streaming computation using Welford's algorithm
        running_mean = 0.0
        running_m2 = 0.0  # For variance calculation  
        total_count = 0
        
        print(f"Processing in chunks of {chunk_size} patterns, sampling every {pixel_stride}th pixel...")
        
        # Open file temporarily for statistics computation
        with h5py.File(str(self.data_path), 'r') as temp_file:
            temp_arr = temp_file['patterns']
            
            # Process all patterns in chunks
            for chunk_start in range(0, total_patterns, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_patterns)
                current_chunk_size = chunk_end - chunk_start
                
                # Progress reporting
                progress = (chunk_start / total_patterns) * 100
                print(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_patterns + chunk_size - 1)//chunk_size} "
                      f"(patterns {chunk_start}-{chunk_end-1}, {progress:.1f}% complete)")
                
                try:
                    # Load chunk of patterns (optimized: load entire chunk at once)
                    chunk_data = np.array(temp_arr[chunk_start:chunk_end])
                    
                    # Apply same dequantization as in training (vectorized)
                    if self.metadata["dtype"] == "uint16":
                        chunk_data = chunk_data.astype("float32") / 65535.0 * self.metadata["data_range"] + self.metadata["data_min"]
                    elif self.metadata["dtype"] == "float16":
                        chunk_data = chunk_data.astype("float32")
                    
                    # Apply log scaling (vectorized) - must match training transform
                    log_chunk = np.log(chunk_data + 1)
                    
                    # Sample pixels efficiently: flatten all patterns then subsample
                    flat_chunk = log_chunk.flatten()
                    sampled_pixels = flat_chunk[::pixel_stride]
                    
                    # Update running statistics using Welford's algorithm (optimized for chunk processing)
                    for value in sampled_pixels:
                        total_count += 1
                        delta = value - running_mean
                        running_mean += delta / total_count
                        delta2 = value - running_mean
                        running_m2 += delta * delta2
                    
                    # Force garbage collection to free memory
                    del chunk_data, log_chunk, flat_chunk, sampled_pixels
                    
                except Exception as e:
                    print(f"Warning: Failed to process chunk {chunk_start}-{chunk_end}: {e}")
                    continue
        
        # Finalize statistics
        self.global_log_mean = float(running_mean)
        self.global_log_std = float(np.sqrt(running_m2 / (total_count - 1)) if total_count > 1 else 1.0)
        
        # Save for future use
        stats = {
            'log_mean': self.global_log_mean,
            'log_std': self.global_log_std,
            'total_patterns_used': total_patterns,
            'pixels_sampled': total_count,
            'pixel_stride': pixel_stride,
            'chunk_size': chunk_size,
            'computed_on': datetime.datetime.now().isoformat(),
            'method': 'full_dataset_streaming_welford'
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Global stats computed and saved: mean={self.global_log_mean:.4f}, std={self.global_log_std:.4f}")
        print(f"Used ALL {total_patterns} patterns with {total_count} pixel samples (every {pixel_stride}th pixel)")
        print(f"Statistics saved to: {stats_path}")
    
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, idx):
        # OPTIMIZED: Fast path with pre-computed normalization
        if torch.is_tensor(idx):
            idx = idx.item()
        
        # Ensure file is open (lazy loading)
        self._ensure_file_open()
        
        # Direct conversion with minimal copies
        pattern = np.array(self.arr[idx])
        x = torch.from_numpy(pattern).float()
        
        # Apply dequantization (cached from init)
        x = self._dequantize_fast(x)
        
        # Apply PRE-COMPUTED normalization (major speedup!)
        x = torch.log(x + 1)
        x = (x - self.global_log_mean) / (self.global_log_std + 1e-8)
        
        # Ensure channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        return (x,)
    
    def _dequantize_fast(self, x):
        """Optimized dequantization with minimal branching."""
        dtype = self.metadata["dtype"]
        if dtype == "uint16":
            return x * (self.metadata["data_range"] / 65535.0) + self.metadata["data_min"]
        elif dtype == "float16":
            return x  # Already float32 from torch.from_numpy().float()
        return x  # float32 ready

class LitAE(pl.LightningModule):
    def __init__(self, latent_dim: int, lr: float, realtime_metrics: bool = False, 
                 loss_config: dict = None, out_shape: tuple[int,int] = (256, 256)):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model with loss configuration
        self.model = Autoencoder(latent_dim, out_shape, loss_config)
        self.train_losses: list[float] = []
        self.validation_metrics: dict = {}
        self.realtime_metrics = realtime_metrics
        
        # Store loss config for logging
        self.loss_config = loss_config or {
            'reconstruction_loss': 'mse',
            'regularization_losses': {
                'lp_reg': 1e-4,
                'contrastive': 5e-5,
                'divergence': 2e-4
            }
        }
        
        # Normalization parameters for scale-aligned loss computation
        # These will be set by the data loader
        self.register_buffer('global_log_mean', torch.tensor(0.0))
        self.register_buffer('global_log_std', torch.tensor(1.0))

    def forward(self, x):
        return self.model(x)
    
    def denormalize_to_log_space(self, x_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized tensor back to log space for scale-aligned loss computation."""
        return x_normalized * (self.global_log_std + 1e-8) + self.global_log_mean
    
    def set_normalization_params(self, log_mean: float, log_std: float):
        """Set normalization parameters from data loader."""
        self.global_log_mean.data = torch.tensor(log_mean)
        self.global_log_std.data = torch.tensor(log_std)

    def training_step(self, batch, batch_idx):
        x, = batch
        z = self.model.embed(x)
        x_hat = self.model.decoder(z)
        
        # SCALE-ALIGNED LOSS: Denormalize BOTH input and output to log space
        # Both x and x_hat are in normalized space, so both need denormalization
        x_log = self.denormalize_to_log_space(x)
        x_hat_log = self.denormalize_to_log_space(x_hat)
        
        # Compute loss using flexible loss system in aligned log space
        loss_dict = self.model.compute_loss(x_log, x_hat_log, z)
        loss = loss_dict['total_loss']

        # OPTIMIZATION: Record losses much less frequently to reduce CPU-GPU sync
        if batch_idx % 50 == 0:  # Reduced from every 10 steps
            self.train_losses.append(loss.detach().cpu().item())

        # Log essential metrics only
        self.log("train_loss", loss, prog_bar=True)
        
        # Log main reconstruction loss (dynamically based on loss type)
        recon_loss_key = f"{self.model.loss_manager.reconstruction_loss.name}_loss"
        if recon_loss_key in loss_dict:
            self.log(f"train_{self.model.loss_manager.reconstruction_loss.name}", 
                    loss_dict[recon_loss_key], prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, = batch
        z = self.model.embed(x)
        x_hat = self.model.decoder(z)
        
        # SCALE-ALIGNED LOSS: Denormalize BOTH input and output to log space
        # Both x and x_hat are in normalized space, so both need denormalization
        x_log = self.denormalize_to_log_space(x)
        x_hat_log = self.denormalize_to_log_space(x_hat)
        
        # Compute loss using flexible loss system in aligned log space
        loss_dict = self.model.compute_loss(x_log, x_hat_log, z)
        loss = loss_dict['total_loss']
        
        # Calculate detailed metrics for validation (use denormalized data)
        metrics = calculate_metrics(x_log, x_hat_log)
        diffraction_metrics = calculate_diffraction_metrics(x_log, x_hat_log)
        
        self.log("val_loss", loss, prog_bar=True)
        
        # Log main reconstruction loss (dynamically based on loss type)
        recon_loss_key = f"{self.model.loss_manager.reconstruction_loss.name}_loss"
        if recon_loss_key in loss_dict:
            self.log(f"val_{self.model.loss_manager.reconstruction_loss.name}", 
                    loss_dict[recon_loss_key], prog_bar=False)
        
        self.log("val_psnr", metrics['psnr'], prog_bar=True)
        self.log("val_ssim", metrics['ssim'], prog_bar=True)
        
        # Log domain-specific diffraction metrics
        self.log("val_peak_preservation", diffraction_metrics['peak_preservation'], prog_bar=False)
        self.log("val_log_correlation", diffraction_metrics['log_correlation'], prog_bar=False)
        self.log("val_range_preservation", diffraction_metrics['range_preservation'], prog_bar=False)
        self.log("val_center_mse", diffraction_metrics['center_region_mse'], prog_bar=False)
        
        # Store validation metrics for potential use
        self.validation_metrics = {
            'loss': loss.item(),
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim'],
            **diffraction_metrics  # Include all diffraction metrics
        }
        
        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler (inspired by reference STEM_AE)."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        
        # Check if scheduler is disabled
        if getattr(self.hparams, 'no_scheduler', False):
            return optimizer
        
        # Simple CyclicLR scheduler matching reference implementation
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.hparams.lr,           # Current learning rate as base
            max_lr=self.hparams.lr * 3.33,    # Reference uses 1e-4 max with 3e-5 base
            step_size_up=15,                  # Exactly as in reference
            cycle_momentum=False              # Exactly as in reference
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Step every batch (matches reference)
            }
        }


def load_dataset(data_path, logger):
    """Load dataset and detect input size."""
    if data_path.suffix == '.h5':
        logger.info(f"Loading HDF5 dataset from: {data_path}")
        dataset = HDF5Dataset(data_path)
        sample = dataset[0][0] if isinstance(dataset[0], tuple) else dataset[0]
        detected_size = sample.shape[-1]
    else:
        logger.info(f"Loading tensor dataset from: {data_path}")
        data = torch.load(data_path)
        dataset = TensorDataset(data)
        detected_size = data.shape[-1] if len(data.shape) == 4 else 256
    
    logger.info(f"Detected input size: {detected_size}x{detected_size}")
    return dataset, detected_size

def create_train_val_split(full_dataset, no_validation, logger):
    """Create train/validation split."""
    if no_validation:
        logger.info(f"Using entire dataset for training: {len(full_dataset)} patterns (no validation)")
        return full_dataset, None
    
    # 80/20 train/validation split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    indices = torch.randperm(total_size)
    
    if isinstance(full_dataset, TensorDataset):
        # Split tensor data directly
        data = full_dataset.tensors[0]
        train_data = data[indices[:train_size]]
        val_data = data[indices[train_size:]]
        train_ds, val_ds = TensorDataset(train_data), TensorDataset(val_data)
    else:
        # Use Subset for zarr datasets
        from torch.utils.data import Subset
        train_ds = Subset(full_dataset, indices[:train_size])
        val_ds = Subset(full_dataset, indices[train_size:])
    
    logger.info(f"Created train/val split: {len(train_ds)} train, {len(val_ds)} validation")
    return train_ds, val_ds

def create_data_loaders(train_ds, val_ds, args):
    """Create optimized training and validation data loaders."""
    # OPTIMIZATION: Enable multiprocessing on Windows with proper HDF5 handling
    import platform
    
    # Use multiprocessing on all platforms - HDF5Dataset now handles pickling properly
    num_workers = args.num_workers
    
    if platform.system() == 'Windows' and num_workers > 0:
        print(f"Windows detected: Enabling {num_workers} worker processes for faster data loading")
        print("Using optimized HDF5 dataset with proper multiprocessing support")
    elif num_workers == 0:
        print("Single-threaded data loading enabled")
        print("PERFORMANCE TIP: Use --num_workers 4 for faster data loading")
    else:
        print(f"Using {num_workers} worker processes for data loading")
    
    dataloader_kwargs = {
        'batch_size': args.batch,
        'num_workers': num_workers,
        'pin_memory': args.pin_memory and torch.cuda.is_available(),  # Only pin if GPU available
        'persistent_workers': args.persistent_workers and num_workers > 0,
        'drop_last': True,     # Ensure consistent batch sizes for mixed precision
    }
    
    # OPTIMIZATION: Tune prefetch and buffer size for better performance
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = 4  # Increase prefetch for better pipeline
    
    # OPTIMIZATION: Use spawn context for all platforms for better HDF5 compatibility
    if num_workers > 0:
        import multiprocessing as mp
        if platform.system() == 'Windows':
            # Windows always uses spawn, so this is explicit
            dataloader_kwargs['multiprocessing_context'] = mp.get_context('spawn')
        else:
            # Use spawn on Unix too for consistent HDF5 behavior
            dataloader_kwargs['multiprocessing_context'] = mp.get_context('spawn')
    
    # Create data loaders with error handling
    try:
        train_dl = DataLoader(train_ds, shuffle=True, **dataloader_kwargs)
        val_dl = DataLoader(val_ds, shuffle=False, **dataloader_kwargs) if val_ds else None
        
        # Test the data loader with a small batch to catch issues early (debug mode only)
        if num_workers > 0 and args.debug:
            print("Testing multiprocessing data loader...")
            test_batch = next(iter(train_dl))
            print(f"✓ Multiprocessing test successful: batch shape {test_batch[0].shape}")
            
    except Exception as e:
        print(f"ERROR: Multiprocessing data loading failed: {e}")
        print("Falling back to single-threaded data loading...")
        # Fallback to single-threaded
        fallback_kwargs = dataloader_kwargs.copy()
        fallback_kwargs['num_workers'] = 0
        fallback_kwargs.pop('multiprocessing_context', None)
        fallback_kwargs.pop('prefetch_factor', None)
        fallback_kwargs['persistent_workers'] = False
        
        train_dl = DataLoader(train_ds, shuffle=True, **fallback_kwargs)
        val_dl = DataLoader(val_ds, shuffle=False, **fallback_kwargs) if val_ds else None
        num_workers = 0
    
    print(f"Final data loading config: batch_size={args.batch}, num_workers={num_workers}")
    if num_workers == 0:
        print("PERFORMANCE TIP: Consider using NVMe SSD storage for better single-threaded I/O")
    
    return train_dl, val_dl

def create_model(args, detected_size):
    """Create the autoencoder model with flexible loss configuration."""
    # Create loss configuration from args
    loss_config = create_loss_config_from_args(args)
    
    model = LitAE(
        args.latent, args.lr, args.realtime_metrics,
        loss_config, (detected_size, detected_size)
    )
    
    # Apply torch.compile for Linux only (maximum efficiency)
    if args.compile:
        import platform
        if platform.system() == 'Linux':
            try:
                print("🚀 Compiling model with torch.compile for maximum efficiency...")
                model.model = torch.compile(
                    model.model, 
                    mode='max-autotune',  # Most aggressive optimization
                    fullgraph=True,       # Compile entire graph for best performance
                    dynamic=False         # Static shapes for maximum speed
                )
                print("✅ Model compilation successful!")
            except Exception as e:
                print(f"⚠️  Model compilation failed: {e}")
                print("Continuing with uncompiled model...")
        else:
            print(f"⚠️  torch.compile only enabled on Linux (current OS: {platform.system()})")
            print("Continuing with uncompiled model...")
    
    return model


def setup_trainer(args, base_name):
    """Setup PyTorch Lightning trainer with epoch checkpointing."""
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="tb_logs",
        default_hp_metric=False
    )
    
    # Setup checkpoint callback to save every N epochs
    checkpoint_dir = args.output_dir / "checkpoints"
    
    if args.no_validation:
        # Monitor training loss when no validation
        filename_pattern = f"{base_name}_epoch{{epoch:03d}}_trainloss{{train_loss:.4f}}"
        monitor_metric = "train_loss"
    else:
        # Monitor validation loss when available
        filename_pattern = f"{base_name}_epoch{{epoch:03d}}_valloss{{val_loss:.4f}}"
        monitor_metric = "val_loss"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=filename_pattern,
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=getattr(args, 'save_every_n_epochs', 1),
        save_on_train_epoch_end=args.no_validation,  # Save after training if no validation
        monitor=monitor_metric,
        mode="min"
    )
    
    # Configure accelerator
    if args.device == "cuda":
        accelerator, devices = "gpu", args.gpus
    elif args.device == "mps":
        accelerator, devices = "mps", 1
    else:
        accelerator, devices = "cpu", 1
    
    # Map precision argument to PyTorch Lightning format
    precision_map = {
        "32": "32-true",
        "16": "16-mixed", 
        "bf16": "bf16-mixed"
    }
    precision = precision_map.get(args.precision, "16-mixed")
    
    return pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        logger=tb_logger,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],
        precision=precision,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=50,  # Reduce logging frequency
        enable_model_summary=False  # Disable model summary to reduce output
    )

def train_model(trainer, model, train_dl, val_dl, logger, start_time, resume_checkpoint=None):
    """Train the model and return training duration."""
    if resume_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
    else:
        logger.info("Starting model training...")
    
    if val_dl is not None:
        trainer.fit(model, train_dl, val_dl, ckpt_path=resume_checkpoint)
    else:
        logger.info("Training without validation - using entire dataset")
        trainer.fit(model, train_dl, ckpt_path=resume_checkpoint)
    
    training_end_time = datetime.datetime.now()
    training_duration = training_end_time - start_time
    logger.info(f"Training completed at: {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Training duration: {training_duration}")
    return training_duration

def save_model_and_results(trainer, model, args, logger, base_name):
    """Save final model checkpoint and loss curve with standardized naming."""
    args.output_dir.mkdir(exist_ok=True)
    
    # Get current epoch from trainer
    current_epoch = trainer.current_epoch + 1  # +1 because epochs are 0-indexed
    
    # Generate final filename with epoch and MSE included
    final_mse = model.train_losses[-1] if model.train_losses else 0.0
    final_base_name = f"{base_name}_epoch{current_epoch:03d}_mse{final_mse:.4f}"
    
    # Save final checkpoint (in addition to epoch checkpoints)
    final_ckpt_path = args.output_dir / f"{final_base_name}_final.ckpt"
    trainer.save_checkpoint(final_ckpt_path)
    logger.info(f"Final model saved to {final_ckpt_path}")
    
    # Save loss curve
    loss_curve_path = args.output_dir / f"{final_base_name}_loss.png"
    plt.figure()
    plt.plot(model.train_losses)
    plt.xlabel("batch")
    plt.ylabel("MSE loss")
    plt.yscale("log")
    plt.title(f"Training Loss (Final MSE: {final_mse:.4f})")
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    logger.info(f"Loss curve saved to {loss_curve_path}")
    
    # Log information about epoch checkpoints
    checkpoint_dir = args.output_dir / "checkpoints"
    if checkpoint_dir.exists():
        epoch_checkpoints = list(checkpoint_dir.glob(f"{base_name}_epoch*.ckpt"))
        logger.info(f"Epoch checkpoints saved: {len(epoch_checkpoints)} files in {checkpoint_dir}")
    
    return final_base_name  # Return for use in other saves

def perform_final_evaluation(model, train_dl, val_dl, args, logger, base_name):
    """Perform final model evaluation with standardized naming."""
    logger.info("="*60)
    logger.info("FINAL MODEL EVALUATION")
    logger.info("="*60)
    
    model.eval()
    model = model.to(args.device)
    
    with torch.no_grad():
        # Get evaluation data
        if val_dl is not None:
            eval_sample = next(iter(val_dl))
            eval_name = "Validation"
        else:
            eval_sample = next(iter(train_dl))
            eval_name = "Training"
        
        if isinstance(eval_sample, (list, tuple)):
            eval_sample = eval_sample[0]
        
        eval_input = eval_sample.to(args.device)
        eval_output = model(eval_input)
        
        # Calculate and log metrics
        final_metrics = calculate_metrics(eval_input, eval_output)
        logger.info(f"{eval_name} MSE:     {final_metrics['mse']:.6f} ± {final_metrics['mse_std']:.6f}")
        logger.info(f"{eval_name} PSNR:    {final_metrics['psnr']:.2f} ± {final_metrics['psnr_std']:.2f} dB")
        logger.info(f"{eval_name} SSIM:    {final_metrics['ssim']:.4f} ± {final_metrics['ssim_std']:.4f}")
        
        # Save comparison images with standardized naming
        from models.summary import save_comparison_images
        final_comparison_path = args.output_dir / f"{base_name}_reconstruction.png"
        save_comparison_images(eval_input, eval_output, final_comparison_path, num_samples=8)
        logger.info(f"Final comparison saved to {final_comparison_path}")
        
        # Log memory usage
        if args.device == "cuda":
            logger.info("CUDA Memory Usage:")
            logger.info(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"  Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            logger.info(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

def finalize_training(logger, start_time, training_duration, args):
    """Log final summary and cleanup."""
    end_time = datetime.datetime.now()
    total_duration = end_time - start_time
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info(f"Script started at:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Script ended at:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration:     {total_duration}")
    logger.info(f"Training duration:  {training_duration}")
    
    # Summary of saved files
    checkpoint_dir = args.output_dir / "checkpoints"
    if checkpoint_dir.exists():
        epoch_checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        logger.info(f"Epoch checkpoints:  {len(epoch_checkpoints)} files saved")
    
    final_checkpoints = list(args.output_dir.glob("*_final.ckpt"))
    logger.info(f"Final checkpoint:   {len(final_checkpoints)} file saved")
    
    logger.info("="*60)
    
    # Cleanup
    if args.device == "cuda":
        torch.cuda.empty_cache()
    
    for handler in logger.handlers:
        handler.flush()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--device", type=str, default="auto", help="Device to use: auto, cpu, cuda, mps")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no_scheduler", action="store_true", help="Disable learning rate scheduler (CyclicLR enabled by default)")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--realtime_metrics", action="store_true", help="Enable real-time metrics calculation during training (may slow down training)")
    # Loss function arguments  
    available_losses = get_available_losses()
    p.add_argument("--loss_function", type=str, default="mse", 
                   choices=available_losses['reconstruction'],
                   help=f"Reconstruction loss function. Available: {available_losses['reconstruction']}")
    
    # Regularization arguments
    p.add_argument("--lambda_act", type=float, default=1e-5, help="Lp regularization coefficient for sparsity")
    p.add_argument("--lambda_sim", type=float, default=0, help="Contrastive similarity regularization coefficient")
    p.add_argument("--lambda_div", type=float, default=0, help="Activation divergence regularization coefficient")
    p.add_argument("--lambda_l2", type=float, default=0, help="L2 regularization coefficient")
    p.add_argument("--lambda_kl", type=float, default=0, help="KL divergence regularization coefficient")
    p.add_argument("--input_size", type=int, default=256, help="Input image size (assumes square images)")
    p.add_argument("--precision", choices=["32", "16", "bf16"], default="16", help="Training precision (32=float32, 16=float16, bf16=bfloat16)")
    p.add_argument("--compile", action="store_true", help="Use torch.compile for faster training (PyTorch 2.0+)")
    p.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    p.add_argument("--pin_memory", action="store_true", default=True, help="Pin memory for faster GPU transfer")
    p.add_argument("--persistent_workers", action="store_true", help="Keep workers alive between epochs")
    p.add_argument("--profile", action="store_true", help="Enable PyTorch profiler for performance analysis")
    p.add_argument("--debug", action="store_true", help="Enable debug output including data loader tests")
    p.add_argument("--no_validation", action="store_true", help="Disable train/validation split - use entire dataset for training")
    p.add_argument("--save_every_n_epochs", type=int, default=1, help="Save checkpoint every N epochs (default: 1)")
    p.add_argument("--accumulate_grad_batches", type=int, default=1, help="Number of batches to accumulate gradients over (default: 1)")
    p.add_argument("--resume_from_checkpoint", type=Path, default=None, help="Path to checkpoint file to resume training from")

    args = p.parse_args()

    # Set up logging with standardized naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(args.output_dir, args, timestamp)
    
    # Record start time
    start_time = datetime.datetime.now()
    logger.info(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Handle device selection more robustly
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
            torch.set_float32_matmul_precision('medium')
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    
    logger.info(f"Using device: {args.device}")
    
    pl.seed_everything(args.seed)

    # Load data and create datasets
    data_path = Path(args.data)
    full_dataset, detected_size = load_dataset(data_path, logger)
    train_ds, val_ds = create_train_val_split(full_dataset, args.no_validation, logger)
    
    # Create data loaders and model
    train_dl, val_dl = create_data_loaders(train_ds, val_ds, args)
    model = create_model(args, detected_size)
    
    # SCALE-ALIGNED LOSS: Set normalization parameters from dataset
    model.set_normalization_params(train_ds.global_log_mean, train_ds.global_log_std)
    logger.info(f"Scale-aligned loss enabled: log_mean={train_ds.global_log_mean:.4f}, log_std={train_ds.global_log_std:.4f}")
    
    # Log loss configuration
    loss_info = model.model.get_loss_info()
    logger.info("Loss Configuration:")
    logger.info(f"  Reconstruction loss: {loss_info['reconstruction']}")
    
    # Log scheduler configuration
    if args.no_scheduler:
        logger.info("  Learning rate scheduler: DISABLED")
    else:
        logger.info(f"  Learning rate scheduler: CyclicLR (base={args.lr:.2e}, max={args.lr*3.33:.2e}, step_size=15)")
    
    # Check for regularization losses
    reg_losses = {k: v for k, v in loss_info.items() if k.startswith('regularization_')}
    if reg_losses:
        logger.info("  Regularization losses:")
        # Map regularization loss names to argument names
        loss_to_arg_map = {
            'l1_reg': 'lambda_act',
            'l2_reg': 'lambda_l2', 
            'contrastive_0.1': 'lambda_sim',
            'divergence': 'lambda_div',
            'kl_div_1.0': 'lambda_kl'
        }
        
        for loss_name in reg_losses.values():
            # Find the correct argument name based on loss name
            arg_name = loss_to_arg_map.get(loss_name, 'lambda_act')  # default to lambda_act
            weight = getattr(args, arg_name, 0)
            logger.info(f"    {loss_name}: {weight}")
    else:
        logger.info("  No regularization losses configured")
    

    # Generate base name for consistent naming across all files
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    base_name = f"ae_e{args.epochs:03d}_{timestamp}"
    logger.info(f"Base filename: {base_name}")
    
    # Validate checkpoint path if provided
    resume_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.exists():
            resume_checkpoint = str(args.resume_from_checkpoint)
            logger.info(f"Checkpoint found: {resume_checkpoint}")
        else:
            logger.error(f"Checkpoint file not found: {args.resume_from_checkpoint}")
            return
    
    # Setup trainer with epoch checkpointing
    trainer = setup_trainer(args, base_name)

    # Train model and complete workflow
    training_duration = train_model(trainer, model, train_dl, val_dl, logger, start_time, resume_checkpoint)
    final_base_name = save_model_and_results(trainer, model, args, logger, base_name)
    perform_final_evaluation(model, train_dl, val_dl, args, logger, final_base_name)
    finalize_training(logger, start_time, training_duration, args)

if __name__ == "__main__":
    main()