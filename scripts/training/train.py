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

# Suppress stem_visualization warnings once at import
try:
    from models.summary import show, calculate_metrics
except ImportError:
    import warnings
    warnings.filterwarnings("ignore", message=".*stem_visualization.*")
    from models.summary import show, calculate_metrics
import zarr
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

class ZarrDataset(Dataset):
    """Optimized dataset for lazy loading of zarr-compressed 4D-STEM data with pre-computed normalization."""
    
    def __init__(self, zarr_path, metadata_path=None):
        self.zarr_path = Path(zarr_path)
        self.arr = zarr.open(str(zarr_path), mode="r")
        
        # Load metadata with fallback defaults
        metadata_path = metadata_path or self.zarr_path.parent / f"{self.zarr_path.stem}_metadata.json"
        self.metadata = self._load_metadata(metadata_path)
        
        # Load or compute global normalization statistics ONCE
        print("Loading/computing normalization statistics...")
        self._load_or_compute_global_stats()
        
        print(f"Loaded zarr dataset: {self.arr.shape} patterns, dtype: {self.metadata['dtype']}")
        print(f"Data range: {self.metadata['data_min']:.3f} to {self.metadata['data_min'] + self.metadata['data_range']:.3f}")
        print(f"Global normalization: mean={self.global_log_mean:.4f}, std={self.global_log_std:.4f}")
    
    def _load_metadata(self, metadata_path):
        """Load metadata with default fallback."""
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        print(f"Warning: No metadata found at {metadata_path}, using defaults")
        return {"data_min": 0.0, "data_max": 1.0, "data_range": 1.0, "dtype": "uint16"}
    
    def _load_or_compute_global_stats(self):
        """Load pre-computed global statistics or compute them once."""
        stats_path = self.zarr_path.parent / f"{self.zarr_path.stem}_normalization_stats.json"
        
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
        """Compute global log-space statistics efficiently using streaming approach."""
        total_patterns = self.arr.shape[0]
        
        # Use much smaller sample size for very large datasets to avoid hanging
        n_samples = min(500, total_patterns // 10)  # Much smaller sample, but representative
        print(f"Using {n_samples} samples to estimate global statistics from {total_patterns} total patterns...")
        
        # Use evenly spaced indices instead of random for better coverage
        step = max(1, total_patterns // n_samples)
        indices = np.arange(0, total_patterns, step)[:n_samples]
        
        # Streaming computation to avoid memory issues
        running_mean = 0.0
        running_m2 = 0.0  # For variance calculation
        total_count = 0
        
        print("Computing statistics with streaming approach...")
        for i, idx in enumerate(indices):
            if i % 100 == 0:  # More frequent updates
                print(f"Processing sample {i+1}/{len(indices)}... ({100*i/len(indices):.1f}%)")
            
            try:
                # Load single pattern
                pattern = np.array(self.arr[idx])
                
                # Apply same dequantization as in training
                if self.metadata["dtype"] == "uint16":
                    pattern = pattern.astype("float32") / 65535.0 * self.metadata["data_range"] + self.metadata["data_min"]
                elif self.metadata["dtype"] == "float16":
                    pattern = pattern.astype("float32")
                
                # Apply log scaling
                log_pattern = np.log(pattern + 1e-6)
                
                # Use Welford's online algorithm for numerical stability and memory efficiency
                flat_pattern = log_pattern.flatten()
                for value in flat_pattern[::100]:  # Subsample pixels for speed (every 100th pixel)
                    total_count += 1
                    delta = value - running_mean
                    running_mean += delta / total_count
                    delta2 = value - running_mean
                    running_m2 += delta * delta2
                
            except Exception as e:
                print(f"Warning: Failed to process pattern {idx}: {e}")
                continue
        
        # Finalize statistics
        self.global_log_mean = float(running_mean)
        self.global_log_std = float(np.sqrt(running_m2 / (total_count - 1)) if total_count > 1 else 1.0)
        
        # Save for future use
        stats = {
            'log_mean': self.global_log_mean,
            'log_std': self.global_log_std,
            'n_samples_used': len(indices),
            'pixels_sampled': total_count,
            'computed_on': datetime.datetime.now().isoformat(),
            'method': 'streaming_welford'
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Global stats computed and saved: mean={self.global_log_mean:.4f}, std={self.global_log_std:.4f}")
        print(f"Used {len(indices)} patterns and {total_count} pixel samples")
    
    def __len__(self):
        return self.arr.shape[0]
    
    def __getitem__(self, idx):
        # OPTIMIZED: Fast path with pre-computed normalization
        if torch.is_tensor(idx):
            idx = idx.item()
        
        # Direct conversion with minimal copies
        pattern = np.array(self.arr[idx])
        x = torch.from_numpy(pattern).float()
        
        # Apply dequantization (cached from init)
        x = self._dequantize_fast(x)
        
        # Apply PRE-COMPUTED normalization (major speedup!)
        x = torch.log(x + 1e-6)
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
                 lambda_act: float = 1e-4, lambda_sim: float = 5e-5, lambda_div: float = 2e-4,
                 out_shape: tuple[int,int] = (256, 256)):
        super().__init__()
        self.save_hyperparameters()
        self.model = Autoencoder(latent_dim, out_shape)
        self.train_losses: list[float] = []
        self.validation_metrics: dict = {}
        self.realtime_metrics = realtime_metrics
        
        # Regularization parameters
        self.lambda_act = lambda_act
        self.lambda_sim = lambda_sim
        self.lambda_div = lambda_div

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, = batch
        z = self.model.embed(x)
        x_hat = self.model.decoder(z)
        
        # Compute regularized loss
        loss_dict = self.model.compute_loss(x, x_hat, z, self.lambda_act, self.lambda_sim, self.lambda_div)
        loss = loss_dict['total_loss']

        # Record for the post-run plot (use .item() to avoid GPU-CPU sync on every step)
        if batch_idx % 10 == 0:  # Only record every 10 steps to reduce overhead
            self.train_losses.append(loss.detach().cpu().item())

        # Log essential metrics only (remove expensive regularization term logging)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mse", loss_dict['mse_loss'], prog_bar=False)

        # REMOVED: Expensive real-time PSNR/SSIM calculations during training
        # These force CPU-GPU synchronization and are computationally expensive
        # Metrics are still calculated during validation and final evaluation
        
        # Optional: Calculate metrics much less frequently (every 100 steps)
        if self.realtime_metrics and batch_idx % 100 == 0:
            with torch.no_grad():
                metrics = calculate_metrics(x, x_hat)
                self.log("train_psnr", metrics['psnr'], prog_bar=False)  # Remove from progress bar
                self.log("train_ssim", metrics['ssim'], prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, = batch
        z = self.model.embed(x)
        x_hat = self.model.decoder(z)
        
        # Compute regularized loss
        loss_dict = self.model.compute_loss(x, x_hat, z, self.lambda_act, self.lambda_sim, self.lambda_div)
        loss = loss_dict['total_loss']
        
        # Calculate detailed metrics for validation
        metrics = calculate_metrics(x, x_hat)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse", loss_dict['mse_loss'], prog_bar=False)
        self.log("val_psnr", metrics['psnr'], prog_bar=True)
        self.log("val_ssim", metrics['ssim'], prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def load_dataset(data_path, logger):
    """Load dataset and detect input size."""
    if data_path.suffix == '.zarr' or data_path.is_dir():
        logger.info(f"Loading zarr dataset from: {data_path}")
        dataset = ZarrDataset(data_path)
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
    # Reduce workers to prevent log duplication - use fewer workers for cleaner output
    num_workers = min(16, args.num_workers)  # Reduce workers to minimize log spam
    
    dataloader_kwargs = {
        'batch_size': args.batch,
        'num_workers': num_workers,
        'pin_memory': args.pin_memory,
        'persistent_workers': args.persistent_workers and num_workers > 0,
        'prefetch_factor': 2,  # Reduce prefetch to minimize memory usage
        'drop_last': True,     # Ensure consistent batch sizes for mixed precision
    }
    
    train_dl = DataLoader(train_ds, shuffle=True, **dataloader_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **dataloader_kwargs) if val_ds else None
    
    print(f"Data loading config: batch_size={args.batch}, num_workers={num_workers}")
    return train_dl, val_dl

def create_model(args, detected_size):
    """Create the autoencoder model."""
    return LitAE(
        args.latent, args.lr, args.realtime_metrics,
        args.lambda_act, args.lambda_sim, args.lambda_div,
        (detected_size, detected_size)
    )

def generate_model_summary(model, train_dl, args):
    """Generate model summary."""
    sample = next(iter(train_dl))
    if isinstance(sample, (list, tuple)):
        sample = sample[0]
    example = sample[:1].to(args.device)
    model = model.to(args.device)
    show(model, example_input=example, output_dir=args.output_dir, include_evaluation=False)

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
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--summary", type=bool, default=True)
    p.add_argument("--realtime_metrics", action="store_true", help="Enable real-time metrics calculation during training (may slow down training)")
    p.add_argument("--lambda_act", type=float, default=1e-5, help="L1 regularization coefficient for sparsity")
    p.add_argument("--lambda_sim", type=float, default=0, help="Contrastive similarity regularization coefficient")
    p.add_argument("--lambda_div", type=float, default=0, help="Activation divergence regularization coefficient")
    p.add_argument("--input_size", type=int, default=256, help="Input image size (assumes square images)")
    p.add_argument("--precision", choices=["32", "16", "bf16"], default="16", help="Training precision (32=float32, 16=float16, bf16=bfloat16)")
    p.add_argument("--compile", action="store_true", help="Use torch.compile for faster training (PyTorch 2.0+)")
    p.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    p.add_argument("--pin_memory", action="store_true", default=True, help="Pin memory for faster GPU transfer")
    p.add_argument("--persistent_workers", action="store_true", help="Keep workers alive between epochs")
    p.add_argument("--profile", action="store_true", help="Enable PyTorch profiler for performance analysis")
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
    
    if args.summary:
        generate_model_summary(model, train_dl, args)

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