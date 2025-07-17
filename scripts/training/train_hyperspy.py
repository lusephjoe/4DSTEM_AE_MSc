"""Train the autoencoder using HyperSpy datasets for efficient memory usage."""
import argparse, torch, pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, DeviceStatsMonitor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.autoencoder import Autoencoder
from models.summary import show, calculate_metrics
from hyperspy_dataset import HyperSpyDataset, ChunkedHyperSpyDataset
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

class TensorDatasetWrapper:
    """Wrapper to make HyperSpy datasets compatible with PyTorch DataLoader multiprocessing."""
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return (self.dataset[idx],)  # Return as tuple to match TensorDataset

class LitAE(pl.LightningModule):
    def __init__(self, latent_dim: int, lr: float, realtime_metrics: bool = False, 
                 lambda_act: float = 1e-4, lambda_sim: float = 5e-5, lambda_div: float = 2e-4,
                 out_shape: tuple[int,int] = (256, 256), use_compile: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.model = Autoencoder(latent_dim, out_shape)
        
        # Compile model for faster training on modern PyTorch
        if use_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode='reduce-overhead')
        
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
        if batch_idx == 0:
            print(f"Starting training step {batch_idx}")
        
        x, = batch  # Match original format - unpack tuple
        
        if batch_idx == 0:
            print(f"Batch unpacked, shape: {x.shape}")
        
        # Use automatic mixed precision for faster training
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.device.type == 'cuda'):
            z = self.model.embed(x)
            x_hat = self.model.decoder(z)
            
            # Compute regularized loss
            loss_dict = self.model.compute_loss(x, x_hat, z, self.lambda_act, self.lambda_sim, self.lambda_div)
            loss = loss_dict['total_loss']

        # Record for the post-run plot (move to CPU asynchronously)
        self.train_losses.append(loss.detach().cpu().item())

        # Log loss components
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mse", loss_dict['mse_loss'], prog_bar=False)
        self.log("train_lp_reg", loss_dict['lp_reg'], prog_bar=False)
        self.log("train_contrastive_reg", loss_dict['contrastive_reg'], prog_bar=False)
        self.log("train_divergence_reg", loss_dict['divergence_reg'], prog_bar=False)

        # Calculate reconstruction metrics every N steps (if enabled)
        if self.realtime_metrics and batch_idx % 10 == 0:
            with torch.no_grad():
                metrics = calculate_metrics(x, x_hat)
                self.log("train_psnr", metrics['psnr'], prog_bar=True)
                self.log("train_ssim", metrics['ssim'], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, = batch  # Match original format - unpack tuple
        
        # Use automatic mixed precision for validation too
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.device.type == 'cuda'):
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
        # Use AdamW for better performance and weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            weight_decay=1e-4,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Add learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


def main():
    p = argparse.ArgumentParser(description="Train autoencoder with HyperSpy datasets")
    p.add_argument("--data", type=Path, required=True, help="Input .hspy file")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)  # Match original default
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--device", type=str, default="auto", help="Device to use: auto, cpu, cuda, mps")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--summary", action="store_true", default=True, help="Generate model summary")
    p.add_argument("--no_summary", action="store_true", help="Skip model summary generation")
    p.add_argument("--realtime_metrics", action="store_true", help="Enable real-time metrics")
    p.add_argument("--lambda_act", type=float, default=1e-5, help="L1 regularization coefficient")
    p.add_argument("--lambda_sim", type=float, default=0, help="Contrastive similarity regularization")
    p.add_argument("--lambda_div", type=float, default=0, help="Activation divergence regularization")
    
    # HyperSpy dataset options
    p.add_argument("--scan_step", type=int, default=1, help="Subsample scan positions")
    p.add_argument("--downsample", type=int, default=1, help="Downsample diffraction patterns")
    p.add_argument("--downsample_mode", choices=["bin", "stride", "gauss"], default="bin")
    p.add_argument("--sigma", type=float, default=0.8, help="Gaussian sigma for gauss mode")
    p.add_argument("--no_normalize", action="store_true", help="Skip normalization")
    p.add_argument("--chunk_size", type=int, default=64, help="Chunk size for chunked dataset")
    p.add_argument("--use_chunked", action="store_true", help="Use chunked dataset")
    
    # Training optimizations
    p.add_argument("--precision", choices=["32", "16", "bf16"], default="16")
    p.add_argument("--compile", action="store_true", help="Use torch.compile")
    p.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")  # Match original
    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--persistent_workers", action="store_true")
    
    args = p.parse_args()

    # Handle device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.set_float32_matmul_precision('medium')
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    
    print(f"Using device: {args.device}")
    if args.device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        if hasattr(torch.cuda, 'memory_stats'):
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    pl.seed_everything(args.seed)

    # Create HyperSpy dataset
    print("Creating HyperSpy dataset...")
    
    dataset_class = ChunkedHyperSpyDataset if args.use_chunked else HyperSpyDataset
    dataset_kwargs = {
        'file_path': args.data,
        'scan_step': args.scan_step,
        'downsample': args.downsample,
        'downsample_mode': args.downsample_mode,
        'sigma': args.sigma,
        'normalize': not args.no_normalize,
        'dtype': torch.float32
    }
    
    if args.use_chunked:
        dataset_kwargs['chunk_size'] = args.chunk_size
    
    dataset = dataset_class(**dataset_kwargs)
    
    # Split dataset exactly like original (using indices)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Create indices for splitting (same as original)
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets using indices
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Dataset split: {train_size} train, {val_size} validation")
    
    # Test dataset loading to catch issues early
    print("Testing dataset loading...")
    try:
        test_sample = dataset[0]
        print(f"✓ Dataset test successful, sample shape: {test_sample.shape}")
    except Exception as e:
        print(f"Warning: Dataset test failed: {e}")
        print("This may cause hanging during training. Consider using --no_summary flag.")
    
    # Create data loaders with wrapper to match original format
    train_wrapped = TensorDatasetWrapper(train_dataset)
    val_wrapped = TensorDatasetWrapper(val_dataset)
    
    # Optimize data loader settings for large datasets
    optimal_workers = min(args.num_workers, 4)  # Limit workers to prevent memory issues
    
    # Disable multiprocessing on Windows due to HyperSpy pickle issues
    import platform
    if platform.system() == "Windows":
        print("Windows detected: disabling multiprocessing due to HyperSpy compatibility issues")
        optimal_workers = 0
        prefetch_factor = 1
        # On Windows, also disable persistent workers and pin memory to avoid hanging
        args.persistent_workers = False
        args.pin_memory = False
        print("Additional Windows optimizations: disabled persistent workers and pin memory")
    elif total_size > 10000:  # For large datasets
        optimal_workers = min(optimal_workers, 2)
        prefetch_factor = 1
        print(f"Large dataset detected ({total_size} samples), using {optimal_workers} workers")
    else:
        prefetch_factor = 2
    
    # Create data loaders with optimized settings
    def create_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory, persistent_workers, prefetch_factor, drop_last=False):
        """Create DataLoader with proper settings."""
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            drop_last=drop_last
        )
    
    train_dl = create_dataloader(
        train_wrapped, 
        args.batch, 
        True, 
        optimal_workers,
        args.pin_memory and args.device == "cuda",
        args.persistent_workers,
        prefetch_factor,
        True
    )
    
    val_dl = create_dataloader(
        val_wrapped, 
        args.batch, 
        False, 
        optimal_workers,
        args.pin_memory and args.device == "cuda",
        args.persistent_workers,
        prefetch_factor
    )

    # Get sample shape
    sample_shape = dataset.get_shape()
    detected_size = sample_shape[1]  # Height (assuming square)
    
    model = LitAE(args.latent, args.lr, args.realtime_metrics, 
                  args.lambda_act, args.lambda_sim, args.lambda_div,
                  (detected_size, detected_size), args.compile)
    
    if args.summary and not args.no_summary:
        # Create sample for model summary with fallback for hanging issues
        try:
            print("Attempting to get first sample for model summary...")
            sample = next(iter(train_dl))
            if isinstance(sample, (list, tuple)):
                sample = sample[0]
            example = sample[:1].to(args.device)
            model = model.to(args.device)
            show(model, example_input=example, output_dir=args.output_dir, include_evaluation=False)
            
        except Exception as e:
            print(f"Warning: Could not create model summary due to data loading issue: {e}")
            print("Skipping model summary and proceeding with training...")
            # Get shape from dataset directly instead
            sample_shape = dataset.get_shape()
            example = torch.randn(1, *sample_shape).to(args.device)
            model = model.to(args.device)
            print(f"Using synthetic example with shape {example.shape} for model initialization")

    # Setup logging
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="tb_logs",
        default_hp_metric=False,
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename='ae-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            save_top_k=3,
            mode='min',
            save_last=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min',
            verbose=True
        )
    ]
    
    if args.device == "cuda":
        callbacks.append(DeviceStatsMonitor())

    # Configure trainer exactly like original (simpler version)
    if args.device == "cuda":
        accelerator = "gpu"
        devices = args.gpus
    elif args.device == "mps":
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1
    
    trainer_kwargs = {
        'max_epochs': args.epochs,
        'accelerator': accelerator,
        'devices': devices,
        'logger': tb_logger,
        'enable_progress_bar': True,
        'num_sanity_val_steps': 0,  # Disable sanity check that can hang with HyperSpy
    }
    
    # Add callbacks and precision only if they exist
    if 'callbacks' in locals():
        trainer_kwargs['callbacks'] = callbacks
    if hasattr(args, 'precision') and args.precision != "32":
        trainer_kwargs['precision'] = args.precision
    
    trainer = pl.Trainer(**trainer_kwargs)

    # Training
    try:
        print("Starting trainer.fit()...")
        print("Testing train DataLoader before training...")
        try:
            first_batch = next(iter(train_dl))
            print(f"✓ Train DataLoader working, first batch shape: {first_batch[0].shape}")
        except Exception as e:
            print(f"✗ Train DataLoader failed: {e}")
            raise
        
        print("Testing val DataLoader before training...")
        try:
            first_val_batch = next(iter(val_dl))
            print(f"✓ Val DataLoader working, first batch shape: {first_val_batch[0].shape}")
        except Exception as e:
            print(f"✗ Val DataLoader failed: {e}")
            raise
        
        print("Both DataLoaders working, starting training...")
        trainer.fit(model, train_dl, val_dl)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Clean up dataset
        dataset.close()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ---------- checkpoint ----------
    out_dir = args.output_dir
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / "ae.ckpt"
    trainer.save_checkpoint(ckpt)
    print(f"Model saved to {ckpt}")

    # ---------- loss curve ----------
    loss_curve_path = args.output_dir / "loss_curve.png"
    plt.figure()
    plt.plot(model.train_losses)
    plt.xlabel("batch")
    plt.ylabel("MSE loss")  # Match original label
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=300)
    print(f"Loss curve saved to {loss_curve_path}")
    
    # ---------- final evaluation ----------
    print("\n" + "="*80)
    print("FINAL MODEL EVALUATION")
    print("="*80)
    
    model.eval()
    # Ensure model is on the correct device
    model = model.to(args.device)
    
    with torch.no_grad():
        # Use mixed precision for evaluation too
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.device == 'cuda'):
            # Evaluate on a batch from validation set
            val_sample = next(iter(val_dl))
            if isinstance(val_sample, (list, tuple)):
                val_sample = val_sample[0]
            val_input = val_sample.to(args.device)
            val_output = model(val_input)
            
            final_metrics = calculate_metrics(val_input, val_output)
        
        print(f"Validation MSE:     {final_metrics['mse']:.6f} ± {final_metrics['mse_std']:.6f}")
        print(f"Validation PSNR:    {final_metrics['psnr']:.2f} ± {final_metrics['psnr_std']:.2f} dB")
        print(f"Validation SSIM:    {final_metrics['ssim']:.4f} ± {final_metrics['ssim_std']:.4f}")
        
        # Save final comparison images
        from models.summary import save_comparison_images
        final_comparison_path = args.output_dir / "final_reconstruction_comparison.png"
        save_comparison_images(val_input, val_output, final_comparison_path, num_samples=8)
        print(f"Final comparison saved to {final_comparison_path}")
        
        # Print CUDA memory usage stats
        if args.device == "cuda":
            print(f"\nCUDA Memory Usage:")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    print("="*80)
    
    # Final cleanup
    if args.device == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()