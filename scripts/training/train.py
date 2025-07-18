"""Train the autoencoder using PyTorch Lightning."""
import argparse, torch, pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.autoencoder import Autoencoder
from models.summary import show, calculate_metrics
import zarr
import json

class ZarrDataset(Dataset):
    """Dataset for lazy loading of zarr-compressed 4D-STEM data."""
    
    def __init__(self, zarr_path, metadata_path=None):
        self.zarr_path = Path(zarr_path)
        self.arr = zarr.open(str(zarr_path), mode="r")
        
        # Load metadata if available
        if metadata_path is None:
            metadata_path = self.zarr_path.parent / f"{self.zarr_path.stem}_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Default metadata if not found
            self.metadata = {
                "data_min": 0.0,
                "data_max": 1.0,
                "data_range": 1.0,
                "dtype": "uint16"
            }
            print(f"Warning: No metadata found at {metadata_path}, using defaults")
        
        self.data_min = self.metadata["data_min"]
        self.data_range = self.metadata["data_range"]
        self.dtype = self.metadata["dtype"]
        
        print(f"Loaded zarr dataset: {self.arr.shape} patterns, dtype: {self.dtype}")
        print(f"Data range: {self.data_min:.3f} to {self.data_min + self.data_range:.3f}")
    
    def __len__(self):
        return self.arr.shape[0]
    
    def __getitem__(self, idx):
        # Convert PyTorch tensor indices to numpy/python integers
        if torch.is_tensor(idx):
            idx = idx.item()
        
        # Load pattern from zarr
        pattern = self.arr[idx]
        
        # Convert to numpy array first, then to tensor
        if not isinstance(pattern, np.ndarray):
            pattern = np.array(pattern)
        
        # Convert to tensor
        x = torch.from_numpy(pattern).float()
        
        # Dequantize if needed
        if self.dtype == "uint16":
            # Convert uint16 back to original range
            x = x / 65535.0 * self.data_range + self.data_min
        elif self.dtype == "float16":
            # Already normalized, just convert to float32
            x = x.float()
        # float32 is already ready to use
        
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        return (x,)  # Return as tuple to match TensorDataset format

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

        # Record for the post-run plot
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

    args = p.parse_args()

    # Handle device selection more robustly
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
            torch.set_float32_matmul_precision('medium')
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    
    print(f"Using device: {args.device}")
    
    pl.seed_everything(args.seed)

    # Detect file format and load data accordingly
    data_path = Path(args.data)
    
    if data_path.suffix == '.zarr' or data_path.is_dir():
        # Handle zarr format
        print(f"Loading zarr dataset from: {data_path}")
        full_dataset = ZarrDataset(data_path)
        
        # Get sample to detect input size
        sample = full_dataset[0]
        detected_size = sample.shape[-1]  # Assume square images
        print(f"Detected input size: {detected_size}x{detected_size}")
        
        # Split dataset indices for train/validation (80/20 split)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        # Create indices for splitting
        indices = torch.randperm(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subset datasets
        from torch.utils.data import Subset
        train_ds = Subset(full_dataset, train_indices)
        val_ds = Subset(full_dataset, val_indices)
        
        print(f"Created train/val split: {len(train_ds)} train, {len(val_ds)} validation")
        
    else:
        # Handle traditional .pt format
        print(f"Loading tensor dataset from: {data_path}")
        data = torch.load(args.data)
        
        # Detect input size from data
        if len(data.shape) == 4:  # (N, C, H, W)
            detected_size = data.shape[-1]  # Assume square images
        else:
            detected_size = args.input_size
        
        print(f"Detected input size: {detected_size}x{detected_size}")
        
        # Split data into train/validation (80/20 split)
        total_size = len(data)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        # Create indices for splitting
        indices = torch.randperm(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_data = data[train_indices]
        val_data = data[val_indices]
        
        train_ds = TensorDataset(train_data)
        val_ds = TensorDataset(val_data)
    
    # Create data loaders
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, 
                         num_workers=args.num_workers, pin_memory=args.pin_memory,
                         persistent_workers=args.persistent_workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, 
                       num_workers=args.num_workers, pin_memory=args.pin_memory,
                       persistent_workers=args.persistent_workers)

    model = LitAE(args.latent, args.lr, args.realtime_metrics, 
                  args.lambda_act, args.lambda_sim, args.lambda_div,
                  (detected_size, detected_size))
    
    if args.summary:          
                                    # grab one batch to build an example tensor
        sample = next(iter(train_dl))
        if isinstance(sample, (list, tuple)):
            sample = sample[0]    
        example = sample[:1].to(args.device)
        # Ensure model is on the correct device for summary
        model = model.to(args.device)
        show(model, example_input=example, output_dir=args.output_dir, include_evaluation=False)

        # ---------- logging ----------
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="tb_logs",
        default_hp_metric=False,
    )

    # Configure trainer accelerator based on device
    if args.device == "cuda":
        accelerator = "gpu"
        devices = args.gpus
    elif args.device == "mps":
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        logger=tb_logger,                 
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dl, val_dl)

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
    plt.ylabel("MSE loss")
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