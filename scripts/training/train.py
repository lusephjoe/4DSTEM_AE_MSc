#!/usr/bin/env python3
"""
Refactored 4D-STEM Autoencoder Training Script

A clean, extensible training pipeline for 4D-STEM autoencoder models with 
object-oriented design and comprehensive configuration management.
"""

import argparse
import logging
import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Add parent directory to path for model imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.autoencoder import Autoencoder
from models.losses import create_loss_config_from_args, get_available_losses


@dataclass
class TrainingConfig:
    """Configuration container for training parameters."""
    
    # Data parameters
    data_path: Path
    output_dir: Path
    use_normalization: bool = True
    no_validation: bool = False
    
    # Model parameters
    latent_dim: int = 128
    input_size: int = 256
    
    # Training parameters
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    no_scheduler: bool = False
    precision: str = "16"
    accumulate_grad_batches: int = 1
    
    # Loss function parameters
    loss_function: str = "mse"
    lambda_act: float = 1e-5
    lambda_sim: float = 0.0
    lambda_div: float = 0.0
    lambda_l2: float = 0.0
    lambda_kl: float = 0.0
    
    # Hardware parameters
    device: str = "auto"
    gpus: int = 1
    compile: bool = False
    
    # Data loading parameters
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    
    # Checkpointing parameters
    resume_from_checkpoint: Optional[Path] = None
    save_every_n_epochs: int = 1
    
    # Debugging parameters
    seed: int = 42
    realtime_metrics: bool = False
    profile: bool = False
    debug: bool = False
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        self.data_path = Path(self.data_path)
        self.output_dir = Path(self.output_dir)
        
        if self.resume_from_checkpoint:
            self.resume_from_checkpoint = Path(self.resume_from_checkpoint)
        
        # Auto-detect device if needed
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


class DatasetManager:
    """Manages dataset loading and creation of data loaders."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger('training.data')
        
    def load_dataset(self):
        """Load and prepare the dataset."""
        from .dataset import HDF5Dataset
        
        if self.config.data_path.suffix == '.h5':
            self.logger.info(f"Loading HDF5 dataset from: {self.config.data_path}")
            dataset = HDF5Dataset(
                self.config.data_path, 
                use_normalization=self.config.use_normalization
            )
            sample = dataset[0][0] if isinstance(dataset[0], tuple) else dataset[0]
            detected_size = sample.shape[-1]
        else:
            self.logger.info(f"Loading tensor dataset from: {self.config.data_path}")
            data = torch.load(self.config.data_path)
            from torch.utils.data import TensorDataset
            dataset = TensorDataset(data)
            detected_size = data.shape[-1] if len(data.shape) == 4 else 256
        
        self.logger.info(f"Detected input size: {detected_size}x{detected_size}")
        return dataset, detected_size
    
    def create_train_val_split(self, full_dataset):
        """Create train/validation split."""
        if self.config.no_validation:
            self.logger.info(f"Using entire dataset for training: {len(full_dataset)} patterns")
            return full_dataset, None
        
        # 80/20 train/validation split
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        indices = torch.randperm(total_size)
        
        from torch.utils.data import TensorDataset, Subset
        
        if isinstance(full_dataset, TensorDataset):
            # Split tensor data directly
            data = full_dataset.tensors[0]
            train_data = data[indices[:train_size]]
            val_data = data[indices[train_size:]]
            train_ds, val_ds = TensorDataset(train_data), TensorDataset(val_data)
        else:
            # Use Subset for other datasets
            train_ds = Subset(full_dataset, indices[:train_size])
            val_ds = Subset(full_dataset, indices[train_size:])
        
        self.logger.info(f"Created train/val split: {len(train_ds)} train, {len(val_ds)} validation")
        return train_ds, val_ds
    
    def create_data_loaders(self, train_ds, val_ds):
        """Create optimized data loaders."""
        from torch.utils.data import DataLoader
        import platform
        import multiprocessing as mp
        
        # Configure multiprocessing
        num_workers = self.config.num_workers
        dataloader_kwargs = {
            'batch_size': self.config.batch_size,
            'num_workers': num_workers,
            'pin_memory': self.config.pin_memory and torch.cuda.is_available(),
            'persistent_workers': self.config.persistent_workers and num_workers > 0,
            'drop_last': True,
        }
        
        if num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = 4
            dataloader_kwargs['multiprocessing_context'] = mp.get_context('spawn')
        
        try:
            train_dl = DataLoader(train_ds, shuffle=True, **dataloader_kwargs)
            val_dl = DataLoader(val_ds, shuffle=False, **dataloader_kwargs) if val_ds else None
            
            # Test multiprocessing if enabled
            if num_workers > 0 and self.config.debug:
                self.logger.info("Testing multiprocessing data loader...")
                test_batch = next(iter(train_dl))
                self.logger.info(f"✓ Multiprocessing test successful: batch shape {test_batch[0].shape}")
                
        except Exception as e:
            self.logger.warning(f"Multiprocessing data loading failed: {e}")
            self.logger.info("Falling back to single-threaded data loading...")
            
            # Fallback to single-threaded
            fallback_kwargs = dataloader_kwargs.copy()
            fallback_kwargs['num_workers'] = 0
            fallback_kwargs.pop('multiprocessing_context', None)
            fallback_kwargs.pop('prefetch_factor', None)
            fallback_kwargs['persistent_workers'] = False
            
            train_dl = DataLoader(train_ds, shuffle=True, **fallback_kwargs)
            val_dl = DataLoader(val_ds, shuffle=False, **fallback_kwargs) if val_ds else None
        
        self.logger.info(f"Data loading config: batch_size={self.config.batch_size}, num_workers={num_workers}")
        return train_dl, val_dl


class ModelManager:
    """Manages model creation and configuration."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger('training.model')
    
    def create_model(self, detected_size: int):
        """Create the autoencoder model with loss configuration."""
        from .lightning_model import LitAE
        
        # Create loss configuration
        loss_config = create_loss_config_from_args(self.config)
        
        model = LitAE(
            latent_dim=self.config.latent_dim,
            lr=self.config.learning_rate,
            realtime_metrics=self.config.realtime_metrics,
            loss_config=loss_config,
            out_shape=(detected_size, detected_size)
        )
        
        # Apply torch.compile if requested
        if self.config.compile:
            model = self._apply_torch_compile(model)
        
        return model
    
    def _apply_torch_compile(self, model):
        """Apply torch.compile optimization if supported."""
        import platform
        
        if platform.system() == 'Linux':
            try:
                self.logger.info("Compiling model with torch.compile...")
                model.model = torch.compile(
                    model.model,
                    mode='max-autotune',
                    fullgraph=True,
                    dynamic=False
                )
                self.logger.info("✅ Model compilation successful!")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
        else:
            self.logger.warning(f"torch.compile only enabled on Linux (current OS: {platform.system()})")
        
        return model


class TrainerManager:
    """Manages PyTorch Lightning trainer setup and configuration."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger('training.trainer')
    
    def setup_trainer(self, base_name: str):
        """Setup PyTorch Lightning trainer with callbacks."""
        # Setup logger
        tb_logger = TensorBoardLogger(
            save_dir=self.config.output_dir,
            name="tb_logs",
            default_hp_metric=False
        )
        
        # Setup checkpoint callback
        checkpoint_callback = self._create_checkpoint_callback(base_name)
        
        # Configure accelerator
        accelerator, devices = self._configure_accelerator()
        
        # Map precision
        precision_map = {
            "32": "32-true",
            "16": "16-mixed", 
            "bf16": "bf16-mixed"
        }
        precision = precision_map.get(self.config.precision, "16-mixed")
        
        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator=accelerator,
            devices=devices,
            logger=tb_logger,
            enable_progress_bar=True,
            callbacks=[checkpoint_callback],
            precision=precision,
            gradient_clip_val=1.0,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            log_every_n_steps=50,
            enable_model_summary=False
        )
        
        return trainer
    
    def _create_checkpoint_callback(self, base_name: str):
        """Create checkpoint callback for saving models."""
        checkpoint_dir = self.config.output_dir / "checkpoints"
        
        if self.config.no_validation:
            filename_pattern = f"{base_name}_epoch{{epoch:03d}}_trainloss{{train_loss:.4f}}"
            monitor_metric = "train_loss"
        else:
            filename_pattern = f"{base_name}_epoch{{epoch:03d}}_valloss{{val_loss:.4f}}"
            monitor_metric = "val_loss"
        
        return ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=filename_pattern,
            save_top_k=-1,
            every_n_epochs=self.config.save_every_n_epochs,
            save_on_train_epoch_end=self.config.no_validation,
            monitor=monitor_metric,
            mode="min"
        )
    
    def _configure_accelerator(self):
        """Configure accelerator settings."""
        if self.config.device == "cuda":
            return "gpu", self.config.gpus
        elif self.config.device == "mps":
            return "mps", 1
        else:
            return "cpu", 1


class TrainingPipeline:
    """Main training pipeline orchestrator."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.start_time = datetime.datetime.now()
        self.logger = self._setup_logging()
        
        # Initialize managers
        self.data_manager = DatasetManager(config)
        self.model_manager = ModelManager(config)
        self.trainer_manager = TrainerManager(config)
        
        # Set random seed
        pl.seed_everything(config.seed)
        
        # Configure device-specific settings
        if config.device == "cuda":
            torch.set_float32_matmul_precision('medium')
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        log_file = self.config.output_dir / f"training_log_{timestamp}.txt"
        
        logger = logging.getLogger('training')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Log session header
        logger.info("=" * 60)
        logger.info("4D-STEM AUTOENCODER TRAINING SESSION")
        logger.info("=" * 60)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Training started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Using device: {self.config.device}")
        logger.info("=" * 60)
        
        return logger
    
    def run(self) -> None:
        """Execute the complete training pipeline."""
        try:
            # Load data
            full_dataset, detected_size = self.data_manager.load_dataset()
            train_ds, val_ds = self.data_manager.create_train_val_split(full_dataset)
            train_dl, val_dl = self.data_manager.create_data_loaders(train_ds, val_ds)
            
            # Create model
            model = self.model_manager.create_model(detected_size)
            
            # Set normalization parameters if using HDF5 dataset
            if hasattr(train_ds, 'global_log_mean'):
                model.set_normalization_params(
                    train_ds.global_log_mean, 
                    train_ds.global_log_std, 
                    train_ds.use_normalization
                )
                self.logger.info(f"Normalization: mean={train_ds.global_log_mean:.4f}, std={train_ds.global_log_std:.4f}")
            
            # Log configuration
            self._log_configuration(model)
            
            # Setup trainer
            base_name = f"ae_e{self.config.epochs:03d}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
            trainer = self.trainer_manager.setup_trainer(base_name)
            
            # Train model
            self._train_model(trainer, model, train_dl, val_dl)
            
            # Save results and evaluate
            self._save_results(trainer, model, base_name)
            self._final_evaluation(model, train_dl, val_dl, base_name)
            
            # Finalize
            self._finalize_training()
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def _log_configuration(self, model):
        """Log training configuration details."""
        loss_info = model.model.get_loss_info()
        self.logger.info("Training Configuration:")
        self.logger.info(f"  Epochs: {self.config.epochs}")
        self.logger.info(f"  Batch size: {self.config.batch_size}")
        self.logger.info(f"  Learning rate: {self.config.learning_rate}")
        self.logger.info(f"  Latent dimension: {self.config.latent_dim}")
        self.logger.info(f"  Reconstruction loss: {loss_info['reconstruction']}")
        
        if self.config.no_scheduler:
            self.logger.info("  Learning rate scheduler: DISABLED")
        else:
            self.logger.info(f"  Learning rate scheduler: CyclicLR")
    
    def _train_model(self, trainer, model, train_dl, val_dl):
        """Execute model training."""
        resume_checkpoint = None
        if self.config.resume_from_checkpoint:
            if self.config.resume_from_checkpoint.exists():
                resume_checkpoint = str(self.config.resume_from_checkpoint)
                self.logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
            else:
                self.logger.error(f"Checkpoint not found: {self.config.resume_from_checkpoint}")
                return
        
        self.logger.info("Starting model training...")
        
        if val_dl is not None:
            trainer.fit(model, train_dl, val_dl, ckpt_path=resume_checkpoint)
        else:
            trainer.fit(model, train_dl, ckpt_path=resume_checkpoint)
        
        training_end_time = datetime.datetime.now()
        training_duration = training_end_time - self.start_time
        self.logger.info(f"Training completed at: {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Training duration: {training_duration}")
    
    def _save_results(self, trainer, model, base_name: str):
        """Save model checkpoint and training artifacts."""
        # Save final checkpoint
        current_epoch = trainer.current_epoch + 1
        final_mse = model.train_losses[-1] if model.train_losses else 0.0
        final_base_name = f"{base_name}_epoch{current_epoch:03d}_mse{final_mse:.4f}"
        
        final_ckpt_path = self.config.output_dir / f"{final_base_name}_final.ckpt"
        trainer.save_checkpoint(final_ckpt_path)
        self.logger.info(f"Final model saved to {final_ckpt_path}")
        
        # Save loss curve
        self._save_loss_curve(model, final_base_name, final_mse)
    
    def _save_loss_curve(self, model, base_name: str, final_mse: float):
        """Save training loss curve visualization."""
        import matplotlib.pyplot as plt
        
        loss_curve_path = self.config.output_dir / f"{base_name}_loss.png"
        plt.figure()
        plt.plot(model.train_losses)
        plt.xlabel("Batch")
        plt.ylabel("MSE Loss")
        plt.yscale("log")
        plt.title(f"Training Loss (Final MSE: {final_mse:.4f})")
        plt.tight_layout()
        plt.savefig(loss_curve_path, dpi=300)
        plt.close()
        self.logger.info(f"Loss curve saved to {loss_curve_path}")
    
    def _final_evaluation(self, model, train_dl, val_dl, base_name: str):
        """Perform final model evaluation."""
        self.logger.info("=" * 60)
        self.logger.info("FINAL MODEL EVALUATION")
        self.logger.info("=" * 60)
        
        model.eval()
        model = model.to(self.config.device)
        
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
            
            eval_input = eval_sample.to(self.config.device)
            eval_output = model(eval_input)
            
            # Calculate metrics
            from models.summary import calculate_metrics, save_comparison_images
            final_metrics = calculate_metrics(eval_input, eval_output)
            
            self.logger.info(f"{eval_name} MSE:  {final_metrics['mse']:.6f} ± {final_metrics['mse_std']:.6f}")
            self.logger.info(f"{eval_name} PSNR: {final_metrics['psnr']:.2f} ± {final_metrics['psnr_std']:.2f} dB")
            self.logger.info(f"{eval_name} SSIM: {final_metrics['ssim']:.4f} ± {final_metrics['ssim_std']:.4f}")
            
            # Save comparison images
            final_comparison_path = self.config.output_dir / f"{base_name}_reconstruction.png"
            save_comparison_images(eval_input, eval_output, final_comparison_path, num_samples=8)
            self.logger.info(f"Final comparison saved to {final_comparison_path}")
    
    def _finalize_training(self):
        """Log final summary and cleanup."""
        end_time = datetime.datetime.now()
        total_duration = end_time - self.start_time
        
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total duration: {total_duration}")
        self.logger.info("=" * 60)
        
        # Cleanup
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
        
        for handler in self.logger.handlers:
            handler.flush()


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train 4D-STEM autoencoder with clean, extensible pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data parameters
    parser.add_argument("--data", type=Path, required=True,
                       help="Path to training data (.h5 or .pt file)")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for results and checkpoints")
    parser.add_argument("--no_normalization", action="store_true",
                       help="Skip z-score normalization, train directly on log data")
    parser.add_argument("--no_validation", action="store_true",
                       help="Use entire dataset for training (no validation split)")
    
    # Model parameters
    parser.add_argument("--latent", type=int, default=128,
                       help="Latent dimension size")
    parser.add_argument("--input_size", type=int, default=256,
                       help="Input image size (assumes square)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--no_scheduler", action="store_true",
                       help="Disable learning rate scheduler")
    parser.add_argument("--precision", choices=["32", "16", "bf16"], default="16",
                       help="Training precision")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                       help="Gradient accumulation batches")
    
    # Loss function parameters
    available_losses = get_available_losses()
    parser.add_argument("--loss_function", type=str, default="mse",
                       choices=available_losses['reconstruction'],
                       help="Reconstruction loss function")
    parser.add_argument("--lambda_act", type=float, default=1e-5,
                       help="Lp regularization coefficient")
    parser.add_argument("--lambda_sim", type=float, default=0,
                       help="Contrastive similarity coefficient")
    parser.add_argument("--lambda_div", type=float, default=0,
                       help="Activation divergence coefficient")
    parser.add_argument("--lambda_l2", type=float, default=0,
                       help="L2 regularization coefficient")
    parser.add_argument("--lambda_kl", type=float, default=0,
                       help="KL divergence coefficient")
    
    # Hardware parameters
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use for training")
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs to use")
    parser.add_argument("--compile", action="store_true",
                       help="Use torch.compile for optimization")
    
    # Data loading parameters
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--pin_memory", action="store_true", default=True,
                       help="Pin memory for faster GPU transfer")
    parser.add_argument("--persistent_workers", action="store_true",
                       help="Keep workers alive between epochs")
    
    # Checkpointing parameters
    parser.add_argument("--resume_from_checkpoint", type=Path,
                       help="Path to checkpoint file to resume from")
    parser.add_argument("--save_every_n_epochs", type=int, default=1,
                       help="Save checkpoint every N epochs")
    
    # Debugging parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--realtime_metrics", action="store_true",
                       help="Enable real-time metrics calculation")
    parser.add_argument("--profile", action="store_true",
                       help="Enable PyTorch profiler")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Convert args to config
    config = TrainingConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        use_normalization=not args.no_normalization,
        no_validation=args.no_validation,
        latent_dim=args.latent,
        input_size=args.input_size,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        no_scheduler=args.no_scheduler,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        loss_function=args.loss_function,
        lambda_act=args.lambda_act,
        lambda_sim=args.lambda_sim,
        lambda_div=args.lambda_div,
        lambda_l2=args.lambda_l2,
        lambda_kl=args.lambda_kl,
        device=args.device,
        gpus=args.gpus,
        compile=args.compile,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_every_n_epochs=args.save_every_n_epochs,
        seed=args.seed,
        realtime_metrics=args.realtime_metrics,
        profile=args.profile,
        debug=args.debug
    )
    
    # Run training pipeline
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()