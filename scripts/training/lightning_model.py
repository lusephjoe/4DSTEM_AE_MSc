"""
PyTorch Lightning model wrapper for 4D-STEM autoencoder.

Extracted from the original train.py for better modularity and testability.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import torch
import pytorch_lightning as pl
import torch.nn.functional as F

# Add parent directory to path for model imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.autoencoder import Autoencoder
from models.summary import calculate_metrics, calculate_diffraction_metrics


class LitAE(pl.LightningModule):
    """PyTorch Lightning wrapper for the autoencoder model."""
    
    def __init__(self, latent_dim: int, lr: float, realtime_metrics: bool = False,
                 loss_config: Optional[Dict[str, Any]] = None, 
                 out_shape: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model with loss configuration
        self.model = Autoencoder(latent_dim, out_shape, loss_config)
        self.train_losses: list[float] = []
        self.validation_metrics: Dict[str, Any] = {}
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
        self.register_buffer('global_log_mean', torch.tensor(0.0))
        self.register_buffer('global_log_std', torch.tensor(1.0))
        self.use_normalization = True  # Will be set based on dataset
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def denormalize_to_log_space(self, x_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized tensor back to log space for scale-aligned loss computation."""
        return x_normalized * (self.global_log_std + 1e-8) + self.global_log_mean
    
    def set_normalization_params(self, log_mean: float, log_std: float, use_normalization: bool = True):
        """Set normalization parameters from data loader."""
        self.global_log_mean.data = torch.tensor(log_mean)
        self.global_log_std.data = torch.tensor(log_std)
        self.use_normalization = use_normalization
    
    def training_step(self, batch, batch_idx):
        x, = batch
        z = self.model.embed(x)
        x_hat = self.model.decoder(z)
        
        # Scale-aligned loss: denormalize both input and output to log space if normalization used
        if self.use_normalization:
            x_log = self.denormalize_to_log_space(x)
            x_hat_log = self.denormalize_to_log_space(x_hat)
        else:
            x_log = x
            x_hat_log = x_hat
        
        # Compute loss using flexible loss system in aligned log space
        loss_dict = self.model.compute_loss(x_log, x_hat_log, z)
        loss = loss_dict['total_loss']
        
        # Comprehensive MSE reporting
        if self.use_normalization:
            mse_input_space = torch.mean((x - x_hat)**2)
            mae_input_space = torch.mean(torch.abs(x - x_hat))
            max_error = torch.max(torch.abs(x - x_hat))
        else:
            mse_input_space = torch.mean((x - x_hat)**2)
            mae_input_space = torch.mean(torch.abs(x - x_hat))
            max_error = torch.max(torch.abs(x - x_hat))
            
            # Relative error as percentage of data range
            data_range = torch.max(x) - torch.min(x)
            relative_rmse_pct = 100 * torch.sqrt(mse_input_space) / (data_range + 1e-8)
        
        # Record losses less frequently to reduce CPU-GPU sync
        if batch_idx % 50 == 0:
            self.train_losses.append(loss.detach().cpu().item())
        
        # Log essential metrics with proper context
        self.log("train_loss", loss, prog_bar=True)
        if self.use_normalization:
            self.log("train_mse_normalized", mse_input_space, prog_bar=True)
            self.log("train_mae_normalized", mae_input_space, prog_bar=False)
            self.log("train_max_error_normalized", max_error, prog_bar=False)
        else:
            self.log("train_mse_log", mse_input_space, prog_bar=True)
            self.log("train_mae_log", mae_input_space, prog_bar=False)
            self.log("train_max_error_log", max_error, prog_bar=False)
            self.log("train_rmse_pct", relative_rmse_pct, prog_bar=True)
        
        # Log main reconstruction loss dynamically
        recon_loss_key = f"{self.model.loss_manager.reconstruction_loss.name}_loss"
        if recon_loss_key in loss_dict:
            self.log(f"train_{self.model.loss_manager.reconstruction_loss.name}",
                    loss_dict[recon_loss_key], prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, = batch
        z = self.model.embed(x)
        x_hat = self.model.decoder(z)
        
        # Scale-aligned loss: denormalize both input and output to log space if normalization used
        if self.use_normalization:
            x_log = self.denormalize_to_log_space(x)
            x_hat_log = self.denormalize_to_log_space(x_hat)
        else:
            x_log = x
            x_hat_log = x_hat
        
        # Compute loss using flexible loss system in aligned log space
        loss_dict = self.model.compute_loss(x_log, x_hat_log, z)
        loss = loss_dict['total_loss']
        
        # Comprehensive MSE reporting
        if self.use_normalization:
            mse_input_space = torch.mean((x - x_hat)**2)
            mae_input_space = torch.mean(torch.abs(x - x_hat))
            max_error = torch.max(torch.abs(x - x_hat))
        else:
            mse_input_space = torch.mean((x - x_hat)**2)
            mae_input_space = torch.mean(torch.abs(x - x_hat))
            max_error = torch.max(torch.abs(x - x_hat))
            
            # Relative error as percentage of data range
            data_range = torch.max(x) - torch.min(x)
            relative_rmse_pct = 100 * torch.sqrt(mse_input_space) / (data_range + 1e-8)
        
        # Calculate detailed metrics for validation (use denormalized data)
        metrics = calculate_metrics(x_log, x_hat_log)
        diffraction_metrics = calculate_diffraction_metrics(x_log, x_hat_log)
        
        # Log comprehensive validation metrics
        self.log("val_loss", loss, prog_bar=True)
        if self.use_normalization:
            self.log("val_mse_normalized", mse_input_space, prog_bar=True)
            self.log("val_mae_normalized", mae_input_space, prog_bar=False)
            self.log("val_max_error_normalized", max_error, prog_bar=False)
        else:
            self.log("val_mse_log", mse_input_space, prog_bar=True)
            self.log("val_mae_log", mae_input_space, prog_bar=False)
            self.log("val_max_error_log", max_error, prog_bar=False)
            self.log("val_rmse_pct", relative_rmse_pct, prog_bar=True)
        
        # Log main reconstruction loss dynamically
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
            **diffraction_metrics
        }
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        
        # Check if scheduler is disabled
        if getattr(self.hparams, 'no_scheduler', False):
            return optimizer
        
        # Simple CyclicLR scheduler matching reference implementation
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.hparams.lr,              # Current learning rate as base
            max_lr=self.hparams.lr * 3.33,       # Reference uses 1e-4 max with 3e-5 base
            step_size_up=15,                     # Exactly as in reference
            cycle_momentum=False                 # Exactly as in reference
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Step every batch (matches reference)
            }
        }