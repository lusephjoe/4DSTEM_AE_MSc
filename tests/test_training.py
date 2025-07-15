"""Test suite for training functionality."""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from scripts.train import LitAE
from models.autoencoder import Autoencoder
import pytorch_lightning as pl


class TestTrainingFunctionality:
    """Test training-related functionality."""
    
    def test_lit_ae_initialization(self):
        """Test LitAE initialization."""
        model = LitAE(
            latent_dim=32, 
            lr=1e-3, 
            realtime_metrics=False,
            lambda_act=1e-4,
            lambda_sim=5e-5,
            lambda_div=2e-4,
            out_shape=(64, 64)
        )
        
        assert model.lambda_act == 1e-4
        assert model.lambda_sim == 5e-5
        assert model.lambda_div == 2e-4
        assert isinstance(model.model, Autoencoder)

    def test_lit_ae_training_step(self, sample_diffraction_data):
        """Test LitAE training step."""
        model = LitAE(
            latent_dim=32, 
            lr=1e-3, 
            realtime_metrics=False,
            out_shape=(64, 64)
        )
        
        batch = (sample_diffraction_data,)
        
        # Test training step
        loss = model.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_lit_ae_validation_step(self, sample_diffraction_data):
        """Test LitAE validation step."""
        model = LitAE(
            latent_dim=32, 
            lr=1e-3, 
            realtime_metrics=False,
            out_shape=(64, 64)
        )
        
        batch = (sample_diffraction_data,)
        
        # Test validation step
        loss = model.validation_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0, "Validation loss should be non-negative"
        assert torch.isfinite(loss), "Validation loss should be finite"

    def test_lit_ae_configure_optimizers(self):
        """Test optimizer configuration."""
        model = LitAE(
            latent_dim=32, 
            lr=1e-3, 
            realtime_metrics=False,
            out_shape=(64, 64)
        )
        
        optimizer = model.configure_optimizers()
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == 1e-3

    def test_realtime_metrics_toggle(self, sample_diffraction_data):
        """Test realtime metrics toggle functionality."""
        # Test with realtime metrics disabled
        model_no_metrics = LitAE(
            latent_dim=32, 
            lr=1e-3, 
            realtime_metrics=False,
            out_shape=(64, 64)
        )
        
        # Test with realtime metrics enabled
        model_with_metrics = LitAE(
            latent_dim=32, 
            lr=1e-3, 
            realtime_metrics=True,
            out_shape=(64, 64)
        )
        
        batch = (sample_diffraction_data,)
        
        # Both should work without error
        loss1 = model_no_metrics.training_step(batch, batch_idx=0)
        loss2 = model_with_metrics.training_step(batch, batch_idx=0)
        
        assert torch.isfinite(loss1) and torch.isfinite(loss2)

    def test_different_regularization_coefficients(self, sample_diffraction_data):
        """Test different regularization coefficient settings."""
        coefficients = [
            (1e-4, 5e-5, 2e-4),  # Default
            (1e-3, 1e-4, 1e-3),  # Higher regularization
            (1e-6, 1e-7, 1e-6),  # Lower regularization
        ]
        
        for lambda_act, lambda_sim, lambda_div in coefficients:
            model = LitAE(
                latent_dim=32, 
                lr=1e-3, 
                realtime_metrics=False,
                lambda_act=lambda_act,
                lambda_sim=lambda_sim,
                lambda_div=lambda_div,
                out_shape=(64, 64)
            )
            
            batch = (sample_diffraction_data,)
            loss = model.training_step(batch, batch_idx=0)
            
            assert torch.isfinite(loss), \
                f"Loss should be finite for coefficients {lambda_act}, {lambda_sim}, {lambda_div}"

    @pytest.mark.slow
    def test_training_with_different_input_sizes(self, input_size):
        """Test training with different input sizes."""
        # Generate data for specific input size
        torch.manual_seed(42)
        data = torch.randn(8, 1, input_size, input_size)
        
        model = LitAE(
            latent_dim=32, 
            lr=1e-3, 
            realtime_metrics=False,
            out_shape=(input_size, input_size)
        )
        
        batch = (data,)
        
        # Should handle different input sizes without error
        loss = model.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss), f"Training failed for input size {input_size}x{input_size}"

    def test_loss_components_logging(self, sample_diffraction_data):
        """Test that all loss components are properly logged."""
        model = LitAE(
            latent_dim=32, 
            lr=1e-3, 
            realtime_metrics=False,
            out_shape=(64, 64)
        )
        
        batch = (sample_diffraction_data,)
        
        # Mock the log method to capture logged values
        logged_values = {}
        
        def mock_log(key, value, **kwargs):
            logged_values[key] = value.item() if isinstance(value, torch.Tensor) else value
        
        model.log = mock_log
        
        # Run training step
        model.training_step(batch, batch_idx=0)
        
        # Check that all expected loss components are logged
        expected_keys = [
            'train_loss', 'train_mse', 'train_l1_reg', 
            'train_contrastive_reg', 'train_divergence_reg'
        ]
        
        for key in expected_keys:
            assert key in logged_values, f"Missing logged value: {key}"
            assert isinstance(logged_values[key], (int, float)), \
                f"Logged value {key} should be a number"

    def test_model_device_consistency(self, device):
        """Test that model handles device placement correctly."""
        model = LitAE(
            latent_dim=32, 
            lr=1e-3, 
            realtime_metrics=False,
            out_shape=(64, 64)
        )
        
        # Move model to device
        model = model.to(device)
        
        # Create data on same device
        data = torch.randn(4, 1, 64, 64, device=device)
        batch = (data,)
        
        # Should work without device mismatch errors
        loss = model.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss), "Training failed with device placement"


class TestTrainingIntegration:
    """Integration tests for training functionality."""
    
    @pytest.mark.integration
    def test_mini_training_loop(self, sample_diffraction_data, temp_output_dir):
        """Test a minimal training loop."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dataset
        dataset = TensorDataset(sample_diffraction_data)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Create model
        model = LitAE(
            latent_dim=32, 
            lr=1e-3, 
            realtime_metrics=False,
            out_shape=(64, 64)
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        
        # Train for one epoch
        trainer.fit(model, dataloader)
        
        # Should complete without error
        assert trainer.state.finished, "Training should complete"

    @pytest.mark.integration
    def test_training_with_validation(self, sample_diffraction_data, temp_output_dir):
        """Test training with validation split."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create train/val split
        train_size = int(0.8 * len(sample_diffraction_data))
        val_size = len(sample_diffraction_data) - train_size
        
        train_indices = torch.randperm(len(sample_diffraction_data))[:train_size]
        val_indices = torch.randperm(len(sample_diffraction_data))[:val_size]
        
        train_data = sample_diffraction_data[train_indices]
        val_data = sample_diffraction_data[val_indices]
        
        # Create datasets
        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)
        
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # Create model
        model = LitAE(
            latent_dim=32, 
            lr=1e-3, 
            realtime_metrics=False,
            out_shape=(64, 64)
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        
        # Train with validation
        trainer.fit(model, train_dataloader, val_dataloader)
        
        # Should complete without error
        assert trainer.state.finished, "Training with validation should complete"

    @pytest.mark.slow
    def test_overfitting_check(self, sample_diffraction_data):
        """Test that model can overfit to a small dataset (sanity check)."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Use only a few samples to test overfitting
        small_data = sample_diffraction_data[:4]
        dataset = TensorDataset(small_data)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        model = LitAE(
            latent_dim=32, 
            lr=1e-2,  # Higher learning rate for faster overfitting
            realtime_metrics=False,
            out_shape=(64, 64)
        )
        
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )
        
        # Train longer
        trainer.fit(model, dataloader)
        
        # Test that model can reconstruct the training data well
        model.eval()
        with torch.no_grad():
            reconstructed = model(small_data)
            mse = torch.nn.functional.mse_loss(reconstructed, small_data)
        
        # Should achieve low reconstruction error on training data
        assert mse < 0.1, f"Model should overfit to small dataset, got MSE: {mse}"


if __name__ == "__main__":
    pytest.main([__file__])