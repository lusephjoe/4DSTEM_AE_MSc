"""
Integration tests for regularization in the full training pipeline.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import numpy as np

from scripts.training.train import TrainingConfig, TrainingPipeline


class TestRegularizationIntegration:
    """Integration tests for regularization across the full training pipeline."""
    
    @pytest.fixture
    def config_with_lp_regularization(self, mock_hdf5_file, temp_dir):
        """Configuration with only Lp regularization enabled."""
        return TrainingConfig(
            data_path=mock_hdf5_file,
            output_dir=temp_dir / "output",
            epochs=2,
            batch_size=8,
            device="cpu",
            num_workers=0,
            no_validation=True,
            lambda_act=1e-4,  # Lp regularization only
            lambda_sim=0.0,
            lambda_div=0.0,
            lambda_l2=0.0,
            lambda_kl=0.0,
            seed=42
        )
    
    @pytest.fixture
    def config_with_contrastive_regularization(self, mock_hdf5_file, temp_dir):
        """Configuration with only contrastive regularization enabled."""
        return TrainingConfig(
            data_path=mock_hdf5_file,
            output_dir=temp_dir / "output",
            epochs=2,
            batch_size=8,
            device="cpu", 
            num_workers=0,
            no_validation=True,
            lambda_act=0.0,
            lambda_sim=5e-5,  # Contrastive regularization only
            lambda_div=0.0,
            lambda_l2=0.0,
            lambda_kl=0.0,
            seed=42
        )
    
    @pytest.fixture
    def config_with_multiple_regularizations(self, mock_hdf5_file, temp_dir):
        """Configuration with multiple regularizations enabled."""
        return TrainingConfig(
            data_path=mock_hdf5_file,
            output_dir=temp_dir / "output",
            epochs=2,
            batch_size=8,
            device="cpu",
            num_workers=0,
            no_validation=True,
            lambda_act=1e-4,   # Lp regularization
            lambda_sim=5e-5,   # Contrastive regularization
            lambda_div=2e-4,   # Divergence regularization
            lambda_l2=1e-6,    # L2 regularization
            lambda_kl=1e-3,    # KL divergence
            seed=42
        )
    
    @pytest.fixture
    def config_no_regularization(self, mock_hdf5_file, temp_dir):  
        """Configuration with all regularizations disabled."""
        return TrainingConfig(
            data_path=mock_hdf5_file,
            output_dir=temp_dir / "output",
            epochs=2,
            batch_size=8,
            device="cpu",
            num_workers=0,
            no_validation=True,
            lambda_act=0.0,    # All disabled
            lambda_sim=0.0,
            lambda_div=0.0,
            lambda_l2=0.0,
            lambda_kl=0.0,
            seed=42
        )
    
    @patch('models.autoencoder.Autoencoder')
    @patch('scripts.training.lightning_model.LitAE')
    @patch('scripts.training.train.create_loss_config_from_args')
    def test_pipeline_with_lp_regularization(self, mock_loss_config, mock_lit_ae, mock_autoencoder,
                                           config_with_lp_regularization):
        """Test full pipeline with Lp regularization."""
        # Setup expected loss config
        expected_loss_config = {
            'reconstruction_loss': 'mse',
            'regularization_losses': {'lp_reg': 1e-4}
        }
        mock_loss_config.return_value = expected_loss_config
        
        # Setup mocks
        mock_model = MagicMock()
        mock_model.train_losses = [0.1, 0.05]
        mock_model.model.get_loss_info.return_value = {'reconstruction': 'mse'}
        mock_lit_ae.return_value = mock_model
        
        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 1
        
        with patch('pytorch_lightning.Trainer') as mock_trainer_class, \
             patch('pytorch_lightning.loggers.TensorBoardLogger'), \
             patch('pytorch_lightning.callbacks.ModelCheckpoint'), \
             patch('models.summary.calculate_metrics') as mock_calc_metrics, \
             patch('models.summary.save_comparison_images'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.savefig'):
            
            mock_trainer_class.return_value = mock_trainer
            mock_calc_metrics.return_value = {
                'mse': 0.01, 'mse_std': 0.001,
                'psnr': 30.0, 'psnr_std': 1.0,
                'ssim': 0.95, 'ssim_std': 0.02
            }
            
            # Run pipeline
            pipeline = TrainingPipeline(config_with_lp_regularization)
            pipeline.run()
            
            # Verify loss config was created with Lp regularization
            mock_loss_config.assert_called_once_with(config_with_lp_regularization)
            
            # Verify model was created with correct loss config
            mock_lit_ae.assert_called_once()
            call_kwargs = mock_lit_ae.call_args[1]
            assert call_kwargs['loss_config'] == expected_loss_config
    
    @patch('models.autoencoder.Autoencoder')
    @patch('scripts.training.lightning_model.LitAE')
    @patch('scripts.training.train.create_loss_config_from_args')
    def test_pipeline_with_multiple_regularizations(self, mock_loss_config, mock_lit_ae, mock_autoencoder,
                                                  config_with_multiple_regularizations):
        """Test full pipeline with multiple regularizations."""
        # Setup expected loss config with all regularizations
        expected_loss_config = {
            'reconstruction_loss': 'mse',
            'regularization_losses': {
                'lp_reg': 1e-4,
                'contrastive': 5e-5,
                'divergence': 2e-4,
                'l2_reg': 1e-6,
                'kl_div': 1e-3
            }
        }
        mock_loss_config.return_value = expected_loss_config
        
        # Setup mocks
        mock_model = MagicMock()
        mock_model.train_losses = [0.1, 0.05]
        mock_model.model.get_loss_info.return_value = {'reconstruction': 'mse'}
        mock_lit_ae.return_value = mock_model
        
        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 1
        
        with patch('pytorch_lightning.Trainer') as mock_trainer_class, \
             patch('pytorch_lightning.loggers.TensorBoardLogger'), \
             patch('pytorch_lightning.callbacks.ModelCheckpoint'), \
             patch('models.summary.calculate_metrics') as mock_calc_metrics, \
             patch('models.summary.save_comparison_images'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.savefig'):
            
            mock_trainer_class.return_value = mock_trainer
            mock_calc_metrics.return_value = {
                'mse': 0.01, 'mse_std': 0.001,
                'psnr': 30.0, 'psnr_std': 1.0,
                'ssim': 0.95, 'ssim_std': 0.02
            }
            
            # Run pipeline
            pipeline = TrainingPipeline(config_with_multiple_regularizations)
            pipeline.run()
            
            # Verify loss config was created with all regularizations
            mock_loss_config.assert_called_once_with(config_with_multiple_regularizations)
            
            # Verify model received all regularizations
            mock_lit_ae.assert_called_once()
            call_kwargs = mock_lit_ae.call_args[1]
            assert call_kwargs['loss_config'] == expected_loss_config
            assert len(call_kwargs['loss_config']['regularization_losses']) == 5
    
    @patch('models.autoencoder.Autoencoder')
    @patch('scripts.training.lightning_model.LitAE')
    @patch('scripts.training.train.create_loss_config_from_args')
    def test_pipeline_no_regularization(self, mock_loss_config, mock_lit_ae, mock_autoencoder,
                                      config_no_regularization):
        """Test full pipeline with no regularizations."""
        # Setup expected loss config with no regularizations
        expected_loss_config = {
            'reconstruction_loss': 'mse',
            'regularization_losses': {}  # Empty
        }
        mock_loss_config.return_value = expected_loss_config
        
        # Setup mocks
        mock_model = MagicMock()
        mock_model.train_losses = [0.1, 0.05]
        mock_model.model.get_loss_info.return_value = {'reconstruction': 'mse'}
        mock_lit_ae.return_value = mock_model
        
        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 1
        
        with patch('pytorch_lightning.Trainer') as mock_trainer_class, \
             patch('pytorch_lightning.loggers.TensorBoardLogger'), \
             patch('pytorch_lightning.callbacks.ModelCheckpoint'), \
             patch('models.summary.calculate_metrics') as mock_calc_metrics, \
             patch('models.summary.save_comparison_images'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.savefig'):
            
            mock_trainer_class.return_value = mock_trainer
            mock_calc_metrics.return_value = {
                'mse': 0.01, 'mse_std': 0.001,
                'psnr': 30.0, 'psnr_std': 1.0,
                'ssim': 0.95, 'ssim_std': 0.02
            }
            
            # Run pipeline
            pipeline = TrainingPipeline(config_no_regularization)
            pipeline.run()
            
            # Verify loss config has no regularizations
            mock_loss_config.assert_called_once_with(config_no_regularization)
            
            # Verify model received empty regularizations
            mock_lit_ae.assert_called_once()
            call_kwargs = mock_lit_ae.call_args[1]
            assert call_kwargs['loss_config'] == expected_loss_config
            assert len(call_kwargs['loss_config']['regularization_losses']) == 0
    
    def test_argument_parser_regularization_integration(self):
        """Test integration of regularization arguments through CLI parser."""
        from scripts.training.train import create_argument_parser, TrainingConfig
        
        parser = create_argument_parser()
        
        # Test parsing with regularization arguments
        args = parser.parse_args([
            "--data", "/path/to/data.h5", 
            "--output_dir", "/path/to/output",
            "--lambda_act", "0.0001",
            "--lambda_sim", "0.00005",
            "--lambda_div", "0.0002", 
            "--lambda_l2", "0.000001",
            "--lambda_kl", "0.001"
        ])
        
        # Convert to config (same as main() function)
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
        
        # Verify regularization parameters made it through
        assert config.lambda_act == 0.0001
        assert config.lambda_sim == 0.00005
        assert config.lambda_div == 0.0002
        assert config.lambda_l2 == 0.000001
        assert config.lambda_kl == 0.001


class TestRegularizationScenarios:
    """Test specific regularization usage scenarios."""
    
    def test_common_regularization_combinations(self, temp_dir):
        """Test commonly used regularization combinations."""
        
        # Scenario 1: Light regularization (common for stable training)
        config_light = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_act=1e-5,   # Light Lp regularization
            lambda_l2=1e-6     # Light L2 regularization
        )
        
        # Scenario 2: Heavy regularization (for preventing overfitting)
        config_heavy = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output", 
            lambda_act=1e-3,   # Strong Lp regularization
            lambda_l2=1e-4,    # Strong L2 regularization
            lambda_div=1e-4    # Activation divergence
        )
        
        # Scenario 3: Contrastive learning setup
        config_contrastive = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_sim=1e-4,   # Contrastive similarity
            lambda_div=5e-5    # Encourage diverse representations
        )
        
        # Test that each scenario creates valid configurations
        from models.losses import create_loss_config_from_args
        
        light_config = create_loss_config_from_args(config_light)
        assert 'lp_reg' in light_config['regularization_losses']
        assert 'l2_reg' in light_config['regularization_losses']
        assert len(light_config['regularization_losses']) == 2
        
        heavy_config = create_loss_config_from_args(config_heavy)
        assert len(heavy_config['regularization_losses']) == 3
        assert heavy_config['regularization_losses']['lp_reg'] == 1e-3
        
        contrastive_config = create_loss_config_from_args(config_contrastive)
        assert 'contrastive' in contrastive_config['regularization_losses']
        assert 'divergence' in contrastive_config['regularization_losses']
    
    def test_progressive_regularization_strengths(self, temp_dir):
        """Test different strengths of the same regularization type."""
        
        # Weak regularization
        config_weak = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_act=1e-6
        )
        
        # Medium regularization  
        config_medium = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_act=1e-4
        )
        
        # Strong regularization
        config_strong = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_act=1e-2
        )
        
        from models.losses import create_loss_config_from_args
        
        weak_config = create_loss_config_from_args(config_weak)
        medium_config = create_loss_config_from_args(config_medium)  
        strong_config = create_loss_config_from_args(config_strong)
        
        # Verify progressive strength
        assert weak_config['regularization_losses']['lp_reg'] < medium_config['regularization_losses']['lp_reg']
        assert medium_config['regularization_losses']['lp_reg'] < strong_config['regularization_losses']['lp_reg']
    
    def test_regularization_with_different_loss_functions(self, temp_dir):
        """Test regularization with different reconstruction loss functions."""
        
        # MSE + regularization
        config_mse = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            loss_function="mse",
            lambda_act=1e-4,
            lambda_l2=1e-6
        )
        
        # L1 + regularization
        config_l1 = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            loss_function="l1",
            lambda_act=1e-4,
            lambda_l2=1e-6
        )
        
        # Huber + regularization
        config_huber = TrainingConfig(
            data_path=temp_dir / "data.h5", 
            output_dir=temp_dir / "output",
            loss_function="huber",
            lambda_act=1e-4,
            lambda_l2=1e-6
        )
        
        from models.losses import create_loss_config_from_args
        
        mse_config = create_loss_config_from_args(config_mse)
        l1_config = create_loss_config_from_args(config_l1)
        huber_config = create_loss_config_from_args(config_huber)
        
        # Different reconstruction losses, same regularization
        assert mse_config['reconstruction_loss'] == 'mse'
        assert l1_config['reconstruction_loss'] == 'l1'
        assert huber_config['reconstruction_loss'] == 'huber'
        
        # Same regularization for all
        for config in [mse_config, l1_config, huber_config]:
            assert config['regularization_losses']['lp_reg'] == 1e-4
            assert config['regularization_losses']['l2_reg'] == 1e-6
    
    def test_edge_case_regularization_values(self, temp_dir):
        """Test edge cases for regularization values."""
        
        # Very small values (near machine precision)
        config_tiny = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_act=1e-15,  # Very small
            lambda_sim=1e-20   # Extremely small
        )
        
        # Moderately large values
        config_large = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_act=0.1,    # Large regularization
            lambda_div=0.05    # Large divergence
        )
        
        from models.losses import create_loss_config_from_args
        
        tiny_config = create_loss_config_from_args(config_tiny)
        large_config = create_loss_config_from_args(config_large)
        
        # Verify extreme values are preserved
        assert tiny_config['regularization_losses']['lp_reg'] == 1e-15
        assert tiny_config['regularization_losses']['contrastive'] == 1e-20
        assert large_config['regularization_losses']['lp_reg'] == 0.1
        assert large_config['regularization_losses']['divergence'] == 0.05