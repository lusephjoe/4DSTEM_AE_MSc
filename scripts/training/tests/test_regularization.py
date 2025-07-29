"""
Comprehensive tests for regularization system in training pipeline.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

from scripts.training.train import TrainingConfig, ModelManager
from models.losses import create_loss_config_from_args


class TestRegularizationConfiguration:
    """Test suite for regularization configuration and setup."""
    
    @pytest.fixture
    def base_config(self, temp_dir):
        """Create base configuration for testing."""
        return TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            epochs=5,
            batch_size=16,
            device="cpu",
            seed=42
        )
    
    def test_default_regularization_config(self, base_config):
        """Test default regularization parameters are correctly set."""
        assert base_config.lambda_act == 1e-5  # Default Lp regularization
        assert base_config.lambda_sim == 0.0   # Default contrastive
        assert base_config.lambda_div == 0.0   # Default divergence
        assert base_config.lambda_l2 == 0.0    # Default L2 regularization  
        assert base_config.lambda_kl == 0.0    # Default KL divergence
    
    def test_custom_regularization_config(self, temp_dir):
        """Test custom regularization parameters are set correctly."""
        config = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_act=1e-4,    # Lp regularization
            lambda_sim=5e-5,    # Contrastive similarity
            lambda_div=2e-4,    # Activation divergence
            lambda_l2=1e-6,     # L2 weight regularization
            lambda_kl=1e-3      # KL divergence
        )
        
        assert config.lambda_act == 1e-4
        assert config.lambda_sim == 5e-5
        assert config.lambda_div == 2e-4
        assert config.lambda_l2 == 1e-6
        assert config.lambda_kl == 1e-3
    
    def test_zero_regularization_config(self, temp_dir):
        """Test all regularizations can be disabled (set to zero)."""
        config = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_act=0.0,
            lambda_sim=0.0,
            lambda_div=0.0,
            lambda_l2=0.0,
            lambda_kl=0.0
        )
        
        assert config.lambda_act == 0.0
        assert config.lambda_sim == 0.0
        assert config.lambda_div == 0.0
        assert config.lambda_l2 == 0.0
        assert config.lambda_kl == 0.0
    
    def test_create_loss_config_no_regularization(self, base_config):
        """Test loss config creation with no active regularizations."""
        # All lambdas are 0.0 except lambda_act (default 1e-5)
        base_config.lambda_act = 0.0
        
        loss_config = create_loss_config_from_args(base_config)
        
        assert loss_config['reconstruction_loss'] == 'mse'
        assert len(loss_config['regularization_losses']) == 0
    
    def test_create_loss_config_with_lp_regularization(self, base_config):
        """Test loss config creation with Lp regularization."""
        base_config.lambda_act = 1e-4
        
        loss_config = create_loss_config_from_args(base_config)
        
        assert 'lp_reg' in loss_config['regularization_losses']
        assert loss_config['regularization_losses']['lp_reg'] == 1e-4
    
    def test_create_loss_config_with_contrastive_regularization(self, base_config):
        """Test loss config creation with contrastive similarity."""
        base_config.lambda_act = 0.0  # Disable default
        base_config.lambda_sim = 5e-5
        
        loss_config = create_loss_config_from_args(base_config)
        
        assert 'contrastive' in loss_config['regularization_losses']
        assert loss_config['regularization_losses']['contrastive'] == 5e-5
    
    def test_create_loss_config_with_divergence_regularization(self, base_config):
        """Test loss config creation with activation divergence."""
        base_config.lambda_act = 0.0  # Disable default
        base_config.lambda_div = 2e-4
        
        loss_config = create_loss_config_from_args(base_config)
        
        assert 'divergence' in loss_config['regularization_losses']
        assert loss_config['regularization_losses']['divergence'] == 2e-4
    
    def test_create_loss_config_with_l2_regularization(self, base_config):
        """Test loss config creation with L2 weight regularization."""
        base_config.lambda_act = 0.0  # Disable default
        base_config.lambda_l2 = 1e-6
        
        loss_config = create_loss_config_from_args(base_config)
        
        assert 'l2_reg' in loss_config['regularization_losses']
        assert loss_config['regularization_losses']['l2_reg'] == 1e-6
    
    def test_create_loss_config_with_kl_regularization(self, base_config):
        """Test loss config creation with KL divergence."""
        base_config.lambda_act = 0.0  # Disable default
        base_config.lambda_kl = 1e-3
        
        loss_config = create_loss_config_from_args(base_config)
        
        assert 'kl_div' in loss_config['regularization_losses']
        assert loss_config['regularization_losses']['kl_div'] == 1e-3
    
    def test_create_loss_config_multiple_regularizations(self, base_config):
        """Test loss config creation with multiple active regularizations."""
        base_config.lambda_act = 1e-4  # Lp regularization
        base_config.lambda_sim = 5e-5  # Contrastive
        base_config.lambda_div = 2e-4  # Divergence
        base_config.lambda_l2 = 1e-6   # L2 regularization
        base_config.lambda_kl = 1e-3   # KL divergence
        
        loss_config = create_loss_config_from_args(base_config)
        
        expected_regularizations = {
            'lp_reg': 1e-4,
            'contrastive': 5e-5,
            'divergence': 2e-4,
            'l2_reg': 1e-6,
            'kl_div': 1e-3
        }
        
        assert loss_config['regularization_losses'] == expected_regularizations
        assert len(loss_config['regularization_losses']) == 5
    
    def test_negative_regularization_values(self, temp_dir):
        """Test that negative regularization values are excluded from configuration."""
        # Negative values should be excluded (only positive regularization makes sense)
        config = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_act=-1e-4,  # Negative value - should be excluded
            lambda_sim=-5e-5,  # Negative value - should be excluded
            lambda_div=2e-4    # Positive value - should be included
        )
        
        loss_config = create_loss_config_from_args(config)
        
        # Negative values should be excluded from regularization losses
        assert 'lp_reg' not in loss_config['regularization_losses']
        assert 'contrastive' not in loss_config['regularization_losses']
        
        # Positive values should still be included
        assert 'divergence' in loss_config['regularization_losses']
        assert loss_config['regularization_losses']['divergence'] == 2e-4


class TestRegularizationInModelManager:
    """Test regularization integration with ModelManager."""
    
    @pytest.fixture
    def model_manager_with_regularization(self, temp_dir):
        """Create ModelManager with regularization configuration."""
        config = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_act=1e-4,
            lambda_sim=5e-5,
            lambda_div=2e-4,
            lambda_l2=1e-6,
            lambda_kl=1e-3
        )
        return ModelManager(config)
    
    @patch('scripts.training.train.create_loss_config_from_args')
    @patch('scripts.training.lightning_model.LitAE')
    def test_model_creation_with_regularization(self, mock_lit_ae, mock_create_loss_config, 
                                              model_manager_with_regularization):
        """Test that regularization config is passed to model creation."""
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
        mock_create_loss_config.return_value = expected_loss_config
        mock_model = MagicMock()
        mock_lit_ae.return_value = mock_model
        
        model = model_manager_with_regularization.create_model(detected_size=256)
        
        # Verify loss config was created from args
        mock_create_loss_config.assert_called_once_with(model_manager_with_regularization.config)
        
        # Verify model was created with loss config
        mock_lit_ae.assert_called_once_with(
            latent_dim=128,
            lr=1e-3,
            realtime_metrics=False,
            loss_config=expected_loss_config,
            out_shape=(256, 256)
        )
    
    @patch('scripts.training.train.create_loss_config_from_args')
    @patch('scripts.training.lightning_model.LitAE')
    def test_model_creation_no_regularization(self, mock_lit_ae, mock_create_loss_config, temp_dir):
        """Test model creation with no regularization."""
        config = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_act=0.0,  # All regularizations disabled
            lambda_sim=0.0,
            lambda_div=0.0,
            lambda_l2=0.0,
            lambda_kl=0.0
        )
        model_manager = ModelManager(config)
        
        expected_loss_config = {
            'reconstruction_loss': 'mse',
            'regularization_losses': {}  # No regularizations
        }
        mock_create_loss_config.return_value = expected_loss_config
        mock_model = MagicMock()
        mock_lit_ae.return_value = mock_model
        
        model = model_manager.create_model(detected_size=256)
        
        # Verify empty regularization losses
        mock_lit_ae.assert_called_once_with(
            latent_dim=128,
            lr=1e-3,
            realtime_metrics=False,
            loss_config=expected_loss_config,
            out_shape=(256, 256)
        )


class TestRegularizationValidation:
    """Test regularization parameter validation and edge cases."""
    
    def test_very_large_regularization_values(self, temp_dir):
        """Test handling of very large regularization values."""
        config = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_act=1.0,      # Very large regularization
            lambda_sim=0.5,
            lambda_div=2.0
        )
        
        loss_config = create_loss_config_from_args(config)
        
        # Large values should be preserved (training may fail, but that's expected)
        assert loss_config['regularization_losses']['lp_reg'] == 1.0
        assert loss_config['regularization_losses']['contrastive'] == 0.5
        assert loss_config['regularization_losses']['divergence'] == 2.0
    
    def test_very_small_regularization_values(self, temp_dir):
        """Test handling of very small but non-zero regularization values."""
        config = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_act=1e-10,    # Very small regularization
            lambda_sim=1e-15,
            lambda_l2=1e-20
        )
        
        loss_config = create_loss_config_from_args(config)
        
        # Small non-zero values should be preserved
        assert loss_config['regularization_losses']['lp_reg'] == 1e-10
        assert loss_config['regularization_losses']['contrastive'] == 1e-15
        assert loss_config['regularization_losses']['l2_reg'] == 1e-20
    
    def test_mixed_zero_and_nonzero_regularizations(self, temp_dir):
        """Test mixed regularization configuration (some enabled, some disabled)."""
        config = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_act=1e-4,     # Enabled
            lambda_sim=0.0,      # Disabled
            lambda_div=2e-4,     # Enabled
            lambda_l2=0.0,       # Disabled
            lambda_kl=1e-3       # Enabled
        )
        
        loss_config = create_loss_config_from_args(config)
        
        # Only non-zero regularizations should be included
        expected_regularizations = {
            'lp_reg': 1e-4,
            'divergence': 2e-4,
            'kl_div': 1e-3
        }
        assert loss_config['regularization_losses'] == expected_regularizations
        
        # Disabled regularizations should not be present
        assert 'contrastive' not in loss_config['regularization_losses']
        assert 'l2_reg' not in loss_config['regularization_losses']
    
    def test_float_precision_regularization_values(self, temp_dir):
        """Test regularization values with different float precisions."""
        config = TrainingConfig(
            data_path=temp_dir / "data.h5",
            output_dir=temp_dir / "output",
            lambda_act=1.23456789e-4,    # High precision
            lambda_sim=5e-5,             # Scientific notation
            lambda_div=0.0002,           # Decimal notation
            lambda_l2=2/3 * 1e-6        # Computed value
        )
        
        loss_config = create_loss_config_from_args(config)
        
        # Values should be preserved with their precision
        assert abs(loss_config['regularization_losses']['lp_reg'] - 1.23456789e-4) < 1e-10
        assert loss_config['regularization_losses']['contrastive'] == 5e-5
        assert loss_config['regularization_losses']['divergence'] == 0.0002
        assert abs(loss_config['regularization_losses']['l2_reg'] - (2/3 * 1e-6)) < 1e-12


class TestRegularizationInArgumentParser:
    """Test regularization in command-line argument parsing."""
    
    @pytest.fixture
    def parser(self):
        """Get argument parser for testing."""
        from scripts.training.train import create_argument_parser
        return create_argument_parser()
    
    def test_default_regularization_args(self, parser):
        """Test default regularization argument values."""
        args = parser.parse_args([
            "--data", "/path/to/data.h5",
            "--output_dir", "/path/to/output"
        ])
        
        assert args.lambda_act == 1e-5  # Default value
        assert args.lambda_sim == 0
        assert args.lambda_div == 0  
        assert args.lambda_l2 == 0
        assert args.lambda_kl == 0
    
    def test_custom_regularization_args(self, parser):
        """Test parsing custom regularization arguments."""
        args = parser.parse_args([
            "--data", "/path/to/data.h5",
            "--output_dir", "/path/to/output",
            "--lambda_act", "0.0001",
            "--lambda_sim", "0.00005", 
            "--lambda_div", "0.0002",
            "--lambda_l2", "0.000001",
            "--lambda_kl", "0.001"
        ])
        
        assert args.lambda_act == 0.0001
        assert args.lambda_sim == 0.00005
        assert args.lambda_div == 0.0002
        assert args.lambda_l2 == 0.000001
        assert args.lambda_kl == 0.001
    
    def test_zero_regularization_args(self, parser):
        """Test explicitly setting regularization to zero."""
        args = parser.parse_args([
            "--data", "/path/to/data.h5", 
            "--output_dir", "/path/to/output",
            "--lambda_act", "0",
            "--lambda_sim", "0.0",
            "--lambda_div", "0.00",
            "--lambda_l2", "0.000",
            "--lambda_kl", "0.0000"
        ])
        
        assert args.lambda_act == 0.0
        assert args.lambda_sim == 0.0
        assert args.lambda_div == 0.0
        assert args.lambda_l2 == 0.0  
        assert args.lambda_kl == 0.0
    
    def test_scientific_notation_regularization_args(self, parser):
        """Test parsing regularization arguments in scientific notation."""
        args = parser.parse_args([
            "--data", "/path/to/data.h5",
            "--output_dir", "/path/to/output", 
            "--lambda_act", "1e-4",
            "--lambda_sim", "5E-5",
            "--lambda_div", "2.5e-4",
            "--lambda_l2", "1.5E-6",
            "--lambda_kl", "1e-03"
        ])
        
        assert args.lambda_act == 1e-4
        assert args.lambda_sim == 5e-5
        assert args.lambda_div == 2.5e-4
        assert args.lambda_l2 == 1.5e-6
        assert args.lambda_kl == 1e-3
    
    def test_negative_regularization_args(self, parser):
        """Test parsing negative regularization arguments."""
        args = parser.parse_args([
            "--data", "/path/to/data.h5",
            "--output_dir", "/path/to/output",
            "--lambda_act", "-0.0001",
            "--lambda_sim", "-0.00005"  # Use decimal notation for negative values
        ])
        
        # Parser should accept negative values (they get filtered out in loss config)
        assert args.lambda_act == -0.0001
        assert args.lambda_sim == -0.00005