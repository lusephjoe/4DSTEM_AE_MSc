"""
Tests for ModelManager class.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import platform

from scripts.training.train import TrainingConfig, ModelManager


class TestModelManager:
    """Test suite for ModelManager class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return TrainingConfig(
            data_path=Path("/path/to/data.h5"),
            output_dir=Path("/path/to/output"),
            latent_dim=128,
            learning_rate=1e-3,
            loss_function="mse",
            lambda_act=1e-5,
            lambda_sim=0.0,
            lambda_div=0.0,
            lambda_l2=0.0,
            lambda_kl=0.0,
            compile=False,
            realtime_metrics=False
        )
    
    @pytest.fixture
    def model_manager(self, config):
        """Create a ModelManager instance."""
        return ModelManager(config)
    
    def test_init(self, config):
        """Test ModelManager initialization."""
        manager = ModelManager(config)
        assert manager.config == config
        assert hasattr(manager, 'logger')
    
    @patch('scripts.training.train.create_loss_config_from_args')
    @patch('scripts.training.lightning_model.LitAE')
    def test_create_model_basic(self, mock_lit_ae, mock_create_loss_config, model_manager):
        """Test basic model creation without compilation."""
        mock_loss_config = {'reconstruction_loss': 'mse'}
        mock_create_loss_config.return_value = mock_loss_config
        mock_model = MagicMock()
        mock_lit_ae.return_value = mock_model
        
        result = model_manager.create_model(detected_size=256)
        
        mock_create_loss_config.assert_called_once_with(model_manager.config)
        mock_lit_ae.assert_called_once_with(
            latent_dim=128,
            lr=1e-3,
            realtime_metrics=False,
            loss_config=mock_loss_config,
            out_shape=(256, 256)
        )
        assert result == mock_model
    
    @patch('scripts.training.train.create_loss_config_from_args')
    @patch('scripts.training.lightning_model.LitAE')
    def test_create_model_with_compilation(self, mock_lit_ae, mock_create_loss_config, model_manager):
        """Test model creation with torch.compile enabled."""
        model_manager.config.compile = True
        
        mock_loss_config = {'reconstruction_loss': 'mse'}
        mock_create_loss_config.return_value = mock_loss_config
        mock_model = MagicMock()
        mock_lit_ae.return_value = mock_model
        
        with patch.object(model_manager, '_apply_torch_compile') as mock_apply_compile:
            mock_apply_compile.return_value = mock_model
            
            result = model_manager.create_model(detected_size=512)
            
            mock_apply_compile.assert_called_once_with(mock_model)
            assert result == mock_model
    
    @patch('scripts.training.train.create_loss_config_from_args')
    @patch('scripts.training.lightning_model.LitAE')
    def test_create_model_different_sizes(self, mock_lit_ae, mock_create_loss_config, model_manager):
        """Test model creation with different input sizes."""
        mock_loss_config = {'reconstruction_loss': 'mse'}
        mock_create_loss_config.return_value = mock_loss_config
        mock_model = MagicMock()
        mock_lit_ae.return_value = mock_model
        
        # Test with 128x128
        model_manager.create_model(detected_size=128)
        mock_lit_ae.assert_called_with(
            latent_dim=128,
            lr=1e-3,
            realtime_metrics=False,
            loss_config=mock_loss_config,
            out_shape=(128, 128)
        )
        
        # Test with 512x512
        model_manager.create_model(detected_size=512)
        mock_lit_ae.assert_called_with(
            latent_dim=128,
            lr=1e-3,
            realtime_metrics=False,
            loss_config=mock_loss_config,
            out_shape=(512, 512)
        )
    
    @patch('scripts.training.train.create_loss_config_from_args')
    @patch('scripts.training.lightning_model.LitAE')
    def test_create_model_with_different_config(self, mock_lit_ae, mock_create_loss_config, model_manager):
        """Test model creation with different configuration parameters."""
        # Modify config
        model_manager.config.latent_dim = 256
        model_manager.config.learning_rate = 5e-4
        model_manager.config.realtime_metrics = True
        
        mock_loss_config = {'reconstruction_loss': 'mae'}
        mock_create_loss_config.return_value = mock_loss_config
        mock_model = MagicMock()
        mock_lit_ae.return_value = mock_model
        
        result = model_manager.create_model(detected_size=256)
        
        mock_lit_ae.assert_called_once_with(
            latent_dim=256,
            lr=5e-4,
            realtime_metrics=True,
            loss_config=mock_loss_config,
            out_shape=(256, 256)
        )
    
    @patch('platform.system')
    @patch('torch.compile')
    def test_apply_torch_compile_linux_success(self, mock_torch_compile, mock_platform, model_manager):
        """Test torch.compile application on Linux with success."""
        mock_platform.return_value = 'Linux'
        mock_model = MagicMock()
        mock_compiled_model = MagicMock()
        mock_torch_compile.return_value = mock_compiled_model
        
        result = model_manager._apply_torch_compile(mock_model)
        
        mock_torch_compile.assert_called_once()
        call_args = mock_torch_compile.call_args[1]
        assert call_args['mode'] == 'max-autotune'
        assert call_args['fullgraph'] is True
        assert call_args['dynamic'] is False
        assert mock_model.model == mock_compiled_model
        assert result == mock_model
    
    @patch('platform.system')
    @patch('torch.compile')
    def test_apply_torch_compile_linux_failure(self, mock_torch_compile, mock_platform, model_manager):
        """Test torch.compile application on Linux with failure."""
        mock_platform.return_value = 'Linux'
        mock_model = MagicMock()
        mock_torch_compile.side_effect = Exception("Compilation failed")
        
        # Should not raise exception, just log warning
        result = model_manager._apply_torch_compile(mock_model)
        
        mock_torch_compile.assert_called_once()
        assert result == mock_model  # Original model returned
    
    @patch('platform.system')
    def test_apply_torch_compile_non_linux(self, mock_platform, model_manager):
        """Test torch.compile application on non-Linux systems."""
        mock_platform.return_value = 'Darwin'  # macOS
        mock_model = MagicMock()
        
        result = model_manager._apply_torch_compile(mock_model)
        
        # Should return original model without compilation
        assert result == mock_model
    
    @patch('platform.system')
    def test_apply_torch_compile_windows(self, mock_platform, model_manager):
        """Test torch.compile application on Windows."""
        mock_platform.return_value = 'Windows'
        mock_model = MagicMock()
        
        result = model_manager._apply_torch_compile(mock_model)
        
        # Should return original model without compilation
        assert result == mock_model
    
    def test_logger_creation(self, model_manager):
        """Test that logger is properly created."""
        assert hasattr(model_manager, 'logger')
        assert model_manager.logger.name == 'training.model'