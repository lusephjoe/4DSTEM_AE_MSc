"""
Tests for TrainingConfig dataclass.
"""
import pytest
from pathlib import Path
from unittest.mock import patch
import torch

from scripts.training.train import TrainingConfig


class TestTrainingConfig:
    """Test suite for TrainingConfig dataclass."""
    
    def test_init_with_required_params(self):
        """Test initialization with only required parameters."""
        config = TrainingConfig(
            data_path=Path("/path/to/data.h5"),
            output_dir=Path("/path/to/output")
        )
        
        assert config.data_path == Path("/path/to/data.h5")
        assert config.output_dir == Path("/path/to/output")
        assert config.use_normalization is True
        assert config.latent_dim == 128
        assert config.epochs == 50
        assert config.batch_size == 128
        assert config.learning_rate == 1e-3
        assert config.seed == 42
    
    def test_init_with_all_params(self):
        """Test initialization with all parameters specified."""
        config = TrainingConfig(
            data_path="/path/to/data.h5",
            output_dir="/path/to/output",
            use_normalization=False,
            no_validation=True,
            latent_dim=256,
            input_size=512,
            epochs=100,
            batch_size=64,
            learning_rate=1e-4,
            no_scheduler=True,
            precision="32",
            accumulate_grad_batches=2,
            loss_function="mae",
            lambda_act=1e-4,
            lambda_sim=1e-3,
            lambda_div=1e-2,
            lambda_l2=1e-5,
            lambda_kl=1e-6,
            device="cuda",
            gpus=2,
            compile=True,
            num_workers=8,
            pin_memory=False,
            persistent_workers=True,
            resume_from_checkpoint="/path/to/checkpoint.ckpt",
            save_every_n_epochs=5,
            seed=123,
            realtime_metrics=True,
            profile=True,
            debug=True
        )
        
        assert config.data_path == Path("/path/to/data.h5")
        assert config.output_dir == Path("/path/to/output")
        assert config.use_normalization is False
        assert config.no_validation is True
        assert config.latent_dim == 256
        assert config.input_size == 512
        assert config.epochs == 100
        assert config.batch_size == 64
        assert config.learning_rate == 1e-4
        assert config.no_scheduler is True
        assert config.precision == "32"
        assert config.accumulate_grad_batches == 2
        assert config.loss_function == "mae"
        assert config.lambda_act == 1e-4
        assert config.lambda_sim == 1e-3
        assert config.lambda_div == 1e-2
        assert config.lambda_l2 == 1e-5
        assert config.lambda_kl == 1e-6
        assert config.device == "cuda"
        assert config.gpus == 2
        assert config.compile is True
        assert config.num_workers == 8
        assert config.pin_memory is False
        assert config.persistent_workers is True
        assert config.resume_from_checkpoint == Path("/path/to/checkpoint.ckpt")
        assert config.save_every_n_epochs == 5
        assert config.seed == 123
        assert config.realtime_metrics is True
        assert config.profile is True
        assert config.debug is True
    
    def test_post_init_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = TrainingConfig(
            data_path="/path/to/data.h5",
            output_dir="/path/to/output",
            resume_from_checkpoint="/path/to/checkpoint.ckpt"
        )
        
        assert isinstance(config.data_path, Path)
        assert isinstance(config.output_dir, Path)
        assert isinstance(config.resume_from_checkpoint, Path)
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_auto_device_detection_cuda(self, mock_mps, mock_cuda):
        """Test automatic device detection chooses CUDA when available."""
        mock_cuda.return_value = True
        mock_mps.return_value = False
        
        config = TrainingConfig(
            data_path="/path/to/data.h5",
            output_dir="/path/to/output",
            device="auto"
        )
        
        assert config.device == "cuda"
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_auto_device_detection_mps(self, mock_mps, mock_cuda):
        """Test automatic device detection chooses MPS when CUDA unavailable."""
        mock_cuda.return_value = False
        mock_mps.return_value = True
        
        config = TrainingConfig(
            data_path="/path/to/data.h5",
            output_dir="/path/to/output",
            device="auto"
        )
        
        assert config.device == "mps"
    
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_auto_device_detection_cpu(self, mock_mps, mock_cuda):
        """Test automatic device detection defaults to CPU."""
        mock_cuda.return_value = False
        mock_mps.return_value = False
        
        config = TrainingConfig(
            data_path="/path/to/data.h5",
            output_dir="/path/to/output",
            device="auto"
        )
        
        assert config.device == "cpu"
    
    def test_manual_device_setting(self):
        """Test that manual device setting is preserved."""
        config = TrainingConfig(
            data_path="/path/to/data.h5",
            output_dir="/path/to/output",
            device="cuda"
        )
        
        assert config.device == "cuda"
    
    def test_none_checkpoint_handling(self):
        """Test that None checkpoint path is handled correctly."""
        config = TrainingConfig(
            data_path="/path/to/data.h5",
            output_dir="/path/to/output",
            resume_from_checkpoint=None
        )
        
        assert config.resume_from_checkpoint is None
    
    def test_config_immutability_after_init(self):
        """Test that config values are properly set after initialization."""
        config = TrainingConfig(
            data_path="/path/to/data.h5",
            output_dir="/path/to/output"
        )
        
        # These should be set by __post_init__
        assert isinstance(config.data_path, Path)
        assert isinstance(config.output_dir, Path)
        assert config.device in ["cpu", "cuda", "mps"]
    
    def test_dataclass_equality(self):
        """Test that two configs with same values are equal."""
        config1 = TrainingConfig(
            data_path="/path/to/data.h5",
            output_dir="/path/to/output"
        )
        
        config2 = TrainingConfig(
            data_path="/path/to/data.h5",
            output_dir="/path/to/output"
        )
        
        # Note: Due to device auto-detection, we need to ensure same device
        config2.device = config1.device
        assert config1 == config2
    
    def test_dataclass_inequality(self):
        """Test that configs with different values are not equal."""
        config1 = TrainingConfig(
            data_path="/path/to/data.h5",
            output_dir="/path/to/output",
            epochs=50
        )
        
        config2 = TrainingConfig(
            data_path="/path/to/data.h5",
            output_dir="/path/to/output",
            epochs=100
        )
        
        assert config1 != config2