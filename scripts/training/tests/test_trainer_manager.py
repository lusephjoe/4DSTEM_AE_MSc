"""
Tests for TrainerManager class.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.training.train import TrainingConfig, TrainerManager


class TestTrainerManager:
    """Test suite for TrainerManager class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return TrainingConfig(
            data_path=Path("/path/to/data.h5"),
            output_dir=Path("/path/to/output"),
            epochs=50,
            device="cuda",
            gpus=1,
            precision="16",
            accumulate_grad_batches=1,
            save_every_n_epochs=1,
            no_validation=False
        )
    
    @pytest.fixture
    def trainer_manager(self, config):
        """Create a TrainerManager instance."""
        return TrainerManager(config)
    
    def test_init(self, config):
        """Test TrainerManager initialization."""
        manager = TrainerManager(config)
        assert manager.config == config
        assert hasattr(manager, 'logger')
    
    @patch('scripts.training.train.TensorBoardLogger')
    @patch('scripts.training.train.ModelCheckpoint')
    @patch('scripts.training.train.pl.Trainer')
    def test_setup_trainer_basic(self, mock_trainer, mock_checkpoint, mock_tb_logger, trainer_manager):
        """Test basic trainer setup."""
        mock_logger = MagicMock()
        mock_tb_logger.return_value = mock_logger
        mock_callback = MagicMock()
        mock_checkpoint.return_value = mock_callback
        mock_pl_trainer = MagicMock()
        mock_trainer.return_value = mock_pl_trainer
        
        result = trainer_manager.setup_trainer("test_model")
        
        # Check TensorBoard logger setup
        mock_tb_logger.assert_called_once_with(
            save_dir=trainer_manager.config.output_dir,
            name="tb_logs",
            default_hp_metric=False
        )
        
        # Check trainer creation
        mock_trainer.assert_called_once()
        trainer_kwargs = mock_trainer.call_args[1]
        assert trainer_kwargs['max_epochs'] == 50
        assert trainer_kwargs['accelerator'] == "gpu"
        assert trainer_kwargs['devices'] == 1
        assert trainer_kwargs['logger'] == mock_logger
        assert trainer_kwargs['callbacks'] == [mock_callback]
        assert trainer_kwargs['precision'] == "16-mixed"
        assert trainer_kwargs['gradient_clip_val'] == 1.0
        assert trainer_kwargs['accumulate_grad_batches'] == 1
        assert trainer_kwargs['log_every_n_steps'] == 50
        assert trainer_kwargs['enable_progress_bar'] is True
        assert trainer_kwargs['enable_model_summary'] is False
        
        assert result == mock_pl_trainer
    
    @patch('scripts.training.train.TensorBoardLogger')
    @patch('scripts.training.train.ModelCheckpoint')  
    @patch('scripts.training.train.pl.Trainer')
    def test_setup_trainer_different_precisions(self, mock_trainer, mock_checkpoint, mock_tb_logger, trainer_manager):
        """Test trainer setup with different precision settings."""
        mock_tb_logger.return_value = MagicMock()
        mock_checkpoint.return_value = MagicMock()
        mock_trainer.return_value = MagicMock()
        
        # Test 32-bit precision
        trainer_manager.config.precision = "32"
        trainer_manager.setup_trainer("test_model")
        trainer_kwargs = mock_trainer.call_args[1]
        assert trainer_kwargs['precision'] == "32-true"
        
        # Test bf16 precision
        trainer_manager.config.precision = "bf16"
        trainer_manager.setup_trainer("test_model")
        trainer_kwargs = mock_trainer.call_args[1]
        assert trainer_kwargs['precision'] == "bf16-mixed"
        
        # Test invalid precision (should default to 16-mixed)
        trainer_manager.config.precision = "invalid"
        trainer_manager.setup_trainer("test_model")
        trainer_kwargs = mock_trainer.call_args[1]
        assert trainer_kwargs['precision'] == "16-mixed"
    
    def test_create_checkpoint_callback_with_validation(self, trainer_manager):
        """Test checkpoint callback creation with validation."""
        trainer_manager.config.no_validation = False
        trainer_manager.config.save_every_n_epochs = 5
        
        with patch('scripts.training.train.ModelCheckpoint') as mock_checkpoint:
            mock_callback = MagicMock()
            mock_checkpoint.return_value = mock_callback
            
            result = trainer_manager._create_checkpoint_callback("test_model")
            
            mock_checkpoint.assert_called_once()
            callback_kwargs = mock_checkpoint.call_args[1]
            assert callback_kwargs['dirpath'] == trainer_manager.config.output_dir / "checkpoints"
            assert "valloss" in callback_kwargs['filename']
            assert callback_kwargs['save_top_k'] == -1
            assert callback_kwargs['every_n_epochs'] == 5
            assert callback_kwargs['save_on_train_epoch_end'] is False
            assert callback_kwargs['monitor'] == "val_loss"
            assert callback_kwargs['mode'] == "min"
            
            assert result == mock_callback
    
    def test_create_checkpoint_callback_no_validation(self, trainer_manager):
        """Test checkpoint callback creation without validation."""
        trainer_manager.config.no_validation = True
        trainer_manager.config.save_every_n_epochs = 2
        
        with patch('scripts.training.train.ModelCheckpoint') as mock_checkpoint:
            mock_callback = MagicMock()
            mock_checkpoint.return_value = mock_callback
            
            result = trainer_manager._create_checkpoint_callback("test_model")
            
            mock_checkpoint.assert_called_once()
            callback_kwargs = mock_checkpoint.call_args[1]
            assert "trainloss" in callback_kwargs['filename']
            assert callback_kwargs['every_n_epochs'] == 2
            assert callback_kwargs['save_on_train_epoch_end'] is True
            assert callback_kwargs['monitor'] == "train_loss"
            assert callback_kwargs['mode'] == "min"
            
            assert result == mock_callback
    
    def test_configure_accelerator_cuda(self, trainer_manager):
        """Test accelerator configuration for CUDA."""
        trainer_manager.config.device = "cuda"
        trainer_manager.config.gpus = 2
        
        accelerator, devices = trainer_manager._configure_accelerator()
        
        assert accelerator == "gpu"
        assert devices == 2
    
    def test_configure_accelerator_mps(self, trainer_manager):
        """Test accelerator configuration for MPS."""
        trainer_manager.config.device = "mps"
        
        accelerator, devices = trainer_manager._configure_accelerator()
        
        assert accelerator == "mps"
        assert devices == 1
    
    def test_configure_accelerator_cpu(self, trainer_manager):
        """Test accelerator configuration for CPU."""
        trainer_manager.config.device = "cpu"
        
        accelerator, devices = trainer_manager._configure_accelerator()
        
        assert accelerator == "cpu" 
        assert devices == 1
    
    @patch('scripts.training.train.TensorBoardLogger')
    @patch('scripts.training.train.ModelCheckpoint')
    @patch('scripts.training.train.pl.Trainer')
    def test_setup_trainer_with_different_config(self, mock_trainer, mock_checkpoint, mock_tb_logger, trainer_manager):
        """Test trainer setup with different configuration values."""
        # Modify config
        trainer_manager.config.epochs = 100
        trainer_manager.config.device = "mps"
        trainer_manager.config.precision = "32"
        trainer_manager.config.accumulate_grad_batches = 4
        
        mock_tb_logger.return_value = MagicMock()
        mock_checkpoint.return_value = MagicMock()
        mock_trainer.return_value = MagicMock()
        
        trainer_manager.setup_trainer("custom_model")
        
        trainer_kwargs = mock_trainer.call_args[1]
        assert trainer_kwargs['max_epochs'] == 100
        assert trainer_kwargs['accelerator'] == "mps"
        assert trainer_kwargs['devices'] == 1
        assert trainer_kwargs['precision'] == "32-true"
        assert trainer_kwargs['accumulate_grad_batches'] == 4
    
    def test_logger_creation(self, trainer_manager):
        """Test that logger is properly created."""
        assert hasattr(trainer_manager, 'logger')
        assert trainer_manager.logger.name == 'training.trainer'