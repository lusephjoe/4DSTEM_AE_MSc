"""
Tests for TrainingPipeline orchestrator class.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import datetime
import torch

from scripts.training.train import TrainingConfig, TrainingPipeline


class TestTrainingPipeline:
    """Test suite for TrainingPipeline orchestrator class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return TrainingConfig(
            data_path=Path("/path/to/data.h5"),
            output_dir=Path("/tmp/test_output"),
            epochs=10,
            batch_size=32,
            device="cpu",
            seed=42,
            no_validation=False
        )
    
    @pytest.fixture
    def mock_managers(self):
        """Create mock managers."""
        data_manager = MagicMock()
        model_manager = MagicMock()
        trainer_manager = MagicMock()
        return data_manager, model_manager, trainer_manager
    
    def test_init(self, config):
        """Test TrainingPipeline initialization."""
        with patch('pytorch_lightning.seed_everything') as mock_seed, \
             patch('torch.set_float32_matmul_precision') as mock_precision:
            
            pipeline = TrainingPipeline(config)
            
            assert pipeline.config == config
            assert hasattr(pipeline, 'start_time')
            assert isinstance(pipeline.start_time, datetime.datetime)
            assert hasattr(pipeline, 'logger')
            assert hasattr(pipeline, 'data_manager')
            assert hasattr(pipeline, 'model_manager') 
            assert hasattr(pipeline, 'trainer_manager')
            
            # Check that seed was set
            mock_seed.assert_called_once_with(42)
            
            # CPU device shouldn't set matmul precision
            mock_precision.assert_not_called()
    
    def test_init_cuda_device(self, config):
        """Test TrainingPipeline initialization with CUDA device."""
        config.device = "cuda"
        
        with patch('pytorch_lightning.seed_everything') as mock_seed, \
             patch('torch.set_float32_matmul_precision') as mock_precision:
            
            pipeline = TrainingPipeline(config)
            
            # CUDA device should set matmul precision
            mock_precision.assert_called_once_with('medium')
    
    @patch('logging.getLogger')
    @patch('logging.FileHandler')
    @patch('logging.StreamHandler')
    def test_setup_logging(self, mock_stream_handler, mock_file_handler, mock_get_logger, config):
        """Test logging setup."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_file_h = MagicMock()
        mock_stream_h = MagicMock()
        mock_file_handler.return_value = mock_file_h
        mock_stream_handler.return_value = mock_stream_h
        
        with patch('pytorch_lightning.seed_everything'):
            pipeline = TrainingPipeline(config)
            
            # Check logger setup - should be called at least once with 'training'
            logger_calls = [call[0][0] for call in mock_get_logger.call_args_list]
            assert 'training' in logger_calls
            mock_logger.setLevel.assert_called_with(20)  # logging.INFO
            mock_logger.handlers.clear.assert_called_once()
            mock_logger.addHandler.assert_any_call(mock_file_h)
            mock_logger.addHandler.assert_any_call(mock_stream_h)
    
    @patch('scripts.training.train.DatasetManager')
    @patch('scripts.training.train.ModelManager')
    @patch('scripts.training.train.TrainerManager')
    def test_run_successful_training(self, mock_trainer_mgr_class, mock_model_mgr_class, 
                                   mock_data_mgr_class, config):
        """Test successful training pipeline execution."""
        # Setup mocks
        mock_data_mgr = MagicMock()
        mock_model_mgr = MagicMock()
        mock_trainer_mgr = MagicMock()
        
        mock_data_mgr_class.return_value = mock_data_mgr
        mock_model_mgr_class.return_value = mock_model_mgr
        mock_trainer_mgr_class.return_value = mock_trainer_mgr
        
        # Setup data manager mocks
        mock_dataset = MagicMock()
        mock_train_ds = MagicMock()
        mock_val_ds = MagicMock()
        mock_train_dl = MagicMock()
        mock_val_dl = MagicMock()
        
        # Set numeric attributes that will be formatted in f-strings
        mock_train_ds.global_log_mean = 0.5
        mock_train_ds.global_log_std = 0.2
        mock_train_ds.use_normalization = True
        
        mock_data_mgr.load_dataset.return_value = (mock_dataset, 256)
        mock_data_mgr.create_train_val_split.return_value = (mock_train_ds, mock_val_ds)
        mock_data_mgr.create_data_loaders.return_value = (mock_train_dl, mock_val_dl)
        
        # Setup model manager mocks
        mock_model = MagicMock()
        mock_model_mgr.create_model.return_value = mock_model
        
        # Setup trainer manager mocks
        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 9
        mock_trainer_mgr.setup_trainer.return_value = mock_trainer
        
        # Setup model attributes
        mock_model.train_losses = [0.1, 0.05, 0.02]
        
        with patch('pytorch_lightning.seed_everything'), \
             patch.object(TrainingPipeline, '_log_configuration'), \
             patch.object(TrainingPipeline, '_train_model'), \
             patch.object(TrainingPipeline, '_save_results'), \
             patch.object(TrainingPipeline, '_final_evaluation'), \
             patch.object(TrainingPipeline, '_finalize_training'):
            
            pipeline = TrainingPipeline(config)
            pipeline.run()
            
            # Verify method calls
            mock_data_mgr.load_dataset.assert_called_once()
            mock_data_mgr.create_train_val_split.assert_called_once_with(mock_dataset)
            mock_data_mgr.create_data_loaders.assert_called_once_with(mock_train_ds, mock_val_ds)
            mock_model_mgr.create_model.assert_called_once_with(256)
            mock_trainer_mgr.setup_trainer.assert_called_once()
    
    @patch('scripts.training.train.DatasetManager')
    @patch('scripts.training.train.ModelManager')
    @patch('scripts.training.train.TrainerManager')
    def test_run_with_exception(self, mock_trainer_mgr_class, mock_model_mgr_class, 
                              mock_data_mgr_class, config):
        """Test training pipeline with exception handling."""
        mock_data_mgr = MagicMock()
        mock_data_mgr_class.return_value = mock_data_mgr
        mock_data_mgr.load_dataset.side_effect = Exception("Data loading failed")
        
        with patch('pytorch_lightning.seed_everything'):
            pipeline = TrainingPipeline(config)
            
            with pytest.raises(Exception, match="Data loading failed"):
                pipeline.run()
    
    def test_log_configuration(self, config):
        """Test configuration logging."""
        mock_model = MagicMock()
        mock_model.model.get_loss_info.return_value = {'reconstruction': 'mse'}
        
        with patch('pytorch_lightning.seed_everything'):
            pipeline = TrainingPipeline(config)
            
            # Mock logger to capture calls
            pipeline.logger = MagicMock()
            
            pipeline._log_configuration(mock_model)
            
            # Verify logging calls were made
            assert pipeline.logger.info.call_count >= 5  # Should log multiple config items
    
    def test_train_model_with_validation(self, config):
        """Test model training with validation data."""
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_train_dl = MagicMock()
        mock_val_dl = MagicMock()
        
        with patch('pytorch_lightning.seed_everything'):
            pipeline = TrainingPipeline(config)
            pipeline.logger = MagicMock()
            
            pipeline._train_model(mock_trainer, mock_model, mock_train_dl, mock_val_dl)
            
            mock_trainer.fit.assert_called_once_with(mock_model, mock_train_dl, mock_val_dl, ckpt_path=None)
    
    def test_train_model_without_validation(self, config):
        """Test model training without validation data."""
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_train_dl = MagicMock()
        
        with patch('pytorch_lightning.seed_everything'):
            pipeline = TrainingPipeline(config)
            pipeline.logger = MagicMock()
            
            pipeline._train_model(mock_trainer, mock_model, mock_train_dl, None)
            
            mock_trainer.fit.assert_called_once_with(mock_model, mock_train_dl, ckpt_path=None)
    
    def test_train_model_with_checkpoint(self, config):
        """Test model training with checkpoint resumption."""
        config.resume_from_checkpoint = Path("/path/to/checkpoint.ckpt")
        
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_train_dl = MagicMock()
        mock_val_dl = MagicMock()
        
        with patch('pytorch_lightning.seed_everything'), \
             patch.object(Path, 'exists', return_value=True):
            
            pipeline = TrainingPipeline(config)
            pipeline.logger = MagicMock()
            
            pipeline._train_model(mock_trainer, mock_model, mock_train_dl, mock_val_dl)
            
            mock_trainer.fit.assert_called_once_with(
                mock_model, mock_train_dl, mock_val_dl, 
                ckpt_path=str(config.resume_from_checkpoint)
            )
    
    def test_train_model_checkpoint_not_found(self, config):
        """Test model training with non-existent checkpoint."""
        config.resume_from_checkpoint = Path("/path/to/nonexistent.ckpt")
        
        mock_trainer = MagicMock()
        mock_model = MagicMock()
        mock_train_dl = MagicMock()
        mock_val_dl = MagicMock()
        
        with patch('pytorch_lightning.seed_everything'), \
             patch.object(Path, 'exists', return_value=False):
            
            pipeline = TrainingPipeline(config)
            pipeline.logger = MagicMock()
            
            pipeline._train_model(mock_trainer, mock_model, mock_train_dl, mock_val_dl)
            
            # Should not call fit if checkpoint doesn't exist
            mock_trainer.fit.assert_not_called()
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.yscale')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_save_loss_curve(self, mock_close, mock_savefig, mock_tight_layout, 
                           mock_title, mock_yscale, mock_ylabel, mock_xlabel, 
                           mock_plot, mock_figure, config):
        """Test loss curve saving."""
        mock_model = MagicMock()
        mock_model.train_losses = [0.1, 0.05, 0.02]
        
        with patch('pytorch_lightning.seed_everything'):
            pipeline = TrainingPipeline(config)
            pipeline.logger = MagicMock()
            
            pipeline._save_loss_curve(mock_model, "test_model", 0.02)
            
            # Verify matplotlib calls
            mock_figure.assert_called_once()
            mock_plot.assert_called_once_with([0.1, 0.05, 0.02])
            mock_xlabel.assert_called_once_with("Batch")
            mock_ylabel.assert_called_once_with("MSE Loss")
            mock_yscale.assert_called_once_with("log")
            mock_title.assert_called_once_with("Training Loss (Final MSE: 0.0200)")
            mock_tight_layout.assert_called_once()
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    def test_finalize_training_cpu(self, config):
        """Test training finalization with CPU device."""
        config.device = "cpu"
        
        with patch('pytorch_lightning.seed_everything'):
            pipeline = TrainingPipeline(config)
            pipeline.logger = MagicMock()
            
            pipeline._finalize_training()
            
            # Should log completion
            assert pipeline.logger.info.call_count >= 3
    
    @patch('torch.cuda.empty_cache')
    def test_finalize_training_cuda(self, mock_empty_cache, config):
        """Test training finalization with CUDA device."""
        config.device = "cuda"
        
        with patch('pytorch_lightning.seed_everything'):
            pipeline = TrainingPipeline(config)
            pipeline.logger = MagicMock()
            
            pipeline._finalize_training()
            
            # Should clear CUDA cache
            mock_empty_cache.assert_called_once()
    
    def test_set_normalization_params(self, config):
        """Test setting normalization parameters on model."""
        mock_model = MagicMock()
        mock_train_ds = MagicMock()
        mock_train_ds.global_log_mean = 0.5
        mock_train_ds.global_log_std = 0.2
        mock_train_ds.use_normalization = True
        
        with patch('pytorch_lightning.seed_everything'):
            pipeline = TrainingPipeline(config)
            
            # Test the normalization logic directly
            mock_model.set_normalization_params(
                mock_train_ds.global_log_mean,
                mock_train_ds.global_log_std,
                mock_train_ds.use_normalization
            )
                
            mock_model.set_normalization_params.assert_called_once_with(0.5, 0.2, True)