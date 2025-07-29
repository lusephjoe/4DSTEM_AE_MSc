"""
Integration tests for the full training pipeline.
"""
import pytest
from pathlib import Path
import tempfile
import shutil
import torch
import numpy as np
import h5py
from unittest.mock import patch, MagicMock

from scripts.training.train import TrainingConfig, TrainingPipeline, main, create_argument_parser


class TestTrainingIntegration:
    """Integration test suite for the complete training pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_hdf5_file(self, temp_dir):
        """Create a mock HDF5 file with test data."""
        data_path = temp_dir / "test_data.h5"
        
        # Create small test dataset
        test_data = np.random.rand(100, 64, 64).astype(np.float32)
        
        with h5py.File(str(data_path), 'w') as f:
            dataset = f.create_dataset('patterns', data=test_data)
            dataset.attrs['dtype'] = 'float32'
            dataset.attrs['data_min'] = float(test_data.min())
            dataset.attrs['data_max'] = float(test_data.max())
            dataset.attrs['data_range'] = float(test_data.max() - test_data.min())
        
        return data_path
    
    @pytest.fixture
    def mock_tensor_file(self, temp_dir):
        """Create a mock PyTorch tensor file with test data."""
        data_path = temp_dir / "test_data.pt"
        test_data = torch.rand(100, 1, 64, 64)
        torch.save(test_data, data_path)
        return data_path
    
    @pytest.fixture
    def config_hdf5(self, mock_hdf5_file, temp_dir):
        """Create test configuration for HDF5 data."""
        return TrainingConfig(
            data_path=mock_hdf5_file,
            output_dir=temp_dir / "output",
            epochs=2,
            batch_size=16,
            latent_dim=32,
            device="cpu",
            num_workers=0,  # Disable multiprocessing for tests
            save_every_n_epochs=1,
            no_validation=True,  # Simplify for integration test
            seed=42
        )
    
    @pytest.fixture
    def config_tensor(self, mock_tensor_file, temp_dir):
        """Create test configuration for tensor data."""
        return TrainingConfig(
            data_path=mock_tensor_file,
            output_dir=temp_dir / "output",
            epochs=2,
            batch_size=16,
            latent_dim=32,
            device="cpu",
            num_workers=0,
            save_every_n_epochs=1,
            no_validation=True,
            seed=42
        )
    
    @patch('models.autoencoder.Autoencoder')
    @patch('scripts.training.lightning_model.LitAE')
    @patch('models.losses.create_loss_config_from_args')
    def test_full_pipeline_hdf5_mock_model(self, mock_loss_config, mock_lit_ae, mock_autoencoder, config_hdf5):
        """Test complete pipeline with HDF5 data and mocked model."""
        # Setup mocks to avoid actual model training
        mock_loss_config.return_value = {'reconstruction_loss': 'mse'}
        
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
            
            # Run the pipeline
            pipeline = TrainingPipeline(config_hdf5)
            pipeline.run()
            
            # Verify output directory was created
            assert config_hdf5.output_dir.exists()
            
            # Verify trainer was called
            mock_trainer.fit.assert_called_once()
            mock_trainer.save_checkpoint.assert_called_once()
    
    @patch('models.autoencoder.Autoencoder')
    @patch('scripts.training.lightning_model.LitAE')
    @patch('models.losses.create_loss_config_from_args')
    def test_full_pipeline_tensor_mock_model(self, mock_loss_config, mock_lit_ae, mock_autoencoder, config_tensor):
        """Test complete pipeline with tensor data and mocked model."""
        # Setup mocks
        mock_loss_config.return_value = {'reconstruction_loss': 'mse'}
        
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
            
            # Run the pipeline
            pipeline = TrainingPipeline(config_tensor)
            pipeline.run()
            
            # Verify output directory was created
            assert config_tensor.output_dir.exists()
            
            # Verify trainer was called
            mock_trainer.fit.assert_called_once()
            mock_trainer.save_checkpoint.assert_called_once()
    
    def test_pipeline_with_validation_split(self, mock_hdf5_file, temp_dir):
        """Test pipeline with validation split enabled."""
        config = TrainingConfig(
            data_path=mock_hdf5_file,
            output_dir=temp_dir / "output",
            epochs=1,
            batch_size=8,
            latent_dim=16,
            device="cpu",
            num_workers=0,
            no_validation=False,  # Enable validation
            seed=42
        )
        
        with patch('models.autoencoder.Autoencoder'), \
             patch('scripts.training.lightning_model.LitAE') as mock_lit_ae, \
             patch('models.losses.create_loss_config_from_args'), \
             patch('pytorch_lightning.Trainer') as mock_trainer_class, \
             patch('pytorch_lightning.loggers.TensorBoardLogger'), \
             patch('pytorch_lightning.callbacks.ModelCheckpoint'), \
             patch('models.summary.calculate_metrics') as mock_calc_metrics, \
             patch('models.summary.save_comparison_images'), \
             patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.savefig'):
            
            mock_model = MagicMock()
            mock_model.train_losses = [0.1]
            mock_model.model.get_loss_info.return_value = {'reconstruction': 'mse'}
            mock_lit_ae.return_value = mock_model
            
            mock_trainer = MagicMock()
            mock_trainer.current_epoch = 0
            mock_trainer_class.return_value = mock_trainer
            mock_calc_metrics.return_value = {
                'mse': 0.01, 'mse_std': 0.001,
                'psnr': 30.0, 'psnr_std': 1.0,
                'ssim': 0.95, 'ssim_std': 0.02
            }
            
            pipeline = TrainingPipeline(config)
            pipeline.run()
            
            # Verify trainer.fit was called with both train and validation loaders
            assert mock_trainer.fit.called
    
    def test_pipeline_error_handling(self, temp_dir):
        """Test pipeline error handling with invalid data path."""
        config = TrainingConfig(
            data_path=Path("/nonexistent/data.h5"),
            output_dir=temp_dir / "output",
            epochs=1,
            batch_size=8,
            device="cpu"
        )
        
        pipeline = TrainingPipeline(config)
        
        with pytest.raises(Exception):
            pipeline.run()
    
    def test_argument_parser_basic(self):
        """Test command-line argument parser with basic arguments."""
        parser = create_argument_parser()
        
        args = parser.parse_args([
            "--data", "/path/to/data.h5",
            "--output_dir", "/path/to/output"
        ])
        
        assert args.data == Path("/path/to/data.h5")
        assert args.output_dir == Path("/path/to/output")
        assert args.epochs == 50  # default
        assert args.batch == 128  # default
        assert args.latent == 128  # default
    
    def test_argument_parser_all_options(self):
        """Test command-line argument parser with all options."""
        parser = create_argument_parser()
        
        args = parser.parse_args([
            "--data", "/path/to/data.h5",
            "--output_dir", "/path/to/output",
            "--epochs", "100",
            "--batch", "64",
            "--latent", "256",
            "--lr", "0.0001",
            "--precision", "32",
            "--loss_function", "l1",
            "--lambda_act", "0.0001",
            "--device", "cuda",
            "--gpus", "2",
            "--num_workers", "8",
            "--seed", "123",
            "--no_validation",
            "--no_scheduler",
            "--compile",
            "--realtime_metrics",
            "--debug"
        ])
        
        assert args.data == Path("/path/to/data.h5")
        assert args.output_dir == Path("/path/to/output")
        assert args.epochs == 100
        assert args.batch == 64
        assert args.latent == 256
        assert args.lr == 0.0001
        assert args.precision == "32"
        assert args.loss_function == "l1"
        assert args.lambda_act == 0.0001
        assert args.device == "cuda"
        assert args.gpus == 2
        assert args.num_workers == 8
        assert args.seed == 123
        assert args.no_validation is True
        assert args.no_scheduler is True
        assert args.compile is True
        assert args.realtime_metrics is True
        assert args.debug is True
    
    @patch('scripts.training.train.TrainingPipeline')
    def test_main_function(self, mock_pipeline_class, mock_hdf5_file, temp_dir):
        """Test main function with mocked pipeline."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        
        test_args = [
            "--data", str(mock_hdf5_file),
            "--output_dir", str(temp_dir / "output"),
            "--epochs", "2",
            "--batch", "16"
        ]
        
        with patch('sys.argv', ['train.py'] + test_args):
            main()
        
        # Verify pipeline was created and run
        mock_pipeline_class.assert_called_once()
        mock_pipeline.run.assert_called_once()
        
        # Verify config was created correctly
        config_arg = mock_pipeline_class.call_args[0][0]
        assert config_arg.data_path == mock_hdf5_file
        assert config_arg.output_dir == temp_dir / "output"
        assert config_arg.epochs == 2
        assert config_arg.batch_size == 16
    
    def test_config_conversion_from_args(self, mock_hdf5_file, temp_dir):
        """Test conversion from argparse namespace to TrainingConfig."""
        parser = create_argument_parser()
        
        args = parser.parse_args([
            "--data", str(mock_hdf5_file),
            "--output_dir", str(temp_dir / "output"),
            "--epochs", "25",
            "--batch", "32",
            "--latent", "64",
            "--lr", "0.001",
            "--no_scheduler",
            "--precision", "16",
            "--loss_function", "mse",
            "--device", "cpu",
            "--seed", "456"
        ])
        
        # Convert args to config (same as in main())
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
        
        # Verify conversion worked correctly
        assert config.data_path == mock_hdf5_file
        assert config.epochs == 25
        assert config.batch_size == 32
        assert config.latent_dim == 64
        assert config.learning_rate == 0.001
        assert config.no_scheduler is True
        assert config.precision == "16"
        assert config.loss_function == "mse"
        assert config.device == "cpu"
        assert config.seed == 456