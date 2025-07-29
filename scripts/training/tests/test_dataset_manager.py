"""
Tests for DatasetManager class.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import torch
import h5py
import numpy as np

from scripts.training.train import TrainingConfig, DatasetManager


class TestDatasetManager:
    """Test suite for DatasetManager class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return TrainingConfig(
            data_path=Path("/path/to/data.h5"),
            output_dir=Path("/path/to/output"),
            batch_size=32,
            num_workers=2,
            use_normalization=True,
            no_validation=False
        )
    
    @pytest.fixture
    def dataset_manager(self, config):
        """Create a DatasetManager instance."""
        return DatasetManager(config)
    
    def test_init(self, config):
        """Test DatasetManager initialization."""
        manager = DatasetManager(config)
        assert manager.config == config
        assert hasattr(manager, 'logger')
    
    @patch('scripts.training.dataset.HDF5Dataset')
    def test_load_dataset_h5_file(self, mock_hdf5_dataset, dataset_manager):
        """Test loading HDF5 dataset."""
        # Setup mock dataset
        mock_dataset = MagicMock()
        mock_sample = torch.rand(1, 256, 256)
        mock_dataset.__getitem__.return_value = (mock_sample,)
        mock_hdf5_dataset.return_value = mock_dataset
        
        dataset, detected_size = dataset_manager.load_dataset()
        
        mock_hdf5_dataset.assert_called_once_with(
            dataset_manager.config.data_path,
            use_normalization=True
        )
        assert dataset == mock_dataset
        assert detected_size == 256
    
    @patch('torch.load')
    @patch('torch.utils.data.TensorDataset')
    def test_load_dataset_pt_file(self, mock_tensor_dataset, mock_torch_load, dataset_manager):
        """Test loading PyTorch tensor dataset."""
        # Change config to use .pt file
        dataset_manager.config.data_path = Path("/path/to/data.pt")
        
        # Setup mock data
        mock_data = torch.rand(1000, 1, 256, 256)
        mock_torch_load.return_value = mock_data
        mock_dataset = MagicMock()
        mock_tensor_dataset.return_value = mock_dataset
        
        dataset, detected_size = dataset_manager.load_dataset()
        
        mock_torch_load.assert_called_once_with(Path("/path/to/data.pt"))
        mock_tensor_dataset.assert_called_once_with(mock_data)
        assert dataset == mock_dataset
        assert detected_size == 256
    
    def test_create_train_val_split_with_validation(self, dataset_manager):
        """Test train/validation split creation."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 1000
        
        with patch('torch.randperm') as mock_randperm, \
             patch('torch.utils.data.Subset') as mock_subset:
            
            mock_randperm.return_value = torch.arange(1000)
            mock_train_ds = MagicMock()
            mock_val_ds = MagicMock()
            mock_subset.side_effect = [mock_train_ds, mock_val_ds]
            
            train_ds, val_ds = dataset_manager.create_train_val_split(mock_dataset)
            
            assert train_ds == mock_train_ds
            assert val_ds == mock_val_ds
            # Should call Subset twice with train/val indices
            assert mock_subset.call_count == 2
    
    def test_create_train_val_split_no_validation(self, dataset_manager):
        """Test no validation split when disabled."""
        dataset_manager.config.no_validation = True
        
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 1000
        
        train_ds, val_ds = dataset_manager.create_train_val_split(mock_dataset)
        
        assert train_ds == mock_dataset
        assert val_ds is None
    
    @patch('torch.utils.data.DataLoader')
    @patch('multiprocessing.get_context')
    def test_create_data_loaders_multiprocessing(self, mock_mp_context, mock_dataloader, dataset_manager):
        """Test data loader creation with multiprocessing."""
        mock_train_ds = MagicMock()
        mock_val_ds = MagicMock()
        mock_train_dl = MagicMock()
        mock_val_dl = MagicMock()
        
        mock_dataloader.side_effect = [mock_train_dl, mock_val_dl]
        mock_context = MagicMock()
        mock_mp_context.return_value = mock_context
        
        train_dl, val_dl = dataset_manager.create_data_loaders(mock_train_ds, mock_val_ds)
        
        assert train_dl == mock_train_dl
        assert val_dl == mock_val_dl
        assert mock_dataloader.call_count == 2
        
        # Check that multiprocessing context was used
        mock_mp_context.assert_called_with('spawn')
    
    @patch('torch.utils.data.DataLoader')
    def test_create_data_loaders_fallback_to_single_threaded(self, mock_dataloader, dataset_manager):
        """Test fallback to single-threaded when multiprocessing fails."""
        mock_train_ds = MagicMock()
        mock_val_ds = MagicMock()
        
        # First call raises exception, second call succeeds
        mock_dataloader.side_effect = [Exception("Multiprocessing failed"), MagicMock(), MagicMock()]
        
        train_dl, val_dl = dataset_manager.create_data_loaders(mock_train_ds, mock_val_ds)
        
        # Should be called 3 times: failed attempt + 2 fallback calls
        assert mock_dataloader.call_count == 3
        
        # Check fallback kwargs don't include multiprocessing settings
        fallback_call = mock_dataloader.call_args_list[-1]  # Last call
        fallback_kwargs = fallback_call[1]
        assert fallback_kwargs['num_workers'] == 0
        assert 'multiprocessing_context' not in fallback_kwargs
        assert 'prefetch_factor' not in fallback_kwargs
        assert fallback_kwargs['persistent_workers'] is False
    
    @patch('torch.utils.data.DataLoader')
    def test_create_data_loaders_no_validation(self, mock_dataloader, dataset_manager):
        """Test data loader creation with no validation dataset."""
        mock_train_ds = MagicMock()
        mock_train_dl = MagicMock()
        
        mock_dataloader.return_value = mock_train_dl
        
        train_dl, val_dl = dataset_manager.create_data_loaders(mock_train_ds, None)
        
        assert train_dl == mock_train_dl
        assert val_dl is None
        assert mock_dataloader.call_count == 1
    
    @patch('torch.cuda.is_available')
    def test_create_data_loaders_pin_memory_cuda_available(self, mock_cuda_available, dataset_manager):
        """Test pin_memory is enabled when CUDA is available."""
        mock_cuda_available.return_value = True
        dataset_manager.config.pin_memory = True
        
        mock_train_ds = MagicMock()
        
        with patch('torch.utils.data.DataLoader') as mock_dataloader:
            dataset_manager.create_data_loaders(mock_train_ds, None)
            
            # Check that pin_memory was set to True
            call_kwargs = mock_dataloader.call_args[1]
            assert call_kwargs['pin_memory'] is True
    
    @patch('torch.cuda.is_available')
    def test_create_data_loaders_pin_memory_cuda_unavailable(self, mock_cuda_available, dataset_manager):
        """Test pin_memory is disabled when CUDA is unavailable."""
        mock_cuda_available.return_value = False
        dataset_manager.config.pin_memory = True
        
        mock_train_ds = MagicMock()
        
        with patch('torch.utils.data.DataLoader') as mock_dataloader:
            dataset_manager.create_data_loaders(mock_train_ds, None)
            
            # Check that pin_memory was set to False
            call_kwargs = mock_dataloader.call_args[1]
            assert call_kwargs['pin_memory'] is False
    
    def test_create_data_loaders_batch_size_and_workers(self, dataset_manager):
        """Test that batch size and num_workers are properly set."""
        dataset_manager.config.batch_size = 64
        dataset_manager.config.num_workers = 8
        
        mock_train_ds = MagicMock()
        
        with patch('torch.utils.data.DataLoader') as mock_dataloader:
            dataset_manager.create_data_loaders(mock_train_ds, None)
            
            call_kwargs = mock_dataloader.call_args[1]
            assert call_kwargs['batch_size'] == 64
            assert call_kwargs['num_workers'] == 8
    
    @patch('torch.utils.data.DataLoader')
    def test_create_data_loaders_debug_mode(self, mock_dataloader, dataset_manager):
        """Test debug mode testing with multiprocessing."""
        dataset_manager.config.debug = True
        dataset_manager.config.num_workers = 4
        
        mock_train_ds = MagicMock()
        mock_train_dl = MagicMock()
        mock_batch = (torch.rand(32, 1, 256, 256),)
        mock_train_dl.__iter__.return_value = iter([mock_batch])
        
        mock_dataloader.return_value = mock_train_dl
        
        train_dl, val_dl = dataset_manager.create_data_loaders(mock_train_ds, None)
        
        # In debug mode, should test the data loader
        assert train_dl == mock_train_dl
        mock_train_dl.__iter__.assert_called_once()