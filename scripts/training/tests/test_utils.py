"""
Test utilities and mock data generators for training tests.
"""
import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import tempfile
import json
from unittest.mock import MagicMock

from scripts.training.train import TrainingConfig


class MockDataGenerator:
    """Utility class for generating mock training data."""
    
    @staticmethod
    def create_synthetic_diffraction_data(
        n_patterns: int = 100,
        height: int = 64,
        width: int = 64,
        noise_level: float = 0.1,
        seed: int = 42
    ) -> np.ndarray:
        """
        Create synthetic 4D-STEM diffraction patterns for testing.
        
        Args:
            n_patterns: Number of diffraction patterns to generate
            height: Height of each pattern
            width: Width of each pattern
            noise_level: Amount of noise to add (0-1)
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_patterns, height, width) with synthetic diffraction data
        """
        np.random.seed(seed)
        
        patterns = []
        for i in range(n_patterns):
            # Create synthetic diffraction pattern with central disk and rings
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height // 2, width // 2
            
            # Central disk (bright field)
            central_disk = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (height/8)**2)
            
            # Add some diffraction rings
            ring1 = 0.3 * np.exp(-((np.sqrt((x - center_x)**2 + (y - center_y)**2) - height/4)**2) / 4)
            ring2 = 0.2 * np.exp(-((np.sqrt((x - center_x)**2 + (y - center_y)**2) - height/3)**2) / 3)
            
            # Combine components
            pattern = central_disk + ring1 + ring2
            
            # Add some random variation per pattern
            pattern += np.random.uniform(0, 0.1, pattern.shape)
            
            # Add noise
            if noise_level > 0:
                pattern += np.random.normal(0, noise_level, pattern.shape)
            
            # Ensure positive values (like real diffraction data)
            pattern = np.maximum(pattern, 0.001)
            
            patterns.append(pattern)
        
        return np.array(patterns, dtype=np.float32)
    
    @staticmethod
    def create_hdf5_dataset(
        file_path: Path,
        n_patterns: int = 100,
        height: int = 64,
        width: int = 64,
        dtype: str = "float32",
        with_metadata: bool = True
    ) -> Path:
        """
        Create an HDF5 dataset file for testing.
        
        Args:
            file_path: Path where to save the HDF5 file
            n_patterns: Number of patterns to generate
            height: Height of each pattern
            width: Width of each pattern
            dtype: Data type for the patterns
            with_metadata: Whether to include metadata attributes
            
        Returns:
            Path to the created HDF5 file
        """
        data = MockDataGenerator.create_synthetic_diffraction_data(
            n_patterns, height, width
        )
        
        if dtype == "uint16":
            # Convert to uint16 (quantized data)
            data_min, data_max = data.min(), data.max()
            data_range = data_max - data_min
            data = ((data - data_min) / data_range * 65535).astype(np.uint16)
        elif dtype == "float16":
            data = data.astype(np.float16)
        
        with h5py.File(str(file_path), 'w') as f:
            dataset = f.create_dataset('patterns', data=data)
            
            if with_metadata:
                dataset.attrs['dtype'] = dtype
                dataset.attrs['data_min'] = float(data.min())
                dataset.attrs['data_max'] = float(data.max())
                dataset.attrs['data_range'] = float(data.max() - data.min())
                dataset.attrs['n_patterns'] = n_patterns
                dataset.attrs['height'] = height
                dataset.attrs['width'] = width
        
        return file_path
    
    @staticmethod
    def create_tensor_dataset(
        file_path: Path,
        n_patterns: int = 100,
        height: int = 64,
        width: int = 64,
        channels: int = 1
    ) -> Path:
        """
        Create a PyTorch tensor dataset file for testing.
        
        Args:
            file_path: Path where to save the tensor file
            n_patterns: Number of patterns to generate
            height: Height of each pattern
            width: Width of each pattern
            channels: Number of channels (usually 1 for diffraction data)
            
        Returns:
            Path to the created tensor file
        """
        data = MockDataGenerator.create_synthetic_diffraction_data(
            n_patterns, height, width
        )
        
        # Add channel dimension
        if channels == 1:
            data = data[:, np.newaxis, :, :]  # Shape: (n, 1, h, w)
        
        tensor_data = torch.from_numpy(data).float()
        torch.save(tensor_data, file_path)
        
        return file_path
    
    @staticmethod
    def create_normalization_stats_file(
        file_path: Path,
        log_mean: float = 0.5,
        log_std: float = 0.2,
        n_patterns: int = 1000
    ) -> Path:
        """
        Create a normalization statistics JSON file for testing.
        
        Args:
            file_path: Path where to save the stats file
            log_mean: Global log-space mean
            log_std: Global log-space standard deviation
            n_patterns: Number of patterns used for stats
            
        Returns:
            Path to the created stats file
        """
        stats = {
            'log_mean': log_mean,
            'log_std': log_std,
            'total_patterns_used': n_patterns,
            'pixels_sampled': n_patterns * 64 * 64 // 50,  # Every 50th pixel
            'pixel_stride': 50,
            'chunk_size': 100,
            'computed_on': "2024-01-01T12:00:00",
            'method': 'full_dataset_streaming_welford'
        }
        
        with open(file_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return file_path


class MockComponents:
    """Factory for creating mock training components."""
    
    @staticmethod
    def create_mock_dataset(
        length: int = 100,
        sample_shape: Tuple[int, ...] = (1, 64, 64),
        has_normalization_params: bool = False
    ) -> MagicMock:
        """Create a mock dataset for testing."""
        dataset = MagicMock()
        dataset.__len__.return_value = length
        
        # Create sample data
        sample_data = torch.rand(*sample_shape)
        dataset.__getitem__.return_value = (sample_data,)
        
        if has_normalization_params:
            dataset.global_log_mean = 0.5
            dataset.global_log_std = 0.2
            dataset.use_normalization = True
        
        return dataset
    
    @staticmethod
    def create_mock_dataloader(
        batch_size: int = 8,
        dataset_length: int = 100,
        sample_shape: Tuple[int, ...] = (1, 64, 64)
    ) -> MagicMock:
        """Create a mock data loader for testing."""
        dataloader = MagicMock()
        
        # Create batch data
        batch_data = (torch.rand(batch_size, *sample_shape),)
        dataloader.__iter__.return_value = iter([batch_data])
        dataloader.__len__.return_value = dataset_length // batch_size
        
        return dataloader
    
    @staticmethod
    def create_mock_model(
        train_losses: Optional[list] = None,
        loss_info: Optional[Dict[str, str]] = None
    ) -> MagicMock:
        """Create a mock Lightning model for testing."""
        model = MagicMock()
        
        if train_losses is None:
            train_losses = [0.1, 0.05, 0.02, 0.01]
        model.train_losses = train_losses
        
        if loss_info is None:
            loss_info = {'reconstruction': 'mse'}
        model.model.get_loss_info.return_value = loss_info
        
        # Mock forward pass
        def mock_forward(x):
            return torch.rand_like(x)
        model.side_effect = mock_forward
        
        return model
    
    @staticmethod
    def create_mock_trainer(
        current_epoch: int = 4,
        fit_success: bool = True
    ) -> MagicMock:
        """Create a mock PyTorch Lightning trainer for testing."""
        trainer = MagicMock()
        trainer.current_epoch = current_epoch
        
        if not fit_success:
            trainer.fit.side_effect = Exception("Training failed")
        
        return trainer
    
    @staticmethod
    def create_mock_logger() -> MagicMock:
        """Create a mock logger for testing."""
        logger = MagicMock()
        return logger


class DataPathsHelper:
    """Utility class for managing test data file paths."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize with base directory for test data."""
        if base_dir is None:
            base_dir = Path(tempfile.mkdtemp())
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_hdf5_path(self, name: str = "test_data") -> Path:
        """Get path for HDF5 test file."""
        return self.base_dir / f"{name}.h5"
    
    def get_tensor_path(self, name: str = "test_data") -> Path:
        """Get path for tensor test file."""
        return self.base_dir / f"{name}.pt"
    
    def get_stats_path(self, data_name: str = "test_data") -> Path:
        """Get path for normalization stats file."""
        return self.base_dir / f"{data_name}_normalization_stats.json"
    
    def get_output_dir(self, name: str = "output") -> Path:
        """Get path for output directory."""
        output_dir = self.base_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def cleanup(self):
        """Clean up all test files."""
        import shutil
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)


class ConfigFactory:
    """Factory for creating different test configurations."""
    
    @staticmethod
    def create_minimal_config(data_path: Path, output_dir: Path) -> TrainingConfig:
        """Create minimal configuration for basic testing."""
        return TrainingConfig(
            data_path=data_path,
            output_dir=output_dir,
            epochs=1,
            batch_size=4,
            latent_dim=8,
            device="cpu",
            num_workers=0,
            seed=42
        )
    
    @staticmethod
    def create_fast_config(data_path: Path, output_dir: Path) -> TrainingConfig:
        """Create configuration optimized for fast testing."""
        return TrainingConfig(
            data_path=data_path,
            output_dir=output_dir,
            epochs=2,
            batch_size=8,
            latent_dim=16,
            device="cpu",
            num_workers=0,
            no_validation=True,  # Skip validation for speed
            seed=42
        )
    
    @staticmethod
    def create_full_config(data_path: Path, output_dir: Path) -> TrainingConfig:
        """Create comprehensive configuration for integration testing."""
        return TrainingConfig(
            data_path=data_path,
            output_dir=output_dir,
            epochs=5,
            batch_size=16,
            latent_dim=32,
            learning_rate=1e-3,
            device="cpu",
            num_workers=0,
            use_normalization=True,
            no_validation=False,
            precision="32",
            loss_function="mse",
            lambda_act=1e-5,
            save_every_n_epochs=1,
            seed=42
        )


def assert_file_exists(file_path: Path, file_type: str = "file"):
    """Assert that a file exists and raise descriptive error if not."""
    assert file_path.exists(), f"Expected {file_type} does not exist: {file_path}"


def assert_directory_structure(base_dir: Path, expected_files: list):
    """Assert that directory contains expected files."""
    assert base_dir.exists(), f"Base directory does not exist: {base_dir}"
    
    for expected_file in expected_files:
        file_path = base_dir / expected_file
        assert file_path.exists(), f"Expected file not found: {file_path}"


def get_tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """Get basic statistics for a tensor."""
    return {
        'mean': float(tensor.mean()),
        'std': float(tensor.std()),
        'min': float(tensor.min()),
        'max': float(tensor.max()),
        'shape': list(tensor.shape)
    }