"""
Pytest configuration and shared fixtures for training tests.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np
import h5py
from unittest.mock import MagicMock

from scripts.training.train import TrainingConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_hdf5_data():
    """Create mock HDF5 dataset data."""
    return np.random.rand(100, 64, 64).astype(np.float32)


@pytest.fixture
def mock_hdf5_file(temp_dir, mock_hdf5_data):
    """Create a mock HDF5 file with test data."""
    data_path = temp_dir / "test_data.h5"
    
    with h5py.File(str(data_path), 'w') as f:
        dataset = f.create_dataset('patterns', data=mock_hdf5_data)
        dataset.attrs['dtype'] = 'float32'
        dataset.attrs['data_min'] = float(mock_hdf5_data.min())
        dataset.attrs['data_max'] = float(mock_hdf5_data.max())
        dataset.attrs['data_range'] = float(mock_hdf5_data.max() - mock_hdf5_data.min())
    
    return data_path


@pytest.fixture
def mock_tensor_data():
    """Create mock tensor dataset data."""
    return torch.rand(100, 1, 64, 64)


@pytest.fixture
def mock_tensor_file(temp_dir, mock_tensor_data):
    """Create a mock PyTorch tensor file with test data."""
    data_path = temp_dir / "test_data.pt"
    torch.save(mock_tensor_data, data_path)
    return data_path


@pytest.fixture
def basic_config(temp_dir):
    """Create a basic test configuration."""
    return TrainingConfig(
        data_path=temp_dir / "data.h5",
        output_dir=temp_dir / "output",
        epochs=10,
        batch_size=32,
        latent_dim=64,
        device="cpu",
        seed=42
    )


@pytest.fixture
def cpu_config(temp_dir):
    """Create a CPU-specific test configuration."""
    return TrainingConfig(
        data_path=temp_dir / "data.h5",
        output_dir=temp_dir / "output",
        epochs=5,
        batch_size=16,
        latent_dim=32,
        device="cpu",
        num_workers=0,  # Disable multiprocessing for tests
        seed=42
    )


@pytest.fixture
def hdf5_config(mock_hdf5_file, temp_dir):
    """Create test configuration for HDF5 data."""
    return TrainingConfig(
        data_path=mock_hdf5_file,
        output_dir=temp_dir / "output",
        epochs=2,
        batch_size=8,
        latent_dim=16,
        device="cpu",
        num_workers=0,
        seed=42
    )


@pytest.fixture
def tensor_config(mock_tensor_file, temp_dir):
    """Create test configuration for tensor data."""
    return TrainingConfig(
        data_path=mock_tensor_file,
        output_dir=temp_dir / "output", 
        epochs=2,
        batch_size=8,
        latent_dim=16,
        device="cpu",
        num_workers=0,
        seed=42
    )


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.train_losses = [0.1, 0.05, 0.02]
    model.model.get_loss_info.return_value = {'reconstruction': 'mse'}
    return model


@pytest.fixture
def mock_trainer():
    """Create a mock PyTorch Lightning trainer."""
    trainer = MagicMock()
    trainer.current_epoch = 4
    return trainer


@pytest.fixture
def mock_dataset():
    """Create a mock dataset."""
    dataset = MagicMock()
    dataset.__len__.return_value = 100
    sample_data = torch.rand(1, 64, 64)
    dataset.__getitem__.return_value = (sample_data,)
    return dataset


@pytest.fixture
def mock_dataloader():
    """Create a mock data loader."""
    dataloader = MagicMock()
    batch_data = (torch.rand(8, 1, 64, 64),)
    dataloader.__iter__.return_value = iter([batch_data])
    return dataloader


@pytest.fixture(autouse=True)
def setup_torch_settings():
    """Setup PyTorch settings for consistent testing."""
    # Set deterministic behavior for reproducible tests
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Disable gradient computation by default in tests
    with torch.no_grad():
        yield


@pytest.fixture
def mock_loss_config():
    """Create a mock loss configuration."""
    return {
        'reconstruction_loss': 'mse',
        'regularization_losses': {
            'lp_reg': 1e-5,
            'contrastive': 0.0,
            'divergence': 0.0
        }
    }


@pytest.fixture
def mock_metrics():
    """Create mock evaluation metrics."""
    return {
        'mse': 0.01,
        'mse_std': 0.001,
        'psnr': 30.0,
        'psnr_std': 1.0,
        'ssim': 0.95,
        'ssim_std': 0.02
    }


@pytest.fixture
def mock_diffraction_metrics():
    """Create mock diffraction-specific metrics."""
    return {
        'peak_preservation': 0.98,
        'log_correlation': 0.96,
        'range_preservation': 0.97,
        'center_region_mse': 0.005
    }


# Test configuration for different scenarios
@pytest.fixture(params=[
    {"device": "cpu", "precision": "32"},
    {"device": "cpu", "precision": "16"},
])
def device_precision_config(request, temp_dir):
    """Parameterized fixture for different device/precision combinations."""
    return TrainingConfig(
        data_path=temp_dir / "data.h5",
        output_dir=temp_dir / "output",
        device=request.param["device"],
        precision=request.param["precision"],
        epochs=1,
        batch_size=4,
        latent_dim=8,
        seed=42
    )


@pytest.fixture(params=[True, False])
def validation_config(request, temp_dir):
    """Parameterized fixture for validation enabled/disabled."""
    return TrainingConfig(
        data_path=temp_dir / "data.h5",
        output_dir=temp_dir / "output",
        no_validation=request.param,
        epochs=1,
        batch_size=4,
        seed=42
    )


@pytest.fixture(params=[True, False])
def normalization_config(request, temp_dir):
    """Parameterized fixture for normalization enabled/disabled."""
    return TrainingConfig(
        data_path=temp_dir / "data.h5",
        output_dir=temp_dir / "output",
        use_normalization=request.param,
        epochs=1,
        batch_size=4,
        seed=42
    )


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location/name."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if "slow" in item.nodeid or item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.slow)