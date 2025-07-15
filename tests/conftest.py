"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def device():
    """Get the appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_diffraction_data():
    """Generate sample 4D-STEM diffraction data."""
    torch.manual_seed(42)
    
    # Create realistic diffraction patterns
    batch_size = 16
    height, width = 64, 64
    
    # Create central bright spot (direct beam)
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    
    data = []
    for i in range(batch_size):
        # Base pattern with central bright spot
        pattern = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 5**2))
        
        # Add some scattered intensity
        pattern += 0.1 * np.random.exponential(0.1, (height, width))
        
        # Add some ring patterns (diffraction rings)
        for radius in [15, 25, 35]:
            ring_mask = np.abs(np.sqrt((x - center_x)**2 + (y - center_y)**2) - radius) < 2
            pattern += 0.3 * ring_mask * np.random.exponential(0.5)
        
        # Add noise
        pattern += 0.05 * np.random.randn(height, width)
        
        # Ensure non-negative
        pattern = np.maximum(pattern, 0)
        
        data.append(pattern)
    
    return torch.tensor(np.array(data), dtype=torch.float32).unsqueeze(1)


@pytest.fixture
def sample_scan_data():
    """Generate sample scan data with spatial correlation."""
    torch.manual_seed(42)
    
    scan_height, scan_width = 8, 8
    pattern_height, pattern_width = 64, 64
    
    data = []
    
    for scan_y in range(scan_height):
        for scan_x in range(scan_width):
            # Create pattern that varies with scan position
            y, x = np.ogrid[:pattern_height, :pattern_width]
            center_y, center_x = pattern_height // 2, pattern_width // 2
            
            # Shift bright spot slightly based on scan position
            shift_y = (scan_y - scan_height // 2) * 2
            shift_x = (scan_x - scan_width // 2) * 2
            
            pattern = np.exp(-((x - center_x - shift_x)**2 + (y - center_y - shift_y)**2) / (2 * 5**2))
            
            # Add some material-dependent features
            if scan_y < scan_height // 2:  # Different material in top half
                pattern += 0.2 * np.exp(-((x - center_x + 10)**2 + (y - center_y)**2) / (2 * 3**2))
            
            # Add noise
            pattern += 0.05 * np.random.randn(pattern_height, pattern_width)
            pattern = np.maximum(pattern, 0)
            
            data.append(pattern)
    
    return torch.tensor(np.array(data), dtype=torch.float32).unsqueeze(1), (scan_height, scan_width)


@pytest.fixture(params=[32, 64, 128, 256])
def input_size(request):
    """Parametrized fixture for different input sizes."""
    return request.param


@pytest.fixture(params=[16, 32, 64])
def latent_dim(request):
    """Parametrized fixture for different latent dimensions."""
    return request.param


@pytest.fixture(params=[1, 4, 8])
def batch_size(request):
    """Parametrized fixture for different batch sizes."""
    return request.param


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests that use large input sizes as slow
        if hasattr(item, 'callspec') and hasattr(item.callspec, 'params'):
            if 'input_size' in item.callspec.params:
                if item.callspec.params['input_size'] >= 256:
                    item.add_marker(pytest.mark.slow)
        
        # Mark tests that use GPU device as gpu tests
        if 'gpu' in item.name.lower() or 'cuda' in item.name.lower():
            item.add_marker(pytest.mark.gpu)