# Training Pipeline Test Suite

This directory contains comprehensive tests for the refactored 4D-STEM autoencoder training pipeline.

## Test Structure

### Unit Tests
- **`test_training_config.py`** - Tests for the `TrainingConfig` dataclass
  - Configuration validation and type conversion
  - Device auto-detection logic
  - Parameter defaults and edge cases

- **`test_dataset_manager.py`** - Tests for the `DatasetManager` class
  - HDF5 and tensor dataset loading
  - Train/validation splitting
  - Data loader creation with multiprocessing fallback

- **`test_model_manager.py`** - Tests for the `ModelManager` class
  - Model creation with different configurations
  - Loss configuration setup
  - torch.compile integration

- **`test_trainer_manager.py`** - Tests for the `TrainerManager` class
  - PyTorch Lightning trainer setup
  - Checkpoint callback configuration
  - Accelerator and precision handling

- **`test_training_pipeline.py`** - Tests for the `TrainingPipeline` orchestrator
  - Complete pipeline execution flow
  - Logging and configuration management
  - Error handling and cleanup

### Regularization Tests
- **`test_regularization.py`** - Comprehensive regularization system tests
  - Configuration validation for all regularization types
  - Loss configuration creation and validation
  - ModelManager integration with regularizations
  - Command-line argument parsing for regularization parameters
  - Edge case handling (negative values, extreme values, precision)

- **`test_regularization_integration.py`** - End-to-end regularization tests
  - Full pipeline execution with different regularization combinations
  - Common regularization usage scenarios (light, heavy, contrastive)
  - Integration with different reconstruction loss functions
  - Progressive regularization strength testing

### Integration Tests
- **`test_integration.py`** - End-to-end pipeline tests
  - Full training runs with mock data
  - Command-line interface testing
  - Different data format handling

### Test Utilities
- **`test_utils.py`** - Test utilities and mock data generators
  - Synthetic diffraction pattern generation
  - Mock component factories
  - File path management utilities

- **`conftest.py`** - Pytest fixtures and shared configuration
  - Common test fixtures
  - Temporary file management
  - Test parametrization

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-cov pytest-mock h5py
```

### Basic Usage
```bash
# Run all tests
pytest scripts/training/tests/

# Run with verbose output
pytest scripts/training/tests/ -v

# Run only fast unit tests
pytest scripts/training/tests/ -m unit

# Run only integration tests  
pytest scripts/training/tests/ -m integration

# Skip slow tests
pytest scripts/training/tests/ -m "not slow"
```

### Coverage Reports
```bash
# Run with coverage
pytest scripts/training/tests/ --cov=scripts.training.train

# Generate HTML coverage report
pytest scripts/training/tests/ --cov=scripts.training.train --cov-report=html
```

### Specific Test Files
```bash
# Test specific component
pytest scripts/training/tests/test_training_config.py

# Test specific function
pytest scripts/training/tests/test_dataset_manager.py::TestDatasetManager::test_load_dataset_h5_file

# Run tests matching pattern
pytest scripts/training/tests/ -k "test_create_model"
```

## Test Categories

### Markers
Tests are marked with the following categories:

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Slower tests involving multiple components
- `@pytest.mark.slow` - Tests that take significant time to run
- `@pytest.mark.gpu` - Tests requiring GPU hardware (if any)

### Performance
- **Unit tests**: Should complete in < 1 second each
- **Integration tests**: May take 5-30 seconds each
- **Full test suite**: Should complete in < 2 minutes

## Mock Data

The test suite uses synthetic diffraction patterns that mimic real 4D-STEM data:

- **Central bright field disk** - Simulates transmitted beam
- **Diffraction rings** - Simulates crystalline scattering
- **Realistic noise** - Gaussian noise at appropriate levels
- **Proper intensity scaling** - Log-normal distribution matching real data

## Test Coverage Goals

- **Line Coverage**: > 90%
- **Branch Coverage**: > 85% 
- **Function Coverage**: 100%

### Current Coverage Areas
Configuration validation and device detection  
Dataset loading for HDF5 and tensor formats  
Data loader creation with multiprocessing fallback  
Model creation and compilation  
Trainer setup with different accelerators  
Complete training pipeline orchestration  
Command-line interface parsing  
Regularization system (all 5 types: Lp, contrastive, divergence, L2, KL)  
Regularization configuration and validation  
Regularization integration with full training pipeline  
Error handling and edge cases  

## Adding New Tests

### Test File Naming
- Unit tests: `test_<component_name>.py`
- Integration tests: `test_integration_<feature>.py`
- Use descriptive class names: `TestComponentName`
- Use descriptive method names: `test_specific_behavior_scenario`

### Test Structure
```python
class TestComponentName:
    """Test suite for ComponentName class."""
    
    @pytest.fixture
    def component_instance(self, basic_config):
        """Create component instance for testing."""
        return ComponentName(basic_config)
    
    def test_basic_functionality(self, component_instance):
        """Test basic functionality works as expected."""
        result = component_instance.method()
        assert result == expected_value
    
    def test_error_handling(self, component_instance):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="Expected error message"):
            component_instance.method_with_invalid_input()
```

### Mock Usage
- Use `unittest.mock.patch` for external dependencies
- Use `pytest.fixtures` for reusable test data
- Prefer dependency injection over global mocks
- Mock at the interface boundary, not internal implementation

## Continuous Integration

Tests are designed to run in CI environments:

- **No external dependencies** - All data is generated synthetically
- **Deterministic** - Fixed random seeds for reproducible results
- **Fast execution** - Optimized for CI time constraints
- **Cross-platform** - Works on Linux, macOS, and Windows

## Debugging Failed Tests

### Common Issues
1. **Import errors** - Check PYTHONPATH includes project root
2. **Mock assertions** - Verify mock call arguments match exactly
3. **Fixture scope** - Ensure fixtures have appropriate scope
4. **Random seeds** - Use fixed seeds for deterministic tests

### Debugging Commands
```bash
# Run with detailed output
pytest scripts/training/tests/ -vvv --tb=long

# Drop into debugger on failure
pytest scripts/training/tests/ --pdb

# Run specific failing test
pytest scripts/training/tests/test_file.py::TestClass::test_method -vvv
```