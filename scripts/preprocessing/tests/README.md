# Convert DM4 Test Suite

Comprehensive test suite for `convert_dm4.py` with fast-running tests for continuous development.

## Test Structure

### Core Test Files

- **`test_downsample_strategy.py`** - Tests for all downsampling algorithms
  - Unit tests for stride, bin mean, Gaussian, and FFT methods
  - Integration tests comparing different strategies
  - Edge cases and error handling

- **`test_dm4_converter.py`** - Tests for the main DM4Converter class
  - Initialization and configuration
  - Individual method testing with mocks
  - Data type conversion and validation
  - HDF5 saving functionality
  - Integration tests with mocked hyperspy data

- **`test_convert_dm4_cli.py`** - Tests for CLI and argument parsing
  - Argument validation and parsing
  - Main function behavior
  - Error handling and edge cases

- **`test_performance.py`** - Performance benchmarks and regression tests
  - Speed comparisons between downsampling methods
  - Memory usage validation
  - Performance baselines for CI

### Test Utilities

- **`run_tests.py`** - Custom test runner with timing and filtering
- **`README.md`** - This documentation file

## Running Tests

### Quick Start

```bash
# Run all tests
python scripts/preprocessing/tests/run_tests.py

# Run only fast tests (no integration tests)
python scripts/preprocessing/tests/run_tests.py --fast

# Run specific test groups
python scripts/preprocessing/tests/run_tests.py --group downsample
python scripts/preprocessing/tests/run_tests.py --group converter
python scripts/preprocessing/tests/run_tests.py --group cli
```

### Individual Test Files

```bash
# Run specific test modules
python -m pytest scripts/preprocessing/tests/test_downsample_strategy.py -v
python -m pytest scripts/preprocessing/tests/test_dm4_converter.py -v
python -m pytest scripts/preprocessing/tests/test_convert_dm4_cli.py -v

# Or with unittest
python scripts/preprocessing/tests/test_downsample_strategy.py
python scripts/preprocessing/tests/test_dm4_converter.py
```

### Performance Tests

```bash
# Run performance benchmarks
python scripts/preprocessing/tests/test_performance.py

# Performance tests show timing information and throughput
```

## Test Categories

### 🚀 Fast Tests (< 1s each)
- Unit tests for individual methods
- Argument parsing validation
- Mock-based converter tests
- Basic downsampling algorithm tests

### 🐌 Integration Tests (1-10s each)
- End-to-end conversion pipeline
- HDF5 file I/O operations
- Large data processing tests

### 📊 Performance Tests (5-30s total)
- Benchmarking different algorithms
- Memory usage validation
- Regression testing for performance

## Key Features

### Comprehensive Coverage
- **Downsampling algorithms**: All 4 strategies tested thoroughly
- **Data types**: Support for uint16, float16, float32
- **Error handling**: Invalid inputs, memory errors, file I/O issues
- **CLI interface**: All arguments and edge cases

### Fast Execution
- Most tests run in milliseconds using small synthetic data
- Mock objects for external dependencies (hyperspy, h5py)
- Separate fast/integration test categories

### Performance Monitoring
- Baseline performance measurements
- Throughput reporting (Mpixels/second)
- Memory efficiency validation
- Regression detection

### Realistic Test Data
- Synthetic data mimicking real diffraction patterns
- Multiple data sizes and dimensions
- Edge cases (single pixels, non-divisible dimensions)

## Test Design Principles

### 1. Fast by Default
- Small test data (64x64 to 256x256 patterns)
- Mock external dependencies
- Separate integration tests that use real I/O

### 2. Comprehensive Coverage
- Test all code paths and error conditions
- Validate both success and failure scenarios
- Check edge cases and boundary conditions

### 3. Maintainable
- Clear test names describing what's being tested
- Separate setup/teardown for clean tests
- Grouped by functionality for easy navigation

### 4. Reliable
- Deterministic test data where needed
- Proper cleanup of temporary files
- Independent tests that don't affect each other

## Adding New Tests

### For New Features
1. Add unit tests in the appropriate test file
2. Add integration tests if the feature involves I/O
3. Update performance tests if algorithm changes are made

### Test Naming Convention
```python
def test_method_condition_expected_result(self):
    """Test method behavior under specific condition."""
```

Examples:
- `test_stride_factor_2_correct_shape()`
- `test_converter_invalid_input_raises_error()`
- `test_cli_missing_args_exits_with_error()`

### Mock Usage
Use mocks for external dependencies to keep tests fast:
```python
@patch('hyperspy.api.load')
def test_conversion_with_mock_data(self, mock_hs_load):
    # Setup mock
    mock_signal = Mock()
    mock_hs_load.return_value = mock_signal
    # Test with controlled data
```

## Continuous Integration

### CI Test Strategy
1. **PR Tests**: Run fast tests only (`--fast` flag)
2. **Main Branch**: Run all tests including integration
3. **Nightly**: Run performance tests and update baselines

### Expected Performance
- **Fast tests**: Complete in < 30 seconds
- **All tests**: Complete in < 2 minutes
- **Performance tests**: Complete in < 1 minute

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure you're in the project root directory
cd /path/to/4DSTEM_AE_MSc
python scripts/preprocessing/tests/run_tests.py
```

**Missing Dependencies**
```bash
# Install test dependencies
pip install pytest numpy h5py
```

**Slow Tests**
```bash
# Run only fast tests during development
python scripts/preprocessing/tests/run_tests.py --fast
```

### Test Data Issues
- Tests use synthetic data, no external files required
- Temporary files are cleaned up automatically
- If tests fail due to permissions, check write access to temp directory

## Contributing

When modifying `convert_dm4.py`:

1. **Run tests first**: `python scripts/preprocessing/tests/run_tests.py --fast`
2. **Add tests for new features**: Follow the existing patterns
3. **Update performance tests**: If changing algorithms
4. **Run full test suite**: Before submitting changes

The test suite is designed to catch regressions quickly while providing confidence in the robustness of the conversion pipeline.