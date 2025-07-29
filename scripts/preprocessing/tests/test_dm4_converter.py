#!/usr/bin/env python3
"""Tests for DM4Converter class in convert_dm4.py"""
import unittest
import numpy as np
import tempfile
import json
import h5py
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory to path to import convert_dm4
sys.path.insert(0, str(Path(__file__).parent.parent))
from convert_dm4 import DM4Converter, DEFAULT_CHUNK_SIZE, MAX_MEMORY_GB, BYTES_PER_FLOAT16


class TestDM4ConverterInit(unittest.TestCase):
    """Test DM4Converter initialization and configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_path = self.temp_dir / "test_input.dm4"
        self.output_path = self.temp_dir / "test_output.h5"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        converter = DM4Converter(self.input_path, self.output_path)
        
        self.assertEqual(converter.input_path, self.input_path)
        self.assertEqual(converter.output_path, self.output_path)
        self.assertEqual(converter.downsample, 1)
        self.assertEqual(converter.mode, "bin")
        self.assertEqual(converter.sigma, 0.8)
        self.assertEqual(converter.scan_step, 1)
        self.assertEqual(converter.chunk_size, DEFAULT_CHUNK_SIZE)
        self.assertEqual(converter.dtype, "float16")
        self.assertEqual(converter.compression_level, 4)
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        converter = DM4Converter(
            self.input_path, self.output_path,
            downsample=4, mode="gauss", sigma=1.2,
            scan_step=2, chunk_size=256,
            dtype="float32", compression_level=9
        )
        
        self.assertEqual(converter.downsample, 4)
        self.assertEqual(converter.mode, "gauss")
        self.assertEqual(converter.sigma, 1.2)
        self.assertEqual(converter.scan_step, 2)
        self.assertEqual(converter.chunk_size, 256)
        self.assertEqual(converter.dtype, "float32")
        self.assertEqual(converter.compression_level, 9)
    
    def test_ensure_h5_extension(self):
        """Test automatic H5 extension handling."""
        # Test with .zarr extension (should be changed to .h5)
        zarr_output = self.temp_dir / "test_output.zarr"
        converter = DM4Converter(self.input_path, zarr_output)
        self.assertEqual(converter.output_path.suffix, ".h5")
        
        # Test with .h5 extension (should be unchanged)
        h5_output = self.temp_dir / "test_output.h5"
        converter = DM4Converter(self.input_path, h5_output)
        self.assertEqual(converter.output_path.suffix, ".h5")
        
        # Test with no extension (should get .h5)
        no_ext_output = self.temp_dir / "test_output"
        converter = DM4Converter(self.input_path, no_ext_output)
        self.assertEqual(converter.output_path.suffix, ".h5")


class TestDM4ConverterMethods(unittest.TestCase):
    """Test individual methods of DM4Converter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_path = self.temp_dir / "test_input.dm4"
        self.output_path = self.temp_dir / "test_output.h5"
        self.converter = DM4Converter(self.input_path, self.output_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_calculate_processing_dimensions_no_processing(self):
        """Test dimension calculation with no downsampling or scan step."""
        self.converter.original_shape = (10, 12, 64, 64)
        
        ny, nx, qy_final, qx_final, total_patterns = self.converter._calculate_processing_dimensions()
        
        self.assertEqual(ny, 10)
        self.assertEqual(nx, 12)
        self.assertEqual(qy_final, 64)
        self.assertEqual(qx_final, 64)
        self.assertEqual(total_patterns, 120)
    
    def test_calculate_processing_dimensions_with_scan_step(self):
        """Test dimension calculation with scan step."""
        self.converter.original_shape = (20, 24, 64, 64)
        self.converter.scan_step = 2
        
        ny, nx, qy_final, qx_final, total_patterns = self.converter._calculate_processing_dimensions()
        
        self.assertEqual(ny, 10)  # 20 // 2
        self.assertEqual(nx, 12)  # 24 // 2
        self.assertEqual(qy_final, 64)
        self.assertEqual(qx_final, 64)
        self.assertEqual(total_patterns, 120)
    
    def test_calculate_processing_dimensions_with_downsample(self):
        """Test dimension calculation with downsampling."""
        self.converter.original_shape = (10, 12, 64, 64)
        self.converter.downsample = 4
        
        ny, nx, qy_final, qx_final, total_patterns = self.converter._calculate_processing_dimensions()
        
        self.assertEqual(ny, 10)
        self.assertEqual(nx, 12)
        self.assertEqual(qy_final, 16)  # 64 // 4
        self.assertEqual(qx_final, 16)  # 64 // 4
        self.assertEqual(total_patterns, 120)
    
    def test_calculate_safe_chunk_size_small_patterns(self):
        """Test chunk size calculation for small patterns."""
        # Small patterns should allow large chunk sizes
        safe_chunk_size = self.converter._calculate_safe_chunk_size(32, 32)
        
        # Should be limited by converter.chunk_size, not memory
        self.assertEqual(safe_chunk_size, self.converter.chunk_size)
    
    def test_calculate_safe_chunk_size_large_patterns(self):
        """Test chunk size calculation for large patterns."""
        # Large patterns should be limited by memory
        safe_chunk_size = self.converter._calculate_safe_chunk_size(512, 512)
        
        # Should be less than default chunk size due to memory constraints
        max_expected = int(MAX_MEMORY_GB * 1024**3 / (512 * 512 * BYTES_PER_FLOAT16))
        self.assertEqual(safe_chunk_size, min(self.converter.chunk_size, max_expected))
        self.assertLessEqual(safe_chunk_size, self.converter.chunk_size)
    
    @patch('h5py.File')
    def test_save_metadata(self, mock_h5py):
        """Test metadata saving functionality."""
        # Set up converter state
        self.converter.original_shape = (10, 12, 64, 64)
        self.converter.final_shape = (120, 32, 32)
        self.converter.data_stats = {
            'data_min': 0.0,
            'data_max': 1.0,
            'data_range': 1.0
        }
        
        self.converter._save_metadata()
        
        # Check that JSON file was created
        expected_path = self.output_path.with_suffix('.json')
        self.assertTrue(expected_path.exists())
        
        # Check JSON content
        with open(expected_path, 'r') as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata['original_shape'], [10, 12, 64, 64])
        self.assertEqual(metadata['final_shape'], [120, 32, 32])
        self.assertEqual(metadata['dtype'], 'float16')
        self.assertEqual(metadata['downsample'], 1)
        self.assertEqual(metadata['data_min'], 0.0)
    
    def test_report_compression_stats_no_file(self):
        """Test compression stats when output file doesn't exist."""
        # Should not raise exception, just print warning
        self.converter._report_compression_stats()
        # Test passes if no exception is raised
    
    def test_report_compression_stats_with_file(self):
        """Test compression stats calculation with actual file."""
        # Create a dummy file
        self.output_path.touch()
        self.output_path.write_bytes(b"dummy data")
        
        # Set up converter state for stats calculation
        self.converter.final_shape = (100, 32, 32)
        
        # Should not raise exception
        self.converter._report_compression_stats()
        # Test passes if no exception is raised


class TestDM4ConverterDataTypes(unittest.TestCase):
    """Test data type conversion functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_path = self.temp_dir / "test_input.dm4"
        self.output_path = self.temp_dir / "test_output.h5"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('dask.array.Array')
    def test_convert_dtype_float16(self, mock_dask_array):
        """Test conversion to float16."""
        converter = DM4Converter(self.input_path, self.output_path, dtype="float16")
        
        # Mock dask array
        mock_data = Mock()
        mock_data.astype = Mock(return_value="converted_data")
        
        result = converter._convert_dtype(mock_data)
        
        mock_data.astype.assert_called_once_with("float16")
        self.assertEqual(result, "converted_data")
    
    @patch('dask.array.Array')
    def test_convert_dtype_float32(self, mock_dask_array):
        """Test conversion to float32."""
        converter = DM4Converter(self.input_path, self.output_path, dtype="float32")
        
        # Mock dask array
        mock_data = Mock()
        mock_data.astype = Mock(return_value="converted_data")
        
        result = converter._convert_dtype(mock_data)
        
        mock_data.astype.assert_called_once_with("float32")
        self.assertEqual(result, "converted_data")
    
    @patch('dask.array.Array')
    def test_convert_dtype_uint16(self, mock_dask_array):
        """Test conversion to uint16."""
        converter = DM4Converter(self.input_path, self.output_path, dtype="uint16")
        
        # Mock dask array
        mock_data = Mock()
        mock_scaled = Mock()
        mock_scaled.astype = Mock(return_value="converted_data")
        mock_data.__mul__ = Mock(return_value=mock_scaled)
        
        result = converter._convert_dtype(mock_data)
        
        mock_data.__mul__.assert_called_once_with(65535)
        mock_scaled.astype.assert_called_once_with("uint16")
        self.assertEqual(result, "converted_data")


class TestDM4ConverterValidation(unittest.TestCase):
    """Test data validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_path = self.temp_dir / "test_input.dm4"
        self.output_path = self.temp_dir / "test_output.h5"
        self.converter = DM4Converter(self.input_path, self.output_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_data_success(self):
        """Test successful data validation."""
        # Mock dask array that validates successfully
        mock_data = Mock()
        mock_data.shape = (100, 32, 32)
        mock_data.chunks = ((50, 50), (32,), (32,))
        
        # Mock successful compute
        mock_slice = np.random.rand(2, 32, 32).astype(np.float16)
        mock_data.__getitem__ = Mock(return_value=Mock())
        mock_data.__getitem__.return_value.compute = Mock(return_value=mock_slice)
        
        result = self.converter._validate_data(mock_data)
        
        self.assertTrue(result)
        mock_data.__getitem__.assert_called_once()
    
    def test_validate_data_failure(self):
        """Test data validation failure."""
        # Mock dask array that fails validation
        mock_data = Mock()
        mock_data.shape = (100, 32, 32)
        mock_data.chunks = ((50, 50), (32,), (32,))
        
        # Mock failed compute
        mock_data.__getitem__ = Mock(return_value=Mock())
        mock_data.__getitem__.return_value.compute = Mock(side_effect=RuntimeError("Test error"))
        
        result = self.converter._validate_data(mock_data)
        
        self.assertFalse(result)


class TestDM4ConverterHDF5(unittest.TestCase):
    """Test HDF5 saving functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_path = self.temp_dir / "test_input.dm4"
        self.output_path = self.temp_dir / "test_output.h5"
        self.converter = DM4Converter(self.input_path, self.output_path)
        self.converter.original_shape = (10, 12, 64, 64)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_hdf5_basic(self):
        """Test basic HDF5 saving functionality."""
        # Create mock processed data
        test_data = np.random.rand(120, 32, 32).astype(np.float16)
        mock_dask_data = Mock()
        mock_dask_data.compute = Mock(return_value=test_data)
        
        # Save HDF5
        data_stats = self.converter._save_hdf5(mock_dask_data)
        
        # Check file was created
        self.assertTrue(self.output_path.exists())
        
        # Check data stats were returned
        self.assertIn('data_min', data_stats)
        self.assertIn('data_max', data_stats)
        self.assertIn('data_range', data_stats)
        
        # Check HDF5 file contents
        with h5py.File(self.output_path, 'r') as f:
            self.assertIn('patterns', f)
            saved_data = f['patterns'][:]
            np.testing.assert_array_equal(saved_data, test_data)
            
            # Check attributes
            attrs = dict(f['patterns'].attrs)
            self.assertEqual(list(attrs['original_shape']), [10, 12, 64, 64])
            self.assertEqual(list(attrs['final_shape']), [120, 32, 32])
            self.assertEqual(attrs['dtype'], 'float16')
    
    def test_save_hdf5_overwrites_existing(self):
        """Test that existing files are overwritten."""
        # Create existing file
        self.output_path.touch()
        self.output_path.write_text("existing content")
        
        # Create mock data and save
        test_data = np.random.rand(10, 8, 8).astype(np.float32)
        mock_dask_data = Mock()
        mock_dask_data.compute = Mock(return_value=test_data)
        
        self.converter._save_hdf5(mock_dask_data)
        
        # File should be HDF5, not text
        with h5py.File(self.output_path, 'r') as f:
            self.assertIn('patterns', f)


class TestDM4ConverterIntegration(unittest.TestCase):
    """Integration tests for DM4Converter using mocked hyperspy data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_path = self.temp_dir / "test_input.dm4"
        self.output_path = self.temp_dir / "test_output.h5"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('hyperspy.api.load')
    def test_convert_pipeline_basic(self, mock_hs_load):
        """Test the complete conversion pipeline with mocked data."""
        # Create mock hyperspy signal
        mock_signal = Mock()
        test_data_shape = (4, 5, 16, 16)  # Small test data
        test_data = np.random.rand(*test_data_shape).astype(np.float32)
        
        # Mock dask array
        import dask.array as da
        mock_dask_array = da.from_array(test_data, chunks=(2, 2, 16, 16))
        
        mock_signal.data = mock_dask_array
        mock_hs_load.return_value = mock_signal
        
        # Create converter
        converter = DM4Converter(
            self.input_path, self.output_path,
            downsample=2, mode="bin",
            dtype="float16", compression_level=1
        )
        
        # Run conversion
        converter.convert()
        
        # Check outputs
        self.assertTrue(self.output_path.exists())
        self.assertTrue(self.output_path.with_suffix('.json').exists())
        
        # Verify HDF5 content
        with h5py.File(self.output_path, 'r') as f:
            saved_data = f['patterns'][:]
            expected_shape = (20, 8, 8)  # 4*5 patterns, 16/2 x 16/2 each
            self.assertEqual(saved_data.shape, expected_shape)
        
        # Verify metadata
        with open(self.output_path.with_suffix('.json'), 'r') as f:
            metadata = json.load(f)
        self.assertEqual(metadata['original_shape'], list(test_data_shape))
        self.assertEqual(metadata['downsample'], 2)
        self.assertEqual(metadata['mode'], 'bin')
    
    @patch('hyperspy.api.load')
    def test_convert_with_scan_step(self, mock_hs_load):
        """Test conversion with scan step parameter."""
        # Mock signal with larger scan grid
        mock_signal = Mock()
        test_data_shape = (8, 6, 16, 16)
        test_data = np.random.rand(*test_data_shape).astype(np.float32)
        
        import dask.array as da
        mock_signal.data = da.from_array(test_data, chunks=(4, 3, 16, 16))
        mock_hs_load.return_value = mock_signal
        
        # Create converter with scan step
        converter = DM4Converter(
            self.input_path, self.output_path,
            scan_step=2, dtype="float32"
        )
        
        converter.convert()
        
        # Check that scan step was applied: 8/2 * 6/2 = 12 patterns
        with h5py.File(self.output_path, 'r') as f:
            saved_data = f['patterns'][:]
            self.assertEqual(saved_data.shape[0], 12)  # 4 * 3 patterns


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)