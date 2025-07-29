#!/usr/bin/env python3
"""Tests for convert_dm4.py CLI and argument parsing"""
import unittest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, Mock
from io import StringIO

# Add parent directory to path to import convert_dm4
sys.path.insert(0, str(Path(__file__).parent.parent))
from convert_dm4 import create_argument_parser, main, DM4Converter


class TestArgumentParser(unittest.TestCase):
    """Test command-line argument parsing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = create_argument_parser()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_file = self.temp_dir / "test_input.dm4"
        self.output_file = self.temp_dir / "test_output.h5"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_required_arguments(self):
        """Test that required arguments are properly parsed."""
        args = self.parser.parse_args([
            "--input", str(self.input_file),
            "--output", str(self.output_file)
        ])
        
        self.assertEqual(args.input, self.input_file)
        self.assertEqual(args.output, self.output_file)
    
    def test_default_values(self):
        """Test that default values are correctly set."""
        args = self.parser.parse_args([
            "--input", str(self.input_file),
            "--output", str(self.output_file)
        ])
        
        self.assertEqual(args.downsample, 1)
        self.assertEqual(args.mode, "bin")
        self.assertEqual(args.sigma, 0.8)
        self.assertEqual(args.scan_step, 1)
        self.assertEqual(args.chunk_size, 128)
        self.assertEqual(args.dtype, "float16")
        self.assertEqual(args.compression_level, 4)
    
    def test_custom_arguments(self):
        """Test parsing of custom argument values."""
        args = self.parser.parse_args([
            "--input", str(self.input_file),
            "--output", str(self.output_file),
            "--downsample", "4",
            "--mode", "gauss",
            "--sigma", "1.2",
            "--scan_step", "2",
            "--chunk_size", "256",
            "--dtype", "float32",
            "--compression_level", "9"
        ])
        
        self.assertEqual(args.downsample, 4)
        self.assertEqual(args.mode, "gauss")
        self.assertEqual(args.sigma, 1.2)
        self.assertEqual(args.scan_step, 2)
        self.assertEqual(args.chunk_size, 256)
        self.assertEqual(args.dtype, "float32")
        self.assertEqual(args.compression_level, 9)
    
    def test_mode_choices(self):
        """Test that mode argument only accepts valid choices."""
        valid_modes = ["bin", "stride", "gauss", "fft"]
        
        for mode in valid_modes:
            with self.subTest(mode=mode):
                args = self.parser.parse_args([
                    "--input", str(self.input_file),
                    "--output", str(self.output_file),
                    "--mode", mode
                ])
                self.assertEqual(args.mode, mode)
        
        # Test invalid mode
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                "--input", str(self.input_file),
                "--output", str(self.output_file),
                "--mode", "invalid_mode"
            ])
    
    def test_dtype_choices(self):
        """Test that dtype argument only accepts valid choices."""
        valid_dtypes = ["uint16", "float16", "float32"]
        
        for dtype in valid_dtypes:
            with self.subTest(dtype=dtype):
                args = self.parser.parse_args([
                    "--input", str(self.input_file),
                    "--output", str(self.output_file),
                    "--dtype", dtype
                ])
                self.assertEqual(args.dtype, dtype)
    
    def test_missing_required_arguments(self):
        """Test that missing required arguments cause parser to exit."""
        # Missing input
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--output", str(self.output_file)])
        
        # Missing output
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--input", str(self.input_file)])
        
        # Missing both
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])
    
    def test_help_message(self):
        """Test that help message is generated properly."""
        with self.assertRaises(SystemExit):
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                self.parser.parse_args(["--help"])
        
        # Should contain key information about the script
        help_output = mock_stdout.getvalue()
        self.assertIn("Convert .dm4 4D-STEM file to HDF5 format", help_output)
        self.assertIn("--input", help_output)
        self.assertIn("--output", help_output)
        self.assertIn("--downsample", help_output)
        self.assertIn("--mode", help_output)
    
    def test_argument_types(self):
        """Test that arguments are parsed to correct types."""
        args = self.parser.parse_args([
            "--input", str(self.input_file),
            "--output", str(self.output_file),
            "--downsample", "4",
            "--sigma", "1.5",
            "--scan_step", "3",
            "--chunk_size", "512",
            "--compression_level", "7"
        ])
        
        self.assertIsInstance(args.input, Path)
        self.assertIsInstance(args.output, Path)
        self.assertIsInstance(args.downsample, int)
        self.assertIsInstance(args.sigma, float)
        self.assertIsInstance(args.scan_step, int)
        self.assertIsInstance(args.chunk_size, int)
        self.assertIsInstance(args.compression_level, int)
    
    def test_metavar_usage(self):
        """Test that metavar values are used in help text."""
        help_text = self.parser.format_help()
        
        # Check for custom metavar values
        self.assertIn("k", help_text)  # downsample metavar
        self.assertIn("σ", help_text)  # sigma metavar
        self.assertIn("n", help_text)  # scan_step metavar
        self.assertIn("1-9", help_text)  # compression_level metavar


class TestMainFunction(unittest.TestCase):
    """Test the main() function and CLI integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_file = self.temp_dir / "test_input.dm4" 
        self.output_file = self.temp_dir / "test_output.h5"
        
        # Create dummy input file
        self.input_file.touch()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('sys.argv')
    @patch.object(DM4Converter, 'convert')
    def test_main_successful_execution(self, mock_convert, mock_argv):
        """Test successful execution of main function."""
        mock_argv.__getitem__ = Mock(side_effect=lambda x: [
            "convert_dm4.py",
            "--input", str(self.input_file),
            "--output", str(self.output_file)
        ][x])
        mock_argv.__len__ = Mock(return_value=5)
        
        # Mock the converter creation and execution
        mock_convert.return_value = None
        
        # Should not raise exception
        try:
            main()
        except SystemExit:
            pass  # Expected from argument parsing in test environment
    
    @patch('sys.argv')
    @patch('builtins.print')
    def test_main_missing_input_file(self, mock_print, mock_argv):
        """Test main function with missing input file."""
        non_existent_file = self.temp_dir / "nonexistent.dm4"
        
        mock_argv.__getitem__ = Mock(side_effect=lambda x: [
            "convert_dm4.py",
            "--input", str(non_existent_file),
            "--output", str(self.output_file)
        ][x])
        mock_argv.__len__ = Mock(return_value=5)
        
        try:
            main()
        except SystemExit:
            pass
        
        # Should print error message
        mock_print.assert_called()
        error_calls = [call for call in mock_print.call_args_list 
                      if 'ERROR' in str(call)]
        self.assertTrue(len(error_calls) > 0)
    
    @patch('sys.argv')
    @patch('builtins.print')
    def test_main_non_dm4_extension_warning(self, mock_print, mock_argv):
        """Test warning for non-.dm4 file extensions."""
        wrong_ext_file = self.temp_dir / "test_input.txt"
        wrong_ext_file.touch()
        
        mock_argv.__getitem__ = Mock(side_effect=lambda x: [
            "convert_dm4.py",
            "--input", str(wrong_ext_file),
            "--output", str(self.output_file)
        ][x])
        mock_argv.__len__ = Mock(return_value=5)
        
        with patch.object(DM4Converter, 'convert'):
            try:
                main()
            except SystemExit:
                pass
        
        # Should print warning message
        warning_calls = [call for call in mock_print.call_args_list 
                        if 'WARNING' in str(call)]
        self.assertTrue(len(warning_calls) > 0)
    
    @patch('sys.argv')
    @patch.object(DM4Converter, 'convert')
    @patch('builtins.print')
    def test_main_keyboard_interrupt(self, mock_print, mock_convert, mock_argv):
        """Test handling of keyboard interrupt."""
        mock_argv.__getitem__ = Mock(side_effect=lambda x: [
            "convert_dm4.py",
            "--input", str(self.input_file),
            "--output", str(self.output_file)
        ][x])
        mock_argv.__len__ = Mock(return_value=5)
        
        mock_convert.side_effect = KeyboardInterrupt()
        
        try:
            main()
        except SystemExit:
            pass
        
        # Should print interruption message
        interrupt_calls = [call for call in mock_print.call_args_list 
                          if 'interrupted' in str(call)]
        self.assertTrue(len(interrupt_calls) > 0)
    
    @patch('sys.argv')
    @patch.object(DM4Converter, 'convert')
    def test_main_converter_exception(self, mock_convert, mock_argv):
        """Test handling of converter exceptions."""
        mock_argv.__getitem__ = Mock(side_effect=lambda x: [
            "convert_dm4.py",
            "--input", str(self.input_file),
            "--output", str(self.output_file)
        ][x])
        mock_argv.__len__ = Mock(return_value=5)
        
        mock_convert.side_effect = RuntimeError("Test conversion error")
        
        # Should re-raise the exception
        with self.assertRaises(RuntimeError):
            try:
                main()
            except SystemExit:
                pass
    
    @patch('sys.argv')
    def test_main_creates_converter_with_correct_params(self, mock_argv):
        """Test that main creates DM4Converter with correct parameters."""
        mock_argv.__getitem__ = Mock(side_effect=lambda x: [
            "convert_dm4.py",
            "--input", str(self.input_file),
            "--output", str(self.output_file),
            "--downsample", "2",
            "--mode", "gauss",
            "--sigma", "1.0",
            "--scan_step", "2",
            "--chunk_size", "256",
            "--dtype", "float32",
            "--compression_level", "6"
        ][x])
        mock_argv.__len__ = Mock(return_value=17)
        
        with patch.object(DM4Converter, '__init__', return_value=None) as mock_init:
            with patch.object(DM4Converter, 'convert'):
                try:
                    main()
                except SystemExit:
                    pass
        
        # Verify converter was created with correct parameters
        mock_init.assert_called_once()
        args, kwargs = mock_init.call_args
        
        self.assertEqual(kwargs['downsample'], 2)
        self.assertEqual(kwargs['mode'], 'gauss')
        self.assertEqual(kwargs['sigma'], 1.0)
        self.assertEqual(kwargs['scan_step'], 2)
        self.assertEqual(kwargs['chunk_size'], 256)
        self.assertEqual(kwargs['dtype'], 'float32')
        self.assertEqual(kwargs['compression_level'], 6)


class TestCLIEdgeCases(unittest.TestCase):
    """Test edge cases in CLI behavior."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = create_argument_parser()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_negative_values(self):
        """Test behavior with negative argument values."""
        input_file = self.temp_dir / "input.dm4"
        output_file = self.temp_dir / "output.h5"
        
        # Most negative values should be accepted (validation happens in converter)
        args = self.parser.parse_args([
            "--input", str(input_file),
            "--output", str(output_file),
            "--downsample", "-1",  # Should be caught by converter logic
            "--sigma", "-0.5",
            "--scan_step", "-2",
            "--compression_level", "-1"
        ])
        
        self.assertEqual(args.downsample, -1)
        self.assertEqual(args.sigma, -0.5)
        self.assertEqual(args.scan_step, -2)
        self.assertEqual(args.compression_level, -1)
    
    def test_very_large_values(self):
        """Test behavior with very large argument values."""
        input_file = self.temp_dir / "input.dm4"
        output_file = self.temp_dir / "output.h5"
        
        args = self.parser.parse_args([
            "--input", str(input_file),
            "--output", str(output_file),
            "--downsample", "10000",
            "--sigma", "100.0",
            "--scan_step", "1000",
            "--chunk_size", "999999",
            "--compression_level", "100"  # Above valid range but parser accepts it
        ])
        
        self.assertEqual(args.downsample, 10000)
        self.assertEqual(args.sigma, 100.0)
        self.assertEqual(args.scan_step, 1000)
        self.assertEqual(args.chunk_size, 999999)
        self.assertEqual(args.compression_level, 100)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)