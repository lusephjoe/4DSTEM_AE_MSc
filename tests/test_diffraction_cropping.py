#!/usr/bin/env python3
"""
Test suite for diffraction pattern cropping functionality.

This module provides comprehensive tests for the automatic diffraction pattern 
cropping implementation, including synthetic data validation and edge cases.

Author: Claude Code Assistant
Date: 2025-01-18
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import tempfile
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.preprocessing.diffraction_cropping import (
    DiffractionCropper, 
    CroppingConfig, 
    create_test_data
)


class TestDiffractionCropping(unittest.TestCase):
    """Test suite for diffraction pattern cropping."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = CroppingConfig(
            target_retention=0.98,
            min_retention=0.95,
            margin_pixels=2,
            chunk_size=50,
            use_gpu=False,
            verbose=False
        )
        self.cropper = DiffractionCropper(self.config)
        
        # Create small test dataset
        self.test_data = create_test_data(n_patterns=100, image_size=128, noise_level=0.05)
    
    def test_synthetic_data_creation(self):
        """Test synthetic data creation."""
        data = create_test_data(n_patterns=10, image_size=64)
        
        self.assertEqual(data.shape, (10, 1, 64, 64))
        self.assertEqual(data.dtype, torch.float32)
        self.assertTrue(torch.all(data >= 0))  # Should be non-negative
    
    def test_centroid_computation(self):
        """Test centroid computation."""
        # Create a simple test pattern with known centroid
        pattern = np.zeros((10, 10))
        pattern[3, 4] = 1.0  # Point source at (3, 4)
        
        center = self.cropper._compute_centroid(pattern)
        
        # Should be close to (3, 4)
        self.assertAlmostEqual(center[0], 3.0, places=5)
        self.assertAlmostEqual(center[1], 4.0, places=5)
    
    def test_radial_profile_computation(self):
        """Test radial profile computation."""
        # Create concentric circle pattern
        size = 64
        pattern = np.zeros((size, size))
        center = (size // 2, size // 2)
        
        y, x = np.ogrid[:size, :size]
        r = np.sqrt((y - center[0])**2 + (x - center[1])**2)
        pattern[r <= 10] = 1.0
        
        radii, profile = self.cropper._compute_radial_profile(pattern, center)
        
        # Profile should be high inside radius 10 and low outside
        self.assertTrue(np.mean(profile[:10]) > np.mean(profile[15:]))
    
    def test_cumulative_intensity_computation(self):
        """Test cumulative intensity computation."""
        # Create test pattern
        pattern = np.ones((10, 10))
        center = (5, 5)
        
        radii, cum_intensity = self.cropper._compute_cumulative_intensity(pattern, center)
        
        # Cumulative intensity should be monotonically increasing
        self.assertTrue(np.all(np.diff(cum_intensity) >= 0))
        
        # Should reach 1.0 at the end (allow some tolerance for floating point)
        self.assertAlmostEqual(cum_intensity[-1], 1.0, places=2)
    
    def test_retention_radius_finding(self):
        """Test retention radius finding."""
        # Create cumulative intensity array
        cum_intensity = np.linspace(0, 1, 100)
        
        # Test finding 50% retention
        radius = self.cropper._find_retention_radius(cum_intensity, 0.5)
        self.assertAlmostEqual(radius, 50, delta=1)
        
        # Test finding 90% retention
        radius = self.cropper._find_retention_radius(cum_intensity, 0.9)
        self.assertAlmostEqual(radius, 90, delta=1)
    
    def test_analysis_pass(self):
        """Test analysis pass functionality."""
        results = self.cropper.analyze_patterns(self.test_data)
        
        # Check required keys
        required_keys = ['global_radius', 'retention_radii', 'retention_stats', 
                        'centers', 'original_shape', 'cropped_shape', 'n_patterns']
        
        for key in required_keys:
            self.assertIn(key, results)
        
        # Check data consistency
        self.assertEqual(results['n_patterns'], self.test_data.shape[0])
        self.assertEqual(len(results['retention_radii']), self.test_data.shape[0])
        self.assertEqual(len(results['centers']), self.test_data.shape[0])
        
        # Check radius is reasonable
        self.assertGreater(results['global_radius'], 0)
        self.assertLessEqual(results['global_radius'], min(self.test_data.shape[2:]) // 2)
    
    def test_cropping_pass(self):
        """Test cropping pass functionality."""
        # Run analysis first
        analysis_results = self.cropper.analyze_patterns(self.test_data)
        
        # Run cropping
        cropped_data = self.cropper.crop_patterns(self.test_data, analysis_results)
        
        # Check output shape
        expected_size = 2 * analysis_results['global_radius']
        expected_shape = (self.test_data.shape[0], 1, expected_size, expected_size)
        
        self.assertEqual(cropped_data.shape, expected_shape)
        self.assertEqual(cropped_data.dtype, self.test_data.dtype)
        
        # Check that cropped data is smaller
        self.assertLess(cropped_data.numel(), self.test_data.numel())
    
    def test_retention_validation(self):
        """Test retention validation functionality."""
        # Run full pipeline
        analysis_results = self.cropper.analyze_patterns(self.test_data)
        cropped_data = self.cropper.crop_patterns(self.test_data, analysis_results)
        
        # Validate retention
        validation_results = self.cropper.validate_retention(
            self.test_data, cropped_data, analysis_results
        )
        
        # Check required keys
        required_keys = ['mean_retention', 'min_retention', 'max_retention', 
                        'retentions', 'n_patterns']
        
        for key in required_keys:
            self.assertIn(key, validation_results)
        
        # Check retention values are reasonable
        self.assertGreaterEqual(validation_results['min_retention'], 0.0)
        self.assertLessEqual(validation_results['max_retention'], 1.0)
        self.assertGreaterEqual(validation_results['mean_retention'], 
                               validation_results['min_retention'])
        
        # For synthetic Airy disks, retention should be high
        self.assertGreater(validation_results['mean_retention'], 0.9)
    
    def test_complete_pipeline(self):
        """Test complete processing pipeline."""
        cropped_data, results = self.cropper.process_dataset(self.test_data)
        
        # Check that all results are present
        self.assertIn('analysis', results)
        self.assertIn('validation', results)
        self.assertIn('processing_time', results)
        
        # Check data consistency
        self.assertEqual(cropped_data.shape[0], self.test_data.shape[0])
        self.assertLess(cropped_data.numel(), self.test_data.numel())
        
        # Check retention meets requirements
        validation = results['validation']
        self.assertGreaterEqual(validation['min_retention'], 0.9)  # Should be high for Airy disks
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with very small data
        small_data = torch.randn(5, 1, 16, 16)
        small_data = torch.abs(small_data)  # Ensure positive
        
        cropped_data, results = self.cropper.process_dataset(small_data)
        
        # Should still work but with small crop
        self.assertEqual(cropped_data.shape[0], small_data.shape[0])
        
        # Test with single pattern
        single_pattern = self.test_data[:1]
        cropped_single, _ = self.cropper.process_dataset(single_pattern)
        
        self.assertEqual(cropped_single.shape[0], 1)
    
    def test_different_center_methods(self):
        """Test different center detection methods."""
        # Test manual center
        config_manual = CroppingConfig(
            center_method="manual",
            manual_center=(64, 64),
            verbose=False
        )
        cropper_manual = DiffractionCropper(config_manual)
        
        results_manual = cropper_manual.analyze_patterns(self.test_data)
        
        # All centers should be the manual center
        for center in results_manual['centers']:
            self.assertEqual(center, (64, 64))
        
        # Test geometric center
        config_center = CroppingConfig(
            center_method="center",
            verbose=False
        )
        cropper_center = DiffractionCropper(config_center)
        
        results_center = cropper_center.analyze_patterns(self.test_data)
        
        # All centers should be image center
        expected_center = (self.test_data.shape[2] // 2, self.test_data.shape[3] // 2)
        for center in results_center['centers']:
            self.assertEqual(center, expected_center)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid retention values
        with self.assertRaises(ValueError):
            config = CroppingConfig(target_retention=1.5)  # > 1.0
        
        with self.assertRaises(ValueError):
            config = CroppingConfig(target_retention=0.0)  # <= 0.0
        
        with self.assertRaises(ValueError):
            config = CroppingConfig(
                target_retention=0.9,
                min_retention=0.95  # min > target
            )
    
    def test_memory_efficiency(self):
        """Test memory efficiency with chunked processing."""
        # Create larger dataset
        large_data = create_test_data(n_patterns=500, image_size=64)
        
        # Use small chunk size
        config = CroppingConfig(chunk_size=10, verbose=False)
        cropper = DiffractionCropper(config)
        
        # Should complete without memory issues
        cropped_data, results = cropper.process_dataset(large_data)
        
        self.assertEqual(cropped_data.shape[0], large_data.shape[0])
        self.assertLess(cropped_data.numel(), large_data.numel())
    
    def test_save_load_functionality(self):
        """Test saving and loading functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.pt"
            
            # Process and save
            cropped_data, results = self.cropper.process_dataset(
                self.test_data, save_path=output_path
            )
            
            # Check files exist
            self.assertTrue(output_path.exists())
            
            results_path = output_path.parent / f"{output_path.stem}_results.json"
            self.assertTrue(results_path.exists())
            
            # Load and verify
            loaded_data = torch.load(output_path)
            self.assertTrue(torch.allclose(cropped_data, loaded_data))


class TestCroppingConfig(unittest.TestCase):
    """Test suite for CroppingConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = CroppingConfig()
        
        self.assertEqual(config.target_retention, 0.98)
        self.assertEqual(config.min_retention, 0.95)
        self.assertEqual(config.margin_pixels, 3)
        self.assertEqual(config.chunk_size, 1000)
        self.assertEqual(config.center_method, "centroid")
        self.assertTrue(config.visualization)
        self.assertTrue(config.verbose)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CroppingConfig(
            target_retention=0.99,
            min_retention=0.97,
            margin_pixels=5,
            chunk_size=500,
            center_method="manual",
            manual_center=(10, 20),
            visualization=False,
            verbose=False
        )
        
        self.assertEqual(config.target_retention, 0.99)
        self.assertEqual(config.min_retention, 0.97)
        self.assertEqual(config.margin_pixels, 5)
        self.assertEqual(config.chunk_size, 500)
        self.assertEqual(config.center_method, "manual")
        self.assertEqual(config.manual_center, (10, 20))
        self.assertFalse(config.visualization)
        self.assertFalse(config.verbose)


class TestPerformance(unittest.TestCase):
    """Test suite for performance characteristics."""
    
    def test_processing_time_scales_linearly(self):
        """Test that processing time scales approximately linearly with data size."""
        import time
        
        config = CroppingConfig(verbose=False, visualization=False)
        cropper = DiffractionCropper(config)
        
        # Test with different sizes
        sizes = [100, 200, 400]
        times = []
        
        for size in sizes:
            test_data = create_test_data(n_patterns=size, image_size=64)
            
            start_time = time.time()
            cropper.process_dataset(test_data)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Check that time increases with size (not necessarily perfectly linear due to overhead)
        self.assertLess(times[0], times[1])
        self.assertLess(times[1], times[2])
    
    def test_memory_usage_bounded(self):
        """Test that memory usage remains bounded with chunked processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large dataset with small chunks
        config = CroppingConfig(chunk_size=10, verbose=False, visualization=False)
        cropper = DiffractionCropper(config)
        
        large_data = create_test_data(n_patterns=1000, image_size=64)
        cropper.process_dataset(large_data)
        
        # Check memory didn't explode
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        # Should use reasonable amount of memory (less than 1GB increase)
        self.assertLess(memory_increase, 1000)  # 1GB limit


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)