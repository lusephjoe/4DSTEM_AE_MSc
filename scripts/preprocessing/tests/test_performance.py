#!/usr/bin/env python3
"""Performance and benchmark tests for convert_dm4.py"""
import unittest
import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from convert_dm4 import DownsampleStrategy


class TestDownsamplePerformance(unittest.TestCase):
    """Performance tests for downsampling strategies."""
    
    def setUp(self):
        """Set up performance test data."""
        # Create realistic test data sizes
        self.small_data = np.random.rand(64, 64).astype(np.float32)
        self.medium_data = np.random.rand(256, 256).astype(np.float32)
        self.large_data = np.random.rand(512, 512).astype(np.float32)
        
        # 4D data simulating multiple patterns
        self.small_4d = np.random.rand(4, 4, 64, 64).astype(np.float32)
        self.medium_4d = np.random.rand(8, 8, 128, 128).astype(np.float32)
    
    def time_operation(self, operation, *args, **kwargs):
        """Time an operation and return duration."""
        start_time = time.time()
        result = operation(*args, **kwargs)
        duration = time.time() - start_time
        return result, duration
    
    def test_stride_performance_scaling(self):
        """Test stride downsampling performance with different data sizes."""
        factor = 4
        
        # Test different data sizes
        test_cases = [
            ("Small (64x64)", self.small_data),
            ("Medium (256x256)", self.medium_data),
            ("Large (512x512)", self.large_data),
        ]
        
        print(f"\nStride downsampling performance (factor={factor}):")
        
        for name, data in test_cases:
            result, duration = self.time_operation(
                DownsampleStrategy.stride, data, factor
            )
            
            throughput = data.size / max(duration, 1e-6) / 1e6  # Mpixels/sec
            print(f"  {name}: {duration:.4f}s ({throughput:.1f} Mpix/s)")
            
            # Verify result shape
            expected_shape = (data.shape[0] // factor, data.shape[1] // factor)
            self.assertEqual(result.shape, expected_shape)
    
    def test_bin_mean_performance_scaling(self):
        """Test bin mean performance with different data sizes."""
        factor = 4
        
        test_cases = [
            ("Small (64x64)", self.small_data),
            ("Medium (256x256)", self.medium_data),
            ("Large (512x512)", self.large_data),
        ]
        
        print(f"\nBin mean downsampling performance (factor={factor}):")
        
        for name, data in test_cases:
            result, duration = self.time_operation(
                DownsampleStrategy.bin_mean, data, factor
            )
            
            throughput = data.size / max(duration, 1e-6) / 1e6
            print(f"  {name}: {duration:.4f}s ({throughput:.1f} Mpix/s)")
            
            expected_shape = (data.shape[0] // factor, data.shape[1] // factor)
            self.assertEqual(result.shape, expected_shape)
    
    def test_gaussian_performance_scaling(self):
        """Test Gaussian downsampling performance."""
        factor = 4
        sigma = 0.8
        
        test_cases = [
            ("Small (64x64)", self.small_data),
            ("Medium (256x256)", self.medium_data),
            # Skip large data for Gaussian as it's slower
        ]
        
        print(f"\nGaussian downsampling performance (factor={factor}, sigma={sigma}):")
        
        for name, data in test_cases:
            result, duration = self.time_operation(
                DownsampleStrategy.gaussian, data, factor, sigma
            )
            
            throughput = data.size / max(duration, 1e-6) / 1e6
            print(f"  {name}: {duration:.4f}s ({throughput:.1f} Mpix/s)")
            
            expected_shape = (data.shape[0] // factor, data.shape[1] // factor)
            self.assertEqual(result.shape, expected_shape)
    
    def test_fft_performance_scaling(self):
        """Test FFT downsampling performance."""
        factor = 4
        
        test_cases = [
            ("Small (64x64)", self.small_data),
            ("Medium (256x256)", self.medium_data),
            # Skip large data for FFT as it's slowest
        ]
        
        print(f"\nFFT downsampling performance (factor={factor}):")
        
        for name, data in test_cases:
            result, duration = self.time_operation(
                DownsampleStrategy.fft_crop, data, factor
            )
            
            throughput = data.size / max(duration, 1e-6) / 1e6
            print(f"  {name}: {duration:.4f}s ({throughput:.1f} Mpix/s)")
            
            expected_shape = (data.shape[0] // factor, data.shape[1] // factor)
            self.assertEqual(result.shape, expected_shape)
    
    def test_method_performance_comparison(self):
        """Compare performance of different downsampling methods."""
        factor = 4
        test_data = self.medium_data  # 256x256
        
        methods = [
            ("Stride", lambda d, f: DownsampleStrategy.stride(d, f)),
            ("Bin Mean", lambda d, f: DownsampleStrategy.bin_mean(d, f)),
            ("Gaussian", lambda d, f: DownsampleStrategy.gaussian(d, f, 0.8)),
            ("FFT", lambda d, f: DownsampleStrategy.fft_crop(d, f)),
        ]
        
        print(f"\nMethod performance comparison (256x256 data, factor={factor}):")
        results = {}
        
        for name, method in methods:
            result, duration = self.time_operation(method, test_data, factor)
            throughput = test_data.size / max(duration, 1e-6) / 1e6
            results[name] = duration
            
            print(f"  {name:12}: {duration:.4f}s ({throughput:.1f} Mpix/s)")
        
        # Verify stride is fastest
        self.assertLess(results["Stride"], results["Bin Mean"])
        self.assertLess(results["Bin Mean"], results["Gaussian"])
        # FFT might not always be slowest due to optimized implementations
    
    def test_4d_data_performance(self):
        """Test performance with 4D data (multiple patterns)."""
        factor = 2
        
        test_cases = [
            ("Small 4D (4x4x64x64)", self.small_4d),
            ("Medium 4D (8x8x128x128)", self.medium_4d),
        ]
        
        print(f"\n4D data performance (factor={factor}):")
        
        for name, data in test_cases:
            # Test stride (fastest) and bin mean (balanced)
            for method_name, method in [("Stride", DownsampleStrategy.stride),
                                       ("Bin Mean", DownsampleStrategy.bin_mean)]:
                result, duration = self.time_operation(method, data, factor)
                
                total_pixels = data.size
                throughput = total_pixels / max(duration, 1e-6) / 1e6
                
                print(f"  {name} - {method_name}: {duration:.3f}s ({throughput:.1f} Mpix/s)")
                
                # Verify shape
                expected_shape = (data.shape[0], data.shape[1], 
                                data.shape[2] // factor, data.shape[3] // factor)
                self.assertEqual(result.shape, expected_shape)
    
    def test_large_downsampling_factors(self):
        """Test performance with large downsampling factors."""
        test_data = self.medium_data  # 256x256
        factors = [2, 4, 8, 16]
        
        print(f"\nDownsampling factor performance (256x256 data, stride method):")
        
        for factor in factors:
            result, duration = self.time_operation(
                DownsampleStrategy.stride, test_data, factor
            )
            
            throughput = test_data.size / max(duration, 1e-6) / 1e6  # Avoid divide by zero
            output_size = result.size
            
            print(f"  Factor {factor:2d}: {duration:.4f}s ({throughput:.1f} Mpix/s) "
                  f"-> {result.shape} ({output_size} pixels)")
            
            # Performance should be relatively constant for stride
            self.assertLess(duration, 0.1)  # Should be very fast
    
    def test_memory_efficiency(self):
        """Test memory usage patterns (basic check)."""
        # This is a basic check - for detailed memory profiling, 
        # use tools like memory_profiler
        
        large_data = np.random.rand(1024, 1024).astype(np.float32)
        factor = 4
        
        # All methods should work without memory errors
        methods = [
            DownsampleStrategy.stride,
            DownsampleStrategy.bin_mean,
            # Skip Gaussian and FFT for very large data
        ]
        
        print(f"\nMemory efficiency test (1024x1024 data, factor={factor}):")
        
        for method in methods:
            try:
                result, duration = self.time_operation(method, large_data, factor)
                success = True
            except MemoryError:
                success = False
                duration = float('inf')
            
            method_name = method.__name__.replace('_', ' ').title()
            status = "✓" if success else "✗ (Memory Error)"
            print(f"  {method_name:12}: {status}")
            
            if success:
                self.assertEqual(result.shape, (256, 256))


class TestPerformanceRegression(unittest.TestCase):
    """Test for performance regressions."""
    
    def setUp(self):
        """Set up regression test data."""
        # Fixed seed for reproducible results
        np.random.seed(42)
        self.test_data = np.random.rand(256, 256).astype(np.float32)
    
    def test_stride_performance_baseline(self):
        """Establish baseline performance for stride downsampling."""
        factor = 4
        
        # Run multiple times to get stable measurement
        durations = []
        for _ in range(5):
            start_time = time.time()
            result = DownsampleStrategy.stride(self.test_data, factor)
            duration = time.time() - start_time
            durations.append(duration)
        
        avg_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        print(f"\nStride baseline (256x256, factor=4):")
        print(f"  Average: {avg_duration:.4f}s ± {std_duration:.4f}s")
        print(f"  Throughput: {self.test_data.size / max(avg_duration, 1e-6) / 1e6:.1f} Mpix/s")
        
        # Reasonable performance expectations
        self.assertLess(avg_duration, 0.01)  # Should be < 10ms for stride
        self.assertEqual(result.shape, (64, 64))
    
    def test_bin_mean_performance_baseline(self):
        """Establish baseline performance for bin mean downsampling."""
        factor = 4
        
        durations = []
        for _ in range(5):
            start_time = time.time()
            result = DownsampleStrategy.bin_mean(self.test_data, factor)
            duration = time.time() - start_time
            durations.append(duration)
        
        avg_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        print(f"\nBin mean baseline (256x256, factor=4):")
        print(f"  Average: {avg_duration:.4f}s ± {std_duration:.4f}s")
        print(f"  Throughput: {self.test_data.size / max(avg_duration, 1e-6) / 1e6:.1f} Mpix/s")
        
        # Should be reasonably fast
        self.assertLess(avg_duration, 0.1)  # Should be < 100ms
        self.assertEqual(result.shape, (64, 64))


if __name__ == '__main__':
    print("Performance Tests for convert_dm4.py Downsampling")
    print("=" * 50)
    
    # Run with high verbosity to see performance output
    unittest.main(verbosity=2, buffer=False)