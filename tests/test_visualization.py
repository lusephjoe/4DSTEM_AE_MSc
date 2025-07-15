"""Test suite for visualization functionality."""

import pytest
import torch
import numpy as np
import os
import tempfile
from models.summary import (
    calculate_metrics, save_comparison_images, create_virtual_field_image,
    save_stem_visualization, show
)
from models.autoencoder import Autoencoder


class TestVisualizationFunctions:
    """Test visualization and metrics functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        torch.manual_seed(42)
        original = torch.randn(16, 1, 64, 64)
        # Create slightly different reconstructed data
        reconstructed = original + 0.1 * torch.randn_like(original)
        return original, reconstructed
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_calculate_metrics(self, sample_data):
        """Test metrics calculation."""
        original, reconstructed = sample_data
        
        metrics = calculate_metrics(original, reconstructed)
        
        # Check that all required metrics are present
        required_keys = ['mse', 'psnr', 'ssim', 'mse_std', 'psnr_std', 'ssim_std']
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], (float, np.float32, np.float64)), \
                f"Metric {key} should be a float"
            assert np.isfinite(metrics[key]), f"Metric {key} should be finite"
        
        # Check metric ranges
        assert metrics['mse'] >= 0, "MSE should be non-negative"
        assert metrics['psnr'] > 0, "PSNR should be positive"
        assert 0 <= metrics['ssim'] <= 1, "SSIM should be between 0 and 1"

    def test_save_comparison_images(self, sample_data, temp_dir):
        """Test comparison image saving."""
        original, reconstructed = sample_data
        output_path = os.path.join(temp_dir, "comparison.png")
        
        # Test with different numbers of samples
        for num_samples in [1, 4, 8]:
            save_comparison_images(original, reconstructed, output_path, num_samples)
            assert os.path.exists(output_path), f"Comparison image not saved for {num_samples} samples"
            os.remove(output_path)  # Clean up for next test

    def test_create_virtual_field_image(self, sample_data):
        """Test virtual field image creation."""
        original, _ = sample_data
        data_np = original.detach().cpu().numpy().squeeze(1)  # Remove channel dimension
        
        # Test with different scan shapes
        scan_shapes = [(4, 4), (8, 2), (2, 8)]
        
        for scan_shape in scan_shapes:
            # Test bright field
            bf_image, bf_region = create_virtual_field_image(
                data_np, scan_shape, field_type='bright'
            )
            assert bf_image.shape == scan_shape, \
                f"Bright field image shape mismatch for {scan_shape}"
            assert len(bf_region) == 4, "Bright field region should have 4 coordinates"
            
            # Test dark field
            df_image, df_region = create_virtual_field_image(
                data_np, scan_shape, field_type='dark'
            )
            assert df_image.shape == scan_shape, \
                f"Dark field image shape mismatch for {scan_shape}"
            assert len(df_region) == 4, "Dark field region should have 4 coordinates"

    def test_save_stem_visualization(self, sample_data, temp_dir):
        """Test STEM visualization saving."""
        original, reconstructed = sample_data
        output_path = os.path.join(temp_dir, "stem_viz.png")
        
        # Test with auto-detected scan shape
        save_stem_visualization(original, reconstructed, output_path)
        assert os.path.exists(output_path), "STEM visualization not saved"
        
        # Test with custom scan shape and regions
        scan_shape = (4, 4)
        bright_field_region = (28, 36, 28, 36)
        dark_field_region = (20, 44, 20, 44)
        
        save_stem_visualization(
            original, reconstructed, output_path, 
            scan_shape, bright_field_region, dark_field_region
        )
        assert os.path.exists(output_path), "STEM visualization with custom regions not saved"

    def test_show_function(self, temp_dir):
        """Test the show function with different input sizes."""
        input_sizes = [32, 64, 128]
        latent_dim = 32
        
        for input_size in input_sizes:
            model = Autoencoder(latent_dim, (input_size, input_size))
            example_input = torch.randn(4, 1, input_size, input_size)
            
            # Test show function (should not raise exceptions)
            try:
                show(model, example_input, temp_dir)
                
                # Check that files were created
                comparison_path = os.path.join(temp_dir, "reconstruction_comparison.png")
                stem_path = os.path.join(temp_dir, "stem_visualization.png")
                
                assert os.path.exists(comparison_path), \
                    f"Comparison image not created for size {input_size}"
                assert os.path.exists(stem_path), \
                    f"STEM visualization not created for size {input_size}"
                
                # Clean up
                if os.path.exists(comparison_path):
                    os.remove(comparison_path)
                if os.path.exists(stem_path):
                    os.remove(stem_path)
                    
            except Exception as e:
                pytest.fail(f"Show function failed for input size {input_size}: {str(e)}")


class TestVisualizationEdgeCases:
    """Test edge cases and error handling in visualization functions."""
    
    def test_metrics_with_identical_images(self):
        """Test metrics calculation with identical images."""
        original = torch.randn(4, 1, 64, 64)
        reconstructed = original.clone()
        
        metrics = calculate_metrics(original, reconstructed)
        
        # MSE should be 0
        assert metrics['mse'] < 1e-6, "MSE should be near zero for identical images"
        # SSIM should be 1
        assert metrics['ssim'] > 0.99, "SSIM should be near 1 for identical images"
        # PSNR should be very high
        assert metrics['psnr'] > 50, "PSNR should be very high for identical images"

    def test_metrics_with_single_sample(self):
        """Test metrics calculation with single sample."""
        original = torch.randn(1, 1, 64, 64)
        reconstructed = original + 0.1 * torch.randn_like(original)
        
        metrics = calculate_metrics(original, reconstructed)
        
        # Should still work with single sample
        assert all(np.isfinite(metrics[key]) for key in metrics.keys()), \
            "All metrics should be finite for single sample"

    def test_virtual_field_with_small_images(self):
        """Test virtual field creation with small images."""
        # Test with very small images
        small_data = np.random.randn(4, 16, 16)
        scan_shape = (2, 2)
        
        bf_image, bf_region = create_virtual_field_image(
            small_data, scan_shape, field_type='bright'
        )
        
        assert bf_image.shape == scan_shape, "Should handle small images"
        assert len(bf_region) == 4, "Should provide valid region"

    def test_scan_shape_detection(self):
        """Test automatic scan shape detection."""
        # Test with different numbers of patterns
        pattern_counts = [16, 25, 36, 49, 64]  # Perfect squares
        
        for n_patterns in pattern_counts:
            data = np.random.randn(n_patterns, 32, 32)
            scan_shape = (int(np.sqrt(n_patterns)), int(np.sqrt(n_patterns)))
            
            bf_image, _ = create_virtual_field_image(data, scan_shape, field_type='bright')
            assert bf_image.shape == scan_shape, \
                f"Scan shape detection failed for {n_patterns} patterns"

    def test_visualization_with_different_dtypes(self):
        """Test visualization with different data types."""
        # Test with float32 and float64
        for dtype in [torch.float32, torch.float64]:
            original = torch.randn(4, 1, 64, 64, dtype=dtype)
            reconstructed = original + 0.1 * torch.randn_like(original)
            
            metrics = calculate_metrics(original, reconstructed)
            
            # Should work regardless of dtype
            assert all(np.isfinite(metrics[key]) for key in metrics.keys()), \
                f"Metrics calculation failed for dtype {dtype}"


if __name__ == "__main__":
    pytest.main([__file__])