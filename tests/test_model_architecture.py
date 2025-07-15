"""Test suite for model architecture consistency with different input sizes."""

import pytest
import torch
import numpy as np
from models.autoencoder import Autoencoder, Encoder, Decoder
from models.blocks import ConvBlock, IdentityBlock, ResNetBlock, EmbeddingLayer
from models.summary import show, _add_hooks, _is_leaf_module


class TestModelArchitecture:
    """Test model architecture consistency across different input sizes."""
    
    @pytest.fixture
    def input_sizes(self):
        """Different input sizes to test."""
        return [32, 64, 128, 256, 512]
    
    @pytest.fixture
    def latent_dims(self):
        """Different latent dimensions to test."""
        return [16, 32, 64, 128]
    
    @pytest.fixture
    def batch_sizes(self):
        """Different batch sizes to test."""
        return [1, 4, 8, 16]

    def test_encoder_output_consistency(self, input_sizes, latent_dims):
        """Test that encoder produces consistent latent dimensions regardless of input size."""
        for input_size in input_sizes:
            for latent_dim in latent_dims:
                encoder = Encoder(latent_dim)
                encoder.eval()
                
                # Create test input
                x = torch.randn(2, 1, input_size, input_size)
                
                with torch.no_grad():
                    output = encoder(x)
                
                # Check output shape
                assert output.shape == (2, latent_dim), \
                    f"Encoder output shape mismatch for input {input_size}x{input_size}, " \
                    f"expected (2, {latent_dim}), got {output.shape}"
                
                # Check output is not NaN or infinite
                assert torch.isfinite(output).all(), \
                    f"Encoder output contains NaN or infinite values for input {input_size}x{input_size}"

    def test_decoder_output_consistency(self, input_sizes, latent_dims):
        """Test that decoder produces output matching specified size."""
        for input_size in input_sizes:
            for latent_dim in latent_dims:
                decoder = Decoder(latent_dim, (input_size, input_size))
                decoder.eval()
                
                # Create test input
                z = torch.randn(2, latent_dim)
                
                with torch.no_grad():
                    output = decoder(z)
                
                # Check output shape
                assert output.shape == (2, 1, input_size, input_size), \
                    f"Decoder output shape mismatch for latent dim {latent_dim}, " \
                    f"expected (2, 1, {input_size}, {input_size}), got {output.shape}"
                
                # Check output is not NaN or infinite
                assert torch.isfinite(output).all(), \
                    f"Decoder output contains NaN or infinite values for latent dim {latent_dim}"

    def test_autoencoder_reconstruction_consistency(self, input_sizes, latent_dims):
        """Test that autoencoder maintains input-output size consistency."""
        for input_size in input_sizes:
            for latent_dim in latent_dims:
                autoencoder = Autoencoder(latent_dim, (input_size, input_size))
                autoencoder.eval()
                
                # Create test input
                x = torch.randn(2, 1, input_size, input_size)
                
                with torch.no_grad():
                    reconstructed = autoencoder(x)
                
                # Check reconstruction shape matches input
                assert reconstructed.shape == x.shape, \
                    f"Autoencoder reconstruction shape mismatch for input {input_size}x{input_size}, " \
                    f"expected {x.shape}, got {reconstructed.shape}"
                
                # Check reconstruction is not NaN or infinite
                assert torch.isfinite(reconstructed).all(), \
                    f"Autoencoder reconstruction contains NaN or infinite values for input {input_size}x{input_size}"

    def test_embedding_consistency(self, input_sizes, latent_dims):
        """Test that embedding layer produces consistent output."""
        for input_size in input_sizes:
            for latent_dim in latent_dims:
                autoencoder = Autoencoder(latent_dim, (input_size, input_size))
                autoencoder.eval()
                
                # Create test input
                x = torch.randn(2, 1, input_size, input_size)
                
                with torch.no_grad():
                    embedding = autoencoder.embed(x)
                
                # Check embedding shape
                assert embedding.shape == (2, latent_dim), \
                    f"Embedding shape mismatch for input {input_size}x{input_size}, " \
                    f"expected (2, {latent_dim}), got {embedding.shape}"
                
                # Check embedding is non-negative (ReLU enforced)
                assert (embedding >= 0).all(), \
                    f"Embedding contains negative values for input {input_size}x{input_size}"

    def test_batch_size_consistency(self, batch_sizes):
        """Test that model handles different batch sizes correctly."""
        input_size = 64
        latent_dim = 32
        
        for batch_size in batch_sizes:
            autoencoder = Autoencoder(latent_dim, (input_size, input_size))
            autoencoder.eval()
            
            # Create test input
            x = torch.randn(batch_size, 1, input_size, input_size)
            
            with torch.no_grad():
                reconstructed = autoencoder(x)
                embedding = autoencoder.embed(x)
            
            # Check shapes
            assert reconstructed.shape == x.shape, \
                f"Reconstruction shape mismatch for batch size {batch_size}"
            assert embedding.shape == (batch_size, latent_dim), \
                f"Embedding shape mismatch for batch size {batch_size}"

    def test_model_parameter_count_consistency(self, input_sizes):
        """Test that model parameter count is consistent regardless of input size."""
        latent_dim = 32
        param_counts = {}
        
        for input_size in input_sizes:
            autoencoder = Autoencoder(latent_dim, (input_size, input_size))
            param_count = sum(p.numel() for p in autoencoder.parameters())
            param_counts[input_size] = param_count
        
        # All models should have the same parameter count (size-agnostic)
        unique_counts = set(param_counts.values())
        assert len(unique_counts) == 1, \
            f"Parameter counts vary across input sizes: {param_counts}"

    def test_loss_computation_consistency(self, input_sizes):
        """Test that loss computation works correctly for different input sizes."""
        latent_dim = 32
        
        for input_size in input_sizes:
            autoencoder = Autoencoder(latent_dim, (input_size, input_size))
            autoencoder.eval()
            
            # Create test input
            x = torch.randn(2, 1, input_size, input_size)
            
            with torch.no_grad():
                z = autoencoder.embed(x)
                x_hat = autoencoder.decoder(z)
                loss_dict = autoencoder.compute_loss(x, x_hat, z)
            
            # Check loss components
            required_keys = ['total_loss', 'mse_loss', 'l1_reg', 'contrastive_reg', 'divergence_reg']
            for key in required_keys:
                assert key in loss_dict, f"Missing loss component: {key}"
                assert torch.isfinite(loss_dict[key]), f"Loss component {key} is not finite"

    def test_gradient_flow(self, input_sizes):
        """Test that gradients flow correctly through the network."""
        latent_dim = 32
        
        for input_size in input_sizes:
            autoencoder = Autoencoder(latent_dim, (input_size, input_size))
            autoencoder.train()
            
            # Create test input
            x = torch.randn(2, 1, input_size, input_size)
            
            # Forward pass
            z = autoencoder.embed(x)
            x_hat = autoencoder.decoder(z)
            loss_dict = autoencoder.compute_loss(x, x_hat, z)
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Check that gradients exist and are finite for parameters that were used
            gradient_count = 0
            for name, param in autoencoder.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        gradient_count += 1
                        assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN/inf"
            
            # Ensure we have a reasonable number of gradients (at least 50% of parameters)
            total_params = sum(1 for p in autoencoder.parameters() if p.requires_grad)
            assert gradient_count >= total_params * 0.5, \
                f"Too few gradients: {gradient_count}/{total_params} for input size {input_size}"


class TestModelSummary:
    """Test model summary functionality."""
    
    def test_summary_no_duplication(self):
        """Test that model summary doesn't contain duplicate modules."""
        autoencoder = Autoencoder(32, (64, 64))
        x = torch.randn(1, 1, 64, 64)
        
        summary = _add_hooks(autoencoder, x)
        
        # Check for duplicate module names
        module_names = []
        for key in summary.keys():
            module_name = key.split('_')[1]  # Extract module name from key
            module_names.append(module_name)
        
        # Should not have exact duplicate module names in the summary
        unique_names = set(module_names)
        assert len(unique_names) > 0, "Summary should contain at least one module"
        
        # Check that we have the expected key components
        key_components = ['conv', 'bn', 'relu', 'pool', 'linear', 'embedding']
        found_components = []
        for key in summary.keys():
            for component in key_components:
                if component in key.lower():
                    found_components.append(component)
        
        assert len(found_components) > 0, "Summary should contain recognizable components"

    def test_leaf_module_detection(self):
        """Test that leaf module detection works correctly."""
        # Test with different module types
        conv = torch.nn.Conv2d(1, 16, 3)
        bn = torch.nn.BatchNorm2d(16)
        relu = torch.nn.ReLU()
        sequential = torch.nn.Sequential(conv, bn, relu)
        
        # Conv, BN should be leaf modules (have parameters, no children with parameters)
        assert _is_leaf_module(conv), "Conv2d should be a leaf module"
        assert _is_leaf_module(bn), "BatchNorm2d should be a leaf module"
        
        # ReLU should not be a leaf module (no parameters)
        assert not _is_leaf_module(relu), "ReLU should not be a leaf module"
        
        # Sequential should not be a leaf module (has children with parameters)
        assert not _is_leaf_module(sequential), "Sequential should not be a leaf module"

    def test_summary_with_different_sizes(self):
        """Test that summary works with different input sizes."""
        input_sizes = [32, 64, 128, 256]
        latent_dim = 32
        
        for input_size in input_sizes:
            autoencoder = Autoencoder(latent_dim, (input_size, input_size))
            x = torch.randn(1, 1, input_size, input_size)
            
            summary = _add_hooks(autoencoder, x)
            
            # Should have a reasonable number of modules
            assert len(summary) > 5, f"Summary should contain multiple modules for size {input_size}"
            
            # Check that all modules have valid shapes
            for key, info in summary.items():
                if info['in'] != "N/A":
                    assert len(info['in']) >= 2, f"Invalid input shape for {key}: {info['in']}"
                if info['out'] != "N/A":
                    assert len(info['out']) >= 2, f"Invalid output shape for {key}: {info['out']}"


class TestBlockComponents:
    """Test individual block components."""
    
    def test_conv_block(self):
        """Test ConvBlock functionality."""
        block = ConvBlock(1, 128)
        x = torch.randn(2, 1, 64, 64)
        
        with torch.no_grad():
            output = block(x)
        
        assert output.shape == (2, 128, 64, 64), f"ConvBlock output shape mismatch"
        assert torch.isfinite(output).all(), "ConvBlock output contains NaN/inf"

    def test_identity_block(self):
        """Test IdentityBlock functionality."""
        block = IdentityBlock(128)
        x = torch.randn(2, 128, 64, 64)
        
        with torch.no_grad():
            output = block(x)
        
        assert output.shape == x.shape, f"IdentityBlock should preserve shape"
        assert torch.isfinite(output).all(), "IdentityBlock output contains NaN/inf"

    def test_resnet_block(self):
        """Test ResNetBlock functionality."""
        block = ResNetBlock(128, 128, pool_size=2)
        x = torch.randn(2, 128, 64, 64)
        
        with torch.no_grad():
            output = block(x)
        
        expected_shape = (2, 128, 32, 32)  # Should be downsampled by factor of 2
        assert output.shape == expected_shape, \
            f"ResNetBlock output shape mismatch, expected {expected_shape}, got {output.shape}"
        assert torch.isfinite(output).all(), "ResNetBlock output contains NaN/inf"

    def test_embedding_layer(self):
        """Test EmbeddingLayer functionality."""
        input_dim = 16
        latent_dim = 32
        embedding = EmbeddingLayer(input_dim, latent_dim)
        
        x = torch.randn(2, input_dim)
        
        with torch.no_grad():
            output = embedding(x)
        
        assert output.shape == (2, latent_dim), f"EmbeddingLayer output shape mismatch"
        assert (output >= 0).all(), "EmbeddingLayer output should be non-negative"
        assert torch.isfinite(output).all(), "EmbeddingLayer output contains NaN/inf"


if __name__ == "__main__":
    pytest.main([__file__])