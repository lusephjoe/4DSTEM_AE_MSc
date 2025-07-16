"""ResNet-based autoencoder for 4D-STEM diffraction patterns."""
import torch
from torch import nn
import torch.nn.functional as F
from .blocks import ResNetBlock, ConvBlock, IdentityBlock, EmbeddingLayer, AdaptiveDecoder, ContrastiveLoss, DivergenceLoss

class Encoder(nn.Module):
    """ResNet-based encoder for image-size agnostic processing."""
    
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        
        # Additional initial conv layers
        self.conv_input = nn.Conv2d(1, 64, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)
        self.relu_input = nn.ReLU(inplace=True)
        
        self.conv_pre = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_pre = nn.BatchNorm2d(128)
        self.relu_pre = nn.ReLU(inplace=True)
        
        # Three ResNet blocks with pooling (4x reduction each)
        # 256x256 -> 64x64 -> 16x16 -> 4x4
        self.resnet1 = ResNetBlock(128, 128, pool_size=4)
        self.resnet2 = ResNetBlock(128, 128, pool_size=4)  
        self.resnet3 = ResNetBlock(128, 128, pool_size=4)
        
        # Additional final conv layers
        self.conv_post = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_post = nn.BatchNorm2d(64)
        self.relu_post = nn.ReLU(inplace=True)
        
        self.conv_final = nn.Conv2d(64, 1, 3, padding=1)
        self.bn_final = nn.BatchNorm2d(1)
        self.relu_final = nn.ReLU(inplace=True)
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Embedding layer
        self.embedding = EmbeddingLayer(16, latent_dim)  # 4*4*1 = 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Additional initial conv layers
        x = self.relu_input(self.bn_input(self.conv_input(x)))
        x = self.relu_pre(self.bn_pre(self.conv_pre(x)))
        
        # Three ResNet blocks (ConvBlock + IdentityBlock + MaxPool2d)
        x = self.resnet1(x)
        x = self.resnet2(x)
        x = self.resnet3(x)
        
        # Additional final conv layers with conditional batch norm
        x = self.conv_post(x)
        if x.size(-1) > 1 or x.size(-2) > 1:  # Only apply batch norm if spatial dims > 1x1
            x = self.bn_post(x)
        x = self.relu_post(x)
        
        x = self.conv_final(x)
        if x.size(-1) > 1 or x.size(-2) > 1:
            x = self.bn_final(x)
        x = self.relu_final(x)
        
        # Adaptive pooling for size-agnostic processing
        x = self.adaptive_pool(x)
        
        # Flatten and embed
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        
        return x

class Decoder(nn.Module):
    """ResNet-based decoder with upsampling."""
    
    def __init__(self, latent_dim: int = 32, out_shape: tuple[int,int] = (256, 256)):
        super().__init__()
        qy, qx = out_shape
        self.target_size = max(qy, qx)  # Use square size for simplicity
        
        # Use adaptive decoder
        self.decoder = AdaptiveDecoder(latent_dim, self.target_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class Autoencoder(nn.Module):
    """ResNet-based autoencoder with regularized loss."""
    
    def __init__(self, latent_dim: int = 32, out_shape: tuple[int,int] = (256, 256)):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, out_shape)
        
        # Loss components
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.contrastive_loss = ContrastiveLoss()
        self.divergence_loss = DivergenceLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        # Make decoder output match input size
        output = self.decoder(z)
        
        # Resize output to match input if needed
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return output

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation without decoding."""
        return self.encoder(x)
    
    def weighted_mse_loss(self, x: torch.Tensor, x_hat: torch.Tensor, background_weight: float = 0.1) -> torch.Tensor:
        """
        Weighted MSE loss that emphasizes diffraction spots over background.
        
        Args:
            x: Ground truth diffraction patterns
            x_hat: Reconstructed diffraction patterns  
            background_weight: Weight for background pixels (default: 0.1)
            
        Returns:
            Weighted MSE loss
        """
        # Identify signal vs background regions using adaptive threshold
        # Use mean + 2*std as threshold to identify diffraction spots
        batch_mean = x.mean(dim=(-2, -1), keepdim=True)
        batch_std = x.std(dim=(-2, -1), keepdim=True)
        threshold = batch_mean + 2 * batch_std
        
        # Create signal mask (True for diffraction spots)
        signal_mask = x > threshold
        
        # Apply different weights: 1.0 for signal, background_weight for background
        weights = torch.where(signal_mask, 1.0, background_weight)
        
        # Compute weighted MSE
        squared_diff = (x - x_hat) ** 2
        weighted_mse = torch.mean(weights * squared_diff)
        
        return weighted_mse
    
    def high_intensity_loss(self, x: torch.Tensor, x_hat: torch.Tensor, threshold: float = 0.8) -> torch.Tensor:
        """
        Focus reconstruction loss on high-intensity diffraction spots.
        
        Args:
            x: Ground truth diffraction patterns
            x_hat: Reconstructed diffraction patterns
            threshold: Quantile threshold for high-intensity regions (default: 0.8)
            
        Returns:
            MSE loss for high-intensity regions only
        """
        # Find threshold value for top percentile of intensities
        threshold_val = torch.quantile(x.flatten(-2, -1), threshold, dim=-1, keepdim=True)
        threshold_val = threshold_val.unsqueeze(-1)  # Add spatial dimensions
        
        # Create mask for high-intensity regions
        mask = x > threshold_val
        
        # Apply loss only to high-intensity regions
        if mask.sum() > 0:
            return F.mse_loss(x[mask], x_hat[mask])
        else:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    
    def multiscale_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale loss to capture both fine details and overall structure.
        
        Args:
            x: Ground truth diffraction patterns
            x_hat: Reconstructed diffraction patterns
            
        Returns:
            Average MSE loss across multiple scales
        """
        loss = 0
        scales = [1, 2, 4]
        
        for scale in scales:
            if scale > 1:
                # Downsample using average pooling
                x_down = F.avg_pool2d(x, scale)
                x_hat_down = F.avg_pool2d(x_hat, scale)
            else:
                x_down, x_hat_down = x, x_hat
            
            loss += F.mse_loss(x_down, x_hat_down)
        
        return loss / len(scales)
    
    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor, 
                    lambda_act: float = 1e-4, lambda_sim: float = 5e-5, lambda_div: float = 2e-4, 
                    lambda_weighted: float = 0.5, lambda_high: float = 0.3, lambda_multi: float = 0.2,
                    ln_parm: int = 1, use_improved_loss: bool = True) -> dict:
        """
        Compute regularized loss with improved components for sparse diffraction patterns.
        
        Args:
            x: Ground truth diffraction patterns
            x_hat: Reconstructed diffraction patterns
            z: Latent representations
            lambda_act: L1 regularization weight
            lambda_sim: Contrastive loss weight
            lambda_div: Divergence loss weight
            lambda_weighted: Weighted MSE loss weight
            lambda_high: High-intensity region loss weight  
            lambda_multi: Multi-scale loss weight
            ln_parm: Norm parameter for regularization
            use_improved_loss: Whether to use improved loss components (default: True)
            
        Returns:
            Dictionary with all loss components
        """
        
        if use_improved_loss:
            # Improved loss components for sparse diffraction patterns
            # Original MSE with reduced weight
            mse_loss = self.mse_loss(x_hat, x) * 0.3
            
            # Weighted loss for sparse patterns
            weighted_loss = self.weighted_mse_loss(x, x_hat) * lambda_weighted
            
            # High-intensity region loss
            high_loss = self.high_intensity_loss(x, x_hat) * lambda_high
            
            # Multi-scale loss
            multi_loss = self.multiscale_loss(x, x_hat) * lambda_multi
            
            # Combined reconstruction loss
            reconstruction_loss = mse_loss + weighted_loss + high_loss + multi_loss
            
        else:
            # Original loss function for comparison
            reconstruction_loss = self.mse_loss(x_hat, x)
            weighted_loss = torch.tensor(0.0, device=x.device)
            high_loss = torch.tensor(0.0, device=x.device)
            multi_loss = torch.tensor(0.0, device=x.device)
            mse_loss = reconstruction_loss
        
        # Lp norm regularization with batch normalization (like m3_learning)
        batch_size = x.shape[0]
        lp_reg = torch.norm(z, ln_parm) / batch_size
        
        # Handle zero case like m3_learning
        if lp_reg == 0:
            lp_reg = torch.tensor(0.5, device=z.device)
        
        # Contrastive similarity regularization
        contrastive_reg = self.contrastive_loss(z)
        
        # Activation divergence regularization
        divergence_reg = self.divergence_loss(z)
        
        # Total loss (subtract divergence to encourage diversity like m3_learning)
        total_loss = (reconstruction_loss + 
                     lambda_act * lp_reg + 
                     lambda_sim * contrastive_reg - 
                     lambda_div * divergence_reg)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'mse_loss': mse_loss,
            'weighted_loss': weighted_loss,
            'high_intensity_loss': high_loss,
            'multiscale_loss': multi_loss,
            'lp_reg': lp_reg,
            'contrastive_reg': contrastive_reg,
            'divergence_reg': divergence_reg
        }