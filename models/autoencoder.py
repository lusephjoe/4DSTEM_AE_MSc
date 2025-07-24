"""ResNet-based autoencoder for 4D-STEM diffraction patterns."""
import torch
from torch import nn
import torch.nn.functional as F
from .blocks import ResNetBlock, ConvBlock, IdentityBlock, EmbeddingLayer, AdaptiveDecoder
from .losses import LossManager, create_loss_config_from_args

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
    """ResNet-based autoencoder with flexible loss system."""
    
    def __init__(self, latent_dim: int = 32, out_shape: tuple[int,int] = (256, 256), 
                 loss_config: dict = None):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, out_shape)
        
        # Initialize flexible loss system
        if loss_config is None:
            loss_config = {
                'reconstruction_loss': 'mse',
                'regularization_losses': {
                    'lp_reg': 1e-4,
                    'contrastive': 5e-5,
                    'divergence': 2e-4
                }
            }
        
        self.loss_manager = LossManager(
            reconstruction_loss=loss_config['reconstruction_loss'],
            regularization_losses=loss_config.get('regularization_losses', {})
        )

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
    
    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor, 
                    **kwargs) -> dict:
        """Compute loss using flexible loss system."""
        return self.loss_manager.compute_loss(x, x_hat, z, **kwargs)
    
    def get_loss_info(self) -> dict:
        """Get information about configured losses."""
        return self.loss_manager.get_loss_names()
    
    def update_loss_config(self, loss_config: dict):
        """Update loss configuration."""
        self.loss_manager = LossManager(
            reconstruction_loss=loss_config['reconstruction_loss'],
            regularization_losses=loss_config.get('regularization_losses', {})
        )