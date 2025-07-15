"""ResNet blocks and other neural network components for 4D-STEM autoencoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with 3 sequential conv layers and skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int = 128, kernel_size: int = 3):
        super().__init__()
        
        # First conv layer adjusts channel dimensions if needed
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second and third conv layers maintain dimensions
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Skip connection projection if input/output channels differ
        self.skip_connection = nn.Identity()
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip_connection(x)
        
        # Three sequential conv layers with conditional batch norm
        out = self.conv1(x)
        if out.size(-1) > 1 or out.size(-2) > 1:  # Only apply batch norm if spatial dims > 1x1
            out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        if out.size(-1) > 1 or out.size(-2) > 1:
            out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        if out.size(-1) > 1 or out.size(-2) > 1:
            out = self.bn3(out)
        
        # Add skip connection
        out += identity
        out = self.relu(out)
        
        return out


class IdentityBlock(nn.Module):
    """Identity block with single conv layer and normalization."""
    
    def __init__(self, channels: int = 128, kernel_size: int = 3):
        super().__init__()
        
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if out.size(-1) > 1 or out.size(-2) > 1:  # Only apply batch norm if spatial dims > 1x1
            out = self.bn(out)
        return self.relu(out)


class ResNetBlock(nn.Module):
    """Complete ResNet block with ConvBlock + IdentityBlock + adaptive pooling sequence."""
    
    def __init__(self, in_channels: int, out_channels: int = 128, pool_size: int = 2):
        super().__init__()
        
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.identity_block = IdentityBlock(out_channels)
        self.pool_size = pool_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = self.identity_block(x)
        
        # Calculate target size - ensure it's at least 1x1
        current_size = x.size(-1)
        target_size = max(1, current_size // self.pool_size)
        
        # Use adaptive pooling for size-agnostic processing
        x = F.adaptive_avg_pool2d(x, (target_size, target_size))
        
        return x


class ResNetUpBlock(nn.Module):
    """ResNet block with upsampling for decoder."""
    
    def __init__(self, in_channels: int, out_channels: int = 128, scale_factor: int = 2):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.identity_block = IdentityBlock(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv_block(x)
        x = self.identity_block(x)
        return x


class EmbeddingLayer(nn.Module):
    """Embedding layer with overcomplete design and non-negative activations."""
    
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        
        self.linear = nn.Linear(input_dim, latent_dim)
        self.relu = nn.ReLU(inplace=True)  # Enforces non-negative activations
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x))


class AdaptiveDecoder(nn.Module):
    """Adaptive decoder that handles different input sizes with fixed parameter count."""
    
    def __init__(self, latent_dim: int = 32, target_size: int = 256):
        super().__init__()
        
        self.target_size = target_size
        
        # Use fixed base size for consistent parameters across different target sizes
        self.base_size = 4  # Always start from 4x4
        self.base_features = self.base_size * self.base_size * 128  # 4*4*128 = 2048
        
        # Linear layer to expand latent to feature map - fixed size
        self.linear = nn.Linear(latent_dim, self.base_features)
        
        # Initial conv layer
        self.conv_initial = nn.Conv2d(128, 128, 3, padding=1)
        
        # Three upsampling ResNet blocks
        self.resnet_up1 = ResNetUpBlock(128, 128, scale_factor=4)  # 4x -> 16x
        self.resnet_up2 = ResNetUpBlock(128, 128, scale_factor=4)  # 16x -> 64x  
        self.resnet_up3 = ResNetUpBlock(128, 128, scale_factor=4)  # 64x -> 256x
        
        # Final conv layer
        self.conv_final = nn.Conv2d(128, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        
        # Expand latent to fixed 4x4 feature map
        x = self.linear(z)
        x = x.view(batch_size, 128, self.base_size, self.base_size)
        
        # Initial conv - ensure we don't get 0x0 tensors
        if x.size(-1) > 0 and x.size(-2) > 0:
            x = self.conv_initial(x)
        else:
            # Fallback for edge cases
            x = torch.zeros(batch_size, 128, 4, 4, device=z.device, dtype=z.dtype)
        
        # Three upsampling blocks
        x = self.resnet_up1(x)
        x = self.resnet_up2(x)
        x = self.resnet_up3(x)
        
        # Final conv and activation
        x = self.conv_final(x)
        x = self.sigmoid(x)
        
        # Ensure exact target size
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size), mode='bilinear', align_corners=False)
        
        return x


class ContrastiveLoss(nn.Module):
    """Contrastive similarity regularization loss."""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Calculate pairwise distances
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Compute cosine similarity matrix
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.t())
        
        # Create positive and negative pairs (simplified)
        loss = 0.0
        count = 0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                sim = similarity_matrix[i, j]
                # Encourage diversity (negative pairs)
                loss += F.relu(sim - self.margin)
                count += 1
        
        return loss / count if count > 0 else torch.tensor(0.0, device=embeddings.device)


class DivergenceLoss(nn.Module):
    """Activation divergence regularization loss."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Encourage different activations across the batch
        # Penalize if activations are too similar across batch
        variance = torch.var(embeddings, dim=0)
        return torch.mean(1.0 / (variance + 1e-8))