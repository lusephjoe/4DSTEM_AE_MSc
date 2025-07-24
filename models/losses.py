"""
Flexible loss function system for autoencoder training.
Supports multiple reconstruction losses and regularization terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import math


class BaseLoss(ABC):
    """Abstract base class for all loss functions."""
    
    @abstractmethod
    def forward(self, x_true: torch.Tensor, x_pred: torch.Tensor, 
                z: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """Compute the loss."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the loss function."""
        pass


class ReconstructionLoss(BaseLoss):
    """Base class for reconstruction losses."""
    pass


class RegularizationLoss(BaseLoss):
    """Base class for regularization losses that require latent representations."""
    pass


# ================================
# RECONSTRUCTION LOSSES
# ================================

class MSELoss(ReconstructionLoss):
    """Mean Squared Error loss."""
    
    def __init__(self):
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x_true: torch.Tensor, x_pred: torch.Tensor, 
                z: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        return self.loss_fn(x_pred, x_true)
    
    @property
    def name(self) -> str:
        return "mse"


class L1Loss(ReconstructionLoss):
    """L1 (Mean Absolute Error) loss."""
    
    def __init__(self):
        self.loss_fn = nn.L1Loss()
    
    def forward(self, x_true: torch.Tensor, x_pred: torch.Tensor, 
                z: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        return self.loss_fn(x_pred, x_true)
    
    @property
    def name(self) -> str:
        return "l1"


class HuberLoss(ReconstructionLoss):
    """Huber loss (smooth L1 loss)."""
    
    def __init__(self, delta: float = 1.0):
        self.loss_fn = nn.HuberLoss(delta=delta)
        self.delta = delta
    
    def forward(self, x_true: torch.Tensor, x_pred: torch.Tensor, 
                z: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        return self.loss_fn(x_pred, x_true)
    
    @property
    def name(self) -> str:
        return f"huber_{self.delta}"


class SSIMLoss(ReconstructionLoss):
    """Structural Similarity Index loss (1 - SSIM)."""
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        self.window_size = window_size
        self.sigma = sigma
        
    def _gaussian_window(self, size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """Create Gaussian window for SSIM computation."""
        coords = torch.arange(size, dtype=torch.float32, device=device)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.unsqueeze(0) * g.unsqueeze(1)
    
    def forward(self, x_true: torch.Tensor, x_pred: torch.Tensor, 
                z: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Create Gaussian window
        window = self._gaussian_window(self.window_size, self.sigma, x_true.device)
        window = window.unsqueeze(0).unsqueeze(0)
        
        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Compute means
        mu1 = F.conv2d(x_true, window, stride=1, padding=self.window_size//2, groups=1)
        mu2 = F.conv2d(x_pred, window, stride=1, padding=self.window_size//2, groups=1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = F.conv2d(x_true * x_true, window, stride=1, padding=self.window_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(x_pred * x_pred, window, stride=1, padding=self.window_size//2, groups=1) - mu2_sq
        sigma12 = F.conv2d(x_true * x_pred, window, stride=1, padding=self.window_size//2, groups=1) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Return 1 - SSIM as loss
        return 1 - ssim_map.mean()
    
    @property
    def name(self) -> str:
        return f"ssim_{self.window_size}_{self.sigma}"


class PerceptualLoss(ReconstructionLoss):
    """Perceptual loss using simple convolutional features."""
    
    def __init__(self, feature_layers: list = [0, 1, 2]):
        self.feature_layers = feature_layers
        # Simple feature extractor using conv layers
        self.features = nn.ModuleList([
            nn.Conv2d(1, 32, 3, padding=1),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Conv2d(64, 128, 3, padding=1)
        ])
        
        # Freeze feature extractor
        for param in self.features.parameters():
            param.requires_grad = False
    
    def _extract_features(self, x: torch.Tensor) -> list:
        """Extract features from multiple layers."""
        features = []
        h = x
        for i, layer in enumerate(self.features):
            h = F.relu(layer(h))
            if i in self.feature_layers:
                features.append(h)
        return features
    
    def forward(self, x_true: torch.Tensor, x_pred: torch.Tensor, 
                z: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Extract features
        features_true = self._extract_features(x_true)
        features_pred = self._extract_features(x_pred)
        
        # Compute L2 loss between features
        loss = 0
        for f_true, f_pred in zip(features_true, features_pred):
            loss += F.mse_loss(f_pred, f_true)
        
        return loss / len(features_true)
    
    @property
    def name(self) -> str:
        return f"perceptual_{len(self.feature_layers)}"


# ================================
# REGULARIZATION LOSSES
# ================================

class L1RegularizationLoss(RegularizationLoss):
    """L1 regularization on latent representations."""
    
    def forward(self, x_true: torch.Tensor, x_pred: torch.Tensor, 
                z: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if z is None:
            raise ValueError("L1 regularization requires latent representation z")
        return torch.mean(torch.abs(z))
    
    @property
    def name(self) -> str:
        return "l1_reg"


class L2RegularizationLoss(RegularizationLoss):
    """L2 regularization on latent representations."""
    
    def forward(self, x_true: torch.Tensor, x_pred: torch.Tensor, 
                z: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if z is None:
            raise ValueError("L2 regularization requires latent representation z")
        return torch.mean(z ** 2)
    
    @property
    def name(self) -> str:
        return "l2_reg"


class LpRegularizationLoss(RegularizationLoss):
    """Lp norm regularization on latent representations."""
    
    def __init__(self, p: int = 1):
        self.p = p
    
    def forward(self, x_true: torch.Tensor, x_pred: torch.Tensor, 
                z: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if z is None:
            raise ValueError("Lp regularization requires latent representation z")
        
        batch_size = z.shape[0]
        lp_reg = torch.norm(z, self.p) / batch_size
        
        # Handle zero case
        if lp_reg == 0:
            lp_reg = torch.tensor(0.5, device=z.device)
        
        return lp_reg
    
    @property
    def name(self) -> str:
        return f"l{self.p}_reg"


class ContrastiveLoss(RegularizationLoss):
    """Contrastive similarity regularization."""
    
    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature
    
    def forward(self, x_true: torch.Tensor, x_pred: torch.Tensor, 
                z: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if z is None:
            raise ValueError("Contrastive loss requires latent representation z")
        
        batch_size = z.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=z.device)
        
        # Normalize embeddings
        z_norm = F.normalize(z, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z_norm, z_norm.t()) / self.temperature
        
        # Create labels (positive pairs are identical samples)
        labels = torch.arange(batch_size, device=z.device)
        
        # Compute contrastive loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    @property
    def name(self) -> str:
        return f"contrastive_{self.temperature}"


class DivergenceLoss(RegularizationLoss):
    """Activation divergence regularization to encourage diversity."""
    
    def forward(self, x_true: torch.Tensor, x_pred: torch.Tensor, 
                z: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if z is None:
            raise ValueError("Divergence loss requires latent representation z")
        
        batch_size = z.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=z.device)
        
        # Compute pairwise distances
        z_expanded = z.unsqueeze(1)  # [B, 1, D]
        z_transposed = z.unsqueeze(0)  # [1, B, D]
        
        # L2 distance between all pairs
        distances = torch.norm(z_expanded - z_transposed, dim=2)  # [B, B]
        
        # Exclude diagonal (distance to self)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=z.device)
        distances = distances[mask]
        
        # Return negative mean distance to encourage diversity
        return -torch.mean(distances)
    
    @property
    def name(self) -> str:
        return "divergence"


class KLDivergenceLoss(RegularizationLoss):
    """KL divergence regularization (useful for VAE-style training)."""
    
    def __init__(self, beta: float = 1.0):
        self.beta = beta
    
    def forward(self, x_true: torch.Tensor, x_pred: torch.Tensor, 
                z: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if z is None:
            raise ValueError("KL divergence requires latent representation z")
        
        # Assume z represents mean of latent distribution
        # For full VAE, you'd need both mean and log_var
        mu = z
        log_var = kwargs.get('log_var', torch.zeros_like(mu))
        
        # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0,I)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return self.beta * torch.mean(kl_loss)
    
    @property
    def name(self) -> str:
        return f"kl_div_{self.beta}"


# ================================
# COMBINED LOSS MANAGER
# ================================

class LossManager:
    """Manager class for combining multiple loss functions."""
    
    def __init__(self, reconstruction_loss: str = "mse", regularization_losses: Optional[Dict[str, float]] = None):
        """
        Initialize loss manager.
        
        Args:
            reconstruction_loss: Name of reconstruction loss ("mse", "l1", "huber", "ssim", "perceptual")
            regularization_losses: Dict of {loss_name: weight} for regularization terms
        """
        self.reconstruction_loss = self._create_reconstruction_loss(reconstruction_loss)
        self.regularization_losses = {}
        
        if regularization_losses:
            for loss_name, weight in regularization_losses.items():
                self.regularization_losses[loss_name] = {
                    'loss': self._create_regularization_loss(loss_name),
                    'weight': weight
                }
    
    def _create_reconstruction_loss(self, loss_name: str) -> ReconstructionLoss:
        """Create reconstruction loss from name."""
        loss_map = {
            "mse": MSELoss,
            "l1": L1Loss,
            "huber": lambda: HuberLoss(delta=1.0),
            "huber_0.5": lambda: HuberLoss(delta=0.5),
            "huber_2.0": lambda: HuberLoss(delta=2.0),
            "ssim": SSIMLoss,
            "perceptual": PerceptualLoss
        }
        
        if loss_name not in loss_map:
            raise ValueError(f"Unknown reconstruction loss: {loss_name}. Available: {list(loss_map.keys())}")
        
        return loss_map[loss_name]()
    
    def _create_regularization_loss(self, loss_name: str) -> RegularizationLoss:
        """Create regularization loss from name."""
        loss_map = {
            "l1_reg": L1RegularizationLoss,
            "l2_reg": L2RegularizationLoss,
            "lp_reg": lambda: LpRegularizationLoss(p=1),
            "l2_norm": lambda: LpRegularizationLoss(p=2),
            "contrastive": ContrastiveLoss,
            "divergence": DivergenceLoss,
            "kl_div": KLDivergenceLoss
        }
        
        if loss_name not in loss_map:
            raise ValueError(f"Unknown regularization loss: {loss_name}. Available: {list(loss_map.keys())}")
        
        return loss_map[loss_name]()
    
    def compute_loss(self, x_true: torch.Tensor, x_pred: torch.Tensor, 
                     z: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and individual components.
        
        Returns:
            Dictionary with 'total_loss' and individual loss components
        """
        # Compute reconstruction loss
        recon_loss = self.reconstruction_loss.forward(x_true, x_pred, z, **kwargs)
        
        # Initialize total loss with reconstruction loss
        total_loss = recon_loss
        
        # Loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss,
            f'{self.reconstruction_loss.name}_loss': recon_loss
        }
        
        # Add regularization losses
        for loss_name, loss_config in self.regularization_losses.items():
            reg_loss = loss_config['loss'].forward(x_true, x_pred, z, **kwargs)
            weight = loss_config['weight']
            
            # Add to total loss with weight
            if loss_name == "divergence":
                # Subtract divergence loss to encourage diversity
                total_loss -= weight * reg_loss
            else:
                total_loss += weight * reg_loss
            
            # Add to loss dictionary
            loss_dict[f'{loss_config["loss"].name}'] = reg_loss
            loss_dict[f'weighted_{loss_config["loss"].name}'] = weight * reg_loss
        
        # Update total loss
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
    
    def get_loss_names(self) -> Dict[str, str]:
        """Get names of all configured losses."""
        names = {
            'reconstruction': self.reconstruction_loss.name
        }
        
        for loss_name, loss_config in self.regularization_losses.items():
            names[f'regularization_{loss_name}'] = loss_config['loss'].name
        
        return names


# ================================
# UTILITY FUNCTIONS
# ================================

def get_available_losses() -> Dict[str, list]:
    """Get list of all available loss functions."""
    return {
        'reconstruction': ["mse", "l1", "huber", "huber_0.5", "huber_2.0", "ssim", "perceptual"],
        'regularization': ["l1_reg", "l2_reg", "lp_reg", "l2_norm", "contrastive", "divergence", "kl_div"]
    }


def create_loss_config_from_args(args) -> Dict[str, Any]:
    """Create loss configuration from command line arguments."""
    config = {
        'reconstruction_loss': getattr(args, 'loss_function', 'mse'),
        'regularization_losses': {}
    }
    
    # Add regularization losses based on args
    if hasattr(args, 'lambda_act') and args.lambda_act > 0:
        config['regularization_losses']['lp_reg'] = args.lambda_act
    
    if hasattr(args, 'lambda_sim') and args.lambda_sim > 0:
        config['regularization_losses']['contrastive'] = args.lambda_sim
    
    if hasattr(args, 'lambda_div') and args.lambda_div > 0:
        config['regularization_losses']['divergence'] = args.lambda_div
    
    if hasattr(args, 'lambda_l2') and args.lambda_l2 > 0:
        config['regularization_losses']['l2_reg'] = args.lambda_l2
    
    if hasattr(args, 'lambda_kl') and args.lambda_kl > 0:
        config['regularization_losses']['kl_div'] = args.lambda_kl
    
    return config