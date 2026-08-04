import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Any, Dict, List, Optional, Tuple, Union

class RoughGridEncoder(nn.Module):
    """FROZEN Work-1 RoughGridEncoder mapping 128^3 TSDF volume to a 16^3x256 latent grid."""
    
    def __init__(self) -> None:
        super().__init__()
        logger.debug("Initializing RoughGridEncoder")
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)  # 128 -> 64
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)  # 64 -> 32
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)  # 32 -> 16
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input TSDF volume of shape (B, 1, 128, 128, 128).
            
        Returns:
            torch.Tensor: Latent grid of shape (B, 256, 16, 16, 16).
        """
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        x = self.act(self.conv2(x))
        x = self.pool2(x)
        x = self.act(self.conv3(x))
        x = self.pool3(x)
        x = self.act(self.conv4(x))
        return x

class NeuralSDFDecoder(nn.Module):
    """FROZEN Work-1 8-layer MLP mapping query point p and sampled latent z_p to signed distance s(p)."""
    
    def __init__(self) -> None:
        super().__init__()
        logger.debug("Initializing NeuralSDFDecoder")
        self.act = nn.Softplus(beta=100.0)
        
        # Input: p (3) + z_p (256) = 259
        self.l1 = nn.Linear(259, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        self.l4 = nn.Linear(512, 512)
        # Skip connection concatenates original input of size 259 -> 512 + 259 = 771
        self.l5 = nn.Linear(771, 512)
        self.l6 = nn.Linear(512, 512)
        self.l7 = nn.Linear(512, 512)
        self.l8 = nn.Linear(512, 1)

    def forward(self, p: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            p (torch.Tensor): Query points of shape (B, N, 3).
            z_p (torch.Tensor): Sampled latent features of shape (B, N, 256).
            
        Returns:
            torch.Tensor: Signed distance values of shape (B, N, 1).
        """
        x_in = torch.cat([p, z_p], dim=-1)  # (B, N, 259)
        
        x = self.act(self.l1(x_in))
        x = self.act(self.l2(x))
        x = self.act(self.l3(x))
        x = self.act(self.l4(x))
        
        x = torch.cat([x, x_in], dim=-1)  # (B, N, 771)
        x = self.act(self.l5(x))
        x = self.act(self.l6(x))
        x = self.act(self.l7(x))
        x = self.l8(x)  # (B, N, 1)
        return x

class NeuralRefiner(nn.Module):
    """FROZEN Work-1 wrapper model containing RoughGridEncoder and NeuralSDFDecoder."""
    
    def __init__(self) -> None:
        super().__init__()
        self.encoder = RoughGridEncoder()
        self.decoder = NeuralSDFDecoder()

    def forward(self, tsdf_volume: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to compute signed distance and retrieve latent grid.
        
        Args:
            tsdf_volume (torch.Tensor): Shape (B, 1, 128, 128, 128).
            points (torch.Tensor): Shape (B, N, 3).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Signed distance values of shape (B, N, 1).
                - Latent grid of shape (B, 256, 16, 16, 16).
        """
        z = self.encoder(tsdf_volume)
        z_p = sample_latent(z, points)
        s = self.decoder(points, z_p)
        return s, z

def sample_latent(z: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Sample latent features from a 16^3 grid via trilinear interpolation.
    
    Args:
        z (torch.Tensor): Latent grid of shape (B, 256, 16, 16, 16).
        points (torch.Tensor): Query points of shape (B, N, 3), normalized to [-1, 1].
        
    Returns:
        torch.Tensor: Sampled features of shape (B, N, 256).
    """
    B, C, D, H, W = z.shape
    # grid_sample expects coordinates of shape (B, D_out, H_out, W_out, 3)
    # points shape: (B, N, 3) -> reshape to (B, N, 1, 1, 3)
    grid = points.unsqueeze(2).unsqueeze(3)  # (B, N, 1, 1, 3)
    
    # sampled shape: (B, C, N, 1, 1)
    sampled = F.grid_sample(z, grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    # Reshape to (B, N, C)
    sampled = sampled.squeeze(-1).squeeze(-1).transpose(1, 2)
    return sampled

def load_frozen(ckpt_path: Optional[str] = None) -> NeuralRefiner:
    """Load the frozen NeuralRefiner model from checkpoint.
    
    Args:
        ckpt_path (Optional[str]): Path to the model checkpoint.
        
    Returns:
        NeuralRefiner: Evaluated, frozen NeuralRefiner instance.
    """
    model = NeuralRefiner()
    if ckpt_path is not None:
        logger.info(f"Loading frozen NeuralRefiner from checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    else:
        logger.warning("No checkpoint path provided. Initializing NeuralRefiner with random weights.")
        
    model.eval()
    model.requires_grad_(False)
    
    # Assert no parameter is trainable
    for name, param in model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} is not frozen!"
        
    return model

def main() -> None:
    """Main entry point for testing or running neural_sdf independently."""
    parser = argparse.ArgumentParser(description="FROZEN Work-1 module: Neural SDF encoder.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file.")
    args = parser.parse_args()
    
    logger.info("Initializing neural_sdf verification...")
    model = load_frozen(args.ckpt)
    logger.info("NeuralRefiner successfully loaded and frozen.")

if __name__ == "__main__":
    main()
