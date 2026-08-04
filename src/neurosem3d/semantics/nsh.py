import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Any, Dict, List, Optional, Tuple, Union

# Dynamic backend detection
HAS_MINKOWSKI = False
try:
    import MinkowskiEngine as ME
    HAS_MINKOWSKI = True
except ImportError:
    pass

HAS_SPCONV = False
try:
    import spconv.pytorch as spconv
    HAS_SPCONV = True
except ImportError:
    pass

class DenseFallbackUNet(nn.Module):
    """Dense 3D U-Net fallback for sparse voxel inputs when ME/spconv are unavailable."""
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        logger.warning("Using DenseFallbackUNet: neither MinkowskiEngine nor spconv was detected.")
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(2, 16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(4, 32)
        
        self.up = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.conv_out = nn.Conv3d(32, out_channels, kernel_size=3, padding=1)
        
    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            coords (torch.Tensor): Shape (N_total, 4) where col 0 is batch_idx.
            feats (torch.Tensor): Shape (N_total, C_in).
            
        Returns:
            torch.Tensor: Gathered output features of shape (N_total, C_out).
        """
        N = coords.shape[0]
        if N == 0:
            return torch.zeros((0, self.conv_out.out_channels), device=feats.device)
            
        batch_size = int(coords[:, 0].max().item() + 1)
        spatial_shape = int(coords[:, 1:].max().item() + 1)
        # Pad to even dimension for pooling
        spatial_shape = max(32, (spatial_shape + 1) // 2 * 2)
        
        # 1. Densify sparse coordinates
        dense_tensor = torch.zeros((batch_size, feats.shape[1], spatial_shape, spatial_shape, spatial_shape), 
                                   device=feats.device, dtype=feats.dtype)
        # Fill coordinates
        # coords[:, 0] is batch, coords[:, 1] is z, coords[:, 2] is y, coords[:, 3] is x
        dense_tensor[coords[:, 0].long(), :, coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()] = feats
        
        # 2. U-Net operations
        # Encoder stage 1
        x1 = F.relu(self.gn1(self.conv1(dense_tensor)))  # (B, 16, res, res, res)
        # Pooling
        x1_pool = F.max_pool3d(x1, kernel_size=2)  # (B, 16, res/2, res/2, res/2)
        
        # Encoder stage 2
        x2 = F.relu(self.gn2(self.conv2(x1_pool)))  # (B, 32, res/2, res/2, res/2)
        
        # Decoder
        x_up = self.up(x2)  # (B, 16, res, res, res)
        x_concat = torch.cat([x_up, x1], dim=1)  # (B, 32, res, res, res)
        out_dense = self.conv_out(x_concat)  # (B, C_out, res, res, res)
        
        # 3. Gather back to sparse coordinates
        out_sparse = out_dense[coords[:, 0].long(), :, coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()]
        return out_sparse

class NeuralSemanticHead(nn.Module):
    """3D Neural Semantic Head with Evidential Dirichlet uncertainty modeling."""
    
    def __init__(
        self,
        in_channels: int = 272,  # 256 (latent) + 1 (SDF) + 15 (P_fuse)
        num_classes_per_level: Dict[str, int] = None
    ) -> None:
        """Initialize NeuralSemanticHead.
        
        Args:
            in_channels (int): input feature dimension.
            num_classes_per_level (Dict[str, int]): dictionary mapping 'coarse', 'middle', 'fine' to label counts.
        """
        super().__init__()
        if num_classes_per_level is None:
            num_classes_per_level = {"coarse": 3, "middle": 8, "fine": 15}
            
        self.num_classes_per_level = num_classes_per_level
        self.in_channels = in_channels
        
        # Sparse U-Net backbone (or dense fallback)
        self.backbone_channels = 64
        self.unet = DenseFallbackUNet(in_channels, self.backbone_channels)
        
        # Evidential Dirichlet concentration heads per tree level
        self.heads = nn.ModuleDict({
            lvl: nn.Linear(self.backbone_channels, K) for lvl, K in num_classes_per_level.items()
        })
        
    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Forward pass to extract logits and Dirichlet concentration parameters alpha.
        
        Args:
            coords (torch.Tensor): Coordinates tensor shape (N_total, 4).
            feats (torch.Tensor): Features tensor shape (N_total, C).
            
        Returns:
            Dict containing level mappings to concentration parameter alpha and logits.
        """
        # Extract features using 3D U-Net backbone
        out_feats = self.unet(coords, feats)  # (N_total, backbone_channels)
        
        outputs = {}
        for lvl, head in self.heads.items():
            logits = head(out_feats)  # (N_total, K)
            
            # Evidential Dirichlet concentration parameter alpha
            # alpha = softplus(logits) + 1.0
            alpha = F.softplus(logits) + 1.0
            
            outputs[lvl] = {
                "logits": logits,
                "alpha": alpha
            }
            
        return outputs

    def decode(
        self,
        outputs: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Decode Dirichlet concentration parameters into categorical predictions and calibrated confidence.
        
        Args:
            outputs (Dict): outputs from forward pass containing 'alpha' and 'logits'.
            
        Returns:
            Dict containing per-level decoded output dictionaries:
                - 'prediction': Argmax class prediction (N_total,).
                - 'confidence': Calibrated confidence u(v) = 1 - K / sum(alpha) (N_total,).
                - 'prob': Categorical probabilities (N_total, K).
        """
        decoded = {}
        for lvl, out in outputs.items():
            alpha = out["alpha"]
            logits = out["logits"]
            K = self.num_classes_per_level[lvl]
            
            # Sum of concentration parameters S
            S = torch.sum(alpha, dim=-1, keepdim=True)  # (N_total, 1)
            
            # Categorical probability p_k = alpha_k / S
            prob = alpha / S
            prediction = torch.argmax(prob, dim=-1)
            
            # Calibrated confidence u(v) = 1.0 - K / S
            confidence = 1.0 - (K / S.squeeze(-1))
            
            decoded[lvl] = {
                "prediction": prediction,
                "confidence": confidence,
                "prob": prob
            }
            
        return decoded

def main() -> None:
    """Main entry point for testing or running nsh independently."""
    parser = argparse.ArgumentParser(description="NEW Work-2 module: 3.2 Sparse 3D Neural Semantic Head + uncertainty.")
    args = parser.parse_args()
    logger.info("Running nsh verify stub")

if __name__ == "__main__":
    main()
