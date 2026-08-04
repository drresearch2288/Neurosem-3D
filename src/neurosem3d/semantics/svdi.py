"""NEW Work-2 module: 3.4 Sparse-voxel distilled inference."""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Any, Dict, List, Optional, Tuple, Union

from neurosem3d.semantics.nsh import NeuralSemanticHead
from neurosem3d.semantics.hierarchy import incremental_relabel, tree_consistent_decode

class StudentDenseFallbackUNet(nn.Module):
    """Student 3D U-Net fallback with reduced width (channels {16, 32}) and 2 stages."""
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(2, 16)
        
        self.conv2 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(2, 16)
        
        self.up = nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2)
        self.conv_out = nn.Conv3d(32, out_channels, kernel_size=3, padding=1)
        
    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        N = coords.shape[0]
        if N == 0:
            return torch.zeros((0, self.conv_out.out_channels), device=feats.device)
            
        batch_size = int(coords[:, 0].max().item() + 1)
        spatial_shape = int(coords[:, 1:].max().item() + 1)
        spatial_shape = max(32, (spatial_shape + 1) // 2 * 2)
        
        dense_tensor = torch.zeros(
            (batch_size, feats.shape[1], spatial_shape, spatial_shape, spatial_shape),
            device=feats.device, dtype=feats.dtype
        )
        dense_tensor[coords[:, 0].long(), :, coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()] = feats
        
        # Encoder Stage 1
        x1 = F.relu(self.gn1(self.conv1(dense_tensor)))  # (B, 16, res, res, res)
        x1_pool = F.max_pool3d(x1, kernel_size=2)         # (B, 16, res/2, res/2, res/2)
        
        # Encoder Stage 2
        x2 = F.relu(self.gn2(self.conv2(x1_pool)))        # (B, 16, res/2, res/2, res/2)
        
        # Decoder Stage 1
        x_up = self.up(x2)                              # (B, 16, res, res, res)
        # Handle shape mismatch in upscale if any
        if x_up.shape[2:] != x1.shape[2:]:
            x_up = F.interpolate(x_up, size=x1.shape[2:], mode='trilinear', align_corners=False)
        x_concat = torch.cat([x_up, x1], dim=1)           # (B, 32, res, res, res)
        
        out_dense = self.conv_out(x_concat)
        
        out_sparse = out_dense[coords[:, 0].long(), :, coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()]
        return out_sparse

class StudentNSH(nn.Module):
    """Compact Student Neural Semantic Head model."""
    
    def __init__(
        self,
        in_channels: int = 272,
        num_classes_per_level: Dict[str, int] = None
    ) -> None:
        super().__init__()
        if num_classes_per_level is None:
            num_classes_per_level = {"coarse": 3, "middle": 8, "fine": 15}
            
        self.num_classes_per_level = num_classes_per_level
        self.in_channels = in_channels
        self.backbone_channels = 32  # Reduced student backbone width
        
        self.unet = StudentDenseFallbackUNet(in_channels, self.backbone_channels)
        
        # evidential heads
        self.heads = nn.ModuleDict({
            lvl: nn.Linear(self.backbone_channels, K) for lvl, K in num_classes_per_level.items()
        })
        
    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        out_feats = self.unet(coords, feats)
        outputs = {}
        for lvl, head in self.heads.items():
            logits = head(out_feats)
            alpha = F.softplus(logits) + 1.0
            outputs[lvl] = {
                "logits": logits,
                "alpha": alpha
            }
        return outputs

    def decode(self, outputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        decoded = {}
        for lvl, out in outputs.items():
            alpha = out["alpha"]
            logits = out["logits"]
            K = self.num_classes_per_level[lvl]
            S = torch.sum(alpha, dim=-1, keepdim=True)
            prob = alpha / S
            prediction = torch.argmax(prob, dim=-1)
            confidence = 1.0 - (K / S.squeeze(-1))
            decoded[lvl] = {
                "prediction": prediction,
                "confidence": confidence,
                "prob": prob
            }
        return decoded

class SVDIRunner:
    """Runner wrapper for inference, quantization, size checks, and fast sub-tree relabeling."""
    
    def __init__(self, student_path: str, taxonomy: Dict[str, Any]) -> None:
        self.taxonomy = taxonomy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = StudentNSH()
        if os.path.exists(student_path):
            self.model.load_state_dict(torch.load(student_path, map_location=self.device))
            logger.info(f"Loaded student model weights from {student_path}")
        else:
            logger.warning(f"Student model path {student_path} not found. Running with random weights.")
            
        self.model.to(self.device)
        self.model.eval()
        
    def get_model_size_mb(self) -> float:
        """Returns student model size in Megabytes."""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024 / 1024
        return size_all_mb
        
    @torch.no_grad()
    def relabel(
        self,
        sparse_input: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
        edited_branch: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Perform SVDI full or incremental relabeling.
        
        Args:
            sparse_input (Tuple): (coords, feats, labels_dict)
            edited_branch (int): if provided, only this sub-tree branch's labels are incrementally relabeled.
            
        Returns:
            Tuple[Dict[str, torch.Tensor], float]: (new_labels_dict, elapsed_time_seconds)
        """
        coords, feats, labels_dict = sparse_input
        coords = coords.to(self.device)
        feats = feats.to(self.device)
        
        # Move inputs to device
        device_labels = {k: v.to(self.device) for k, v in labels_dict.items()}
        
        start_time = time.perf_counter()
        
        if edited_branch is None:
            # Full student model execution
            outputs = self.model(coords, feats)
            decoded = self.model.decode(outputs)
            
            # Constrained tree-consistent decode
            logits_per_level = {k: v["logits"] for k, v in outputs.items()}
            new_labels = tree_consistent_decode(logits_per_level, self.taxonomy)
        else:
            # Incremental relabeling on the sparse grid (restricted sub-tree)
            # Find the level of the edited_branch
            if edited_branch in self.taxonomy.get("mid_to_coarse", {}):
                level = "middle"
            elif edited_branch in [val for val in self.taxonomy.get("mid_to_coarse", {}).values()]:
                level = "coarse"
            else:
                level = "fine"
                
            from neurosem3d.semantics.hierarchy import get_descendant_parts
            descendants = get_descendant_parts(edited_branch, self.taxonomy, level)
            
            # Find mask of voxels belonging to descendants of this branch
            mask = torch.zeros_like(device_labels["fine"], dtype=torch.bool)
            for desc in descendants:
                mask |= (device_labels["fine"] == desc)
                mask |= (device_labels["middle"] == desc)
                mask |= (device_labels["coarse"] == desc)
                
            new_labels = {k: v.clone() for k, v in device_labels.items()}
            
            if mask.any():
                masked_labels = {k: v[mask] for k, v in device_labels.items()}
                updated_masked = incremental_relabel(masked_labels, self.taxonomy)
                for k in new_labels.keys():
                    new_labels[k][mask] = updated_masked[k]
            
        elapsed = time.perf_counter() - start_time
        logger.info(f"Relabel execution time: {elapsed:.6f} seconds (edited_branch={edited_branch})")
        return new_labels, elapsed

def main() -> None:
    parser = argparse.ArgumentParser(description="NEW Work-2 module: 3.4 Sparse-voxel distilled inference.")
    args = parser.parse_args()
    logger.info("Running SVDI execution stub")

if __name__ == "__main__":
    main()
