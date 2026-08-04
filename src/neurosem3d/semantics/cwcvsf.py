import argparse
import numpy as np
import torch
from loguru import logger
from typing import Any, Dict, List, Optional, Tuple, Union

def cwcvsf_torch(
    c_depth: torch.Tensor,
    c_angle: torch.Tensor,
    c_mask: torch.Tensor,
    projected_label: torch.Tensor,
    num_classes: int = 15,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Confidence-Weighted Cross-View Semantic Fusion (CW-CVSF) in PyTorch.
    
    Computes a weighted soft probability distribution over class labels for each voxel.
    
    Args:
        c_depth (torch.Tensor): depth confidence shape (N, 8) or (B, N, 8).
        c_angle (torch.Tensor): angle confidence shape (N, 8) or (B, N, 8).
        c_mask (torch.Tensor): mask confidence shape (N, 8) or (B, N, 8).
        projected_label (torch.Tensor): projected class labels shape (N, 8) or (B, N, 8).
        num_classes (int): number of semantic classes (K).
        eps (float): epsilon to avoid division by zero.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - P_fuse: Soft probability distribution of shape (..., N, K) on the K-simplex.
            - zero_weight_mask: Boolean tensor of shape (..., N) flagging voxels with zero weight.
    """
    # 1. Compute view weights per voxel (product of cues)
    weights = c_depth * c_angle * c_mask  # (..., 8)
    
    # 2. Accumulate weights for each class using scatter
    # We want to sum weights into class bins.
    # projected_label shape: (..., 8), values in range [0, num_classes-1]
    # We can use one-hot encoding for vectorised binning:
    one_hot = torch.nn.functional.one_hot(projected_label.long(), num_classes=num_classes).float() # (..., 8, K)
    
    # Multiply by weights and sum along the view dimension (axis -2)
    # weights: (..., 8) -> expand to (..., 8, 1)
    weighted_one_hot = one_hot * weights.unsqueeze(-1)  # (..., 8, K)
    P_fuse = torch.sum(weighted_one_hot, dim=-2)  # (..., K)
    
    # 3. Normalise probabilities
    sum_weights = torch.sum(weights, dim=-1, keepdim=True)  # (..., 1)
    P_fuse = P_fuse / (sum_weights + eps)
    
    # Flag zero total weight voxels
    zero_weight_mask = (sum_weights.squeeze(-1) == 0)
    
    # For zero weight voxels, assign uniform distribution (or assign to class 0 ignore)
    if torch.any(zero_weight_mask):
        uniform = torch.ones(num_classes, device=P_fuse.device, dtype=P_fuse.dtype) / num_classes
        P_fuse[zero_weight_mask] = uniform
        
    return P_fuse, zero_weight_mask

def cwcvsf_numpy(
    c_depth: np.ndarray,
    c_angle: np.ndarray,
    c_mask: np.ndarray,
    projected_label: np.ndarray,
    num_classes: int = 15,
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """Confidence-Weighted Cross-View Semantic Fusion (CW-CVSF) in NumPy.
    
    Computes a weighted soft probability distribution over class labels for each voxel.
    
    Args:
        c_depth (np.ndarray): depth confidence shape (N, 8).
        c_angle (np.ndarray): angle confidence shape (N, 8).
        c_mask (np.ndarray): mask confidence shape (N, 8).
        projected_label (np.ndarray): projected class labels shape (N, 8).
        num_classes (int): number of semantic classes (K).
        eps (float): epsilon to avoid division by zero.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - P_fuse: Soft probability distribution of shape (N, K) on the K-simplex.
            - zero_weight_mask: Boolean array of shape (N,) flagging voxels with zero weight.
    """
    weights = c_depth * c_angle * c_mask  # (N, 8)
    sum_weights = np.sum(weights, axis=1, keepdims=True)  # (N, 1)
    
    # Vectorised accumulation using np.eye indexing
    one_hot = np.eye(num_classes)[projected_label.astype(np.int32)]  # (N, 8, K)
    weighted_one_hot = one_hot * weights[:, :, np.newaxis]  # (N, 8, K)
    P_fuse = np.sum(weighted_one_hot, axis=1)  # (N, K)
    
    P_fuse = P_fuse / (sum_weights + eps)
    
    zero_weight_mask = (sum_weights.squeeze(1) == 0)
    if np.any(zero_weight_mask):
        P_fuse[zero_weight_mask] = 1.0 / num_classes
        
    return P_fuse, zero_weight_mask

def fuse_semantics(
    c_depth: np.ndarray,
    c_angle: np.ndarray,
    c_mask: np.ndarray,
    projected_label: np.ndarray,
    num_classes: int = 15
) -> np.ndarray:
    """Legacy preprocessing wrapper for cwcvsf_numpy returning only P_fuse."""
    P_fuse, _ = cwcvsf_numpy(c_depth, c_angle, c_mask, projected_label, num_classes)
    return P_fuse

def majority_vote(
    projected_label: Union[torch.Tensor, np.ndarray],
    visible: Union[torch.Tensor, np.ndarray],
    num_classes: int = 15
) -> Union[torch.Tensor, np.ndarray]:
    """Work 1 reference plurality voting implementation.
    
    Assigns each voxel the mode label across visible views.
    
    Args:
        projected_label: projected labels of shape (..., 8).
        visible: visibility boolean mask of shape (..., 8).
        num_classes (int): number of classes.
        
    Returns:
        Voxel label indices of shape (...).
    """
    if isinstance(projected_label, torch.Tensor):
        assert isinstance(visible, torch.Tensor)
        # Apply visibility mask to one-hot encoded labels
        one_hot = torch.nn.functional.one_hot(projected_label.long(), num_classes=num_classes).float()  # (..., 8, K)
        visible_votes = one_hot * visible.unsqueeze(-1).float()  # (..., 8, K)
        vote_sums = torch.sum(visible_votes, dim=-2)  # (..., K)
        
        # Plurality argmax
        vote_sums[..., 0] = -1e5  # force non-ignore class unless all votes are 0
        all_zero = (torch.sum(visible.float(), dim=-1) == 0)
        modes = torch.argmax(vote_sums, dim=-1)
        modes[all_zero] = 0  # ignore class
        return modes
    else:
        # NumPy path
        one_hot = np.eye(num_classes)[projected_label.astype(np.int32)]  # (..., 8, K)
        visible_votes = one_hot * visible[..., np.newaxis].astype(float)  # (..., 8, K)
        vote_sums = np.sum(visible_votes, axis=-2)  # (..., K)
        
        vote_sums[..., 0] = -1e5
        all_zero = (np.sum(visible.astype(float), axis=-1) == 0)
        modes = np.argmax(vote_sums, axis=-1)
        modes[all_zero] = 0
        return modes

def main() -> None:
    """Main entry point for testing or running cwcvsf independently."""
    parser = argparse.ArgumentParser(description="Confidence-Weighted Cross-View Semantic Fusion.")
    args = parser.parse_args()
    logger.info("Running cwcvsf verify stub")

if __name__ == "__main__":
    main()
