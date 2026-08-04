import argparse
import torch
import torch.nn.functional as F
from loguru import logger
from typing import Any, Dict, List, Optional, Tuple, Union

def compute_cues(
    voxel_xyz: torch.Tensor,
    normal_v: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    depth_map: torch.Tensor,
    sam_mask: torch.Tensor,
    stability_score: float = 0.95,
    delta: float = 0.05
) -> Dict[str, torch.Tensor]:
    """Compute confidence cues for a set of voxels and a given view.
    
    Args:
        voxel_xyz (torch.Tensor): occupied voxel centers of shape (N, 3).
        normal_v (torch.Tensor): normals of shape (N, 3).
        K (torch.Tensor): camera intrinsics (3, 3).
        T (torch.Tensor): camera extrinsics (4, 4) transforming world to camera coords.
        depth_map (torch.Tensor): depth map image (H, W).
        sam_mask (torch.Tensor): SAM segmentation mask (H, W).
        stability_score (float): general stability score of the SAM mask.
        delta (float): tolerance parameter for depth difference.
        
    Returns:
        Dict[str, torch.Tensor]: containing:
            - 'c_depth': depth confidence (N,).
            - 'c_angle': grazing-angle confidence (N,).
            - 'c_mask': SAM mask/stability confidence (N,).
            - 'projected_label': SAM label at the projected pixel (N,).
    """
    N = voxel_xyz.shape[0]
    H, W = depth_map.shape
    device = voxel_xyz.device
    
    # 1. Transform voxels to camera coordinates
    # voxel_xyz: (N, 3)
    R = T[:3, :3]
    t = T[:3, 3]
    
    v_cam = voxel_xyz @ R.t() + t.unsqueeze(0)  # (N, 3)
    d_v_i = v_cam[:, 2]  # depth in camera space
    
    # 2. Project voxels onto image plane
    # proj = v_cam @ K.t()
    proj = v_cam @ K.t()
    u = proj[:, 0] / (proj[:, 2] + 1e-8)
    v_coord = proj[:, 1] / (proj[:, 2] + 1e-8)
    
    # 3. Check image bounds
    in_bounds = (u >= 0) & (u < W) & (v_coord >= 0) & (v_coord < H) & (d_v_i > 0)
    
    # Round to nearest pixel coordinates
    u_px = torch.clamp(u.round().long(), 0, W - 1)
    v_px = torch.clamp(v_coord.round().long(), 0, H - 1)
    
    # 4. Compute c_depth
    # D_i(pi_i(v))
    obs_depth = depth_map[v_px, u_px]  # (N,)
    depth_diff = torch.abs(d_v_i - obs_depth)
    c_depth = torch.exp(-depth_diff / delta)
    c_depth[~in_bounds] = 0.0
    
    # Work 1 visibility band check: hard truncation gate (depth_diff <= 0.1)
    visibility_mask = depth_diff <= 0.1
    c_depth[~visibility_mask] = 0.0
    
    # 5. Compute c_angle
    # Ray direction in camera space: v_cam / norm(v_cam)
    norm_v_cam = torch.norm(v_cam, p=2, dim=-1, keepdim=True) + 1e-8
    ray_dir_cam = v_cam / norm_v_cam
    
    # Transform normals to camera space
    normal_cam = normal_v @ R.t()
    normal_cam = normal_cam / (torch.norm(normal_cam, p=2, dim=-1, keepdim=True) + 1e-8)
    
    # Dot product between surface normal and vector pointing towards camera (-ray_dir_cam)
    dot_prod = torch.sum(normal_cam * (-ray_dir_cam), dim=-1)
    c_angle = torch.clamp(dot_prod, min=0.0)
    c_angle[~in_bounds] = 0.0
    c_angle[~visibility_mask] = 0.0
    
    # 6. Compute c_mask & retrieve labels
    # Use stability_score if projected voxel falls within a SAM mask (pixel is non-zero)
    pixel_labels = sam_mask[v_px, u_px]
    c_mask = torch.where((pixel_labels > 0) & in_bounds & visibility_mask, torch.tensor(stability_score, device=device), torch.tensor(0.0, device=device))
    
    # Labels outside bounds/visibility are set to 0 (ignore)
    projected_label = torch.where(in_bounds & visibility_mask, pixel_labels, torch.zeros_like(pixel_labels))
    
    return {
        "c_depth": c_depth,
        "c_angle": c_angle,
        "c_mask": c_mask,
        "projected_label": projected_label
    }

def main() -> None:
    """Main entry point for testing or running confidence independently."""
    parser = argparse.ArgumentParser(description="Confidence calculation utilities.")
    args = parser.parse_args()
    logger.info("Running confidence verify stub")

if __name__ == "__main__":
    main()
