"""
Tests for confidence cues extraction
"""

import pytest
import torch
from loguru import logger
from neurosem3d.data.confidence import compute_cues

def test_confidence_cues_bounds_and_visibility() -> None:
    """Test that all confidence cues are in [0, 1] and visibility band works."""
    logger.info("Starting confidence cues verification...")
    
    # 1. Setup mock data
    device = torch.device("cpu")
    N = 10
    
    # Voxel coordinates: one voxel at origin, others spread out
    voxel_xyz = torch.tensor([
        [0.0, 0.0, 2.0],    # perfectly matching depth
        [0.0, 0.0, 2.05],   # inside visibility band (diff = 0.05 <= 0.1)
        [0.0, 0.0, 2.15],   # outside visibility band (diff = 0.15 > 0.1)
        [0.0, 0.0, 1.85],   # outside visibility band (diff = 0.15 > 0.1)
        [0.0, 0.0, 1.95],   # inside visibility band (diff = 0.05 <= 0.1)
        [10.0, 0.0, 2.0],   # way out of image bounds
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 2.0]
    ], dtype=torch.float32, device=device)
    
    # Normals: pointing towards the camera (so normal = [0, 0, -1])
    normal_v = torch.tensor([
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],    # grazing angle (normal orthogonal to view ray)
        [1.0, 0.0, 0.0],    # grazing angle
        [0.0, 0.0, 1.0],    # facing away from camera (dot < 0)
        [0.0, 0.0, -1.0]
    ], dtype=torch.float32, device=device)
    
    # Camera at origin looking along +Z
    K = torch.tensor([
        [100.0, 0.0, 32.0],
        [0.0, 100.0, 32.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device)
    
    # Extrinsics: Identity (since camera is at origin looking along +Z)
    T = torch.eye(4, dtype=torch.float32, device=device)
    
    # Depth map (64x64) with constant depth of 2.0
    depth_map = torch.ones(64, 64, dtype=torch.float32, device=device) * 2.0
    
    # SAM mask (64x64)
    sam_mask = torch.zeros(64, 64, dtype=torch.long, device=device)
    sam_mask[30:34, 30:34] = 3  # draw a small mask in center
    
    # 2. Compute cues
    cues = compute_cues(
        voxel_xyz=voxel_xyz,
        normal_v=normal_v,
        K=K,
        T=T,
        depth_map=depth_map,
        sam_mask=sam_mask,
        stability_score=0.95,
        delta=0.05
    )
    
    c_depth = cues["c_depth"]
    c_angle = cues["c_angle"]
    c_mask = cues["c_mask"]
    
    # Assertions
    # 1. Check all values are within [0, 1]
    assert torch.all(c_depth >= 0.0) and torch.all(c_depth <= 1.0)
    assert torch.all(c_angle >= 0.0) and torch.all(c_angle <= 1.0)
    assert torch.all(c_mask >= 0.0) and torch.all(c_mask <= 1.0)
    
    # 2. Check voxel 0: perfect match
    assert c_depth[0] == 1.0, f"Expected c_depth=1.0 for perfect match, got {c_depth[0]}"
    
    # 3. Check voxel 2 & 3: outside visibility band (|diff| > 0.1) -> should be 0.0
    assert c_depth[2] == 0.0, f"Expected voxel 2 (diff=0.15) to be cut off, got c_depth={c_depth[2]}"
    assert c_depth[3] == 0.0, f"Expected voxel 3 (diff=0.15) to be cut off, got c_depth={c_depth[3]}"
    assert c_angle[2] == 0.0, "Angle cue should also be 0 when visibility gate fails"
    assert c_mask[2] == 0.0, "Mask cue should also be 0 when visibility gate fails"
    
    # 4. Check voxel 1 & 4: inside visibility band (|diff| <= 0.1) -> should be exp(-0.05/0.05) = exp(-1) = 0.3679
    import math
    expected_val = math.exp(-1.0)
    assert torch.allclose(c_depth[1], torch.tensor(expected_val), atol=1e-4)
    assert torch.allclose(c_depth[4], torch.tensor(expected_val), atol=1e-4)
    
    # 5. Check voxel 5: out of image bounds -> all cues 0
    assert c_depth[5] == 0.0
    assert c_angle[5] == 0.0
    assert c_mask[5] == 0.0
    
    # 6. Check voxel 8: facing away from camera (dot < 0) -> c_angle 0
    assert c_angle[8] == 0.0
    
    logger.success("Confidence cues tests successfully passed.")

