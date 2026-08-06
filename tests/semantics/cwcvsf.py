"""
Tests for Confidence-Weighted Cross-View Semantic Fusion (CW-CVSF).
"""

import pytest
import numpy as np
import torch
from loguru import logger
from neurosem3d.semantics.cwcvsf import cwcvsf_torch, cwcvsf_numpy, majority_vote

def test_cwcvsf_properties() -> None:
    """Test CW-CVSF properties: simplex constraints, equivalence to majority vote, and confidence-driven shift."""
    logger.info("Starting CW-CVSF properties verification...")
    
    # Setup mock data (N=3 occupied voxels, 8 views, class size K=15)
    N = 3
    num_views = 8
    num_classes = 15
    
    # 1. Projected labels
    # Voxel 0: 6 votes for class 1, 2 votes for class 2
    # Voxel 1: 4 votes for class 3, 4 votes for class 4
    # Voxel 2: 8 votes for class 5
    projected_label = np.array([
        [1, 1, 1, 1, 1, 1, 2, 2],
        [3, 3, 3, 3, 4, 4, 4, 4],
        [5, 5, 5, 5, 5, 5, 5, 5]
    ], dtype=np.int32)
    
    # 2. Setup equal confidence cues (all = 1.0)
    c_depth = np.ones((N, num_views), dtype=np.float32)
    c_angle = np.ones((N, num_views), dtype=np.float32)
    c_mask = np.ones((N, num_views), dtype=np.float32)
    
    # Run numpy CW-CVSF
    P_fuse_np, zero_mask_np = cwcvsf_numpy(c_depth, c_angle, c_mask, projected_label, num_classes)
    
    # Test (a): Each row sums to 1.0
    row_sums_np = np.sum(P_fuse_np, axis=-1)
    assert np.allclose(row_sums_np, 1.0), f"Row sums: {row_sums_np}"
    
    # Test (b): All cues = 1.0 -> argmax(P_fuse) equals majority_vote
    # In majority vote, we specify a boolean visibility mask (all visible = True)
    visible_np = np.ones((N, num_views), dtype=bool)
    maj_labels = majority_vote(projected_label, visible_np, num_classes)
    
    assert np.all(np.argmax(P_fuse_np, axis=-1) == maj_labels)
    logger.info("Verified all cues equal to 1.0 reduces soft vote to plurality mode.")
    
    # Run PyTorch path to test equivalence
    c_depth_t = torch.from_numpy(c_depth)
    c_angle_t = torch.from_numpy(c_angle)
    c_mask_t = torch.from_numpy(c_mask)
    projected_label_t = torch.from_numpy(projected_label)
    
    P_fuse_t, zero_mask_t = cwcvsf_torch(c_depth_t, c_angle_t, c_mask_t, projected_label_t, num_classes)
    assert np.allclose(P_fuse_t.cpu().numpy(), P_fuse_np)
    
    # Test (c): Increasing c_angle for correct-label view shifts probability
    # Look at voxel 0: currently class 1 has 6 votes, class 2 has 2 votes.
    # If we drastically increase the weight of class 2 views and decrease class 1 weights, P_fuse should shift to class 2.
    c_angle_modified = np.ones((N, num_views), dtype=np.float32)
    # class 1 views: indices 0..5. set weight to 0.05
    c_angle_modified[0, 0:6] = 0.05
    # class 2 views: indices 6..7. set weight to 1.0
    c_angle_modified[0, 6:8] = 1.0
    
    P_fuse_shifted, _ = cwcvsf_numpy(c_depth, c_angle_modified, c_mask, projected_label, num_classes)
    
    # Voxel 0 should now favor class 2
    assert P_fuse_shifted[0, 2] > P_fuse_shifted[0, 1]
    assert np.argmax(P_fuse_shifted[0]) == 2
    logger.info(f"Verified cue modification shifted top prediction: original mode 1 -> shifted mode {np.argmax(P_fuse_shifted[0])}.")
    
    # Test zero weight case: should output uniform distribution
    c_depth_zero = np.zeros((1, num_views), dtype=np.float32)
    c_angle_zero = np.zeros((1, num_views), dtype=np.float32)
    c_mask_zero = np.zeros((1, num_views), dtype=np.float32)
    projected_label_zero = np.zeros((1, num_views), dtype=np.int32)
    
    P_fuse_zero, zero_mask = cwcvsf_numpy(c_depth_zero, c_angle_zero, c_mask_zero, projected_label_zero, num_classes)
    assert zero_mask[0] == True
    assert np.allclose(P_fuse_zero[0], 1.0 / num_classes)
    
    logger.success("All CW-CVSF properties verified successfully.")

