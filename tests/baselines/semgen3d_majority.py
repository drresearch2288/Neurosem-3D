"""Unit tests for B1 SemGen-3D majority-vote baseline."""

import pytest
import os
import torch
import numpy as np
from loguru import logger

from neurosem3d.baselines.semgen3d_majority import Semgen3dMajority
from neurosem3d.semantics.cwcvsf import cwcvsf_torch, majority_vote

def test_soft_fusion_equivalence() -> None:
    """Verify majority vote equals the soft-fusion argmax when all cues = 1.0."""
    logger.info("Verifying soft-fusion and majority-vote equivalence under cues=1.0...")
    
    N = 200
    num_views = 8
    num_classes = 15
    
    # Generate random projected labels (range 1 to 14, class 0 is ignore)
    torch.manual_seed(42)
    projected_label = torch.randint(1, num_classes, (N, num_views), dtype=torch.long)
    
    # All cues set to 1.0
    c_depth = torch.ones((N, num_views), dtype=torch.float32)
    c_angle = torch.ones((N, num_views), dtype=torch.float32)
    c_mask = torch.ones((N, num_views), dtype=torch.float32)
    
    # Soft fusion
    P_fuse, _ = cwcvsf_torch(c_depth, c_angle, c_mask, projected_label, num_classes=num_classes)
    
    # Work 1 plurality tie-break expects non-ignore class unless all votes are 0
    # In P_fuse, class 0 is also sum of weights for class 0. We force non-ignore class
    # to match majority_vote's tie-break setting.
    P_fuse_adjusted = P_fuse.clone()
    P_fuse_adjusted[..., 0] = -1e5
    soft_argmax = torch.argmax(P_fuse_adjusted, dim=-1)
    
    # Hard majority vote with all views visible
    visible = torch.ones((N, num_views), dtype=torch.bool)
    majority_val = majority_vote(projected_label, visible, num_classes=num_classes)
    
    # Assert equivalence
    torch.testing.assert_close(soft_argmax, majority_val)

def test_semgen3d_majority_run() -> None:
    """Verify Semgen3dMajority baseline runs correctly on dummy object."""
    logger.info("Verifying Semgen3dMajority.run on dummy object...")
    
    confidence_dir = "neurosem3d/data/processed/confidence"
    if not os.path.exists(confidence_dir):
        confidence_dir = "data/processed/confidence"
        
    baseline = Semgen3dMajority(confidence_dir=confidence_dir)
    res = baseline.run("dummy_obj_0")
    
    assert "fine" in res
    assert "middle" in res
    assert "coarse" in res
    assert "u" in res
    
    assert res["u"] is None
    
    # Check that shapes of all level predictions are aligned
    fine_shape = res["fine"].shape
    assert res["middle"].shape == fine_shape
    assert res["coarse"].shape == fine_shape
