"""Unit tests for B2 Enhanced Instant3D baseline."""

import pytest
import os
import torch
import numpy as np
from loguru import logger

from neurosem3d.baselines.enhanced_instant3d import EnhancedInstant3d

def test_enhanced_instant3d_validity() -> None:
    """Verify that EnhancedInstant3d output coordinates are aligned, and labels are in taxonomy range."""
    logger.info("Running EnhancedInstant3d validity test...")
    
    confidence_dir = "neurosem3d/data/processed/confidence"
    if not os.path.exists(confidence_dir):
        confidence_dir = "data/processed/confidence"
        
    baseline = EnhancedInstant3d(confidence_dir=confidence_dir)
    res = baseline.run("dummy_obj_0")
    
    assert "fine" in res
    assert "middle" in res
    assert "coarse" in res
    assert "u" in res
    
    assert res["u"] is None
    
    # Load coordinates from confidence NPZ directly to verify shape alignment
    confidence_path = os.path.join(confidence_dir, "dummy_obj_0.npz")
    with np.load(confidence_path) as data:
        voxel_xyz = data["voxel_xyz"]
        
    N_voxels = voxel_xyz.shape[0]
    
    # Assert coordinates align 1-to-1 with predictions (meaning they are a subset of occupied voxels)
    assert res["fine"].shape == (N_voxels,)
    assert res["middle"].shape == (N_voxels,)
    assert res["coarse"].shape == (N_voxels,)
    
    # Assert labels are within taxonomy range
    # Valid fine class labels in TEST_TAXONOMY/PartNet: 0..5
    valid_fine = torch.all((res["fine"] >= 0) & (res["fine"] <= 5))
    assert valid_fine.item() is True
    
    # Valid middle class labels: {0, 10, 11, 12}
    middle_labels = torch.unique(res["middle"])
    for ml in middle_labels:
        assert ml.item() in {0, 10, 11, 12}
        
    # Valid coarse class labels: {0, 20, 21}
    coarse_labels = torch.unique(res["coarse"])
    for cl in coarse_labels:
        assert cl.item() in {0, 20, 21}
