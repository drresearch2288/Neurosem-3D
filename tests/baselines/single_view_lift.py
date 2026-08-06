"""Unit tests for B4 Single-view 2D-lift (SAM) baseline."""

import pytest
import os
import torch
import numpy as np
from loguru import logger

from neurosem3d.baselines.single_view_lift import SingleViewLift

def test_single_view_lift_validity() -> None:
    """Verify B4 single_view_lift runs end-to-end and only visible voxels receive non-zero labels."""
    logger.info("Running B4 SingleViewLift validity test...")
    
    # Resolve confidence dir dynamically
    confidence_dir = "neurosem3d/data/processed/confidence"
    if not os.path.exists(confidence_dir):
        confidence_dir = "data/processed/confidence"
        
    baseline = SingleViewLift(confidence_dir=confidence_dir, default_view_index=0)
    
    # Run end-to-end
    res = baseline.run("dummy_obj_0", view_index=0)
    
    # Check expected dict keys
    for key in ["fine", "middle", "coarse", "u"]:
        assert key in res
    assert res["u"] is None
    
    # Load raw data to verify visibility constraints
    confidence_path = os.path.join(confidence_dir, "dummy_obj_0.npz")
    with np.load(confidence_path) as data:
        c_depth = data["c_depth"]
        
    visible = c_depth[:, 0] >= np.exp(-1.0)
    visible_t = torch.from_numpy(visible)
    
    # Assert that all non-zero labels belong ONLY to visible voxels
    non_zero_mask = res["fine"] != 0
    
    # Every non-zero voxel must be visible from the chosen view
    assert torch.all(visible_t[non_zero_mask]).item() is True
