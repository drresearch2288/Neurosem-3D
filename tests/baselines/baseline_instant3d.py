"""Unit tests for B3 BaselineInstant3D and RealNeRFComparison baselines."""

import pytest
import os
import numpy as np
from loguru import logger

from neurosem3d.baselines.baseline_instant3d import BaselineInstant3D, RealNeRFComparison

def test_b3_baselines_end_to_end() -> None:
    """Verify both BaselineInstant3D and RealNeRFComparison baselines run correctly and return expected keys."""
    logger.info("Running B3 Baselines end-to-end tests...")
    
    # Resolve confidence dir dynamically
    confidence_dir = "neurosem3d/data/processed/confidence"
    if not os.path.exists(confidence_dir):
        confidence_dir = "data/processed/confidence"
        
    b3_mono = BaselineInstant3D(confidence_dir=confidence_dir)
    b3_nerf = RealNeRFComparison(confidence_dir=confidence_dir)
    
    # Load voxel count for dummy_obj_0 to verify output shape
    confidence_path = os.path.join(confidence_dir, "dummy_obj_0.npz")
    with np.load(confidence_path) as data:
        voxel_xyz = data["voxel_xyz"]
    N_voxels = voxel_xyz.shape[0]
    
    # Check Monocular Baseline
    res_mono = b3_mono.run("dummy_obj_0")
    for key in ["fine", "middle", "coarse", "u"]:
        assert key in res_mono
    assert res_mono["u"] is None
    assert res_mono["fine"].shape == (N_voxels,)
    assert res_mono["middle"].shape == (N_voxels,)
    assert res_mono["coarse"].shape == (N_voxels,)
    
    # Check RealNeRF Baseline
    res_nerf = b3_nerf.run("dummy_obj_0")
    for key in ["fine", "middle", "coarse", "u"]:
        assert key in res_nerf
    assert res_nerf["u"] is None
    assert res_nerf["fine"].shape == (N_voxels,)
    assert res_nerf["middle"].shape == (N_voxels,)
    assert res_nerf["coarse"].shape == (N_voxels,)
