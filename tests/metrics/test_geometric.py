"""
Tests for geometric metrics (Chamfer, Volumetric IoU, Normal Consistency).
"""

import pytest
import numpy as np
import trimesh
from loguru import logger
from neurosem3d.metrics.geometric import chamfer_distance, volumetric_iou, normal_consistency


def test_volumetric_iou_hand_check() -> None:
    """Test volumetric IoU on a small, hand-checkable 2x2 grid."""
    logger.info("Testing volumetric IoU hand-checked case")
    
    # 2x2x2 grids
    occ_pred = np.array([
        [[1, 0], [0, 1]],
        [[1, 1], [0, 0]]
    ])
    
    occ_ref = np.array([
        [[1, 1], [0, 0]],
        [[1, 0], [0, 1]]
    ])
    
    # Intersection elements at: (0,0,0) and (1,0,0) -> 2 elements
    # Union elements at: (0,0,0), (0,0,1), (0,1,1), (1,0,0), (1,0,1), (1,1,1) -> 6 elements
    # IoU = 2 / 6 = 1/3 ~ 0.333333
    res = volumetric_iou(occ_pred, occ_ref)
    assert pytest.approx(res["volumetric_iou"]) == 1.0 / 3.0


def test_mesh_metrics_with_sphere() -> None:
    """Test Chamfer and normal consistency using standard shapes (spheres)."""
    logger.info("Testing Chamfer and Normal Consistency using spheres")
    
    # Identical spheres should have CD ~ 0 and NC ~ 1
    sphere1 = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    sphere2 = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    
    cd_res = chamfer_distance(sphere1, sphere2, num_samples=1000)
    nc_res = normal_consistency(sphere1, sphere2, num_samples=1000)
    
    assert cd_res["chamfer_distance"] < 0.2
    assert nc_res["normal_consistency"] > 0.8
