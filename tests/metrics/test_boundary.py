"""
Tests for boundary metrics (Boundary IoU and Boundary F1).
"""

import pytest
import numpy as np
from loguru import logger
from neurosem3d.metrics.boundary import boundary_metrics


def test_boundary_metrics_3x3_toy() -> None:
    """Test boundary metrics on a simple 3x3 coordinate grid.
    
    Formula check:
        coords represent a 3x3 grid on z=0 plane.
        gt has labels 1 and 2 separated by a vertical boundary.
    """
    logger.info("Testing boundary metrics 3x3 grid toy case")
    
    # 3x3 coordinate grid: coordinates range from 0 to 2
    coords = np.array([
        [0, 0, 0], [0, 1, 0], [0, 2, 0],
        [1, 0, 0], [1, 1, 0], [1, 2, 0],
        [2, 0, 0], [2, 1, 0], [2, 2, 0]
    ], dtype=np.float32)
    
    # Let's set labels: column 0 is class 1, column 1 is class 2, column 2 is class 2
    gt = np.array([
        1, 2, 2,
        1, 2, 2,
        1, 2, 2
    ], dtype=np.int16)
    
    # Make predicted labels: introduce one mistake at the boundary (1, 1, 0)
    pred = np.array([
        1, 2, 2,
        1, 1, 2,  # index 4 (middle) is predicted as 1 instead of 2
        1, 2, 2
    ], dtype=np.int16)
    
    # boundary_metrics will find boundary voxels, compute the shell, and evaluate
    res = boundary_metrics(coords, pred, gt, radius=1.5, num_classes=3)
    
    assert "boundary_iou" in res
    assert "boundary_f1" in res
    assert 0.0 <= res["boundary_iou"] <= 1.0
    assert 0.0 <= res["boundary_f1"] <= 1.0
