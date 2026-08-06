"""
Tests for editing metrics (Edit Leakage, Sub-Assembly Success Rate).
"""

import pytest
import numpy as np
from loguru import logger
from neurosem3d.metrics.editing import edit_leakage, sub_assembly_success_rate


def test_edit_leakage_hand_check() -> None:
    """Test edit leakage on a 5-voxel toy example.
    
    Formula check:
        labels_before    = [1, 1, 1, 2, 2]
        labels_after     = [1, 2, 1, 2, 1]
        target_part_mask = [T, T, T, F, F] (Indices 0, 1, 2 are target)
        
        Outside target (mask is False): indices 3 and 4 (total 2).
        Modified outside target: index 4 (before 2 -> after 1) (total 1).
        
        Leakage = 1 / 2 * 100 = 50.0%
    """
    logger.info("Testing edit leakage hand-checked case")
    labels_before = np.array([1, 1, 1, 2, 2])
    labels_after = np.array([1, 2, 1, 2, 1])
    target_part_mask = np.array([True, True, True, False, False])
    
    res = edit_leakage(labels_before, labels_after, target_part_mask)
    assert res["edit_leakage"] == 50.0


def test_sub_assembly_success_rate_hand_check() -> None:
    """Test sub-assembly success rate on a 4-edit case."""
    logger.info("Testing sub-assembly success rate hand-checked case")
    leakages = [0.5, 1.2, 0.8, 2.5]
    
    # Successful if leakage < 1.0: 0.5 and 0.8 (2 out of 4) -> 0.5
    res = sub_assembly_success_rate(leakages, threshold_tau=1.0)
    assert res["sub_assembly_success_rate"] == 0.5
