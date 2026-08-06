"""Tests for NeuroSem-3D hierarchical decoding, editing, and resolution.

This module contains unit tests for verifying:
(a) Decoded labels always satisfy the parent-child union property.
(b) The `remove_part` operation removes exactly the targeted sub-tree branch's voxels.
(c) Raising the uncertainty threshold monotonically reduces the count of flipped boundary voxels.
(d) Other utility and editing operations like `scale_part` and `incremental_relabel`.
"""

import pytest
import torch
from loguru import logger
from typing import Any, Dict

from neurosem3d.semantics.hierarchy import (
    tree_consistent_decode,
    remove_part,
    scale_part,
    uncertainty_gated_resolve,
    incremental_relabel,
    get_descendant_parts
)

# Test taxonomy configured with disjoint keys to prevent key collisions:
# Fine labels: 0..5 (map to Middle 10..12)
# Middle labels: 10..12 (map to Coarse 20..21)
TEST_TAXONOMY = {
    "fine_to_mid": {
        0: 10,
        1: 10,
        2: 11,
        3: 11,
        4: 12,
        5: 12,
    },
    "mid_to_coarse": {
        10: 20,
        11: 20,
        12: 21,
    }
}

def test_tree_consistent_decode() -> None:
    """Test (a): Verify decoded labels always satisfy parent=child-union constraint."""
    logger.info("Running Test (a): tree_consistent_decode validation...")
    
    # 100 voxels, with enough classes to match taxonomy indexes
    N = 100
    coarse_K = 22
    mid_K = 13
    fine_K = 6
    
    # Generate random logits
    torch.manual_seed(42)
    coarse_logits = torch.randn(N, coarse_K)
    mid_logits = torch.randn(N, mid_K)
    fine_logits = torch.randn(N, fine_K)
    
    # Mask invalid classes (not in taxonomy) to prevent them from being predicted
    coarse_logits[:, :20] = -1e9
    mid_logits[:, :10] = -1e9
    
    logits_per_level = {
        "coarse": coarse_logits,
        "middle": mid_logits,
        "fine": fine_logits
    }
    
    # Run tree-consistent decoding
    preds = tree_consistent_decode(logits_per_level, TEST_TAXONOMY)
    
    coarse_preds = preds["coarse"]
    mid_preds = preds["middle"]
    fine_preds = preds["fine"]
    
    # Verify shape
    assert coarse_preds.shape == (N,)
    assert mid_preds.shape == (N,)
    assert fine_preds.shape == (N,)
    
    # Check parent-child consistency
    for i in range(N):
        c_val = coarse_preds[i].item()
        m_val = mid_preds[i].item()
        f_val = fine_preds[i].item()
        
        # Mapping from fine to middle must match the predicted middle label
        expected_mid = TEST_TAXONOMY["fine_to_mid"].get(f_val, f_val)
        assert m_val == expected_mid, f"Voxel {i}: fine label {f_val} maps to {expected_mid}, but middle prediction is {m_val}"
        
        # Mapping from middle to coarse must match the predicted coarse label
        expected_coarse = TEST_TAXONOMY["mid_to_coarse"].get(m_val, m_val)
        assert c_val == expected_coarse, f"Voxel {i}: middle label {m_val} maps to {expected_coarse}, but coarse prediction is {c_val}"

def test_remove_part() -> None:
    """Test (b): Verify remove_part on a branch removes exactly that branch's voxels and nothing else."""
    logger.info("Running Test (b): remove_part validation...")
    
    # Create synthetic coordinates and labels
    coords = torch.tensor([
        [0, 1, 1, 1],  # Voxel 0
        [0, 2, 2, 2],  # Voxel 1
        [0, 3, 3, 3],  # Voxel 2
        [0, 4, 4, 4],  # Voxel 3
        [0, 5, 5, 5]   # Voxel 4
    ], dtype=torch.int32)
    
    # Labels corresponding to taxonomy:
    # Voxels 0, 1 -> fine [0, 1] (mid 10, coarse 20)
    # Voxel 2     -> fine 2 (mid 11, coarse 20)
    # Voxels 3, 4 -> fine [4, 5] (mid 12, coarse 21)
    labels_dict = {
        "fine": torch.tensor([0, 1, 2, 4, 5], dtype=torch.long),
        "middle": torch.tensor([10, 10, 11, 12, 12], dtype=torch.long),
        "coarse": torch.tensor([20, 20, 20, 21, 21], dtype=torch.long)
    }
    
    # Remove middle branch 12 (descendants are 12, 4, 5)
    new_coords, new_labels = remove_part(
        coords, labels_dict, branch_id=12, taxonomy=TEST_TAXONOMY, level="middle"
    )
    
    # Only voxels 0, 1, 2 should remain
    assert new_coords.shape[0] == 3
    torch.testing.assert_close(new_coords, coords[:3])
    torch.testing.assert_close(new_labels["fine"], labels_dict["fine"][:3])
    torch.testing.assert_close(new_labels["middle"], labels_dict["middle"][:3])
    torch.testing.assert_close(new_labels["coarse"], labels_dict["coarse"][:3])
    
    # Remove coarse branch 20 (descendants 20, 10, 11, 0, 1, 2, 3)
    new_coords_coarse, new_labels_coarse = remove_part(
        coords, labels_dict, branch_id=20, taxonomy=TEST_TAXONOMY, level="coarse"
    )
    
    # Only voxels 3 and 4 should remain
    assert new_coords_coarse.shape[0] == 2
    torch.testing.assert_close(new_coords_coarse, coords[3:])
    torch.testing.assert_close(new_labels_coarse["fine"], labels_dict["fine"][3:])

def test_uncertainty_gated_resolve_monotonicity() -> None:
    """Test (c): Verify raising u threshold monotonically reduces the count of voxels flipped near a boundary."""
    logger.info("Running Test (c): uncertainty_gated_resolve monotonicity check...")
    
    # Synthetic boundary voxels
    labels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    parent_labels = torch.tensor([10, 10, 11, 11, 12], dtype=torch.long)
    
    # Calibrated confidence values (confidence = 1 - u)
    confidence = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float32)
    
    # We define uncertainty thresholds u from 0.0 to 1.0
    u_thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    flip_counts = []
    
    for u in u_thresholds:
        # confidence threshold is 1 - u
        threshold_conf = 1.0 - u
        
        resolved = uncertainty_gated_resolve(labels, parent_labels, confidence, threshold_conf)
        flipped = (resolved != labels).sum().item()
        flip_counts.append(flipped)
        logger.debug(f"u={u:.1f} (conf_threshold={threshold_conf:.1f}) -> flipped count={flipped}")
        
    # Verify that as u increases, flip_counts monotonically decreases (or stays equal)
    for i in range(len(flip_counts) - 1):
        assert flip_counts[i] >= flip_counts[i+1], (
            f"Non-monotonic flip count detected: {flip_counts[i]} < {flip_counts[i+1]} "
            f"when raising u threshold from {u_thresholds[i]} to {u_thresholds[i+1]}."
        )

def test_scale_part() -> None:
    """Verify scale_part scales coordinates around their center and keeps the batch index."""
    logger.info("Testing scale_part...")
    coords = torch.tensor([
        [0, 10, 10, 10],
        [0, 20, 10, 10],
        [0, 15, 15, 15]
    ], dtype=torch.int32)
    
    labels_dict = {
        "fine": torch.tensor([0, 1, 5], dtype=torch.long),
        "middle": torch.tensor([10, 10, 12], dtype=torch.long),
        "coarse": torch.tensor([20, 20, 21], dtype=torch.long)
    }
    
    # Scale middle branch 10 by factor of 2.0. Center of voxels 0 and 1 is (15, 10, 10)
    # Voxel 0: (10, 10, 10) -> (15 + (10-15)*2, 10, 10) = (5, 10, 10)
    # Voxel 1: (20, 10, 10) -> (15 + (20-15)*2, 10, 10) = (25, 10, 10)
    new_coords = scale_part(coords, labels_dict, branch_id=10, scale_factor=2.0, taxonomy=TEST_TAXONOMY, level="middle")
    
    expected_coords = torch.tensor([
        [0, 5, 10, 10],
        [0, 25, 10, 10],
        [0, 15, 15, 15]
    ], dtype=torch.int32)
    
    torch.testing.assert_close(new_coords, expected_coords)

def test_incremental_relabel() -> None:
    """Verify incremental_relabel propagates fine label changes up the hierarchy."""
    logger.info("Testing incremental_relabel...")
    
    # Hand-edited fine labels
    labels_dict = {
        "fine": torch.tensor([0, 1, 2, 4, 5], dtype=torch.long),
        "middle": torch.tensor([0, 0, 0, 0, 0], dtype=torch.long),
        "coarse": torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    }
    
    relabelled = incremental_relabel(labels_dict, TEST_TAXONOMY)
    
    expected_mid = torch.tensor([10, 10, 11, 12, 12], dtype=torch.long)
    expected_coarse = torch.tensor([20, 20, 20, 21, 21], dtype=torch.long)
    
    torch.testing.assert_close(relabelled["middle"], expected_mid)
    torch.testing.assert_close(relabelled["coarse"], expected_coarse)
