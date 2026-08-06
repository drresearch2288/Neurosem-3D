"""
Tests for sparse_grid representations
"""

import pytest
import numpy as np
import torch
from loguru import logger
from neurosem3d.data.sparse_grid import SparseVoxels, build_from_sdf, densify, neighbour_query

def test_sparse_voxels_assertions() -> None:
    """Test shape constraints and value bounds on sparse voxels representation."""
    logger.info("Starting sparse grid assertions verification...")
    
    # 1. Create a mock s_grid (SDF) of size 128^3
    res = 128
    s_grid = np.ones((res, res, res), dtype=np.float32)
    # create a surface band at center
    s_grid[60:68, 60:68, 60:68] = 0.05
    
    # 2. Build coordinates from SDF
    coords = build_from_sdf(s_grid, band=0.1)
    N = coords.shape[0]
    
    # Assert coordinates are within [0, 128)
    assert N > 0, "Occupied voxel count should be greater than 0"
    assert np.all(coords >= 0) and np.all(coords < 128), "Voxel coordinates must be within range [0, 128)"
    
    # 3. Create mock features [N, 256 + 1 + 15] = [N, 272]
    K = 15
    feat_dim = 256 + 1 + K
    feats = np.random.randn(N, feat_dim).astype(np.float32)
    
    # Instantiate SparseVoxels
    sparse_repr = SparseVoxels(coords, feats)
    
    # Assert features dimensions
    assert sparse_repr.features.shape[0] == N, "Features row dimension must match coordinates count"
    assert sparse_repr.features.shape[1] == 272, f"Expected feature dimension to be 272, got {sparse_repr.features.shape[1]}"
    
    # 4. Verify densification works
    dense_grid = densify(coords, feats[:, 0], res=128)
    assert dense_grid.shape == (128, 128, 128), f"Densified grid shape must be (128, 128, 128), got {dense_grid.shape}"
    assert np.allclose(dense_grid[coords[:, 0], coords[:, 1], coords[:, 2]], feats[:, 0])
    
    # 5. Verify neighbour_query runs without issues
    # Use a small subset to verify KDTree neighbors
    subset_coords = coords[:50]
    neighbors = neighbour_query(subset_coords, distance=1)
    assert len(neighbors) == 50, "KDTree query must return neighbors for each voxel"
    
    logger.success("Sparse grid representation verification PASSED.")

