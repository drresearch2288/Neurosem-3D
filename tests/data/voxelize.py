"""
Tests for voxelize
"""

import pytest
import numpy as np
from loguru import logger
from neurosem3d.data.voxelize import voxelize_part_tree

def test_voxelize_surface_and_hierarchy() -> None:
    """Test near-surface band restriction and hierarchical tree consistency."""
    logger.info("Starting voxelize test...")
    
    # 1. Create a dummy SDF grid (64^3) with known values
    res = 64
    coords = np.linspace(-1.0, 1.0, res)
    cx, cy, cz = np.meshgrid(coords, coords, coords, indexing='ij')
    dist = np.sqrt(cx**2 + cy**2 + cz**2)
    s_grid = dist - 0.5  # surface at radius 0.5
    
    # 2. Define dummy tree hierarchy
    # Fine parts: 1..8
    # Middle parts: 10, 11 (1..4 -> 10, 5..8 -> 11)
    # Coarse parts: 100, 101 (10 -> 100, 11 -> 101)
    tree = {
        "parent_map": {
            1: 10, 2: 10, 3: 10, 4: 10,
            5: 11, 6: 11, 7: 11, 8: 11,
            10: 100, 11: 101
        }
    }
    
    level_label_maps = {
        "fine": {i: i for i in range(1, 9)},
        "middle": {10: 10, 11: 11},
        "coarse": {100: 100, 101: 101}
    }
    
    # 3. Call voxelize
    res_voxel = 64
    threshold = 0.1
    results = voxelize_part_tree(
        mesh=None,
        tree=tree,
        level_label_maps=level_label_maps,
        s_grid=s_grid,
        res=res_voxel,
        sdf_threshold=threshold
    )
    
    coarse = results["coarse"]
    middle = results["middle"]
    fine = results["fine"]
    ignore_mask = results["ignore_mask"]
    
    # Assert (a): All non-ignore voxels lie within the near-surface band
    # Non-ignore is where ignore_mask is False (so ~ignore_mask)
    valid_voxels = ~ignore_mask
    assert np.all(np.abs(s_grid[valid_voxels]) <= threshold), "Some non-ignored voxels are outside the near-surface band!"
    assert np.all(coarse[ignore_mask] == 0), "Ignored voxels must have class 0"
    assert np.all(fine[ignore_mask] == 0), "Ignored voxels must have class 0"
    
    # Assert (b): Tree-consistency holds between coarse and fine after aggregation
    # For every voxel, the coarse label must correspond to the parent mapping of its fine label
    parent_map = tree["parent_map"]
    
    valid_indices = np.where(valid_voxels)
    for idx in zip(*valid_indices):
        f_val = fine[idx]
        m_val = middle[idx]
        c_val = coarse[idx]
        
        # Verify parent mappings
        assert parent_map[f_val] == m_val, f"Middle label {m_val} does not match parent of fine label {f_val}"
        assert parent_map[m_val] == c_val, f"Coarse label {c_val} does not match parent of middle label {m_val}"
        
    logger.info("Voxelise tests successfully verified near-surface band and hierarchy consistency.")

