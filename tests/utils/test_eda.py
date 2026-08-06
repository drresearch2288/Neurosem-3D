"""Unit tests for eda_dataset_stats.py script."""

import os
import sys
import json
import pytest
import tempfile
import numpy as np

# Adjust path to import from scripts directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))
from eda_dataset_stats import main, get_category_from_obj_id


def test_get_category_from_obj_id() -> None:
    """Test mapping logic for categories based on object IDs."""
    assert get_category_from_obj_id("chair_model_12") == "Chair"
    assert get_category_from_obj_id("lamp_model_3") == "Lamp"
    assert get_category_from_obj_id("cabinet_model_4") == "Cabinet"
    assert get_category_from_obj_id("gear_model_5") == "Gear"
    
    # Fallback checks
    cat = get_category_from_obj_id("dummy_obj_0")
    assert cat in ["Chair", "Lamp", "Cabinet", "Gear"]


def test_eda_dataset_stats_main() -> None:
    """Smoke test running eda_dataset_stats.py main function with mock inputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Setup mock splits
        splits_dir = os.path.join(temp_dir, "splits")
        os.makedirs(splits_dir, exist_ok=True)
        
        splits = {
            "train": ["mock_chair_0", "mock_lamp_1"],
            "val": ["mock_cabinet_2"],
            "test": ["mock_gear_3", "mock_chair_4"]
        }
        
        for split_name, obj_list in splits.items():
            with open(os.path.join(splits_dir, f"{split_name}.json"), "w") as f:
                json.dump(obj_list, f)
                
        # 2. Setup mock gt_labels
        gt_labels_dir = os.path.join(temp_dir, "gt_labels")
        os.makedirs(gt_labels_dir, exist_ok=True)
        
        # Create a small dummy npz label grid for each object
        dummy_grid = np.zeros((16, 16, 16), dtype=np.int16)
        dummy_grid[2:5, 2:5, 2:5] = 4  # e.g., thin parts
        dummy_grid[8:12, 8:12, 8:12] = 1  # thick parts
        
        for obj_list in splits.values():
            for obj_id in obj_list:
                npz_path = os.path.join(gt_labels_dir, f"{obj_id}.npz")
                np.savez_compressed(npz_path, fine=dummy_grid)
                
        # 3. Setup output dirs
        figures_dir = os.path.join(temp_dir, "figures")
        tables_dir = os.path.join(temp_dir, "tables")
        
        # Mock sys.argv
        orig_argv = sys.argv
        sys.argv = [
            "eda_dataset_stats.py",
            "--splits-dir", splits_dir,
            "--gt-labels-dir", gt_labels_dir,
            "--figures-dir", figures_dir,
            "--tables-dir", tables_dir
        ]
        
        try:
            main()
            
            # Assert files are successfully generated
            assert os.path.exists(os.path.join(figures_dir, "split_counts.png"))
            assert os.path.exists(os.path.join(figures_dir, "taxonomy_depth_breadth.png"))
            assert os.path.exists(os.path.join(figures_dir, "thin_parts_voxel_distribution.png"))
            assert os.path.exists(os.path.join(tables_dir, "taxonomy_stats.csv"))
            
            # Verify CSV content
            df = np.loadtxt(os.path.join(tables_dir, "taxonomy_stats.csv"), delimiter=",", dtype=str, skiprows=1)
            # Should have 4 rows corresponding to the 4 categories
            assert len(df) == 4
            
        finally:
            sys.argv = orig_argv
