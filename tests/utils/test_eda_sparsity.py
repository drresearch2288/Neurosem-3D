"""Unit tests for eda_sparsity.py script."""

import os
import sys
import json
import pytest
import tempfile
import numpy as np

# Adjust path to import from scripts directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))
from eda_sparsity import main, get_category_from_obj_id


def test_get_category_from_obj_id() -> None:
    """Test category mapping based on object IDs."""
    assert get_category_from_obj_id("chair_test") == "Chair"
    assert get_category_from_obj_id("lamp_test") == "Lamp"
    assert get_category_from_obj_id("cabinet_test") == "Cabinet"
    assert get_category_from_obj_id("gear_test") == "Gear"


def test_eda_sparsity_main() -> None:
    """Smoke test running eda_sparsity.py main function with mock inputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Setup mock splits
        splits_dir = os.path.join(temp_dir, "splits")
        os.makedirs(splits_dir, exist_ok=True)
        
        splits = {
            "train": ["mock_chair_0", "mock_lamp_1"],
            "val": [],
            "test": []
        }
        for split_name, obj_list in splits.items():
            with open(os.path.join(splits_dir, f"{split_name}.json"), "w") as f:
                json.dump(obj_list, f)
                
        # 2. Setup mock sparse NPZ files
        sparse_dir = os.path.join(temp_dir, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)
        
        # 128^3 resolution coordinates representation
        dummy_coords = np.random.randint(0, 128, (100000, 3))
        
        for obj_list in splits.values():
            for obj_id in obj_list:
                npz_path = os.path.join(sparse_dir, f"{obj_id}.npz")
                np.savez_compressed(npz_path, coords=dummy_coords)
                
        # 3. Setup output dirs
        figures_dir = os.path.join(temp_dir, "figures")
        tables_dir = os.path.join(temp_dir, "tables")
        
        # Mock sys.argv
        orig_argv = sys.argv
        sys.argv = [
            "eda_sparsity.py",
            "--splits-dir", splits_dir,
            "--sparse-dir", sparse_dir,
            "--figures-dir", figures_dir,
            "--tables-dir", tables_dir
        ]
        
        try:
            main()
            
            # Assert files are successfully generated
            assert os.path.exists(os.path.join(figures_dir, "sparsity_occupancy_histogram.png"))
            assert os.path.exists(os.path.join(figures_dir, "sparsity_category_occupancy.png"))
            assert os.path.exists(os.path.join(figures_dir, "sparsity_memory_comparison.png"))
            assert os.path.exists(os.path.join(tables_dir, "occupancy.csv"))
            
            # Verify CSV content structure
            df = np.loadtxt(os.path.join(tables_dir, "occupancy.csv"), delimiter=",", dtype=str, skiprows=1)
            # 2 classes (Chair, Lamp) + 1 overall row = 3 rows
            assert len(df) == 3
            
        finally:
            sys.argv = orig_argv
