"""Unit tests for eda_confidence.py script."""

import os
import sys
import json
import pytest
import tempfile
import numpy as np

# Adjust path to import from scripts directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))
from eda_confidence import main


def test_eda_confidence_main() -> None:
    """Smoke test running eda_confidence.py main function with mock inputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Setup mock splits
        splits_dir = os.path.join(temp_dir, "splits")
        os.makedirs(splits_dir, exist_ok=True)
        
        splits = {
            "train": ["mock_obj_0"],
            "val": [],
            "test": []
        }
        for split_name, obj_list in splits.items():
            with open(os.path.join(splits_dir, f"{split_name}.json"), "w") as f:
                json.dump(obj_list, f)
                
        # 2. Setup mock sparse NPZ
        sparse_dir = os.path.join(temp_dir, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)
        
        N_points = 50
        coords = np.random.randint(0, 128, (N_points, 3))
        # Assign different labels to create boundaries
        fine = np.ones(N_points, dtype=np.int64)
        fine[:N_points//2] = 2
        
        np.savez_compressed(
            os.path.join(sparse_dir, "mock_obj_0.npz"),
            coords=coords,
            fine=fine
        )
        
        # 3. Setup mock confidence NPZ
        confidence_dir = os.path.join(temp_dir, "confidence")
        os.makedirs(confidence_dir, exist_ok=True)
        
        c_depth = np.random.uniform(0.7, 1.0, (N_points, 8))
        c_angle = np.random.uniform(0.6, 1.0, (N_points, 8))
        c_mask = np.random.uniform(0.8, 1.0, (N_points, 8))
        projected_label = np.random.randint(1, 4, (N_points, 8))
        
        np.savez_compressed(
            os.path.join(confidence_dir, "mock_obj_0.npz"),
            c_depth=c_depth,
            c_angle=c_angle,
            c_mask=c_mask,
            projected_label=projected_label
        )
        
        # 4. Setup output dirs
        figures_dir = os.path.join(temp_dir, "figures")
        tables_dir = os.path.join(temp_dir, "tables")
        
        # Mock sys.argv
        orig_argv = sys.argv
        sys.argv = [
            "eda_confidence.py",
            "--splits-dir", splits_dir,
            "--sparse-dir", sparse_dir,
            "--confidence-dir", confidence_dir,
            "--figures-dir", figures_dir,
            "--tables-dir", tables_dir
        ]
        
        try:
            main()
            
            # Assert files are successfully generated
            assert os.path.exists(os.path.join(figures_dir, "confidence_histograms_all.png"))
            assert os.path.exists(os.path.join(figures_dir, "confidence_boundary_vs_interior.png"))
            assert os.path.exists(os.path.join(figures_dir, "entropy_vs_boundary_distance.png"))
            assert os.path.exists(os.path.join(tables_dir, "confidence_summary.csv"))
            
            # Verify CSV content structure
            df = np.loadtxt(os.path.join(tables_dir, "confidence_summary.csv"), delimiter=",", dtype=str, skiprows=1)
            assert len(df) == 4
            
        finally:
            sys.argv = orig_argv
