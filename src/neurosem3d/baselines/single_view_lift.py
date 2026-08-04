"""Baseline: B4 Single-view 2D-lift (SAM)."""

import os
import time
import argparse
import numpy as np
import torch
from loguru import logger
from typing import Dict, Any, Optional

from neurosem3d.semantics.hierarchy import incremental_relabel

class SingleViewLift:
    """Baseline B4: Single-View 2D-Lift monocular projection baseline.
    
    Lifts only a single view's SAM segmentations into 3D using depth and camera parameters.
    No cross-view fusion or learned head.
    """
    
    def __init__(self, confidence_dir: str = "data/processed/confidence", default_view_index: int = 0) -> None:
        """Initialize SingleViewLift.
        
        Args:
            confidence_dir (str): directory containing the confidence npz files.
            default_view_index (int): default view index (e.g. 0 for front view).
        """
        self.confidence_dir = confidence_dir
        self.default_view_index = default_view_index
        logger.debug(f"Initializing SingleViewLift baseline (default_view_index={default_view_index})")
        
    def run(self, obj_id: str, view_index: Optional[int] = None) -> Dict[str, Optional[torch.Tensor]]:
        """Run single view lifting on the given object.
        
        Args:
            obj_id (str): object identifier.
            view_index (int): view index to lift. If None, uses default_view_index.
            
        Returns:
            Dict[str, Optional[torch.Tensor]]: containing fine, middle, coarse labels and u=None.
        """
        start_time = time.perf_counter()
        
        if view_index is None:
            view_index = self.default_view_index
            
        # Load precomputed projected labels and depth cues
        npz_path = os.path.join(self.confidence_dir, f"{obj_id}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Confidence cues file not found: {npz_path}")
            
        with np.load(npz_path) as data:
            c_depth = data["c_depth"].astype(np.float32)            # (N, 8)
            projected_label = data["projected_label"].astype(np.int32)  # (N, 8)
            
        # Get visibility mask for chosen view
        visible = c_depth[:, view_index] >= np.exp(-1.0)
        visible_t = torch.from_numpy(visible)
        projected_label_t = torch.from_numpy(projected_label[:, view_index])
        
        # Label only voxels visible from this view. Unseen voxels get label 0.
        fine_labels = torch.where(visible_t, projected_label_t, torch.zeros_like(projected_label_t))
        
        elapsed = time.perf_counter() - start_time
        
        category = "default"
        if "_" in obj_id:
            category = obj_id.split("_")[0]
        logger.info(f"Baseline B4 Single-View Lift | View: {view_index} | Category: '{category}' | Object: '{obj_id}' | Latency: {elapsed:.6f}s")
        
        # Map to taxonomy hierarchy
        taxonomy_map = {
            "fine_to_mid": {
                0: 10, 1: 10, 2: 11, 3: 11, 4: 12, 5: 12,
            },
            "mid_to_coarse": {
                10: 20, 11: 20, 12: 21,
            }
        }
        
        labels_dict = {
            "fine": fine_labels,
            "middle": torch.zeros_like(fine_labels),
            "coarse": torch.zeros_like(fine_labels)
        }
        relabelled = incremental_relabel(labels_dict, taxonomy_map)
        
        return {
            "coarse": relabelled["coarse"],
            "middle": relabelled["middle"],
            "fine": relabelled["fine"],
            "u": None
        }

def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline: B4 Single-view 2D-lift (SAM).")
    parser.add_argument("--obj_id", type=str, default="dummy_obj_0", help="Object ID to run on.")
    parser.add_argument("--view_index", type=int, default=0, help="View index to lift.")
    args = parser.parse_args()
    
    baseline = SingleViewLift()
    try:
        res = baseline.run(args.obj_id, view_index=args.view_index)
        logger.info(f"Successfully ran single view lift. Fine shape: {res['fine'].shape}")
    except Exception as e:
        logger.error(f"Error running baseline B4: {e}")

if __name__ == "__main__":
    main()
