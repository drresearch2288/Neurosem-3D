"""Baseline: B1 SemGen-3D majority-vote (predecessor)."""

import os
import time
import argparse
import numpy as np
import torch
from loguru import logger
from typing import Any, Dict, Optional, Tuple

from neurosem3d.semantics.cwcvsf import majority_vote
from neurosem3d.semantics.hierarchy import incremental_relabel

class Semgen3dMajority:
    """SemGen-3D (Work 1) Baseline using equal-weight majority-vote semantics."""
    
    def __init__(self, confidence_dir: str = "data/processed/confidence") -> None:
        """Initialize Semgen3dMajority.
        
        Args:
            confidence_dir (str): directory containing the confidence npz files.
        """
        self.confidence_dir = confidence_dir
        logger.debug("Initializing Semgen3dMajority baseline")
        
    def run(self, obj_id: str) -> Dict[str, Optional[torch.Tensor]]:
        """Run majority vote semantics on the given object.
        
        Args:
            obj_id (str): object identifier.
            
        Returns:
            Dict[str, Optional[torch.Tensor]]: containing fine, middle, coarse labels and u=None.
        """
        start_time = time.perf_counter()
        
        # Load precomputed projected labels and confidence cues
        npz_path = os.path.join(self.confidence_dir, f"{obj_id}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Confidence cues file not found: {npz_path}")
            
        with np.load(npz_path) as data:
            c_depth = data["c_depth"].astype(np.float32)            # (N, 8)
            projected_label = data["projected_label"].astype(np.int32)  # (N, 8)
            
        # Hard visibility gate (Work 1): |d_v_i - D_i| <= delta
        # Since c_depth = exp(-|d_v_i - D_i| / delta), the gate is c_depth >= exp(-1.0)
        visible = c_depth >= np.exp(-1.0)
        
        # Convert to torch Tensors for majority_vote
        projected_label_t = torch.from_numpy(projected_label)
        visible_t = torch.from_numpy(visible)
        
        # Run majority vote (Work 1 plurality with lower-index tie-break)
        fine_labels = majority_vote(projected_label_t, visible_t, num_classes=15)
        
        elapsed = time.perf_counter() - start_time
        
        # Determine category for per-category logging
        category = "default"
        if "_" in obj_id:
            category = obj_id.split("_")[0]
        logger.info(f"Baseline B1 | Category: '{category}' | Object: '{obj_id}' | Latency: {elapsed:.6f}s")
        
        # Define taxonomy map for mapping fine labels to middle and coarse levels
        taxonomy_map = {
            "fine_to_mid": {
                0: 10, 1: 10, 2: 11, 3: 11, 4: 12, 5: 12,
            },
            "mid_to_coarse": {
                10: 20, 11: 20, 12: 21,
            }
        }
        
        # Re-propagate up the taxonomy to construct consistent middle/coarse baseline targets
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
    parser = argparse.ArgumentParser(description="Baseline: B1 SemGen-3D majority-vote (predecessor).")
    parser.add_argument("--obj_id", type=str, default="dummy_obj_0", help="Object ID to run on.")
    args = parser.parse_args()
    
    baseline = Semgen3dMajority()
    try:
        res = baseline.run(args.obj_id)
        logger.info(f"Successfully ran majority vote. Fine shape: {res['fine'].shape}")
    except Exception as e:
        logger.error(f"Error running baseline: {e}")

if __name__ == "__main__":
    main()
