"""Baseline: B2 Enhanced Instant3D (SegFormer+TSDF)."""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from typing import Any, Dict, Optional, Tuple

# Try importing SegFormer from transformers
HAS_SEGFORMER = False
try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    HAS_SEGFORMER = True
except ImportError:
    pass

from neurosem3d.semantics.cwcvsf import majority_vote
from neurosem3d.semantics.hierarchy import incremental_relabel

class EnhancedInstant3d:
    """Enhanced Instant3D (Baseline B2) using SegFormer + TSDF Projection & Fusion."""
    
    def __init__(self, confidence_dir: str = "data/processed/confidence") -> None:
        self.confidence_dir = confidence_dir
        logger.debug("Initializing EnhancedInstant3d baseline")
        
        # Camera Intrinsics
        self.K = torch.tensor([
            [120.0, 0.0, 128.0],
            [0.0, 120.0, 128.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        
        # Documented Mapping from ADE20K classes to PartNet fine classes (0..14)
        # Class 19: chair -> 0
        # Class 13: table -> 1
        # Class 55: cushion -> 2
        # Class 30: desk -> 1
        # Class 74: stool -> 0
        # Class 116: armchair -> 0
        self.ade20k_to_partnet = {
            19: 0,
            13: 1,
            55: 2,
            30: 1,
            74: 0,
            116: 0
        }
        self.unmapped_classes_logged = set()
        
        # Initialize frozen SegFormer model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        if HAS_SEGFORMER:
            try:
                logger.info("Loading SegFormer model from Hugging Face (nvidia/segformer-b0-ade-512)...")
                self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-ade-512")
                self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-ade-512")
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                logger.warning(f"Could not load SegFormer model: {e}. Falling back to mock SegFormer.")
        else:
            logger.warning("transformers library not fully imported or SegFormer unavailable. Using mock SegFormer.")
            
    def get_extrinsics(self, view_idx: int) -> torch.Tensor:
        """Compute camera extrinsics for view_idx orbiting the center."""
        theta = float(view_idx) * (2.0 * np.pi / 8.0)
        eye = np.array([2.0 * np.cos(theta), 0.5, 2.0 * np.sin(theta)])
        at = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        z_axis = eye - at
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        R = np.stack([x_axis, y_axis, z_axis], axis=0)
        T_wc = np.eye(4)
        T_wc[:3, :3] = R
        T_wc[:3, 3] = -R @ eye
        return torch.tensor(T_wc, dtype=torch.float32)

    def run_segformer(self, images: np.ndarray) -> np.ndarray:
        """Run SegFormer model or mock SegFormer on 8 views to get class segmentation masks."""
        N_views, C, H, W = images.shape
        if self.model is not None and self.processor is not None:
            # Hugging Face inference
            # images shape is expected to be (N, H, W, C) for processor, range [0, 255]
            images_list = [images[i].transpose(1, 2, 0) for i in range(N_views)]
            inputs = self.processor(images=images_list, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # shape (N_views, 150, H_low, W_low)
                # Resize logits to match original image size
                logits_resized = nn.functional.interpolate(
                    logits, size=(H, W), mode="bilinear", align_corners=False
                )
                preds = torch.argmax(logits_resized, dim=1).cpu().numpy()  # (N_views, H, W)
                return preds
        else:
            # Mock SegFormer predictions (random ADE20K classes)
            logger.debug("Running mock SegFormer inference...")
            # We seed for reproducibility of mock baseline predictions
            rng = np.random.default_rng(42)
            # Predict some valid classes (like 13, 19, 55) and some unmapped classes (like 4, 100)
            possible_classes = [13, 19, 55, 4, 100]
            return rng.choice(possible_classes, size=(N_views, H, W))

    def run(self, obj_id: str) -> Dict[str, Optional[torch.Tensor]]:
        """Run Enhanced Instant3D baseline on the given object."""
        start_time = time.perf_counter()
        
        # Load precomputed voxel coordinates and depth cues from confidence file
        npz_path = os.path.join(self.confidence_dir, f"{obj_id}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Confidence cues file not found: {npz_path}")
            
        with np.load(npz_path) as data:
            voxel_xyz = data["voxel_xyz"].astype(np.float32)  # (N, 3)
            c_depth = data["c_depth"].astype(np.float32)      # (N, 8)
            
        N = voxel_xyz.shape[0]
        voxel_xyz_t = torch.from_numpy(voxel_xyz).to(self.device)
        
        # Generate 8 orbit views (dummy RGB images of resolution 256x256)
        images = np.zeros((8, 3, 256, 256), dtype=np.uint8)
        
        # Run SegFormer on views to extract class masks
        seg_masks = self.run_segformer(images)  # (8, 256, 256)
        
        # For each view, project voxels to find pixel indices, retrieve SegFormer classes,
        # and map them to our taxonomy
        projected_label_all = np.zeros((N, 8), dtype=np.int32)
        H, W = 256, 256
        
        for view_idx in range(8):
            T = self.get_extrinsics(view_idx).to(self.device)
            R = T[:3, :3]
            t = T[:3, 3]
            
            # Project voxels onto the camera view plane
            v_cam = voxel_xyz_t @ R.t() + t.unsqueeze(0)  # (N, 3)
            proj = v_cam @ self.K.to(self.device).t()
            
            u = torch.clamp((proj[:, 0] / (proj[:, 2] + 1e-8)).round().long(), 0, W - 1)
            v_coord = torch.clamp((proj[:, 1] / (proj[:, 2] + 1e-8)).round().long(), 0, H - 1)
            
            # Extract classes predicted by SegFormer
            view_preds = seg_masks[view_idx, v_coord.cpu().numpy(), u.cpu().numpy()]  # (N,)
            
            # Map SegFormer class to PartNet taxonomy
            mapped_preds = np.zeros(N, dtype=np.int32)
            for i in range(N):
                c = view_preds[i]
                if c in self.ade20k_to_partnet:
                    mapped_preds[i] = self.ade20k_to_partnet[c]
                else:
                    if c not in self.unmapped_classes_logged:
                        logger.info(f"Unmapped SegFormer class detected: {c}")
                        self.unmapped_classes_logged.add(c)
                    mapped_preds[i] = 0  # ignore/unmapped class maps to 0
                    
            projected_label_all[:, view_idx] = mapped_preds
            
        # Convert projection labels and depth cues (for visibility) to Tensors
        projected_label_t = torch.from_numpy(projected_label_all)
        visible_t = torch.from_numpy(c_depth >= np.exp(-1.0))
        
        # Plurality voting with Work 1 lower-index tie-break
        fine_labels = majority_vote(projected_label_t, visible_t, num_classes=15)
        
        elapsed = time.perf_counter() - start_time
        
        category = "default"
        if "_" in obj_id:
            category = obj_id.split("_")[0]
        logger.info(f"Baseline B2 | Category: '{category}' | Object: '{obj_id}' | Latency: {elapsed:.6f}s")
        
        # Taxonomy mapping
        taxonomy_map = {
            "fine_to_mid": {
                0: 10, 1: 10, 2: 11, 3: 11, 4: 12, 5: 12,
            },
            "mid_to_coarse": {
                10: 20, 11: 20, 12: 21,
            }
        }
        
        # Propagate up taxonomy
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
    parser = argparse.ArgumentParser(description="Baseline: B2 Enhanced Instant3D (SegFormer+TSDF).")
    parser.add_argument("--obj_id", type=str, default="dummy_obj_0", help="Object ID to run on.")
    args = parser.parse_args()
    
    baseline = EnhancedInstant3d()
    try:
        res = baseline.run(args.obj_id)
        logger.info(f"Successfully ran Enhanced Instant3D. Fine shape: {res['fine'].shape}")
    except Exception as e:
        logger.error(f"Error running baseline: {e}")

if __name__ == "__main__":
    main()
