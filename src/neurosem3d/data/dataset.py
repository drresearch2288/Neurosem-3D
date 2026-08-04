import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger
from typing import Any, Dict, List, Optional, Tuple, Union

from neurosem3d.semantics.cwcvsf import fuse_semantics

class NeuroSemDataset(Dataset):
    """PyTorch Dataset for NeuroSem-3D sparse semantic fields."""
    
    def __init__(
        self,
        split: str,
        sparse_dir: str = "data/processed/sparse",
        confidence_dir: str = "data/processed/confidence",
        splits_dir: str = "data/splits",
        augment: bool = False,
        jitter_std: float = 0.02,
        mask_dropout_prob: float = 0.2
    ) -> None:
        """Initialize NeuroSemDataset.
        
        Args:
            split (str): One of 'train', 'val', 'test'.
            sparse_dir (str): Folder containing sparse voxel npz files.
            confidence_dir (str): Folder containing confidence cues npz files.
            splits_dir (str): Folder containing splits JSON files.
            augment (bool): Whether to apply training-time augmentation.
            jitter_std (float): Standard deviation of Gaussian noise for cues jitter.
            mask_dropout_prob (float): Probability of dropping view votes.
        """
        self.split = split
        self.sparse_dir = sparse_dir
        self.confidence_dir = confidence_dir
        self.augment = augment
        self.jitter_std = jitter_std
        self.mask_dropout_prob = mask_dropout_prob
        
        # Load splits file
        splits_path = os.path.join(splits_dir, f"{split}.json")
        if os.path.exists(splits_path):
            with open(splits_path, "r") as f:
                self.object_ids = json.load(f)
        else:
            logger.warning(f"Splits file {splits_path} not found. Using fallback mock objects.")
            self.object_ids = [f"dummy_obj_{i}" for i in range(5)]
            
        logger.info(f"Loaded NeuroSemDataset '{split}' with {len(self.object_ids)} items.")

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        obj_id = self.object_ids[idx]
        
        # 1. Load sparse file
        sparse_path = os.path.join(self.sparse_dir, f"{obj_id}.npz")
        with np.load(sparse_path) as data:
            coords = data["coords"].astype(np.int32)
            feats = data["feats"].astype(np.float32)
            coarse = data["coarse"].astype(np.int64)
            middle = data["middle"].astype(np.int64)
            fine = data["fine"].astype(np.int64)
            ignore_mask = data["ignore_mask"].astype(np.bool_)
            
        # 2. Load confidence file
        confidence_path = os.path.join(self.confidence_dir, f"{obj_id}.npz")
        with np.load(confidence_path) as data:
            c_depth = data["c_depth"].astype(np.float32)
            c_angle = data["c_angle"].astype(np.float32)
            c_mask = data["c_mask"].astype(np.float32)
            proj_label = data["projected_label"].astype(np.int32)
            
        # 3. Training-time augmentation
        if self.augment:
            # Random jitter of confidence cues
            if self.jitter_std > 0:
                c_depth = np.clip(c_depth + np.random.normal(0, self.jitter_std, c_depth.shape), 0.0, 1.0)
                c_angle = np.clip(c_angle + np.random.normal(0, self.jitter_std, c_angle.shape), 0.0, 1.0)
                c_mask = np.clip(c_mask + np.random.normal(0, self.jitter_std, c_mask.shape), 0.0, 1.0)
                
            # Mask dropout: drop one random view's votes
            if np.random.rand() < self.mask_dropout_prob:
                drop_view = np.random.randint(8)
                c_depth[:, drop_view] = 0.0
                c_angle[:, drop_view] = 0.0
                c_mask[:, drop_view] = 0.0
                
                # Re-compute P_fuse using modified cues
                P_fuse = fuse_semantics(c_depth, c_angle, c_mask, proj_label, num_classes=15)
                # Overwrite P_fuse features in feats: feats[:, 257:] is P_fuse
                feats[:, 257:] = P_fuse
                
        # Convert to torch Tensors
        coords_t = torch.from_numpy(coords)
        feats_t = torch.from_numpy(feats)
        coarse_t = torch.from_numpy(coarse)
        middle_t = torch.from_numpy(middle)
        fine_t = torch.from_numpy(fine)
        ignore_mask_t = torch.from_numpy(ignore_mask)
        
        return coords_t, feats_t, coarse_t, middle_t, fine_t, ignore_mask_t, obj_id

def sparse_collate_fn(batch: List[Tuple]) -> Dict[str, Any]:
    """Collate batch of sparse voxel coordinates and features for MinkowskiEngine/spconv.
    
    Args:
        batch (List[Tuple]): List of tuples returned by NeuroSemDataset.__getitem__.
        
    Returns:
        Dict[str, Any]: batched tensors:
            - 'coords': shape (N_total, 4) containing [batch_idx, z, y, x].
            - 'feats': shape (N_total, C).
            - 'coarse': shape (N_total,).
            - 'middle': shape (N_total,).
            - 'fine': shape (N_total,).
            - 'ignore_mask': shape (N_total,).
            - 'obj_ids': List[str] of original object ids.
    """
    coords_list = []
    feats_list = []
    coarse_list = []
    middle_list = []
    fine_list = []
    ignore_mask_list = []
    obj_ids = []
    
    for b_idx, item in enumerate(batch):
        coords_i, feats_i, coarse_i, middle_i, fine_i, ignore_mask_i, obj_id = item
        
        # Prepend batch index to coordinates: (N, 3) -> (N, 4)
        N = coords_i.shape[0]
        batch_idx_col = torch.full((N, 1), b_idx, dtype=torch.int32)
        coords_i_batched = torch.cat([batch_idx_col, coords_i], dim=1)
        
        coords_list.append(coords_i_batched)
        feats_list.append(feats_i)
        coarse_list.append(coarse_i)
        middle_list.append(middle_i)
        fine_list.append(fine_i)
        ignore_mask_list.append(ignore_mask_i)
        obj_ids.append(obj_id)
        
    return {
        "coords": torch.cat(coords_list, dim=0),
        "feats": torch.cat(feats_list, dim=0),
        "coarse": torch.cat(coarse_list, dim=0),
        "middle": torch.cat(middle_list, dim=0),
        "fine": torch.cat(fine_list, dim=0),
        "ignore_mask": torch.cat(ignore_mask_list, dim=0),
        "obj_ids": obj_ids
    }

def main() -> None:
    """Main entry point for testing or running dataset independently."""
    parser = argparse.ArgumentParser(description="PyTorch dataset implementations.")
    args = parser.parse_args()
    logger.info("Running dataset verify stub")

if __name__ == "__main__":
    main()
