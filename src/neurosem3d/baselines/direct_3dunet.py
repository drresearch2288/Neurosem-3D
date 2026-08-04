"""Baseline: B5 Direct 3D U-Net."""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from typing import Dict, Any, Optional

from neurosem3d.semantics.nsh import NeuralSemanticHead
from neurosem3d.semantics.hierarchy import tree_consistent_decode
from neurosem3d.data.dataset import NeuroSemDataset, sparse_collate_fn
from neurosem3d.semantics.losses import compute_losses

class DirectUNet3D(NeuralSemanticHead):
    """Direct 3D U-Net taking only latent z(v) and s(v) as inputs (in_channels=257)."""
    
    def __init__(self, num_classes_per_level: Dict[str, int] = None) -> None:
        super().__init__(in_channels=257, num_classes_per_level=num_classes_per_level)

class Direct3dunet:
    """Direct 3D U-Net (Baseline B5) runner."""
    
    def __init__(
        self,
        model_path: str = "results/models/b5_direct_unet.pt",
        sparse_dir: str = "data/processed/sparse"
    ) -> None:
        self.sparse_dir = sparse_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DirectUNet3D()
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded Direct 3D U-Net baseline from {model_path}")
        else:
            logger.warning(f"Direct 3D U-Net model not found at {model_path}. Running with random weights.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Taxonomy mapping
        self.taxonomy = {
            "fine_to_mid": {
                0: 10, 1: 10, 2: 11, 3: 11, 4: 12, 5: 12,
            },
            "mid_to_coarse": {
                10: 20, 11: 20, 12: 21,
            }
        }
        
    @torch.no_grad()
    def run(self, obj_id: str) -> Dict[str, Optional[torch.Tensor]]:
        """Run Direct 3D U-Net baseline inference on the given object.
        
        Args:
            obj_id (str): object identifier.
            
        Returns:
            Dict[str, Optional[torch.Tensor]]: containing fine, middle, coarse labels and uncertainty u.
        """
        start_time = time.perf_counter()
        
        # Load sparse voxel coords and features
        sparse_path = os.path.join(self.sparse_dir, f"{obj_id}.npz")
        if not os.path.exists(sparse_path):
            raise FileNotFoundError(f"Sparse voxel file not found: {sparse_path}")
            
        with np.load(sparse_path) as data:
            coords = data["coords"].astype(np.int32)
            feats = data["feats"].astype(np.float32)
            
        N = coords.shape[0]
        
        # Add batch dimension to coords: (N, 3) -> (N, 4)
        batch_col = np.zeros((N, 1), dtype=np.int32)
        coords_batched = np.concatenate([batch_col, coords], axis=1)
        
        coords_t = torch.from_numpy(coords_batched).to(self.device)
        # B5 uses only the first 257 channels: z(v) (256-d) + s(v) (1-d)
        feats_257 = torch.from_numpy(feats[:, :257]).to(self.device)
        
        # Model forward pass
        outputs = self.model(coords_t, feats_257)
        decoded = self.model.decode(outputs)
        
        # Retrieve logits and decode tree-consistently
        logits_per_level = {k: v["logits"] for k, v in outputs.items()}
        labels_dict = tree_consistent_decode(logits_per_level, self.taxonomy)
        
        # Compute calibrated uncertainty u(v) = K / sum(alpha)
        # In NSH.decode, confidence = 1.0 - K / S. So u = 1.0 - confidence
        u = 1.0 - decoded["fine"]["confidence"]
        
        elapsed = time.perf_counter() - start_time
        
        category = "default"
        if "_" in obj_id:
            category = obj_id.split("_")[0]
        logger.info(f"Baseline B5 Direct 3D U-Net | Category: '{category}' | Object: '{obj_id}' | Latency: {elapsed:.6f}s")
        
        return {
            "coarse": labels_dict["coarse"],
            "middle": labels_dict["middle"],
            "fine": labels_dict["fine"],
            "u": u
        }

def train_baseline(model_path: str = "results/models/b5_direct_unet.pt", epochs: int = 2) -> None:
    """Train the Direct 3D U-Net baseline on the train split and save to model_path."""
    logger.info("Training Direct 3D U-Net baseline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Load splits/dataset
    dataset = NeuroSemDataset(split="train")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=sparse_collate_fn)
    
    model = DirectUNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    taxonomy_map = {
        0: 10, 1: 10, 2: 11, 3: 11, 4: 12, 5: 12,
        10: 20, 11: 20, 12: 21
    }
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            coords = batch["coords"].to(device)
            feats = batch["feats"].to(device)
            # Slice only 257 features
            feats_257 = feats[:, :257]
            
            targets = {
                "coarse": (batch["coarse"] % 3).to(device),
                "middle": (batch["middle"] % 8).to(device),
                "fine": (batch["fine"] % 15).to(device)
            }
            ignore_mask = batch["ignore_mask"].to(device)
            
            optimizer.zero_grad()
            outputs = model(coords, feats_257)
            
            # Compute losses
            loss, _ = compute_losses(
                outputs, targets, ignore_mask,
                betas=(0.5, 0.3, 0.1), parent_map=taxonomy_map, coords=coords
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        logger.info(f"Direct U-Net Train Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved trained Direct 3D U-Net model to {model_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline: B5 Direct 3D U-Net.")
    parser.add_argument("--train", action="store_true", help="Train the baseline model.")
    parser.add_argument("--model_path", type=str, default="results/models/b5_direct_unet.pt", help="Path to save model.")
    args = parser.parse_args()
    
    if args.train:
        train_baseline(args.model_path)
    else:
        baseline = Direct3dunet(model_path=args.model_path)
        try:
            res = baseline.run("dummy_obj_0")
            logger.info(f"Successfully ran Direct 3D U-Net baseline. Fine shape: {res['fine'].shape}")
        except Exception as e:
            logger.error(f"Error running baseline: {e}")

if __name__ == "__main__":
    main()
