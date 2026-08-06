"""Unit tests for B5 Direct 3D U-Net baseline."""

import pytest
import os
import torch
from loguru import logger

from neurosem3d.baselines.direct_3dunet import Direct3dunet, DirectUNet3D
from neurosem3d.semantics.losses import compute_losses

def test_direct_3dunet_features() -> None:
    """Verify B5 model uses input feature dimension of 257 exactly (no soft labels)."""
    logger.info("Verifying DirectUNet3D input feature dimension...")
    
    model = DirectUNet3D()
    assert model.in_channels == 257
    
    # Check that the UNet backbone first layer expects 257 input channels
    first_conv = model.unet.conv1
    assert first_conv.in_channels == 257

def test_direct_3dunet_training_step() -> None:
    """Verify that a single training step on a tiny batch decreases the loss."""
    logger.info("Running training step loss decrease test...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectUNet3D().to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Tiny batch mock input
    N = 10
    coords = torch.randint(0, 32, (N, 4), dtype=torch.int32).to(device)
    coords[:, 0] = 0
    feats_257 = torch.randn(N, 257, dtype=torch.float32).to(device)
    
    targets = {
        "coarse": torch.randint(0, 3, (N,)).to(device),
        "middle": torch.randint(0, 8, (N,)).to(device),
        "fine": torch.randint(0, 15, (N,)).to(device)
    }
    ignore_mask = torch.zeros(N, dtype=torch.bool).to(device)
    
    taxonomy_map = {
        0: 10, 1: 10, 2: 11, 3: 11, 4: 12, 5: 12,
        10: 20, 11: 20, 12: 21
    }
    
    # Run a few steps to let it fit
    initial_loss = None
    for step in range(15):
        optimizer.zero_grad()
        outputs = model(coords, feats_257)
        loss, _ = compute_losses(outputs, targets, ignore_mask, parent_map=taxonomy_map, coords=coords)
        loss.backward()
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
            
    final_loss = loss.item()
    logger.info(f"Initial Loss: {initial_loss:.6f} | Final Loss: {final_loss:.6f}")
    assert final_loss < initial_loss, "Loss should decrease after training steps."

def test_direct_3dunet_run() -> None:
    """Verify running inference yields expected output keys and shapes."""
    logger.info("Running Direct3dunet.run end-to-end test...")
    
    # Resolve sparse directory dynamically
    sparse_dir = "neurosem3d/data/processed/sparse"
    if not os.path.exists(sparse_dir):
        sparse_dir = "data/processed/sparse"
        
    baseline = Direct3dunet(model_path="nonexistent.pt", sparse_dir=sparse_dir)
    res = baseline.run("dummy_obj_0")
    
    for key in ["fine", "middle", "coarse", "u"]:
        assert key in res
        
    assert res["fine"].shape == res["u"].shape
