"""
Tests for NSH composite loss functions.
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from neurosem3d.semantics.losses import compute_losses

def test_losses_properties() -> None:
    """Test composite loss non-negativity, perfect prediction limits, and tree consistency."""
    logger.info("Starting NSH loss checks...")
    
    device = torch.device("cpu")
    N = 4
    
    # 1. Setup perfect mock outputs & targets
    # coarse classes: 3, middle classes: 8, fine classes: 15
    num_classes_per_level = {"coarse": 3, "middle": 8, "fine": 15}
    
    # Perfect targets
    # Voxel indices 0..3: child classes [1, 2, 3, 4] all map to middle class 0, which maps to coarse class 1
    targets = {
        "coarse": torch.tensor([1, 1, 1, 1], dtype=torch.long, device=device),
        "middle": torch.tensor([0, 0, 0, 0], dtype=torch.long, device=device),
        "fine": torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)
    }
    
    # Perfect outputs (extremely high logits for the target classes)
    outputs = {}
    for lvl in ["coarse", "middle", "fine"]:
        K = num_classes_per_level[lvl]
        target = targets[lvl]
        
        # Create perfect one-hot logits
        logits = F.one_hot(target, num_classes=K).float() * 100.0  # huge logit differences
        alpha = F.softplus(logits) + 1.0
        
        outputs[lvl] = {
            "logits": logits,
            "alpha": alpha
        }
        
    ignore_mask = torch.tensor([False, False, False, False], dtype=torch.bool, device=device)
    parent_map = {
        1: 0, 2: 0, 3: 0, 4: 0,
        0: 1
    }
    
    # 2. Compute losses
    L_total, terms = compute_losses(
        outputs=outputs,
        targets=targets,
        ignore_mask=ignore_mask,
        betas=(0.5, 0.3, 0.1),
        parent_map=parent_map
    )
    
    # Assertions
    # 1. Non-negativity and finiteness
    assert L_total.item() >= 0.0
    for name, val in terms.items():
        assert val.item() >= 0.0, f"{name} should be non-negative, got {val.item()}"
        assert torch.isfinite(val), f"{name} is not finite!"
        
    # 2. With perfect predictions L_sem -> 0
    assert terms["L_sem"].item() < 1e-4, f"L_sem should be close to 0 for perfect predictions, got {terms['L_sem'].item()}"
    
    # 3. L_hier = 0 when child/parent labels are tree-consistent
    # Since fine predictions map perfectly to middle predictions (e.g. class 1 & 2 maps to middle 0; class 5 & 6 maps to middle 1),
    # and the middle logits are also perfectly 0 & 1, L_hier should be 0.
    assert terms["L_hier"].item() < 1e-4, f"L_hier should be close to 0 for tree-consistent predictions, got {terms['L_hier'].item()}"
    
    logger.success("NSH composite loss properties verified successfully.")

