"""
Tests for neural_sdf
"""

import pytest
import torch
from loguru import logger
from neurosem3d.backbone.neural_sdf import load_frozen

def test_neural_sdf_frozen() -> None:
    """Test that load_frozen produces a model with no trainable parameters."""
    logger.info("Testing that load_frozen sets requires_grad to False on all parameters.")
    model = load_frozen(ckpt_path=None)
    
    total_grad_params = sum(p.requires_grad for p in model.parameters())
    assert total_grad_params == 0, f"Expected 0 trainable parameters, got {total_grad_params}"
    logger.info("Successfully verified model is fully frozen.")

