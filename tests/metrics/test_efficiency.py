"""
Tests for efficiency metrics (GPU memory, model size).
"""

import pytest
import torch
import torch.nn as nn
from loguru import logger
from neurosem3d.metrics.efficiency import peak_gpu_mem_gb, model_size_mb


def test_model_size_mb_hand_check() -> None:
    """Test model size calculation on a small linear layer.
    
    Formula check:
        nn.Linear(10, 20)
        Parameters:
            - weight: shape (20, 10) -> 200 elements
            - bias: shape (20,) -> 20 elements
            Total = 220 float32 elements (4 bytes each) = 880 bytes.
            Size in MB = 880 / 1,048,576 = 0.0008392333984375 MB
    """
    logger.info("Testing model size MB hand-checked case")
    model = nn.Linear(10, 20)
    
    res = model_size_mb(model)
    expected_size_mb = 880.0 / (1024 * 1024)
    assert pytest.approx(res["model_size_mb"]) == expected_size_mb


def test_peak_gpu_mem_gb() -> None:
    """Verify peak GPU memory returns a float value (can be 0.0 on CPU)."""
    logger.info("Testing peak GPU memory retrieval")
    res = peak_gpu_mem_gb()
    assert isinstance(res["peak_gpu_mem_gb"], float)
    assert res["peak_gpu_mem_gb"] >= 0.0
