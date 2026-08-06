"""Unit tests for NeuroSem-3D SVDI model, distillation, and SVDIRunner."""

import pytest
import torch
import torch.nn as nn
from loguru import logger
from typing import Dict, Any

from neurosem3d.semantics.nsh import NeuralSemanticHead
from neurosem3d.semantics.svdi import StudentNSH, SVDIRunner

# Test taxonomy
TEST_TAXONOMY = {
    "fine_to_mid": {
        0: 10,
        1: 10,
        2: 11,
        3: 11,
        4: 12,
        5: 12,
    },
    "mid_to_coarse": {
        10: 20,
        11: 20,
        12: 21,
    }
}

def get_model_size_mb(model: nn.Module) -> float:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024

def test_student_teacher_shapes() -> None:
    """Assert student output shapes match teacher output shapes."""
    logger.info("Verifying student vs teacher output shapes...")
    
    in_channels = 272
    num_classes_per_level = {"coarse": 3, "middle": 8, "fine": 15}
    
    teacher = NeuralSemanticHead(in_channels=in_channels, num_classes_per_level=num_classes_per_level)
    student = StudentNSH(in_channels=in_channels, num_classes_per_level=num_classes_per_level)
    
    # Mock sparse input
    N = 100
    coords = torch.randint(0, 32, (N, 4), dtype=torch.int32)
    coords[:, 0] = 0  # Single batch item
    feats = torch.randn(N, in_channels)
    
    with torch.no_grad():
        teacher_out = teacher(coords, feats)
        student_out = student(coords, feats)
        
    # Check levels
    assert set(teacher_out.keys()) == set(student_out.keys())
    
    for lvl in teacher_out.keys():
        t_lvl = teacher_out[lvl]
        s_lvl = student_out[lvl]
        
        assert t_lvl["logits"].shape == s_lvl["logits"].shape
        assert t_lvl["alpha"].shape == s_lvl["alpha"].shape
        
    # Log model sizes
    t_size = get_model_size_mb(teacher)
    s_size = get_model_size_mb(student)
    logger.info(f"Teacher size: {t_size:.4f} MB")
    logger.info(f"Student size: {s_size:.4f} MB")
    assert s_size < t_size, f"Student size ({s_size:.4f}MB) should be smaller than teacher ({t_size:.4f}MB)"

def test_incremental_relabel_isolated() -> None:
    """Assert incremental relabel touches only the edited branch's voxels."""
    logger.info("Verifying incremental relabel isolation...")
    
    runner = SVDIRunner(student_path="nonexistent.pt", taxonomy=TEST_TAXONOMY)
    
    # Create mock sparse input
    N = 10
    coords = torch.randint(0, 32, (N, 4), dtype=torch.int32)
    coords[:, 0] = 0
    feats = torch.randn(N, 272)
    
    # Set labels where only some voxels belong to branch 12 (descendants 12, 4, 5)
    # Voxels 0..4 belong to branch 10 (coarse 20)
    # Voxels 5..9 belong to branch 12 (coarse 21)
    labels_dict = {
        "fine": torch.tensor([0, 1, 0, 1, 0, 4, 5, 4, 5, 4], dtype=torch.long),
        "middle": torch.tensor([10, 10, 10, 10, 10, 12, 12, 12, 12, 12], dtype=torch.long),
        "coarse": torch.tensor([20, 20, 20, 20, 20, 21, 21, 21, 21, 21], dtype=torch.long)
    }
    
    sparse_input = (coords, feats, labels_dict)
    
    # Relabel only branch 12
    new_labels, elapsed = runner.relabel(sparse_input, edited_branch=12)
    
    # Check that voxels 0..4 (which do not belong to branch 12) are completely untouched
    for k in labels_dict.keys():
        torch.testing.assert_close(new_labels[k][:5], labels_dict[k][:5].to(new_labels[k].device))
        
    # Log measured latency
    logger.info(f"Measured SVDIRunner incremental relabel latency: {elapsed * 1000:.3f} ms")
    assert elapsed < 0.5, f"Relabel execution took too long: {elapsed:.4f}s"
