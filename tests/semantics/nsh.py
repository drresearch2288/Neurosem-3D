"""
Tests for Neural Semantic Head (NSH)
"""

import pytest
import torch
from loguru import logger
from neurosem3d.semantics.nsh import NeuralSemanticHead

def test_nsh_shapes_and_decoding() -> None:
    """Test NSH forward output shapes, coordinate alignment, and decoding."""
    logger.info("Starting NSH verification tests...")
    
    # 1. Setup mock sparse tensor (N=5 voxels, batch index 0, C=272)
    device = torch.device("cpu")
    N = 5
    coords = torch.tensor([
        [0, 10, 10, 10],
        [0, 11, 10, 10],
        [0, 10, 12, 10],
        [0, 10, 10, 13],
        [0, 15, 15, 15]
    ], dtype=torch.int32, device=device)
    
    feats = torch.randn(N, 272, dtype=torch.float32, device=device)
    
    # 2. Instantiate NSH model
    num_classes_per_level = {"coarse": 3, "middle": 8, "fine": 15}
    model = NeuralSemanticHead(in_channels=272, num_classes_per_level=num_classes_per_level)
    
    # Run forward pass
    outputs = model(coords, feats)
    
    # Assert output alignment
    for lvl in ["coarse", "middle", "fine"]:
        K = num_classes_per_level[lvl]
        assert lvl in outputs, f"Expected {lvl} in outputs"
        
        logits = outputs[lvl]["logits"]
        alpha = outputs[lvl]["alpha"]
        
        # Output coordinates match voxel count N
        assert logits.shape == (N, K), f"Logits shape should be {(N, K)}, got {logits.shape}"
        assert alpha.shape == (N, K), f"Alpha shape should be {(N, K)}, got {alpha.shape}"
        assert torch.all(alpha >= 1.0), "Dirichlet concentration alphas must be >= 1.0 (softplus + 1)"
        
    # 3. Test decoding
    decoded = model.decode(outputs)
    
    for lvl in ["coarse", "middle", "fine"]:
        K = num_classes_per_level[lvl]
        pred = decoded[lvl]["prediction"]
        conf = decoded[lvl]["confidence"]
        prob = decoded[lvl]["prob"]
        
        assert pred.shape == (N,), f"Predictions shape should be {(N,)}, got {pred.shape}"
        assert conf.shape == (N,), f"Confidence shape should be {(N,)}, got {conf.shape}"
        assert prob.shape == (N, K), f"Prob shape should be {(N, K)}, got {prob.shape}"
        
        # Simplex checks
        assert torch.allclose(torch.sum(prob, dim=-1), torch.ones(N)), "Class probabilities must sum to 1"
        assert torch.all(conf >= 0.0) and torch.all(conf <= 1.0), "Calibrated confidence must be in range [0, 1]"
        
    logger.success("NSH forward and decoding verification passed successfully.")

