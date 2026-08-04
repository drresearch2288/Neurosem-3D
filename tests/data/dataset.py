"""
Tests for NeuroSemDataset and sparse collation.
"""

import pytest
import torch
from loguru import logger
from neurosem3d.data.dataset import NeuroSemDataset, sparse_collate_fn

def test_dataset_item_and_collation() -> None:
    """Test loading items from NeuroSemDataset and batch collation."""
    logger.info("Starting dataset shape and collation verification...")
    
    # Instantiate dataset for 'test' split (uses dummy files created in previous steps)
    dataset = NeuroSemDataset(
        split="test",
        sparse_dir="neurosem3d/data/processed/sparse",
        confidence_dir="neurosem3d/data/processed/confidence",
        splits_dir="neurosem3d/data/splits",
        augment=True
    )
    
    # Assert size
    assert len(dataset) > 0, "Dataset should contain dummy objects"
    
    # 1. Load one item
    item = dataset[0]
    coords, feats, coarse, middle, fine, ignore_mask, obj_id = item
    
    N = coords.shape[0]
    logger.info(f"Loaded object '{obj_id}' with N={N} sparse voxels.")
    
    # 2. Shape consistency checks
    assert coords.shape == (N, 3), f"Expected coordinates shape {(N, 3)}, got {coords.shape}"
    assert feats.shape == (N, 272), f"Expected features shape {(N, 272)}, got {feats.shape}"
    assert coarse.shape == (N,), f"Expected coarse labels shape {(N,)}, got {coarse.shape}"
    assert middle.shape == (N,), f"Expected middle labels shape {(N,)}, got {middle.shape}"
    assert fine.shape == (N,), f"Expected fine labels shape {(N,)}, got {fine.shape}"
    assert ignore_mask.shape == (N,), f"Expected ignore mask shape {(N,)}, got {ignore_mask.shape}"
    
    # 3. Test Collation function
    batch = [dataset[0], dataset[0]]  # duplicate for testing collation of size 2
    collated = sparse_collate_fn(batch)
    
    # Collation shape checks
    assert collated["coords"].shape == (2 * N, 4), f"Expected collated coords shape {(2 * N, 4)}, got {collated['coords'].shape}"
    assert collated["feats"].shape == (2 * N, 272), f"Expected collated feats shape {(2 * N, 272)}, got {collated['feats'].shape}"
    assert collated["coarse"].shape == (2 * N,), f"Expected collated coarse shape {(2 * N,)}, got {collated['coarse'].shape}"
    assert len(collated["obj_ids"]) == 2, "Expected 2 object IDs in collated batch"
    
    # Batch indices (col 0 of coordinates) should be prepended correctly
    assert torch.all(collated["coords"][:N, 0] == 0)
    assert torch.all(collated["coords"][N:, 0] == 1)
    
    logger.success("Dataset and collation verification tests PASSED.")

