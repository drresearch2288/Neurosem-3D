import argparse
import numpy as np
import torch
from loguru import logger
from typing import Any, Dict, List, Optional, Set, Tuple, Union

def tree_consistent_decode(
    logits_per_level: Dict[str, torch.Tensor],
    taxonomy: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """Enforce tree-consistent assignments top-down using logits per taxonomy level.
    
    Args:
        logits_per_level (Dict[str, torch.Tensor]): containing:
            - 'coarse': tensor of shape (N, K_coarse).
            - 'middle': tensor of shape (N, K_mid).
            - 'fine': tensor of shape (N, K_fine).
        taxonomy (Dict[str, Any]): Taxonomy mappings. Expected keys:
            - 'fine_to_mid': Dict[int, int]
            - 'mid_to_coarse': Dict[int, int]
            
    Returns:
        Dict[str, torch.Tensor]: Top-down tree-consistent predictions for each level.
    """
    logger.info("Performing tree-consistent top-down decoding...")
    
    coarse_logits = logits_per_level["coarse"]
    mid_logits = logits_per_level["middle"]
    fine_logits = logits_per_level["fine"]
    
    N = coarse_logits.shape[0]
    device = coarse_logits.device
    
    fine_to_mid = taxonomy.get("fine_to_mid", {})
    mid_to_coarse = taxonomy.get("mid_to_coarse", {})
    
    # 1. Decode Coarse Level
    coarse_preds = torch.argmax(coarse_logits, dim=-1)  # (N,)
    
    # 2. Decode Middle Level (constrained by Coarse)
    mid_K = mid_logits.shape[-1]
    # Build mask of valid middle classes for the chosen coarse class
    mid_mask = torch.full((N, mid_K), -1e9, device=device)
    for m in range(mid_K):
        # Allow middle class if it maps to the predicted coarse class
        m_coarse = mid_to_coarse.get(m, m)
        valid = (coarse_preds == m_coarse)
        mid_mask[valid, m] = 0.0
        
    masked_mid_logits = mid_logits + mid_mask
    mid_preds = torch.argmax(masked_mid_logits, dim=-1)  # (N,)
    
    # 3. Decode Fine Level (constrained by Middle)
    fine_K = fine_logits.shape[-1]
    fine_mask = torch.full((N, fine_K), -1e9, device=device)
    for f in range(fine_K):
        f_mid = fine_to_mid.get(f, f)
        valid = (mid_preds == f_mid)
        fine_mask[valid, f] = 0.0
        
    masked_fine_logits = fine_logits + fine_mask
    fine_preds = torch.argmax(masked_fine_logits, dim=-1)  # (N,)
    
    return {
        "coarse": coarse_preds,
        "middle": mid_preds,
        "fine": fine_preds
    }

def get_descendant_parts(branch_id: int, taxonomy: Dict[str, Any], level: str) -> Set[int]:
    """Helper to find all fine/middle sub-tree descendants of a branch_id at a specific level."""
    from collections import deque
    
    fine_to_mid = taxonomy.get("fine_to_mid", {})
    mid_to_coarse = taxonomy.get("mid_to_coarse", {})
    
    descendants = {branch_id}
    
    if level == "coarse":
        # branch_id is coarse. find middle classes mapping to it
        mid_descendants = {m for m, c in mid_to_coarse.items() if c == branch_id}
        descendants.update(mid_descendants)
        # find fine classes mapping to those middle classes
        fine_descendants = {f for f, m in fine_to_mid.items() if m in mid_descendants}
        descendants.update(fine_descendants)
    elif level == "middle":
        # branch_id is middle. find fine classes mapping to it
        fine_descendants = {f for f, m in fine_to_mid.items() if m == branch_id}
        descendants.update(fine_descendants)
        
    return descendants

def remove_part(
    coords: torch.Tensor,
    labels_dict: Dict[str, torch.Tensor],
    branch_id: int,
    taxonomy: Dict[str, Any],
    level: str = "middle"
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Remove a whole sub-tree branch of the geometry and labels.
    
    Args:
        coords (torch.Tensor): Coordinates tensor shape (N, 3) or (N, 4).
        labels_dict (Dict[str, torch.Tensor]): Dictionary of per-level label tensors.
        branch_id (int): Part ID of the branch to remove.
        taxonomy (Dict[str, Any]): Taxonomy hierarchy.
        level (str): Taxonomy level of the branch_id ('coarse', 'middle', 'fine').
        
    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Edited coords and labels.
    """
    logger.info(f"Removing sub-tree branch: {branch_id} at level {level}")
    
    # 1. Determine which label indices belong to this branch
    descendants = get_descendant_parts(branch_id, taxonomy, level)
    
    # 2. Find voxels that belong to this branch
    # A voxel belongs to the branch if its label at the specified level is branch_id
    # or if its label at lower levels is a descendant
    target_labels = labels_dict[level]
    mask_to_remove = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
    for desc in descendants:
        mask_to_remove |= (labels_dict["fine"] == desc)
        mask_to_remove |= (labels_dict["middle"] == desc)
        mask_to_remove |= (labels_dict["coarse"] == desc)
        
    keep_mask = ~mask_to_remove
    
    # Filter coords and labels
    new_coords = coords[keep_mask]
    new_labels = {lvl: val[keep_mask] for lvl, val in labels_dict.items()}
    
    return new_coords, new_labels

def scale_part(
    coords: torch.Tensor,
    labels_dict: Dict[str, torch.Tensor],
    branch_id: int,
    scale_factor: float,
    taxonomy: Dict[str, Any],
    level: str = "middle"
) -> torch.Tensor:
    """Scale the coordinates of a whole sub-tree branch relative to its center.
    
    Args:
        coords (torch.Tensor): Coordinates shape (N, 3) or (N, 4).
        labels_dict (Dict[str, torch.Tensor]): Per-level labels.
        branch_id (int): Part ID to scale.
        scale_factor (float): Scale multiplier.
        taxonomy (Dict[str, Any]): Taxonomy.
        level (str): Level of the part.
        
    Returns:
        torch.Tensor: Scaled coordinates.
    """
    logger.info(f"Scaling branch {branch_id} by {scale_factor}")
    
    descendants = get_descendant_parts(branch_id, taxonomy, level)
    
    # Identify voxels in branch
    part_mask = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
    for desc in descendants:
        part_mask |= (labels_dict["fine"] == desc)
        part_mask |= (labels_dict["middle"] == desc)
        part_mask |= (labels_dict["coarse"] == desc)
        
    if not torch.any(part_mask):
        return coords.clone()
        
    new_coords = coords.clone().float()
    
    # We scale coordinates around their center. Batch dimension column (index 0) is preserved if coords is (N, 4)
    start_col = 1 if coords.shape[1] == 4 else 0
    xyz = new_coords[:, start_col:]
    
    part_xyz = xyz[part_mask]
    center = torch.mean(part_xyz, dim=0, keepdim=True)
    
    xyz[part_mask] = center + (part_xyz - center) * scale_factor
    
    # Cast back to original coordinate types (indices are int)
    new_coords[:, start_col:] = xyz.round().int()
    return new_coords.to(coords.dtype)

def uncertainty_gated_resolve(
    labels: torch.Tensor,
    parent_labels: torch.Tensor,
    confidence: torch.Tensor,
    threshold: float
) -> torch.Tensor:
    """Assign low-confidence boundary voxels to the enclosing parent part.
    
    Args:
        labels (torch.Tensor): predicted fine/middle labels (N,).
        parent_labels (torch.Tensor): predicted enclosing middle/coarse labels (N,).
        confidence (torch.Tensor): NSH calibrated confidence u(v) (N,).
        threshold (float): confidence gate threshold.
        
    Returns:
        torch.Tensor: resolved labels.
    """
    # Low confidence boundary voxels: confidence < threshold
    low_conf = confidence < threshold
    resolved = labels.clone()
    resolved[low_conf] = parent_labels[low_conf]
    return resolved

def incremental_relabel(
    labels_dict: Dict[str, torch.Tensor],
    taxonomy: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """Recompute middle and coarse labels from fine labels using taxonomy parent map.
    
    Args:
        labels_dict (Dict[str, torch.Tensor]): Dictionary of per-level label tensors.
        taxonomy (Dict[str, Any]): Taxonomy.
        
    Returns:
        Dict[str, torch.Tensor]: Re-consistent labels dictionary.
    """
    fine = labels_dict["fine"]
    fine_to_mid = taxonomy.get("fine_to_mid", {})
    mid_to_coarse = taxonomy.get("mid_to_coarse", {})
    
    device = fine.device
    N = fine.shape[0]
    
    # Recompute middle
    middle = torch.zeros(N, dtype=fine.dtype, device=device)
    for f, m in fine_to_mid.items():
        middle[fine == f] = m
        
    # Recompute coarse
    coarse = torch.zeros(N, dtype=fine.dtype, device=device)
    for m, c in mid_to_coarse.items():
        coarse[middle == m] = c
        
    return {
        "coarse": coarse,
        "middle": middle,
        "fine": fine
    }

def main() -> None:
    """Main entry point for testing or running hierarchy independently."""
    parser = argparse.ArgumentParser(description="NEW Work-2 module: 3.3 Hierarchical decode + editing ops.")
    args = parser.parse_args()
    logger.info("Running hierarchy verify stub")

if __name__ == "__main__":
    main()
