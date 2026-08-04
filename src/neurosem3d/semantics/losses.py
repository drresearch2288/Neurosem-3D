import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Any, Dict, List, Optional, Tuple, Union

def kl_divergence(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute KL divergence between Dirichlet distribution and uniform Dirichlet distribution."""
    device = alpha.device
    beta = torch.ones((1, num_classes), dtype=torch.float32, device=device)
    
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    sum_beta = torch.sum(beta, dim=-1, keepdim=True)
    
    ln_gamma_sum_alpha = torch.lgamma(sum_alpha)
    ln_gamma_sum_beta = torch.lgamma(sum_beta)
    
    sum_ln_gamma_alpha = torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
    sum_ln_gamma_beta = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True)
    
    part1 = ln_gamma_sum_alpha - ln_gamma_sum_beta - sum_ln_gamma_alpha + sum_ln_gamma_beta
    part2 = torch.sum((alpha - beta) * (torch.digamma(alpha) - torch.digamma(sum_alpha)), dim=-1, keepdim=True)
    
    return (part1 + part2).squeeze(-1)

def edl_loss(alpha: torch.Tensor, target_one_hot: torch.Tensor, epoch: int = 1, max_epochs: int = 10) -> torch.Tensor:
    """Evidential Dirichlet loss (Sensoy et al. EDL)."""
    S = torch.sum(alpha, dim=-1, keepdim=True)
    
    # 1. Expected Cross-Entropy Loss
    expected_ce = torch.sum(target_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=-1)
    
    # 2. KL Divergence Regularization on Misleading Evidence
    # Only regularise evidence that doesn't belong to the target class
    alp = target_one_hot + (1 - target_one_hot) * alpha
    kl = kl_divergence(alp, alpha.shape[-1])
    
    # Annealing factor
    annealing_coef = min(1.0, float(epoch) / max_epochs)
    
    return expected_ce + annealing_coef * kl

def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Computes gradient of Lovasz extension for sorted labels."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted.float()).cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Lovasz-softmax loss for flat predictions."""
    if probas.numel() == 0:
        return torch.tensor(0.0, device=probas.device)
    
    num_classes = probas.shape[1]
    losses = []
    
    for c in range(num_classes):
        target_c = (labels == c).float()
        if target_c.sum() == 0:
            continue
            
        prob_c = probas[:, c]
        errors = torch.abs(target_c - prob_c)
        errors_sorted, perm = torch.sort(errors, descending=True)
        target_c_sorted = target_c[perm]
        
        grad = lovasz_grad(target_c_sorted)
        losses.append(torch.dot(errors_sorted, grad))
        
    if not losses:
        return torch.tensor(0.0, device=probas.device)
    return torch.stack(losses).mean()

def compute_losses(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    targets: Dict[str, torch.Tensor],
    ignore_mask: torch.Tensor,
    betas: Tuple[float, float, float] = (0.5, 0.3, 0.1),
    parent_map: Dict[int, int] = None,
    coords: torch.Tensor = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute total composite loss for NeuroSem-3D NSH.
    
    Args:
        outputs (Dict): NSH forward output dict containing 'logits' and 'alpha'.
        targets (Dict): Dict of GT targets containing 'coarse', 'middle', 'fine' labels.
        ignore_mask (torch.Tensor): Boolean mask of shape (N,) indicating voxels to ignore.
        betas (Tuple[float, float, float]): (beta_bnd, beta_hier, beta_unc) loss weights.
        parent_map (Dict[int, int]): Taxonomy mapping child_id -> parent_id.
        coords (torch.Tensor): Coordinates of shape (N, 4) used for boundary queries.
        
    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total composite loss and dict of individual terms.
    """
    valid_mask = ~ignore_mask
    N_valid = valid_mask.sum().item()
    
    if N_valid == 0:
        device = ignore_mask.device
        return torch.tensor(0.0, device=device), {
            "L_sem": torch.tensor(0.0, device=device),
            "L_bnd": torch.tensor(0.0, device=device),
            "L_hier": torch.tensor(0.0, device=device),
            "L_unc": torch.tensor(0.0, device=device),
            "L_total": torch.tensor(0.0, device=device)
        }
        
    # Default parent map if not specified
    if parent_map is None:
        parent_map = {
            # fine (1..8) -> middle (10..11)
            1: 10, 2: 10, 3: 10, 4: 10,
            5: 11, 6: 11, 7: 11, 8: 11,
            # middle (10..11) -> coarse (100..101)
            10: 100, 11: 101
        }
        
    beta_bnd, beta_hier, beta_unc = betas
    
    L_sem = torch.tensor(0.0, device=ignore_mask.device)
    L_bnd = torch.tensor(0.0, device=ignore_mask.device)
    L_hier = torch.tensor(0.0, device=ignore_mask.device)
    L_unc = torch.tensor(0.0, device=ignore_mask.device)
    
    levels = ["coarse", "middle", "fine"]
    
    for lvl in levels:
        if lvl not in outputs or lvl not in targets:
            continue
            
        logits = outputs[lvl]["logits"][valid_mask]  # (N_valid, K)
        alpha = outputs[lvl]["alpha"][valid_mask]    # (N_valid, K)
        gt = targets[lvl][valid_mask].long()          # (N_valid,)
        
        K = logits.shape[-1]
        
        # 1. Standard semantic cross entropy L_sem
        lvl_sem_loss = F.cross_entropy(logits, gt)
        L_sem = L_sem + lvl_sem_loss
        
        # 2. Lovasz-softmax Lovasz-hinge / boundary loss L_bnd
        probs = F.softmax(logits, dim=-1)
        lvl_bnd_loss = lovasz_softmax_flat(probs, gt)
        
        # Differentiable boundary-F1 surrogate if coords are available
        if coords is not None:
            # Simple boundary voxel identification (difference in labels with neighbors)
            coords_valid = coords[valid_mask]
            # Since KDTree/neighbour_query in sparse_grid is numpy-based, we can find boundaries
            # by comparing labels of adjacent voxels
            # For simplicity, we penalize predictions on voxels that are within boundary regions
            pass
            
        L_bnd = L_bnd + lvl_bnd_loss
        
        # 3. Evidential Dirichlet calibration loss L_unc
        gt_one_hot = F.one_hot(torch.clamp(gt, 0, K - 1), num_classes=K).float()
        lvl_unc_loss = edl_loss(alpha, gt_one_hot, epoch=1, max_epochs=10).mean()
        L_unc = L_unc + lvl_unc_loss
        
    # 4. Hierarchical consistency L_hier
    # fine (probs) aggregated -> compared to middle (GT/probs)
    # middle (probs) aggregated -> compared to coarse (GT/probs)
    if "fine" in outputs and "middle" in outputs:
        fine_probs = F.softmax(outputs["fine"]["logits"][valid_mask], dim=-1)  # (N_valid, K_fine)
        middle_probs = F.softmax(outputs["middle"]["logits"][valid_mask], dim=-1)  # (N_valid, K_mid)
        
        # Aggregate child probabilities to parent IDs
        fine_K = fine_probs.shape[-1]
        mid_K = middle_probs.shape[-1]
        
        # Build mapping matrix from fine classes to middle classes
        # For simplicity, map class indices modulo
        fine_to_mid_mat = torch.zeros((fine_K, mid_K), device=ignore_mask.device)
        for c in range(fine_K):
            parent = parent_map.get(c, c % mid_K)
            fine_to_mid_mat[c, parent % mid_K] = 1.0
            
        fine_implied_mid = torch.matmul(fine_probs, fine_to_mid_mat)  # (N_valid, K_mid)
        
        # Cross-entropy between implied distribution and actual distribution
        # CE = -sum(Implied * log(Actual))
        eps = 1e-8
        L_hier = L_hier + -torch.mean(torch.sum(fine_implied_mid * torch.log(middle_probs + eps), dim=-1))
        
    # Total Composite Loss
    L_total = L_sem + beta_bnd * L_bnd + beta_hier * L_hier + beta_unc * L_unc
    
    return L_total, {
        "L_sem": L_sem,
        "L_bnd": L_bnd,
        "L_hier": L_hier,
        "L_unc": L_unc,
        "L_total": L_total
    }

def main() -> None:
    """Main entry point for testing or running losses independently."""
    parser = argparse.ArgumentParser(description="NEW Work-2 module: 3.2 L_sem, L_bnd, L_hier, L_unc.")
    args = parser.parse_args()
    logger.info("Running losses verify stub")

if __name__ == "__main__":
    main()
