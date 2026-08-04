"""
Boundary metrics: Boundary IoU and Boundary F1 evaluated in a thin shell around ground truth part boundaries.
"""

import numpy as np
from loguru import logger
from scipy.spatial import cKDTree
from typing import Dict, Any, Union


def compute_boundary_shell(
    coords: np.ndarray,
    gt: np.ndarray,
    radius: float = 2.0
) -> np.ndarray:
    """Identify the mask of occupied voxels that lie within a thin shell around GT part boundaries.
    
    Args:
        coords: Voxel coordinates shape (N, 3).
        gt: Ground truth labels shape (N,).
        radius: Shell thickness in voxel grid units.
        
    Returns:
        np.ndarray: Boolean mask of shape (N,) for voxels in the boundary shell.
    """
    N = coords.shape[0]
    if N == 0:
        return np.zeros(0, dtype=bool)
        
    # 1. Identify direct boundary voxels (26-connectivity)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=1.75)
    
    gt_boundary = np.zeros(N, dtype=bool)
    for i, j in pairs:
        if gt[i] > 0 and gt[j] > 0 and gt[i] != gt[j]:
            gt_boundary[i] = True
            gt_boundary[j] = True
            
    # 2. Expand boundary to a shell using the configurable radius
    if np.any(gt_boundary):
        boundary_coords = coords[gt_boundary]
        boundary_tree = cKDTree(boundary_coords)
        distances, _ = boundary_tree.query(coords)
        shell_mask = (distances <= radius) & (gt > 0)
        return shell_mask
    else:
        return np.zeros(N, dtype=bool)


def boundary_metrics(
    coords: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    radius: float = 2.0,
    num_classes: int = 15,
    ignore_label: int = 0
) -> Dict[str, float]:
    """Calculate the Boundary IoU and Boundary F1-score inside a thin boundary shell.
    
    Formulas:
        Boundary IoU = part_mIoU(pred[shell], gt[shell])
        Boundary F1 = macro_F1(pred[shell], gt[shell])
        
    Args:
        coords: Voxel coordinates shape (N, 3).
        pred: Predicted class label array shape (N,).
        gt: Ground truth class label array shape (N,).
        radius: Shell thickness.
        num_classes: Number of semantic classes.
        ignore_label: Label to ignore.
        
    Returns:
        Dict[str, float]: dict with keys 'boundary_iou', 'boundary_f1'
    """
    logger.info(f"Computing boundary metrics with radius={radius}")
    
    shell_mask = compute_boundary_shell(coords, gt, radius)
    if not np.any(shell_mask):
        return {"boundary_iou": 1.0, "boundary_f1": 1.0}
        
    pred_shell = pred[shell_mask]
    gt_shell = gt[shell_mask]
    
    # Calculate IoU and F1 per class in the shell
    valid_classes = 0
    total_iou = 0.0
    total_f1 = 0.0
    
    for k in range(num_classes):
        if k == ignore_label:
            continue
            
        gt_k = (gt_shell == k)
        if not np.any(gt_k):
            continue
            
        pred_k = (pred_shell == k)
        
        tp = np.logical_and(pred_k, gt_k).sum()
        fp = np.logical_and(pred_k, ~gt_k).sum()
        fn = np.logical_and(~pred_k, gt_k).sum()
        
        # IoU
        union = tp + fp + fn
        iou = float(tp) / float(union) if union > 0 else 0.0
        total_iou += iou
        
        # F1
        precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        total_f1 += f1
        
        valid_classes += 1
        
    m_iou = total_iou / valid_classes if valid_classes > 0 else 0.0
    m_f1 = total_f1 / valid_classes if valid_classes > 0 else 0.0
    
    return {
        "boundary_iou": m_iou,
        "boundary_f1": m_f1
    }
