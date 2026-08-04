"""
Semantic evaluation metrics: mean semantic accuracy, part mIoU, and per-part accuracy for thin parts.
"""

import numpy as np
from loguru import logger
from typing import Dict, Any, Union


def part_m_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int = 15,
    ignore_label: int = 0
) -> Dict[str, Union[float, Dict[int, float]]]:
    """Calculate the Part mean Intersection over Union (part mIoU) matching Work 1 Eq. 20.
    
    Formula:
        mIoU = 1/K_valid * sum_{k in K_valid} |P_k ∩ G_k| / |P_k ∪ G_k|
        where P_k is the predicted voxels for class k, and G_k is the ground truth voxels.
        
    Args:
        pred: Predicted class label array.
        gt: Ground truth class label array.
        num_classes: Total number of classes.
        ignore_label: Label index to ignore in calculation (e.g. background/unsegmented 0).
        
    Returns:
        Dict[str, Union[float, Dict[int, float]]]: mIoU and per-class IoU dictionary.
    """
    logger.info("Computing Part mIoU")
    
    per_class_iou = {}
    valid_classes = 0
    total_iou = 0.0
    
    for k in range(num_classes):
        if k == ignore_label:
            continue
            
        pred_k = (pred == k)
        gt_k = (gt == k)
        
        # Only evaluate classes that appear in GT
        if not np.any(gt_k):
            continue
            
        intersection = np.logical_and(pred_k, gt_k).sum()
        union = np.logical_or(pred_k, gt_k).sum()
        
        iou = float(intersection) / float(union) if union > 0 else 0.0
        per_class_iou[k] = iou
        total_iou += iou
        valid_classes += 1
        
    mean_iou = total_iou / valid_classes if valid_classes > 0 else 0.0
    
    return {
        "part_mIoU": mean_iou,
        "per_class_iou": per_class_iou
    }


def mean_semantic_accuracy(
    pred: np.ndarray,
    gt: np.ndarray,
    ignore_label: int = 0
) -> Dict[str, float]:
    """Calculate overall voxel classification accuracy.
    
    Formula:
        Accuracy = sum_{v} 1[pred(v) == gt(v)] * 1[gt(v) != ignore] / sum_{v} 1[gt(v) != ignore]
        
    Args:
        pred: Predicted class label array.
        gt: Ground truth class label array.
        ignore_label: Label index to ignore.
        
    Returns:
        Dict[str, float]: dict with key 'mean_semantic_accuracy'
    """
    logger.info("Computing Mean Semantic Accuracy")
    valid_mask = (gt != ignore_label)
    if not np.any(valid_mask):
        return {"mean_semantic_accuracy": 1.0}
        
    correct = (pred[valid_mask] == gt[valid_mask]).sum()
    total = valid_mask.sum()
    acc = float(correct) / float(total)
    
    return {"mean_semantic_accuracy": acc}


def per_part_accuracy(
    pred: np.ndarray,
    gt: np.ndarray,
    part_class_idx: int
) -> Dict[str, float]:
    """Calculate semantic accuracy restricted to a specific part.
    
    Formula:
        Accuracy = count(pred == part_class_idx AND gt == part_class_idx) / count(gt == part_class_idx)
        
    Args:
        pred: Predicted class label array.
        gt: Ground truth class label array.
        part_class_idx: Class index of the targeted part.
        
    Returns:
        Dict[str, float]: dict with accuracy for the specified class.
    """
    gt_mask = (gt == part_class_idx)
    if not np.any(gt_mask):
        return {"accuracy": 0.0}
        
    correct = (pred[gt_mask] == part_class_idx).sum()
    total = gt_mask.sum()
    acc = float(correct) / float(total)
    
    return {"accuracy": acc}
