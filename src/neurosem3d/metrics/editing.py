"""
Editing metrics: Edit Leakage, Sub-Assembly Success Rate, and Relabel Latency.
"""

import numpy as np
from loguru import logger
from typing import Dict, List, Union


def edit_leakage(
    labels_before: np.ndarray,
    labels_after: np.ndarray,
    target_part_mask: np.ndarray
) -> Dict[str, float]:
    """Calculate the edit leakage percentage.
    
    Formula:
        leakage = sum_{v in outside_target} 1[labels_before(v) != labels_after(v)] / count(outside_target) * 100
        where outside_target is the set of voxels where target_part_mask is False.
        
    Args:
        labels_before: Voxel labels before applying the edit shape (N,).
        labels_after: Voxel labels after applying the edit shape (N,).
        target_part_mask: Boolean mask flagging voxels belonging to the target part shape (N,).
        
    Returns:
        Dict[str, float]: dict with key 'edit_leakage'
    """
    logger.info("Computing Edit Leakage")
    outside_mask = ~target_part_mask
    
    total_outside = outside_mask.sum()
    if total_outside == 0:
        return {"edit_leakage": 0.0}
        
    modified_outside = np.logical_and(outside_mask, labels_before != labels_after).sum()
    leakage = float(modified_outside) / float(total_outside) * 100.0
    
    return {"edit_leakage": leakage}


def sub_assembly_success_rate(
    leakages: Union[List[float], np.ndarray],
    threshold_tau: float = 1.0
) -> Dict[str, float]:
    """Calculate the sub-assembly success rate based on a leakage threshold tau.
    
    Formula:
        success_rate = count(leakages < threshold_tau) / count(leakages)
        
    Args:
        leakages: List or array of leakage percentages.
        threshold_tau: Tolerance threshold for leakage (default 1.0%).
        
    Returns:
        Dict[str, float]: dict with key 'sub_assembly_success_rate'
    """
    logger.info(f"Computing Sub-Assembly Success Rate with tau={threshold_tau}")
    leakages_arr = np.array(leakages)
    
    if len(leakages_arr) == 0:
        return {"sub_assembly_success_rate": 1.0}
        
    success = (leakages_arr < threshold_tau).sum()
    rate = float(success) / float(len(leakages_arr))
    
    return {"sub_assembly_success_rate": rate}
