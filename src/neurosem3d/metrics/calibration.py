"""
Calibration metrics: Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), and Error Detection AUROC.
"""

import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score
from typing import Dict, Any


def expected_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    gt: np.ndarray,
    num_bins: int = 15,
    ignore_label: int = 0
) -> Dict[str, float]:
    """Calculate the Expected Calibration Error (ECE) for classification.
    
    Formula:
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
        where B_m is the set of samples whose confidence falls in the m-th bin.
        
    Args:
        confidences: Predicted class probability/confidence score shape (N,).
        predictions: Predicted class label array shape (N,).
        gt: Ground truth class label array shape (N,).
        num_bins: Number of confidence bins (default 15).
        ignore_label: Label to ignore.
        
    Returns:
        Dict[str, float]: dict with key 'ece'
    """
    logger.info(f"Computing Expected Calibration Error with {num_bins} bins")
    valid_mask = (gt != ignore_label)
    if not np.any(valid_mask):
        return {"ece": 0.0}
        
    conf = confidences[valid_mask]
    pred = predictions[valid_mask]
    labels = gt[valid_mask]
    
    N = len(labels)
    ece = 0.0
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    for m in range(num_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]
        
        # Identify samples in bin
        in_bin = (conf > bin_lower) & (conf <= bin_upper)
        if m == 0:
            in_bin |= (conf == bin_lower)
            
        bin_size = in_bin.sum()
        if bin_size > 0:
            bin_accuracy = np.mean(pred[in_bin] == labels[in_bin])
            bin_confidence = np.mean(conf[in_bin])
            ece += (bin_size / N) * np.abs(bin_accuracy - bin_confidence)
            
    return {"ece": float(ece)}


def negative_log_likelihood(
    probs: np.ndarray,
    gt: np.ndarray,
    ignore_label: int = 0,
    eps: float = 1e-15
) -> Dict[str, float]:
    """Calculate the Negative Log-Likelihood (NLL).
    
    Formula:
        NLL = -1/N * sum_{i=1}^N log(probs[i, gt[i]] + eps)
        
    Args:
        probs: Soft probability distributions shape (N, K).
        gt: Ground truth class labels shape (N,).
        ignore_label: Label to ignore.
        eps: Small epsilon to prevent log(0).
        
    Returns:
        Dict[str, float]: dict with key 'nll'
    """
    logger.info("Computing Negative Log Likelihood")
    valid_mask = (gt != ignore_label)
    if not np.any(valid_mask):
        return {"nll": 0.0}
        
    probs_valid = probs[valid_mask]
    gt_valid = gt[valid_mask]
    
    # Clip probs to prevent log(0)
    probs_valid = np.clip(probs_valid, eps, 1.0)
    
    # Extract prob of correct class
    row_indices = np.arange(len(gt_valid))
    correct_probs = probs_valid[row_indices, gt_valid.astype(int)]
    nll = -np.mean(np.log(correct_probs))
    
    return {"nll": float(nll)}


def error_detection_auroc(
    uncertainty: np.ndarray,
    predictions: np.ndarray,
    gt: np.ndarray,
    ignore_label: int = 0
) -> Dict[str, float]:
    """Calculate Area Under the ROC Curve for predicting prediction errors using uncertainty.
    
    Formula:
        Computes ROC AUC on binary targets error_flag = 1[predictions != gt]
        using uncertainty score u(v) as predictions.
        
    Args:
        uncertainty: Uncertainty scores u(v) shape (N,).
        predictions: Predicted class labels shape (N,).
        gt: Ground truth labels shape (N,).
        ignore_label: Label to ignore.
        
    Returns:
        Dict[str, float]: dict with key 'error_detection_auroc'
    """
    logger.info("Computing Error Detection AUROC")
    valid_mask = (gt != ignore_label)
    if not np.any(valid_mask):
        return {"error_detection_auroc": 0.5}
        
    y_true = (predictions[valid_mask] != gt[valid_mask]).astype(int)
    y_score = uncertainty[valid_mask]
    
    # If all predictions are correct or all are incorrect, AUROC is technically undefined/trivial.
    if len(np.unique(y_true)) < 2:
        logger.warning("Error detection target has only one unique class. Returning default AUROC of 0.5.")
        return {"error_detection_auroc": 0.5}
        
    auroc = roc_auc_score(y_true, y_score)
    return {"error_detection_auroc": float(auroc)}
