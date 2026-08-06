"""
Tests for calibration metrics (ECE, NLL, Error Detection AUROC).
"""

import pytest
import numpy as np
from loguru import logger
from neurosem3d.metrics.calibration import expected_calibration_error, negative_log_likelihood, error_detection_auroc


def test_ece_hand_check() -> None:
    """Test Expected Calibration Error on a 4-bin toy example.
    
    Formula check:
        confidences = [0.2, 0.45, 0.7, 0.9]
        predictions = [1, 2, 2, 1]
        gt          = [1, 1, 2, 2]
        num_bins    = 4
        
        Bin 1 (0.0 <= conf <= 0.25): [0.2]
            size = 1, acc = 1.0 (1==1), conf = 0.2
            error_1 = 1/4 * |1.0 - 0.2| = 0.20
        Bin 2 (0.25 < conf <= 0.50): [0.45]
            size = 1, acc = 0.0 (2!=1), conf = 0.45
            error_2 = 1/4 * |0.0 - 0.45| = 0.1125
        Bin 3 (0.50 < conf <= 0.75): [0.70]
            size = 1, acc = 1.0 (2==2), conf = 0.7
            error_3 = 1/4 * |1.0 - 0.7| = 0.075
        Bin 4 (0.75 < conf <= 1.00): [0.90]
            size = 1, acc = 0.0 (1!=2), conf = 0.9
            error_4 = 1/4 * |0.0 - 0.9| = 0.225
            
        ECE = 0.20 + 0.1125 + 0.075 + 0.225 = 0.6125
    """
    logger.info("Testing Expected Calibration Error hand-checked case")
    confidences = np.array([0.2, 0.45, 0.7, 0.9])
    predictions = np.array([1, 2, 2, 1])
    gt = np.array([1, 1, 2, 2])
    
    res = expected_calibration_error(confidences, predictions, gt, num_bins=4, ignore_label=0)
    assert pytest.approx(res["ece"]) == 0.6125


def test_nll_hand_check() -> None:
    """Test Negative Log Likelihood on a 2-sample case."""
    logger.info("Testing Negative Log Likelihood hand-checked case")
    probs = np.array([
        [0.8, 0.2],
        [0.1, 0.9]
    ])
    gt = np.array([0, 1])
    
    expected_nll = -0.5 * (np.log(0.8) + np.log(0.9))
    res = negative_log_likelihood(probs, gt, ignore_label=-1)
    assert pytest.approx(res["nll"]) == expected_nll


def test_auroc_hand_check() -> None:
    """Test Error Detection AUROC on a 3-sample case.
    
    Errors occur where predictions != gt:
        pred = [0, 0, 1]
        gt   = [0, 1, 1]
        targets = [0, 1, 0] (error at index 1)
        
    Uncertainty score:
        u = [0.1, 0.9, 0.2]
        Since the score for the true error (0.9) is larger than correct predictions (0.1, 0.2),
        the AUROC should be perfectly 1.0.
    """
    logger.info("Testing Error Detection AUROC hand-checked case")
    predictions = np.array([0, 0, 1])
    gt = np.array([0, 1, 1])
    uncertainty = np.array([0.1, 0.9, 0.2])
    
    res = error_detection_auroc(uncertainty, predictions, gt, ignore_label=-1)
    assert res["error_detection_auroc"] == 1.0
