"""
Tests for semantic metrics (Part mIoU, Accuracy, per-class metrics).
"""

import pytest
import numpy as np
from loguru import logger
from neurosem3d.metrics.semantic import part_m_iou, mean_semantic_accuracy, per_part_accuracy


def test_part_m_iou_hand_check() -> None:
    """Test part mIoU on a simple hand-checkable array.
    
    Formula check:
        pred = [1, 1, 2, 2, 0]
        gt   = [1, 2, 2, 1, 0]
        
        Class 1:
            P_1 = [T, T, F, F, F]
            G_1 = [T, F, F, T, F]
            Intersection = 1, Union = 3 -> IoU = 1/3
        Class 2:
            P_2 = [F, F, T, T, F]
            G_2 = [F, T, T, F, F]
            Intersection = 1, Union = 3 -> IoU = 1/3
            
        mIoU = (1/3 + 1/3) / 2 = 1/3 ~ 0.333333
    """
    logger.info("Testing part mIoU hand-checked case")
    pred = np.array([1, 1, 2, 2, 0])
    gt = np.array([1, 2, 2, 1, 0])
    
    res = part_m_iou(pred, gt, num_classes=3, ignore_label=0)
    assert pytest.approx(res["part_mIoU"]) == 1.0 / 3.0
    assert pytest.approx(res["per_class_iou"][1]) == 1.0 / 3.0
    assert pytest.approx(res["per_class_iou"][2]) == 1.0 / 3.0


def test_mean_semantic_accuracy_hand_check() -> None:
    """Test mean semantic accuracy on a simple hand-checkable array.
    
    Formula check:
        pred = [1, 1, 2, 2, 0]
        gt   = [1, 2, 2, 1, 0]
        
        Valid elements (gt != 0): indices 0, 1, 2, 3 (total 4)
        Correct predictions: indices 0 (1==1) and 2 (2==2) (total 2)
        Accuracy = 2 / 4 = 0.5
    """
    logger.info("Testing mean semantic accuracy hand-checked case")
    pred = np.array([1, 1, 2, 2, 0])
    gt = np.array([1, 2, 2, 1, 0])
    
    res = mean_semantic_accuracy(pred, gt, ignore_label=0)
    assert res["mean_semantic_accuracy"] == 0.5


def test_per_part_accuracy_hand_check() -> None:
    """Test per-part accuracy on a simple hand-checkable array."""
    pred = np.array([1, 1, 2, 2, 0])
    gt = np.array([1, 2, 2, 1, 0])
    
    # Class 1: gt == 1 at indices 0 and 3. pred at these indices is 1 and 2.
    # Accuracy for class 1 = 1 / 2 = 0.5
    res1 = per_part_accuracy(pred, gt, part_class_idx=1)
    assert res1["accuracy"] == 0.5
    
    # Class 2: gt == 2 at indices 1 and 2. pred at these indices is 1 and 2.
    # Accuracy for class 2 = 1 / 2 = 0.5
    res2 = per_part_accuracy(pred, gt, part_class_idx=2)
    assert res2["accuracy"] == 0.5
