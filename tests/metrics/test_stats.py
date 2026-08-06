"""
Tests for statistical utility metrics (paired t-test, bootstrap CI, Bonferroni).
"""

import pytest
import numpy as np
from loguru import logger
from neurosem3d.metrics.stats import paired_t_test, bootstrap_ci, bonferroni


def test_paired_t_test() -> None:
    """Test paired t-test on a small matched dataset."""
    logger.info("Testing paired t-test")
    a = [10.0, 12.0, 11.0]
    b = [9.0, 10.0, 10.0]
    
    res = paired_t_test(a, b)
    assert "t_statistic" in res
    assert "p_value" in res
    assert res["t_statistic"] > 0  # since a is systematically higher than b


def test_bootstrap_ci_hand_check() -> None:
    """Test bootstrap confidence interval for mean of simple array."""
    logger.info("Testing bootstrap CI bounds")
    values = [1.0, 2.0, 3.0, 4.0]
    
    res = bootstrap_ci(values, n_resamples=100)
    assert res["mean"] == 2.5
    assert res["ci_lower"] <= 2.5 <= res["ci_upper"]


def test_bonferroni_hand_check() -> None:
    """Test Bonferroni correction adjustment.
    
    Formula check:
        p_values = [0.01, 0.05, 0.2] (length m = 3)
        adjusted = [0.03, 0.15, 0.6]
    """
    logger.info("Testing Bonferroni correction")
    p_values = [0.01, 0.05, 0.2]
    
    res = bonferroni(p_values)
    expected = np.array([0.03, 0.15, 0.6])
    assert np.allclose(res["adjusted_p_values"], expected)
