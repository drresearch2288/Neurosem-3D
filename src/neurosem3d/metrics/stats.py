"""
Statistical utility metrics: Paired t-tests, bootstrap confidence intervals, and Bonferroni correction.
"""

import numpy as np
from scipy.stats import ttest_rel
from loguru import logger
from typing import Dict, List, Union


def paired_t_test(a: Union[List[float], np.ndarray], b: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """Perform a paired student's t-test comparing two distributions (e.g. baseline vs proposed).
    
    Formula:
        t = mean(d) / (std(d) / sqrt(n))
        where d = a - b, and n is the sample size.
        
    Args:
        a: Array of values from distribution A.
        b: Array of values from distribution B.
        
    Returns:
        Dict[str, float]: dict with keys 't_statistic', 'p_value'
    """
    logger.info("Performing paired t-test")
    a_arr = np.array(a)
    b_arr = np.array(b)
    
    if len(a_arr) != len(b_arr) or len(a_arr) == 0:
        raise ValueError("Inputs must have identical non-zero lengths for a paired t-test.")
        
    stat, pval = ttest_rel(a_arr, b_arr)
    return {
        "t_statistic": float(stat),
        "p_value": float(pval)
    }


def bootstrap_ci(
    values: Union[List[float], np.ndarray],
    n_resamples: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """Calculate the bootstrap confidence interval for the mean.
    
    Formula:
        Resample values with replacement, calculate means, and take percentiles.
        
    Args:
        values: Data points to calculate the mean confidence interval for.
        n_resamples: Number of bootstrap draws (default 1000).
        confidence_level: Level of significance (default 0.95).
        
    Returns:
        Dict[str, float]: dict with keys 'mean', 'ci_lower', 'ci_upper'
    """
    logger.info(f"Computing bootstrap confidence interval with {n_resamples} resamples")
    arr = np.array(values)
    
    if len(arr) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        
    bootstrap_means = np.zeros(n_resamples)
    for i in range(n_resamples):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        bootstrap_means[i] = np.mean(sample)
        
    alpha = 1.0 - confidence_level
    lower_pct = alpha / 2 * 100
    upper_pct = (1.0 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_pct)
    ci_upper = np.percentile(bootstrap_means, upper_pct)
    mean_val = np.mean(arr)
    
    return {
        "mean": float(mean_val),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper)
    }


def bonferroni(p_values: Union[List[float], np.ndarray]) -> Dict[str, np.ndarray]:
    """Apply the Bonferroni correction for multiple hypothesis testing.
    
    Formula:
        p_adj = min(p_val * m, 1.0)
        where m is the number of tested hypotheses.
        
    Args:
        p_values: List or array of raw p-values.
        
    Returns:
        Dict[str, np.ndarray]: dict with key 'adjusted_p_values'
    """
    logger.info("Applying Bonferroni correction")
    pvals_arr = np.array(p_values)
    m = len(pvals_arr)
    
    adjusted = np.clip(pvals_arr * m, 0.0, 1.0)
    return {"adjusted_p_values": adjusted}
