"""
Efficiency metrics: latency, peak GPU memory allocated, and model size.
"""

import torch
from loguru import logger
from typing import Dict


def peak_gpu_mem_gb() -> Dict[str, float]:
    """Retrieve peak GPU memory allocated in Gigabytes.
    
    Formula:
        mem_gb = torch.cuda.max_memory_allocated() / (1024^3)
        
    Returns:
        Dict[str, float]: dict with key 'peak_gpu_mem_gb'
    """
    if torch.cuda.is_available():
        mem_bytes = torch.cuda.max_memory_allocated()
        mem_gb = float(mem_bytes) / (1024 ** 3)
    else:
        logger.warning("CUDA is not available. Peak GPU memory will be reported as 0.0.")
        mem_gb = 0.0
        
    return {"peak_gpu_mem_gb": mem_gb}


def model_size_mb(model: torch.nn.Module) -> Dict[str, float]:
    """Calculate the memory footprint of a PyTorch model in Megabytes.
    
    Formula:
        size_mb = sum_{param} (param.nelement() * param.element_size()) / 1024^2
        
    Args:
        model: PyTorch module.
        
    Returns:
        Dict[str, float]: dict with key 'model_size_mb'
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return {"model_size_mb": float(size_mb)}
