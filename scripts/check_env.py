"""
Script to check critical environment package imports and verify CUDA availability.
"""

import sys
from loguru import logger

def main() -> None:
    logger.info("Starting environment validation check...")
    
    packages = [
        ("torch", lambda: __import__("torch").__version__),
        ("torchvision", lambda: __import__("torchvision").__version__),
        ("diffusers", lambda: __import__("diffusers").__version__),
        ("transformers", lambda: __import__("transformers").__version__),
        ("accelerate", lambda: __import__("accelerate").__version__),
        ("safetensors", lambda: __import__("safetensors").__version__),
        ("trimesh", lambda: __import__("trimesh").__version__),
        ("open3d", lambda: __import__("open3d").__version__),
        ("skimage", lambda: __import__("skimage").__version__),
        ("pymeshlab", lambda: __import__("pymeshlab").__version__),
        ("numpy", lambda: __import__("numpy").__version__),
        ("scipy", lambda: __import__("scipy").__version__),
        ("sklearn", lambda: __import__("sklearn").__version__),
        ("hydra", lambda: __import__("hydra").__version__),
        ("omegaconf", lambda: __import__("omegaconf").__version__),
        ("matplotlib", lambda: __import__("matplotlib").__version__),
        ("seaborn", lambda: __import__("seaborn").__version__),
        ("pandas", lambda: __import__("pandas").__version__),
    ]

    success = True
    for name, get_version in packages:
        try:
            ver = get_version()
            logger.info(f"[OK] {name} is installed. Version: {ver}")
        except ImportError as e:
            logger.error(f"[FAIL] {name} could not be imported. Error: {e}")
            success = False

    # Check spconv and MinkowskiEngine
    for sparse_pkg in ["spconv", "MinkowskiEngine"]:
        try:
            mod = __import__(sparse_pkg)
            logger.info(f"[OK] Sparse library '{sparse_pkg}' is available. Version: {getattr(mod, '__version__', 'N/A')}")
        except ImportError:
            logger.warning(f"[INFO] Sparse library '{sparse_pkg}' is NOT available in this environment.")

    # Verify CUDA availability
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        logger.info(f"CUDA Available: {cuda_avail}")
        if cuda_avail:
            logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
        else:
            logger.warning("CUDA is NOT available in PyTorch. CPU-only mode will be used (configured for Intel Iris Xe Graphics).")
    except ImportError:
        logger.error("PyTorch not installed, cannot verify CUDA.")
        success = False

    if success:
        logger.success("Environment check PASSED!")
        sys.exit(0)
    else:
        logger.error("Environment check FAILED (missing critical requirements or CUDA).")
        sys.exit(1)

if __name__ == "__main__":
    main()
