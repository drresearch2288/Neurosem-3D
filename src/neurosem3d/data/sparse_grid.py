import argparse
import numpy as np
import torch
from loguru import logger
from typing import Any, Dict, List, Optional, Tuple, Union

# Dynamic import checks for spconv and MinkowskiEngine
HAS_MINKOWSKI = False
try:
    import MinkowskiEngine as ME
    HAS_MINKOWSKI = True
except ImportError:
    pass

HAS_SPCONV = False
try:
    import spconv.pytorch as spconv
    HAS_SPCONV = True
except ImportError:
    pass

class SparseVoxels:
    """Sparse voxels representation storing near-surface voxel indices and features."""
    
    def __init__(self, coords: np.ndarray, features: Optional[np.ndarray] = None) -> None:
        """Initialize SparseVoxels.
        
        Args:
            coords (np.ndarray): Voxel coordinates of shape (N, 3), int32.
            features (Optional[np.ndarray]): Features of shape (N, C), float32.
        """
        self.coords = coords.astype(np.int32)
        self.features = features
        logger.debug(f"Created SparseVoxels with N={len(coords)} voxels.")

def build_from_sdf(s_grid: np.ndarray, band: float = 0.1) -> np.ndarray:
    """Find coordinates of near-surface voxels (|SDF| <= band).
    
    Args:
        s_grid (np.ndarray): Dense signed distance grid (res, res, res).
        band (float): Near-surface band distance.
        
    Returns:
        np.ndarray: Voxel indices of shape (N, 3), int32.
    """
    indices = np.where(np.abs(s_grid) <= band)
    coords = np.stack(indices, axis=-1).astype(np.int32)
    return coords

def to_minkowski(coords: torch.Tensor, feats: torch.Tensor) -> Any:
    """Convert coords and feats into MinkowskiEngine.SparseTensor.
    
    Args:
        coords (torch.Tensor): Coordinates of shape (N, 3) or (N, 4) with batch index.
        feats (torch.Tensor): Features of shape (N, C).
        
    Returns:
        MinkowskiEngine.SparseTensor or Dict: Sparse tensor representation.
    """
    if HAS_MINKOWSKI:
        # MinkowskiEngine expects coords as (N, 4) where col 0 is batch index
        if coords.shape[1] == 3:
            batch_col = torch.zeros((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)
            coords = torch.cat([batch_col, coords], dim=1)
        return ME.SparseTensor(feats, coordinates=coords.int())
    else:
        logger.warning("MinkowskiEngine is not installed. Returning a fallback dict.")
        return {"coords": coords, "feats": feats}

def to_spconv(coords: torch.Tensor, feats: torch.Tensor, spatial_shape: Tuple[int, int, int] = (128, 128, 128)) -> Any:
    """Convert coords and feats into spconv.SparseConvTensor.
    
    Args:
        coords (torch.Tensor): Coordinates of shape (N, 4) [batch_idx, z, y, x].
        feats (torch.Tensor): Features of shape (N, C).
        spatial_shape (Tuple[int, int, int]): Dimensions of the dense space.
        
    Returns:
        spconv.SparseConvTensor or Dict: Sparse tensor representation.
    """
    if HAS_SPCONV:
        if coords.shape[1] == 3:
            batch_col = torch.zeros((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)
            coords = torch.cat([batch_col, coords], dim=1)
        return spconv.SparseConvTensor(feats, indices=coords.int(), spatial_shape=spatial_shape, batch_size=1)
    else:
        logger.warning("spconv is not installed. Returning a fallback dict.")
        return {"coords": coords, "feats": feats, "spatial_shape": spatial_shape}

def densify(coords: np.ndarray, values: np.ndarray, res: int = 128, fill_value: float = 0.0) -> np.ndarray:
    """Reconstruct a dense grid from sparse coordinates and values.
    
    Args:
        coords (np.ndarray): Coordinates of shape (N, 3).
        values (np.ndarray): Values of shape (N, C) or (N,).
        res (int): Grid resolution.
        fill_value (float): Background value.
        
    Returns:
        np.ndarray: Dense grid of shape (res, res, res, C) or (res, res, res).
    """
    if len(values.shape) > 1:
        C = values.shape[1]
        dense = np.full((res, res, res, C), fill_value, dtype=values.dtype)
        dense[coords[:, 0], coords[:, 1], coords[:, 2]] = values
    else:
        dense = np.full((res, res, res), fill_value, dtype=values.dtype)
        dense[coords[:, 0], coords[:, 1], coords[:, 2]] = values
    return dense

def neighbour_query(coords: np.ndarray, distance: int = 1) -> List[np.ndarray]:
    """Find neighbor voxel indices for each coordinate.
    
    Args:
        coords (np.ndarray): Sparse coordinates shape (N, 3).
        distance (int): Maximum grid step size.
        
    Returns:
        List[np.ndarray]: List of 1D arrays containing neighbor indices for each voxel.
    """
    from scipy.spatial import KDTree
    tree = KDTree(coords)
    # query_ball_tree returns a list of lists containing neighbor indices
    neighbors = tree.query_ball_tree(tree, r=distance)
    return [np.array(n, dtype=np.int32) for n in neighbors]

def main() -> None:
    """Main entry point for testing or running sparse_grid independently."""
    parser = argparse.ArgumentParser(description="Sparse grid representations.")
    args = parser.parse_args()
    logger.info("Running sparse_grid verify stub")

if __name__ == "__main__":
    main()
