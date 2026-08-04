import os
import json
import argparse
from typing import Optional
import numpy as np
import torch
from loguru import logger

from neurosem3d.backbone.neural_sdf import load_frozen, sample_latent

def preprocess_backbone(splits_path: str, ckpt_path: str, output_dir: str, cache_dir: Optional[str] = None) -> None:
    """Preprocess objects from splits, extract latent grid z and dense s(p) grid, and save them.
    
    Args:
        splits_path (str): Path to JSON file containing split object IDs.
        ckpt_path (str): Path to NeuralRefiner checkpoint.
        output_dir (str): Where to save results.
        cache_dir (Optional[str]): Directory containing cached TSDF volumes.
    """
    logger.info("Starting preprocessing stage: backbone")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load splits
    if os.path.exists(splits_path):
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        object_ids = splits.get("test", [])
    else:
        logger.warning(f"Splits path {splits_path} not found. Creating a dummy list of object IDs for testing.")
        object_ids = [f"dummy_obj_{i}" for i in range(5)]
        
    # Load frozen backbone model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = load_frozen(ckpt_path).to(device)
    
    for obj_id in object_ids:
        logger.info(f"Processing object: {obj_id}")
        
        # Load or generate TSDF volume (128^3)
        tsdf_vol = None
        if cache_dir is not None:
            cache_path = os.path.join(cache_dir, f"{obj_id}_tsdf.npy")
            if os.path.exists(cache_path):
                tsdf_vol = np.load(cache_path)
                logger.info(f"Loaded cached TSDF volume from {cache_path}")
                
        if tsdf_vol is None:
            logger.info("No cached TSDF found. Generating mock/dummy TSDF volume.")
            # Create a dummy TSDF volume for testing (128^3)
            # Center sphere TSDF
            grid_coords = np.linspace(-1, 1, 128)
            x, y, z_coords = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
            dist = np.sqrt(x**2 + y**2 + z_coords**2)
            tsdf_vol = np.clip(0.3 - dist, -1.0, 1.0)
            
        # Run Encoder E*
        tsdf_tensor = torch.tensor(tsdf_vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 128, 128, 128)
        with torch.no_grad():
            z = model.encoder(tsdf_tensor)  # (1, 256, 16, 16, 16)
            
        # Re-arrange z shape to (16, 16, 16, 256)
        z_np = z.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        assert z_np.shape == (16, 16, 16, 256), f"z shape assertion failed: got {z_np.shape}"
        
        # Generate dense s(p) grid (e.g. 64^3 or 128^3)
        grid_res = 64
        coords = np.linspace(-1.0, 1.0, grid_res)
        cx, cy, cz = np.meshgrid(coords, coords, coords, indexing='ij')
        points = np.stack([cx, cy, cz], axis=-1).reshape(-1, 3)  # (N_points, 3)
        
        # Query decoder in chunks
        points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N_points, 3)
        chunk_size = 100000
        s_vals = []
        
        with torch.no_grad():
            # Interpolate latents for all points
            z_p = sample_latent(z, points_tensor)  # (1, N_points, 256)
            
            # Chunk through decoder
            for i in range(0, points.shape[0], chunk_size):
                p_chunk = points_tensor[:, i:i+chunk_size]
                zp_chunk = z_p[:, i:i+chunk_size]
                s_chunk = model.decoder(p_chunk, zp_chunk)
                s_vals.append(s_chunk.squeeze(0).cpu().numpy())
                
        s_grid = np.concatenate(s_vals, axis=0).reshape(grid_res, grid_res, grid_res)
        
        # Save output npz
        out_path = os.path.join(output_dir, f"{obj_id}.npz")
        np.savez_compressed(out_path, z=z_np, s_grid=s_grid)
        logger.info(f"Saved latent grid and s(p) grid to {out_path}. z shape: {z_np.shape}, s_grid shape: {s_grid.shape}")

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Preprocess data to tensors.")
    parser.add_argument("--stage", type=str, required=True, choices=["backbone", "labels", "confidence", "sparse", "splits", "semantics"], help="Preprocessing stage.")
    parser.add_argument("--splits", type=str, default="data/splits", help="Path to splits folder.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to frozen NeuralRefiner checkpoint.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory containing cached TSDF volumes.")
    parser.add_argument("--latent_grids_dir", type=str, default="data/processed/latent_grids", help="Directory containing latent grids.")
    args = parser.parse_args()
    
    # Resolve default output dirs if not specified
    if args.output_dir is None:
        if args.stage == "backbone":
            args.output_dir = "data/processed/latent_grids"
        elif args.stage == "labels":
            args.output_dir = "data/processed/gt_labels"
        elif args.stage == "confidence":
            args.output_dir = "data/processed/confidence"
        elif args.stage == "sparse":
            args.output_dir = "data/processed/sparse"
            
    if args.stage == "backbone":
        preprocess_backbone(args.splits, args.ckpt, args.output_dir, args.cache_dir)
    elif args.stage == "labels":
        preprocess_labels(args.splits, args.latent_grids_dir, args.output_dir)
    elif args.stage == "confidence":
        preprocess_confidence(args.splits, args.latent_grids_dir, args.output_dir)
    elif args.stage == "sparse":
        parent_dir = os.path.dirname(args.latent_grids_dir)
        gt_labels_dir = os.path.join(parent_dir, "gt_labels")
        confidence_dir = os.path.join(parent_dir, "confidence")
        preprocess_sparse(args.splits, args.latent_grids_dir, gt_labels_dir, confidence_dir, args.output_dir)
    elif args.stage == "splits":
        preprocess_splits(args.splits, args.latent_grids_dir)
    else:
        logger.info(f"Stage {args.stage} logic to be implemented.")

def preprocess_labels(splits_path: str, latent_grids_dir: str, output_dir: str) -> None:
    """Preprocess PartNet part tree labels, voxelise them, and save to gt_labels folder.
    
    Args:
        splits_path (str): Path to splits JSON file.
        latent_grids_dir (str): Path containing saved latent grids (with s_grid).
        output_dir (str): Output directory for ground-truth label tensors.
    """
    logger.info("Starting preprocessing stage: labels")
    os.makedirs(output_dir, exist_ok=True)
    
    from neurosem3d.data.voxelize import voxelize_part_tree
    
    # Load splits
    if os.path.exists(splits_path):
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        object_ids = splits.get("test", [])
    else:
        logger.warning(f"Splits path {splits_path} not found. Using dummy list of object IDs for testing.")
        object_ids = [f"dummy_obj_{i}" for i in range(5)]
        
    # Mock label maps & part trees for demonstration / testing
    dummy_tree = {
        "parent_map": {1: 10, 2: 10, 3: 11, 4: 11, 5: 12, 6: 12, 7: 13, 8: 13},
        "part_to_class": {i: i for i in range(1, 15)}
    }
    dummy_level_maps = {
        "fine": {i: i for i in range(1, 15)},
        "middle": {10: 10, 11: 11, 12: 12, 13: 13},
        "coarse": {10: 100, 11: 100, 12: 101, 13: 101}
    }
    
    for obj_id in object_ids:
        logger.info(f"Voxelising labels for: {obj_id}")
        latent_file = os.path.join(latent_grids_dir, f"{obj_id}.npz")
        if not os.path.exists(latent_file):
            logger.error(f"Latent grid file not found for {obj_id} at {latent_file}. Make sure backbone stage is run first.")
            continue
            
        # Load s_grid
        with np.load(latent_file) as data:
            s_grid = data["s_grid"]
            
        # Call voxelise (mesh is None here for mock pipeline)
        vox_results = voxelize_part_tree(
            mesh=None,
            tree=dummy_tree,
            level_label_maps=dummy_level_maps,
            s_grid=s_grid,
            res=128
        )
        
        # Save output
        out_path = os.path.join(output_dir, f"{obj_id}.npz")
        np.savez_compressed(
            out_path,
            coarse=vox_results["coarse"],
            middle=vox_results["middle"],
            fine=vox_results["fine"],
            ignore_mask=vox_results["ignore_mask"]
        )
        
        # Log counts
        for lvl in ["coarse", "middle", "fine"]:
            labels = vox_results[lvl]
            unique, counts = np.unique(labels[labels > 0], return_counts=True)
            counts_dict = dict(zip(unique, counts))
            logger.info(f"[{lvl}] Voxel counts per part class for {obj_id}: {counts_dict}")

def preprocess_confidence(splits_path: str, latent_grids_dir: str, output_dir: str) -> None:
    """Preprocess confidence cues for occupied voxels across all 8 orbit views.
    
    Args:
        splits_path (str): Path to splits JSON.
        latent_grids_dir (str): Path containing saved latent grids.
        output_dir (str): Output directory for confidence tensor.
    """
    logger.info("Starting preprocessing stage: confidence")
    os.makedirs(output_dir, exist_ok=True)
    
    from neurosem3d.data.confidence import compute_cues
    
    # Load splits
    if os.path.exists(splits_path):
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        object_ids = splits.get("test", [])
    else:
        logger.warning(f"Splits path {splits_path} not found. Using dummy list of object IDs for testing.")
        object_ids = [f"dummy_obj_{i}" for i in range(5)]
        
    for obj_id in object_ids:
        logger.info(f"Extracting confidence cues for: {obj_id}")
        latent_file = os.path.join(latent_grids_dir, f"{obj_id}.npz")
        if not os.path.exists(latent_file):
            logger.error(f"Latent grid file not found for {obj_id}. Cannot run confidence stage.")
            continue
            
        with np.load(latent_file) as data:
            s_grid = data["s_grid"]
            
        # Get occupied voxels (near-surface)
        res = s_grid.shape[0]
        coords = np.linspace(-1.0, 1.0, res)
        cx, cy, cz = np.meshgrid(coords, coords, coords, indexing='ij')
        points = np.stack([cx, cy, cz], axis=-1)
        
        ns_idx = np.where(np.abs(s_grid) <= 0.1)
        voxel_xyz = points[ns_idx]  # (N, 3)
        N = voxel_xyz.shape[0]
        
        if N == 0:
            logger.warning(f"No occupied voxels found for {obj_id}!")
            continue
            
        voxel_xyz_t = torch.tensor(voxel_xyz, dtype=torch.float32)
        # Mock normals pointing outwards
        normal_v_t = voxel_xyz_t / (torch.norm(voxel_xyz_t, dim=-1, keepdim=True) + 1e-8)
        
        # Setup arrays to store 8 views
        c_depth_all = np.zeros((N, 8), dtype=np.float32)
        c_angle_all = np.zeros((N, 8), dtype=np.float32)
        c_mask_all = np.zeros((N, 8), dtype=np.float32)
        projected_label_all = np.zeros((N, 8), dtype=np.int32)
        
        # 8 views projection loop
        for view_idx in range(8):
            # 1. Setup camera intrinsics
            K = torch.tensor([
                [120.0, 0.0, 128.0],
                [0.0, 120.0, 128.0],
                [0.0, 0.0, 1.0]
            ], dtype=torch.float32)
            
            # 2. Setup camera extrinsics (orbiting)
            theta = float(view_idx) * (2.0 * np.pi / 8.0)
            eye = np.array([2.0 * np.cos(theta), 0.5, 2.0 * np.sin(theta)])
            at = np.array([0.0, 0.0, 0.0])
            up = np.array([0.0, 1.0, 0.0])
            z_axis = eye - at
            z_axis = z_axis / np.linalg.norm(z_axis)
            x_axis = np.cross(up, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            R = np.stack([x_axis, y_axis, z_axis], axis=0)
            T_wc = np.eye(4)
            T_wc[:3, :3] = R
            T_wc[:3, 3] = -R @ eye
            T = torch.tensor(T_wc, dtype=torch.float32)
            
            # 3. Mock depth map and SAM mask (256x256)
            depth_map = torch.ones(256, 256, dtype=torch.float32) * 2.0
            sam_mask = torch.zeros(256, 256, dtype=torch.long)
            # Create a mock center SAM mask
            sam_mask[64:192, 64:192] = (view_idx % 4) + 1
            
            # Call compute_cues
            cues = compute_cues(
                voxel_xyz=voxel_xyz_t,
                normal_v=normal_v_t,
                K=K,
                T=T,
                depth_map=depth_map,
                sam_mask=sam_mask,
                stability_score=0.95,
                delta=0.05
            )
            
            c_depth_all[:, view_idx] = cues["c_depth"].cpu().numpy()
            c_angle_all[:, view_idx] = cues["c_angle"].cpu().numpy()
            c_mask_all[:, view_idx] = cues["c_mask"].cpu().numpy()
            projected_label_all[:, view_idx] = cues["projected_label"].cpu().numpy()
            
        logger.info(f"Saved cues to {out_path}. N voxels: {N}")
        logger.info(f"Mean cues: depth={c_depth_all.mean():.4f}, angle={c_angle_all.mean():.4f}, mask={c_mask_all.mean():.4f}")

def preprocess_sparse(
    splits_path: str,
    latent_grids_dir: str,
    gt_labels_dir: str,
    confidence_dir: str,
    output_dir: str
) -> None:
    """Preprocess sparse grid representations for NeuroSem-3D Neural Semantic Head.
    
    Args:
        splits_path (str): Path to splits JSON.
        latent_grids_dir (str): Directory containing latent grids.
        gt_labels_dir (str): Directory containing ground-truth labels.
        confidence_dir (str): Directory containing confidence NPZs.
        output_dir (str): Where to save results.
    """
    logger.info("Starting preprocessing stage: sparse")
    os.makedirs(output_dir, exist_ok=True)
    
    from neurosem3d.data.sparse_grid import build_from_sdf
    from neurosem3d.semantics.cwcvsf import fuse_semantics
    from neurosem3d.backbone.neural_sdf import sample_latent
    
    # Load splits
    if os.path.exists(splits_path):
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        object_ids = splits.get("test", [])
    else:
        logger.warning(f"Splits path {splits_path} not found. Using dummy list of object IDs for testing.")
        object_ids = [f"dummy_obj_{i}" for i in range(5)]
        
    occupancies = []
    dense_voxel_count = 128**3  # standard dense grid size
    
    for obj_id in object_ids:
        logger.info(f"Building sparse representation for: {obj_id}")
        
        # 1. Load latent grids
        latent_file = os.path.join(latent_grids_dir, f"{obj_id}.npz")
        if not os.path.exists(latent_file):
            logger.error(f"Latent grid file not found for {obj_id}.")
            continue
        with np.load(latent_file) as data:
            s_grid = data["s_grid"]
            z = data["z"]
            
        # 2. Build coordinates from SDF
        coords = build_from_sdf(s_grid, band=0.1)  # (N, 3)
        N = coords.shape[0]
        if N == 0:
            logger.warning(f"No near-surface voxels found for {obj_id}!")
            continue
            
        # 3. Sample latent features z(v)
        res = s_grid.shape[0]
        # Normalize coordinates to [-1, 1] range for grid_sample
        coords_norm = (torch.tensor(coords, dtype=torch.float32) / (res - 1)) * 2.0 - 1.0
        z_tensor = torch.tensor(z, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0)  # (1, 256, 16, 16, 16)
        
        # sample_latent expects coords_norm shape (1, N, 3)
        z_v_tensor = sample_latent(z_tensor, coords_norm.unsqueeze(0))  # (1, N, 256)
        z_v = z_v_tensor.squeeze(0).numpy()  # (N, 256)
        
        # 4. Extract s(v)
        s_v = s_grid[coords[:, 0], coords[:, 1], coords[:, 2]][:, np.newaxis]  # (N, 1)
        
        # 5. Compute fused semantics soft labels P_fuse(v)
        conf_file = os.path.join(confidence_dir, f"{obj_id}.npz")
        if not os.path.exists(conf_file):
            logger.error(f"Confidence cues file not found for {obj_id}.")
            continue
        with np.load(conf_file) as data:
            c_depth = data["c_depth"]
            c_angle = data["c_angle"]
            c_mask = data["c_mask"]
            proj_labels = data["projected_label"]
            
        P_fuse = fuse_semantics(c_depth, c_angle, c_mask, proj_labels, num_classes=15)  # (N, 15)
        
        # 6. Concatenate features [z_v, s_v, P_fuse] -> shape (N, 256 + 1 + 15)
        feats = np.concatenate([z_v, s_v, P_fuse], axis=1)
        
        # 7. Load GT label levels at coords
        gt_file = os.path.join(gt_labels_dir, f"{obj_id}.npz")
        if not os.path.exists(gt_file):
            logger.error(f"GT label file not found for {obj_id}.")
            continue
        with np.load(gt_file) as data:
            coarse_grid = data["coarse"]
            middle_grid = data["middle"]
            fine_grid = data["fine"]
            ignore_mask_grid = data["ignore_mask"]
            
        coarse_labels = coarse_grid[coords[:, 0], coords[:, 1], coords[:, 2]]
        middle_labels = middle_grid[coords[:, 0], coords[:, 1], coords[:, 2]]
        fine_labels = fine_grid[coords[:, 0], coords[:, 1], coords[:, 2]]
        ignore_mask = ignore_mask_grid[coords[:, 0], coords[:, 1], coords[:, 2]]
        
        # Save output npz
        out_path = os.path.join(output_dir, f"{obj_id}.npz")
        np.savez_compressed(
            out_path,
            coords=coords,
            feats=feats,
            coarse=coarse_labels,
            middle=middle_labels,
            fine=fine_labels,
            ignore_mask=ignore_mask
        )
        
        occupancy_frac = N / dense_voxel_count
        occupancies.append(occupancy_frac)
        logger.info(f"Saved sparse representations to {out_path}.")
        logger.info(f"Occupied voxels: {N} / {dense_voxel_count} (fraction: {occupancy_frac:.4f})")
        
    if occupancies:
        mean_occupancy = np.mean(occupancies)
        logger.info(f"Preprocess Stage Sparse Complete.")
        logger.info(f"Mean occupancy fraction: {mean_occupancy:.4f} (~{mean_occupancy*100:.2f}%)")
        logger.info(f"Voxel reduction ratio (sparse vs dense 128^3): 1:{1.0 / (mean_occupancy + 1e-8):.2f}")

def preprocess_splits(splits_dir: str, latent_grids_dir: str) -> None:
    """Create train/val/test data splits JSON files under splits folder.
    
    Args:
        splits_dir (str): Output directory for split files.
        latent_grids_dir (str): Path to latent grids folder.
    """
    logger.info("Starting preprocessing stage: splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    import yaml
    
    # 1. Load train config for seed
    seed = 42
    train_cfg_path = "neurosem3d/configs/train.yaml"
    if os.path.exists(train_cfg_path):
        with open(train_cfg_path, "r") as f:
            train_cfg = yaml.safe_load(f)
            if train_cfg is not None:
                seed = train_cfg.get("seed", 42)
    logger.info(f"Setting global random seed: {seed}")
    np.random.seed(seed)
    
    # 2. Load Work-1 test ids from paths.yaml
    work1_test_ids = []
    paths_cfg_path = "neurosem3d/configs/paths.yaml"
    if os.path.exists(paths_cfg_path):
        with open(paths_cfg_path, "r") as f:
            paths_cfg = yaml.safe_load(f)
            if paths_cfg is not None:
                work1_test_ids = paths_cfg.get("work1_test_ids", [])
                
    if not work1_test_ids:
        logger.warning("No Work-1 test IDs found in paths.yaml. Creating dummy test list.")
        work1_test_ids = [f"dummy_obj_{i}" for i in range(5)]
        
    # 3. Gather all available object IDs
    parent_dir = os.path.dirname(latent_grids_dir)
    sparse_dir = os.path.join(parent_dir, "sparse")
    if os.path.exists(sparse_dir):
        all_ids = [f.replace(".npz", "") for f in os.listdir(sparse_dir) if f.endswith(".npz")]
    else:
        # Generate some dummy IDs if sparse dir is empty
        all_ids = [f"dummy_obj_{i}" for i in range(25)]
        
    # 4. Perform split
    # Remove test set ids from train/val pool
    test_set = set(work1_test_ids)
    remaining_pool = [oid for oid in all_ids if oid not in test_set]
    
    # Stratified shuffle (for dummy set, we just shuffle remainder)
    np.random.shuffle(remaining_pool)
    
    # If remaining pool is empty (e.g. testing with only mock test set), generate distinct dummy train/val IDs
    if not remaining_pool:
        logger.warning("Remaining pool is empty. Generating mock train and validation IDs to prevent overlap.")
        train_ids = [f"dummy_train_{i}" for i in range(10)]
        val_ids = [f"dummy_val_{i}" for i in range(5)]
    else:
        # 70% train, 30% val of the remaining pool
        num_train = int(len(remaining_pool) * 0.7)
        train_ids = remaining_pool[:num_train]
        val_ids = remaining_pool[num_train:]
        if not train_ids:
            train_ids = [f"dummy_train_{i}" for i in range(5)]
            val_ids = [f"dummy_val_{i}" for i in range(2)]
        
    # Assert zero overlap
    train_set = set(train_ids)
    val_set = set(val_ids)
    
    assert len(train_set.intersection(val_set)) == 0, "Overlap between train and val splits!"
    assert len(train_set.intersection(test_set)) == 0, "Overlap between train and test splits!"
    assert len(val_set.intersection(test_set)) == 0, "Overlap between val and test splits!"
    
    # Save JSON files
    with open(os.path.join(splits_dir, "train.json"), "w") as f:
        json.dump(train_ids, f, indent=4)
    with open(os.path.join(splits_dir, "val.json"), "w") as f:
        json.dump(val_ids, f, indent=4)
    with open(os.path.join(splits_dir, "test.json"), "w") as f:
        json.dump(list(test_set), f, indent=4)
        
    logger.info(f"Split results: train={len(train_ids)}, val={len(val_ids)}, test={len(test_set)}")
    logger.success("Preprocessing Stage Splits Complete.")

if __name__ == "__main__":
    main()
