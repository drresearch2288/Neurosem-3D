# NeuroSem-3D

**Learned, Uncertainty-Aware, Hierarchical 3D Semantic Field for Editable Text-to-3D Generation.**

This repository contains the official implementation of NeuroSem-3D (Work 2), an incremental extension of SemGen-3D (Work 1).

## Delta: Work 1 vs. Work 2

| Component | SemGen-3D (Work 1) | NeuroSem-3D (Work 2) |
| :--- | :--- | :--- |
| **Geometry** | SDXL + ControlNet $\rightarrow$ DPT $\rightarrow$ TSDF $\rightarrow$ SDF+MLP | **FROZEN** (Reused exactly) |
| **Semantic Fusion** | Equal-weight majority vote | Confidence-Weighted Cross-View Semantic Fusion (CW-CVSF) |
| **3D Semantic Rep** | Dense implicit MLP | PartNet-supervised sparse 3D Neural Semantic Head (NSH) with uncertainty |
| **Editing** | Flat part-level editing | Hierarchical decoder for sub-assembly editing |
| **Inference** | Dense evaluation | Sparse-voxel distilled inference path (SVDI) |

## Structure

- `configs/`: Hydra configuration files.
- `data/`: Dataset storage (ShapeNet, Objaverse, PartNet).
- `src/neurosem3d/`: Source code.
  - `backbone/`: Frozen Work-1 modules.
  - `semantics/`: New Work-2 modules (CW-CVSF, NSH, SVDI).
  - `data/`, `baselines/`, `metrics/`, `utils/`.
- `scripts/`: Execution scripts.
- `tests/`: Pytest mirror of the source directory.
