"""Unit tests for NSH training script."""

import pytest
import os
import torch
import shutil
from loguru import logger

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))
from train_nsh import main

def test_train_nsh_fast() -> None:
    """Verify train_nsh.py runs in smoke-test (--fast) mode, saving checkpoint and logging accuracy."""
    logger.info("Running NSH training smoke-test...")
    
    # Save original args to restore after test
    import sys
    orig_argv = sys.argv
    
    # Configure path references for test runner running from project root
    # Set sys.argv to mock calling python train_nsh.py --fast
    sys.argv = ["train_nsh.py", "--fast"]
    
    # Create models backup folder to avoid overwriting trained models if any
    teacher_path = "neurosem3d/results/models/nsh_teacher.pt"
    backup_path = "neurosem3d/results/models/nsh_teacher.pt.backup"
    if os.path.exists(teacher_path):
        shutil.copyfile(teacher_path, backup_path)
        
    try:
        # Run training main (change directory to neurosem3d context temporarily)
        orig_cwd = os.getcwd()
        os.chdir("neurosem3d")
        
        main()
        
        # Check that checkpoint is written
        assert os.path.exists("results/models/nsh_teacher.pt"), "Teacher checkpoint was not written!"
        
        # Check that checkpoint is valid and contains metadata
        checkpoint = torch.load("results/models/nsh_teacher.pt", map_location="cpu")
        assert "state_dict" in checkpoint
        assert "git_commit" in checkpoint
        assert "config" in checkpoint
        assert "seed" in checkpoint
        
        # Check that CSV history log is written
        assert os.path.exists("results/logs/train_history.csv"), "Training CSV history was not written!"
        
    finally:
        # Restore directory, argv, and model backup
        os.chdir(orig_cwd)
        sys.argv = orig_argv
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, teacher_path)
            os.remove(backup_path)

def test_train_nsh_sweep_fast() -> None:
    """Verify train_nsh.py runs in sweep mode with smoke-test configs."""
    logger.info("Running NSH training sweep smoke-test...")
    
    import sys
    orig_argv = sys.argv
    sys.argv = ["train_nsh.py", "--sweep", "--fast"]
    
    teacher_path = "neurosem3d/results/models/nsh_teacher.pt"
    backup_path = "neurosem3d/results/models/nsh_teacher.pt.backup"
    if os.path.exists(teacher_path):
        shutil.copyfile(teacher_path, backup_path)
        
    try:
        orig_cwd = os.getcwd()
        os.chdir("neurosem3d")
        
        main()
        
        # Verify sweep results table is written
        assert os.path.exists("results/tables/loss_weight_sweep.csv"), "Sweep results CSV was not written!"
        
        # Verify winning betas saved in train.yaml config
        assert os.path.exists("configs/train.yaml"), "train.yaml config missing!"
        with open("configs/train.yaml", "r") as f:
            import yaml
            cfg_yaml = yaml.safe_load(f)
        assert "betas" in cfg_yaml
        assert "beta1" in cfg_yaml["betas"]
        assert "beta2" in cfg_yaml["betas"]
        assert "beta3" in cfg_yaml["betas"]
        
    finally:
        os.chdir(orig_cwd)
        sys.argv = orig_argv
        if os.path.exists(backup_path):
            shutil.copyfile(backup_path, teacher_path)
            os.remove(backup_path)
