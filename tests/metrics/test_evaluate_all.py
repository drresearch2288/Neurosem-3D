"""
Unit tests for evaluate_all.py and check_geometry_preserved.py.
"""

import os
import json
import pytest
import shutil
import sys
from loguru import logger

# Add scripts directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))

def test_evaluate_all_fast() -> None:
    """Test that scripts/evaluate_all.py runs correctly in --fast mode."""
    logger.info("Testing scripts/evaluate_all.py in fast mode...")
    
    # Backup existing evaluation folder
    eval_dir = "neurosem3d/results/evaluation"
    backup_dir = "neurosem3d/results/evaluation_backup"
    
    if os.path.exists(eval_dir):
        shutil.copytree(eval_dir, backup_dir, dirs_exist_ok=True)
        
    try:
        orig_cwd = os.getcwd()
        os.chdir("neurosem3d")
        
        # Import main from evaluate_all
        import sys as sys_argv
        orig_argv = sys_argv.argv
        sys_argv.argv = ["evaluate_all.py", "--fast", "--methods", "proposed_student,baseline_b1"]
        
        from evaluate_all import main as eval_main
        eval_main()
        
        # Assert files are created
        assert os.path.exists("results/evaluation/results.json"), "results.json not written!"
        assert os.path.exists("results/evaluation/results.csv"), "results.csv not written!"
        assert os.path.exists("results/evaluation/target_gap_report.md"), "target_gap_report.md not written!"
        
        # Load and check JSON structure
        with open("results/evaluation/results.json", "r") as f:
            data = json.load(f)
            
        assert "summary" in data
        assert "proposed_student" in data["summary"]
        assert "baseline_b1" in data["summary"]
        
    finally:
        os.chdir(orig_cwd)
        sys_argv.argv = orig_argv
        if os.path.exists(backup_dir):
            shutil.copytree(backup_dir, eval_dir, dirs_exist_ok=True)
            shutil.rmtree(backup_dir)

def test_check_geometry_preserved_fast() -> None:
    """Test that scripts/check_geometry_preserved.py runs correctly in --fast mode."""
    logger.info("Testing scripts/check_geometry_preserved.py in fast mode...")
    
    eval_dir = "neurosem3d/results/evaluation"
    backup_dir = "neurosem3d/results/evaluation_backup"
    
    if os.path.exists(eval_dir):
        shutil.copytree(eval_dir, backup_dir, dirs_exist_ok=True)
        
    try:
        orig_cwd = os.getcwd()
        os.chdir("neurosem3d")
        
        import sys as sys_argv
        orig_argv = sys_argv.argv
        sys_argv.argv = ["check_geometry_preserved.py", "--fast"]
        
        from check_geometry_preserved import main as geom_main
        geom_main()
        
        # Assert files are created
        assert os.path.exists("results/evaluation/geometry_preservation.json"), "geometry_preservation.json not written!"
        
        with open("results/evaluation/geometry_preservation.json", "r") as f:
            data = json.load(f)
            
        assert "chamfer_distance" in data
        assert "verdict" in data
        assert data["verdict"] == "PASS"
        
    finally:
        os.chdir(orig_cwd)
        sys_argv.argv = orig_argv
        if os.path.exists(backup_dir):
            shutil.copytree(backup_dir, eval_dir, dirs_exist_ok=True)
            shutil.rmtree(backup_dir)
