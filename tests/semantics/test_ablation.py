"""
Unit tests for make_tables.py and run_ablation.py.
"""

import os
import pytest
import shutil
import sys
from loguru import logger

# Add scripts directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))

def test_run_ablation_fast() -> None:
    """Test that scripts/run_ablation.py runs correctly in --fast mode."""
    logger.info("Testing scripts/run_ablation.py in fast mode...")
    
    tables_dir = "neurosem3d/results/tables"
    backup_file = "neurosem3d/results/tables/ablation.csv.backup"
    
    if os.path.exists(os.path.join(tables_dir, "ablation.csv")):
        shutil.copyfile(os.path.join(tables_dir, "ablation.csv"), backup_file)
        
    try:
        orig_cwd = os.getcwd()
        os.chdir("neurosem3d")
        
        import sys as sys_argv
        orig_argv = sys_argv.argv
        sys_argv.argv = ["run_ablation.py", "--fast"]
        
        from run_ablation import main as abl_main
        abl_main()
        
        # Assert ablation.csv is written
        assert os.path.exists("results/tables/ablation.csv"), "ablation.csv not written!"
        
    finally:
        os.chdir(orig_cwd)
        sys_argv.argv = orig_argv
        if os.path.exists(backup_file):
            shutil.copyfile(backup_file, os.path.join(tables_dir, "ablation.csv"))
            os.remove(backup_file)

def test_make_tables() -> None:
    """Test that scripts/make_tables.py runs correctly."""
    logger.info("Testing scripts/make_tables.py...")
    
    try:
        orig_cwd = os.getcwd()
        os.chdir("neurosem3d")
        
        import sys as sys_argv
        orig_argv = sys_argv.argv
        sys_argv.argv = ["make_tables.py"]
        
        from make_tables import main as tab_main
        tab_main()
        
        # Assert all tables 1-7 are written
        for i in range(1, 8):
            path = f"results/tables/table{i}_"
            # Find file prefix
            found = False
            for f in os.listdir("results/tables"):
                if f.startswith(f"table{i}_") and f.endswith(".tex"):
                    found = True
                    break
            assert found, f"Table {i} .tex file not written!"
            
    finally:
        os.chdir(orig_cwd)
        sys_argv.argv = orig_argv
