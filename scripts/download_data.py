"""
Download datasets (ShapeNet, Objaverse, PartNet).
"""

import argparse
from loguru import logger

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download datasets (ShapeNet, Objaverse, PartNet).")
    args = parser.parse_args()
    logger.info("Executing download_data")

if __name__ == "__main__":
    main()
