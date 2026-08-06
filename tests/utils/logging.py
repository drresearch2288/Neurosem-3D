"""
Tests for logging
"""

import pytest
from loguru import logger
# from neurosem3d.utils.logging import Logging

def test_logging() -> None:
    """Basic test for logging."""
    logger.info("Testing logging")
    assert True
