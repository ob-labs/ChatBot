"""
Download utilities module.

This module provides utilities for downloading files and repositories.
"""

import os
import shutil
import subprocess

from src.common.logger import get_logger

logger = get_logger(__name__)


def clone_github_repo(github_url: str, dest_dir: str) -> bool:
    """
    Clone a GitHub repository to the destination directory.

    Args:
        github_url: GitHub repository URL
        dest_dir: Destination directory

    Returns:
        True if cloning was successful
    """
    try:
        # Remove dest_dir if it already exists
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", github_url, dest_dir],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to clone repository {github_url}: {e}")
        return False
