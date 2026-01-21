"""
Compression utilities for handling archive files.

This module provides functions for extracting various archive formats
and checking if files are archives.
"""

import bz2
import gzip
import lzma
import os
import shutil
import tarfile
import zipfile
from typing import List

from src.common.logger import get_logger

logger = get_logger(__name__)


# Supported archive extensions
ARCHIVE_EXTENSIONS: List[str] = [
    ".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".gz", ".bz2", ".xz"
]


def extract_archive(archive_path: str, dest_dir: str) -> bool:
    """
    Extract an archive file to the destination directory.

    Args:
        archive_path: Path to the archive file
        dest_dir: Destination directory

    Returns:
        True if extraction was successful, False otherwise
    """
    try:
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_dir)
        elif archive_path.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(dest_dir)
        elif archive_path.endswith(".gz"):
            output_path = os.path.join(dest_dir, os.path.basename(archive_path)[:-3])
            with gzip.open(archive_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif archive_path.endswith(".bz2"):
            output_path = os.path.join(dest_dir, os.path.basename(archive_path)[:-4])
            with bz2.open(archive_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif archive_path.endswith(".xz"):
            output_path = os.path.join(dest_dir, os.path.basename(archive_path)[:-3])
            with lzma.open(archive_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to extract archive {archive_path}: {e}")
        return False


def is_archive_file(filename: str) -> bool:
    """
    Check if a file is an archive based on its extension.

    Args:
        filename: Name or path of the file to check

    Returns:
        True if the file has an archive extension, False otherwise
    """
    return any(filename.lower().endswith(ext) for ext in ARCHIVE_EXTENSIONS)
