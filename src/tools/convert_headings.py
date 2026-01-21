"""
Markdown heading conversion tool.

Provides functionality to convert markdown headings from alternative format
(=== and ---) to standard format (# and ##).
"""
import os
import re
import sys
from typing import Optional

from src.common.file_path import is_markdown_file
from src.common.logger import get_logger

logger = get_logger(__name__)

# Regex patterns for heading conversion
PATTERN_HEADING_1 = re.compile(r"^(.*)\n(\=+)$", re.MULTILINE)
PATTERN_HEADING_2 = re.compile(r"^(.*)\n(\-+)$", re.MULTILINE)


def convert_headings_in_file(
    input_path: str,
    output_path: Optional[str] = None,
) -> None:
    """
    Convert markdown headings from alternative format to standard format.

    Converts headings from format:
    - === (underlined) to # (level 1)
    - --- (underlined) to ## (level 2)

    Args:
        input_path: Path to input markdown file
        output_path: Path to output file. If None, overwrites input file.
    """
    logger.info(f"Converting headings in file: {input_path}")

    with open(input_path, "r", encoding="utf-8") as file:
        content = file.read()

    original_length = len(content)

    # Convert headings
    content = PATTERN_HEADING_1.sub(r"# \1", content)
    content = PATTERN_HEADING_2.sub(r"## \1", content)

    logger.debug(
        f"Converted headings, file length: {original_length} -> {len(content)}"
    )

    # Write output
    final_path = output_path or input_path
    with open(final_path, "w", encoding="utf-8") as file:
        file.write(content)

    logger.info(f"Successfully wrote converted file: {final_path}")


def convert_headings_in_directory(dir_path: str) -> int:
    """
    Recursively convert headings in all markdown files in a directory.

    Args:
        dir_path: Path to directory containing markdown files

    Returns:
        Number of files processed
    """
    logger.info(f"Walking directory to convert headings: {dir_path}")

    file_count = 0
    for root, _, files in os.walk(dir_path):
        for file in files:
            if is_markdown_file(file):
                file_path = os.path.join(root, file)
                convert_headings_in_file(file_path)
                file_count += 1

    logger.info(f"Converted headings in {file_count} markdown files")
    return file_count


def process_path(path: str) -> int:
    """
    Process a file or directory path for heading conversion.

    Args:
        path: Path to a file or directory

    Returns:
        Number of files processed (1 for a single file, N for directory)
    """
    if not os.path.exists(path):
        logger.warning(f"File or directory {path} does not exist.")
        print(f"File or directory {path} does not exist.")
        return 0

    if os.path.isdir(path):
        return convert_headings_in_directory(path)
    else:
        convert_headings_in_file(path)
        return 1


def main() -> None:
    """Main entry point for the heading conversion tool."""
    if len(sys.argv) < 2:
        logger.error("Usage: python convert_headings.py <file or directory>...")
        print("Usage: python convert_headings.py <file or directory>...")
        sys.exit(1)

    logger.info(f"Processing {len(sys.argv) - 1} arguments")

    total_files = 0
    for arg in sys.argv[1:]:
        total_files += process_path(arg)

    print(f"Successfully converted headings in {total_files} file(s).")


if __name__ == "__main__":
    main()
