import os
import re
import sys

from src.common.logger import get_logger

logger = get_logger(__name__)


def convert_headings(input_file_path: str, output_file_path: str = None):
    """
    Convert markdown headings from alternative format to standard format.

    Converts headings from format:
    - === (underlined) to # (level 1)
    - --- (underlined) to ## (level 2)

    Args:
        input_file_path: Path to input markdown file
        output_file_path: Path to output file (defaults to input file path)
    """
    logger.info(f"Converting headings in file: {input_file_path}")
    pattern_headline_1 = re.compile(r"^(.*)\n(\=+)$", re.MULTILINE)
    pattern_headline_2 = re.compile(r"^(.*)\n(\-+)$", re.MULTILINE)

    with open(input_file_path, "r", encoding="utf-8") as file:
        content = file.read()

    original_length = len(content)
    content = pattern_headline_1.sub(r"# \1", content)
    content = pattern_headline_2.sub(r"## \1", content)
    logger.debug(f"Converted headings, file length: {original_length} -> {len(content)}")

    output_path = output_file_path or input_file_path
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(content)
    logger.info(f"Successfully wrote converted file: {output_path}")


def walk_dir(dir_path: str):
    """
    Recursively convert headings in all markdown files in a directory.

    Args:
        dir_path: Path to directory containing markdown files
    """
    logger.info(f"Walking directory to convert headings: {dir_path}")
    file_count = 0
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".md"):
                convert_headings(os.path.join(root, file))
                file_count += 1
    logger.info(f"Converted headings in {file_count} markdown files")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python3 convert_headings.py <file or directory>...")
        print("Usage: python3 convert_headings.py <file or directory>...")
        sys.exit(1)
    logger.info(f"Processing {len(sys.argv) - 1} arguments")
    for arg in sys.argv[1:]:
        if not os.path.exists(arg):
            logger.warning(f"File or directory {arg} does not exist.")
            print(f"File or directory {arg} does not exist.")
            continue
        if os.path.isdir(arg):
            walk_dir(arg)
        else:
            convert_headings(arg)
