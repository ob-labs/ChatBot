"""
File path utilities module.

This module provides utility functions for file path operations.
"""


def is_markdown_file(filename: str) -> bool:
    """
    Check if a file is a markdown file.

    Args:
        filename: The filename to check.

    Returns:
        True if the file has a markdown extension (.md or .mdx).
    """
    return filename.lower().endswith((".md", ".mdx"))
