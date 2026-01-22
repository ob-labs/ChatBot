"""
Document processing utilities for markdown files.

This module provides functions and classes for parsing and loading markdown
documents with chunking support.
"""

import os
import re
from typing import Iterator, Optional

import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel

from src.common.file_path import is_markdown_file
from src.common.logger import get_logger
from src.rag.ob import DEFAULT_COMPONENT

logger = get_logger(__name__)


# Document chunking settings
DEFAULT_MAX_CHUNK_SIZE = 4096

# Markdown header levels for text splitting
MARKDOWN_HEADERS = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
    ("####", "Header4"),
    ("#####", "Header5"),
    ("######", "Header6"),
]

# Global markdown splitter instance
_markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=MARKDOWN_HEADERS)


class DocumentMeta(BaseModel):
    """
    Metadata model for document chunks.

    Attributes:
        doc_url: URL or file path to the source document.
        doc_name: Display name of the document.
        component: Component name the document belongs to.
        chunk_title: Title of this specific chunk.
        enhanced_title: Hierarchical title with full path.
    """

    class Config:
        extra = "allow"

    doc_url: str
    doc_name: str
    component: str = DEFAULT_COMPONENT
    chunk_title: str
    enhanced_title: str


def parse_md(
    file_path: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> Iterator[Document]:
    """
    Parse a markdown file and split it into document chunks.

    Uses hierarchical header-based splitting to preserve document structure.
    Large chunks exceeding max_chunk_size are further split into sub-chunks.

    Args:
        file_path: Path to the markdown file.
        max_chunk_size: Maximum size of each chunk in characters.

    Yields:
        Document objects with populated metadata.

    Raises:
        IOError: If the file cannot be read.
    """
    logger.debug(f"Parsing markdown file: {file_path}, max_chunk_size: {max_chunk_size}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
    except Exception as e:
        logger.error(f"Error reading markdown file {file_path}: {e}")
        raise

    chunks = _markdown_splitter.split_text(file_content)
    logger.debug(f"Split file into {len(chunks)} initial chunks")

    chunk_count = 0
    for chunk in chunks:
        meta = _create_chunk_metadata(chunk, file_path)

        if len(chunk.page_content) <= max_chunk_size:
            chunk.metadata = meta.model_dump()
            chunk_count += 1
            yield chunk
        else:
            # Split large chunks into smaller sub-chunks
            for sub_chunk in _split_large_chunk(chunk, meta, max_chunk_size):
                chunk_count += 1
                yield sub_chunk

    logger.info(f"Parsed {file_path}: generated {chunk_count} chunks")


def _create_chunk_metadata(chunk: Document, file_path: str) -> DocumentMeta:
    """
    Create metadata for a document chunk.

    Args:
        chunk: The document chunk from splitter.
        file_path: Original file path.

    Returns:
        DocumentMeta instance with populated fields.
    """
    subtitles = list(chunk.metadata.values())
    if not subtitles:
        subtitles.append(file_path.split("/")[-1])

    return DocumentMeta(
        doc_url=file_path,
        chunk_title=subtitles[-1],
        enhanced_title=" -> ".join(subtitles),
        doc_name=chunk.metadata.get("Header1", subtitles[-1]),
    )


def _split_large_chunk(
    chunk: Document,
    meta: DocumentMeta,
    max_chunk_size: int,
) -> Iterator[Document]:
    """
    Split a large chunk into smaller sub-chunks.

    Args:
        chunk: The large chunk to split.
        meta: Metadata to apply to all sub-chunks.
        max_chunk_size: Maximum size per sub-chunk.

    Yields:
        Document objects representing sub-chunks.
    """
    content = chunk.page_content
    sub_chunk_count = (len(content) + max_chunk_size - 1) // max_chunk_size
    logger.debug(f"Splitting large chunk into {sub_chunk_count} sub-chunks")

    for i in range(0, len(content), max_chunk_size):
        sub_chunk = Document(content[i:i + max_chunk_size])
        sub_chunk.metadata = meta.model_dump()
        yield sub_chunk


class MarkdownDocumentsLoader:
    """
    Loader for markdown documents from a directory.

    Recursively walks through a directory, finds all markdown files,
    and loads them as Document objects with metadata.

    Example:
        loader = MarkdownDocumentsLoader("/path/to/docs")
        for doc in loader.load():
            process(doc)
    """

    def __init__(
        self,
        doc_base: str,
        skip_patterns: Optional[list[str]] = None,
    ):
        """
        Initialize the loader.

        Args:
            doc_base: Base directory containing markdown files.
            skip_patterns: List of regex patterns for files to skip.
        """
        self.doc_base = doc_base
        self.skip_patterns = skip_patterns or []
        logger.info(
            f"Initializing MarkdownDocumentsLoader: "
            f"doc_base={doc_base}, skip_patterns={self.skip_patterns}"
        )

    def load(
        self,
        show_progress: bool = True,
        limit: int = 0,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    ) -> Iterator[Document]:
        """
        Load markdown documents from the base directory.

        Args:
            show_progress: Whether to show progress bar.
            limit: Maximum number of files to process (0 = no limit).
            max_chunk_size: Maximum size of each chunk in characters.

        Yields:
            Document objects with DocumentMeta metadata.
        """
        logger.info(
            f"Loading documents from {self.doc_base}, "
            f"limit={limit}, max_chunk_size={max_chunk_size}"
        )

        files = self._discover_files()
        logger.info(f"Found {len(files)} markdown files to process")

        files_iter = tqdm.tqdm(files) if show_progress else files

        for count, file_path in enumerate(files_iter, start=1):
            yield from parse_md(file_path, max_chunk_size=max_chunk_size)

            if limit > 0 and count >= limit:
                logger.info(f"Limit reached: {limit}, exiting.")
                print(f"Limit reached: {limit}, exiting.")
                exit(0)

        logger.info(f"Document loading completed, processed {len(files)} files")

    def _discover_files(self) -> list[str]:
        """
        Discover all markdown files in the base directory.

        Returns:
            List of file paths to process.
        """
        files: list[str] = []

        for root, _, filenames in os.walk(self.doc_base):
            for filename in filenames:
                if not is_markdown_file(filename):
                    continue

                file_path = os.path.join(root, filename)

                if self._should_skip(file_path):
                    logger.debug(f"Skipping file matching pattern: {file_path}")
                    continue

                files.append(file_path)

        return files

    def _should_skip(self, file_path: str) -> bool:
        """
        Check if a file should be skipped based on patterns.

        Args:
            file_path: Path to check.

        Returns:
            True if file should be skipped.
        """
        return any(re.search(pattern, file_path) for pattern in self.skip_patterns)
