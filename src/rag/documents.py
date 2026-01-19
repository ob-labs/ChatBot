import os
import re
from typing import Iterator

import tqdm
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel

from src.common.logger import get_logger

logger = get_logger(__name__)

# Default component name
DEFAULT_COMPONENT = "observer"

# Default chunk size for document splitting
DEFAULT_MAX_CHUNK_SIZE = 4096


class DocumentMeta(BaseModel):
    """
    Document metadata model.

    Attributes:
        doc_url: URL or path to the document
        doc_name: Name of the document
        component: Component name (default: "observer")
        chunk_title: Title of the chunk
        enhanced_title: Enhanced title with hierarchy
    """

    class Config:
        extra = "allow"

    doc_url: str
    doc_name: str
    component: str = DEFAULT_COMPONENT
    chunk_title: str
    enhanced_title: str


# Component name to code mapping
component_mapping = {
    "observer": 1,
    "ocp": 2,
    "oms": 3,
    "obd": 4,
    "operator": 5,
    "odp": 6,
    "odc": 7,
}


headers_to_split_on = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
    ("####", "Header4"),
    ("#####", "Header5"),
    ("######", "Header6"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
)


def parse_md(
    file_path: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> Iterator[Document]:
    """
    Parse a markdown file and split it into document chunks.

    Args:
        file_path: Path to the markdown file
        max_chunk_size: Maximum size of each chunk in characters

    Yields:
        Document objects with metadata
    """
    logger.debug(f"Parsing markdown file: {file_path}, max_chunk_size: {max_chunk_size}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
            chunks = splitter.split_text(file_content)
            logger.debug(f"Split file into {len(chunks)} initial chunks")
            chunk_count = 0
            for chunk in chunks:
                subtitles = list(chunk.metadata.values())
                if len(subtitles) == 0:
                    subtitles.append(file_path.split("/")[-1])
                meta = DocumentMeta(
                    doc_url=file_path,
                    chunk_title=subtitles[-1],
                    enhanced_title=" -> ".join(subtitles),
                    doc_name=chunk.metadata.get("Header1", subtitles[-1]),
                )
                if len(chunk.page_content) <= max_chunk_size:
                    chunk.metadata = meta.model_dump()
                    chunk_count += 1
                    yield chunk
                else:
                    # Split large chunks into smaller sub-chunks
                    sub_chunk_count = (len(chunk.page_content) + max_chunk_size - 1) // max_chunk_size
                    logger.debug(f"Splitting large chunk into {sub_chunk_count} sub-chunks")
                    for i in range(0, len(chunk.page_content), max_chunk_size):
                        sub_chunk = Document(chunk.page_content[i : i + max_chunk_size])
                        sub_chunk.metadata = meta.model_dump()
                        chunk_count += 1
                        yield sub_chunk
            logger.info(f"Parsed {file_path}: generated {chunk_count} chunks")
    except Exception as e:
        logger.error(f"Error parsing markdown file {file_path}: {e}")
        raise


class MarkdownDocumentsLoader:
    """
    Loader for markdown documents from a directory.

    Recursively walks through a directory, finds all markdown files,
    and loads them as Document objects with metadata.
    """

    def __init__(
        self,
        doc_base: str,
        skip_patterns: list[str] = [],
    ):
        """
        Initialize the loader.

        Args:
            doc_base: Base directory containing markdown files
            skip_patterns: List of regex patterns for files to skip
        """
        logger.info(f"Initializing MarkdownDocumentsLoader: doc_base={doc_base}, skip_patterns={skip_patterns}")
        self.doc_base = doc_base
        self.skip_patterns = skip_patterns

    def load(
        self,
        show_progress: bool = True,
        limit: int = 0,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    ) -> Iterator[Document]:
        """
        Load markdown documents from the base directory.

        Args:
            show_progress: Whether to show progress bar
            limit: Maximum number of files to process (0 = no limit)
            max_chunk_size: Maximum size of each chunk in characters

        Yields:
            Document objects with DocumentMeta metadata
        """
        logger.info(f"Loading documents from {self.doc_base}, limit={limit}, max_chunk_size={max_chunk_size}")
        files_to_process: list[str] = []
        for root, _, files in os.walk(self.doc_base):
            for file in files:
                if file.endswith(".md") or file.endswith(".mdx"):
                    file_path = os.path.join(root, file)
                    if any(re.search(regex, file_path) for regex in self.skip_patterns):
                        logger.debug(f"Skipping file matching pattern: {file_path}")
                        continue
                    files_to_process.append(file_path)

        logger.info(f"Found {len(files_to_process)} markdown files to process")
        progress_wrapper = tqdm.tqdm if show_progress else lambda x: x
        files_to_process: list[str] = progress_wrapper(files_to_process)

        count = 0
        for file_path in files_to_process:
            for chunk in parse_md(file_path, max_chunk_size=max_chunk_size):
                yield chunk
            count += 1
            if limit > 0 and count >= limit:
                logger.info(f"Limit reached: {limit}, exiting.")
                print(f"Limit reached: {limit}, exiting.")
                exit(0)
        logger.info(f"Document loading completed, processed {count} files")
