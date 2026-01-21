"""
RAG (Retrieval-Augmented Generation) module for document processing and chat.

This module provides:
- Document metadata models and parsing utilities
- Markdown document loading with chunking support
- RAG streaming handler for retrieval-augmented chat

Module Structure:
- Constants and Configuration
- Document Models and Parsing
- Document Loading
- RAG Stream Processing
"""

import os
import re
import time
import uuid
from typing import Iterator, Optional, Union

import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from langchain_oceanbase.vectorstores import OceanbaseVectorStore
from pydantic import BaseModel
from pyobvector import ObListPartition, RangeListPartInfo
from sqlalchemy import Column, Integer

from src.agents.base import AgentBase
from src.agents.comp_analyzing_agent import prompt as caa_prompt
from src.agents.intent_guard_agent import prompt as guard_prompt
from src.agents.rag_agent import prompt as rag_prompt
from src.agents.rag_agent import prompt_en as rag_prompt_en
from src.agents.universe_rag_agent import prompt as universal_rag_prompt
from src.agents.universe_rag_agent import prompt_en as universal_rag_prompt_en
from src.common.config import (
    EmbeddingConfig,
    get_enable_rerank,
    get_table_name,
)
from src.common.db import ConnectionParams
from src.common.file_path import is_markdown_file
from src.common.logger import get_logger
from src.frontend.i18n import t
from src.rag.ob import (
    DEFAULT_COMPONENT,
    DEFAULT_RERANK_LIMIT,
    DEFAULT_SEARCH_LIMIT,
    component_mapping,
    get_part_list,
    replace_doc_url,
    supported_components,
)
from src.rag.embedding import get_embedding

logger = get_logger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================

# Document chunking settings
DEFAULT_MAX_CHUNK_SIZE = 4096

# Citation processing settings
BUFFER_SIZE_THRESHOLD = 128
DOC_CITE_PATTERN = r"(\[+\@(\d+)\]+)"

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


# =============================================================================
# Document Models
# =============================================================================

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


# =============================================================================
# Markdown Parsing Utilities
# =============================================================================

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


# =============================================================================
# Document Loading
# =============================================================================

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


# =============================================================================
# Document Embedding
# =============================================================================

class DocumentEmbedder:
    """
    Embed documents into a vector database.

    This class provides methods to embed markdown documents into an OceanBase
    vector store, with support for batched insertion and component-based
    partitioning.
    """

    def __init__(
        self,
        embedding_config: EmbeddingConfig,
        table_name: str = "",
        echo: bool = False,
    ):
        """
        Initialize the document embedder.

        Args:
            embedding_config: Embedding model configuration
            table_name: Name of the vector store table. If empty, uses config.
            echo: Whether to echo SQL queries
        """
        self._table_name = table_name or get_table_name()
        self._echo = echo
        self._embedding_config = embedding_config
        self._conn_args = ConnectionParams.from_env().to_dict()

        # Initialize embedding model based on config
        self._embeddings = get_embedding(self._embedding_config)

        # Initialize vector store
        self._vector_store = self._create_vector_store()

        logger.info(
            f"DocumentEmbedder initialized: table={table_name}, echo={echo}, "
            f"embedding_type={embedding_config.embedded_type}"
        )

    def _create_vector_store(self) -> OceanbaseVectorStore:
        """
        Create the OceanBase vector store.

        Returns:
            Configured OceanbaseVectorStore instance
        """
        return OceanbaseVectorStore(
            embedding_function=self._embeddings,
            table_name=self._table_name,
            connection_args=self._conn_args,
            metadata_field="metadata",
            extra_columns=[Column("component_code", Integer, primary_key=True)],
            partitions=ObListPartition(
                is_list_columns=False,
                list_part_infos=get_part_list(),
                list_expr="component_code",
            ),
            echo=self._echo,
        )

    def insert_batch(
        self,
        docs: list[Document],
        component: str = "observer",
    ) -> None:
        """
        Insert a batch of documents into the vector store.

        Args:
            docs: List of Document objects to insert
            component: Component name for partitioning

        Raises:
            ValueError: If the component is not found in component_mapping
        """
        code = component_mapping.get(component, 0)

        logger.info(f"Inserting batch of {len(docs)} documents for component {component}")
        self._vector_store.add_documents(
            docs,
            ids=[str(uuid.uuid4()) for _ in range(len(docs))],
            extras=[{"component_code": code} for _ in docs],
            partition_name=component,
        )
        logger.info(f"Successfully inserted {len(docs)} documents for component {component}")

    def embed_from_directory(
        self,
        doc_base: str,
        component: str = "observer",
        skip_patterns: Optional[list[str]] = None,
        batch_size: int = 64,
        limit: int = 0,
    ) -> int:
        """
        Embed documents from a directory into the vector store.

        Args:
            doc_base: Path to the directory containing markdown documents
            component: Component name for partitioning
            skip_patterns: List of regex patterns for files to skip
            batch_size: Number of documents to insert in each batch
            limit: Maximum number of documents to process

        Returns:
            Total number of documents embedded
        """
        if skip_patterns is None:
            skip_patterns = []

        logger.info(
            f"Starting document embedding: doc_base={doc_base}, "
            f"component={component}, limit={limit}, batch_size={batch_size}"
        )

        loader = MarkdownDocumentsLoader(
            doc_base=doc_base,
            skip_patterns=skip_patterns,
        )

        batch = []
        total_docs = 0

        for doc in loader.load(limit=limit):
            batch.append(doc)
            if len(batch) >= batch_size:
                self.insert_batch(batch, component=component)
                total_docs += len(batch)
                batch = []

        # Insert remaining documents
        if batch:
            self.insert_batch(batch, component=component)
            total_docs += len(batch)

        logger.info(f"Document embedding completed: total documents embedded={total_docs}")
        return total_docs

    def doc_search(
        self,
        query: str,
        partition_names: Optional[list[str]] = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[Document]:
        """
        Search for documents related to the query using text similarity.

        Args:
            query: Search query text
            partition_names: Optional list of partition names to search in
            limit: Maximum number of documents to return

        Returns:
            List of relevant documents
        """
        logger.info(f"Searching documents: query length={len(query)}, partition_names={partition_names}, limit={limit}")
        docs = self._vector_store.similarity_search(
            query=query,
            k=limit,
            partition_names=partition_names,
        )
        logger.debug(f"Found {len(docs)} documents")
        return docs

    def doc_search_by_vector(
        self,
        vector: list[float],
        partition_names: Optional[list[str]] = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[Document]:
        """
        Search for documents related to the query using vector similarity.

        Args:
            vector: Query embedding vector
            partition_names: Optional list of partition names to search in
            limit: Maximum number of documents to return

        Returns:
            List of relevant documents
        """
        logger.debug(f"Searching documents by vector: vector_dim={len(vector)}, partition_names={partition_names}, limit={limit}")
        docs = self._vector_store.similarity_search_by_vector(
            embedding=vector,
            k=limit,
            partition_names=partition_names,
        )
        logger.debug(f"Found {len(docs)} documents")
        return docs

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query string into a vector.

        Args:
            query: The query string to embed

        Returns:
            The embedding vector
        """
        if self._embeddings is None:
            raise ValueError("Embeddings model is not initialized")
        return self._embeddings.embed_query(query)

    def rerank(
        self,
        query: str,
        docs: list[Document],
    ) -> list[Document]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: The query string
            docs: List of documents to rerank

        Returns:
            Reranked list of documents
        """
        if self._embeddings is None or not hasattr(self._embeddings, "rerank"):
            logger.warning("Rerank not available, returning original documents")
            return docs
        return self._embeddings.rerank(query, docs)

    def has_rerank(self) -> bool:
        """
        Check if the embedding model supports reranking.

        Returns:
            True if reranking is available
        """
        return (
            self._embeddings is not None
            and hasattr(self._embeddings, "rerank")
            and callable(getattr(self._embeddings, "rerank", None))
        )


# =============================================================================
# RAG Stream Processing
# =============================================================================

class RAGStreamHandler:
    """
    Handler for RAG (Retrieval-Augmented Generation) streaming responses.

    This class encapsulates the complete RAG pipeline:
    1. Query intent analysis
    2. Component detection
    3. Document search and retrieval
    4. Optional reranking
    5. LLM response generation
    6. Citation extraction and reference formatting

    Example:
        handler = RAGStreamHandler(
            query="How to optimize OceanBase?",
            chat_history=[],
            llm_model="gpt-4",
        )
        for chunk in handler.stream():
            print(chunk)
    """

    def __init__(
        self,
        query: str,
        chat_history: list[dict],
        llm_model: str,
        embedding_config: EmbeddingConfig,
        *,
        suffixes: Optional[list[str]] = None,
        universal_rag: bool = False,
        rerank: bool = False,
        search_docs: bool = True,
        lang: str = "zh",
        show_refs: bool = True,
    ):
        """
        Initialize the RAG stream handler.

        Args:
            query: User query string.
            chat_history: List of previous chat messages.
            llm_model: Name of the LLM model to use.
            embedding_config: Embedding model configuration.
            suffixes: Additional suffixes to append to response.
            universal_rag: Whether to use universal RAG mode.
            rerank: Whether to enable document reranking.
            search_docs: Whether to search documents.
            lang: Language code ("zh" or "en").
            show_refs: Whether to show reference list.
        """
        self.query = query
        self.chat_history = chat_history
        self.llm_model = llm_model
        self.suffixes = suffixes or []
        self.universal_rag = universal_rag
        self.rerank = rerank
        self.search_docs = search_docs
        self.lang = lang
        self.show_refs = show_refs

        # Internal state
        self._start_time: float = 0
        self._docs: list[Document] = []
        self._rag_agent: Optional[AgentBase] = None

        # Initialize document embedder
        self._doc_embedder = DocumentEmbedder(embedding_config=embedding_config)

    def stream(self) -> Iterator[Union[str, AIMessageChunk]]:
        """
        Execute the RAG pipeline and stream responses.

        Yields:
            Progress messages (str) and response chunks (AIMessageChunk).
        """
        self._start_time = time.time()
        logger.info(
            f"Starting RAG stream: query_len={len(self.query)}, "
            f"universal_rag={self.universal_rag}, rerank={self.rerank}, "
            f"search_docs={self.search_docs}, lang={self.lang}"
        )

        # Handle no-search mode
        if not self.search_docs:
            yield from self._handle_no_search_mode()
            return

        # Execute appropriate RAG pipeline
        if self.universal_rag:
            yield from self._execute_universal_rag()
        else:
            result = yield from self._execute_oceanbase_rag()
            if result == "early_return":
                return

        # Generate LLM response
        yield from self._generate_response()

        # Log completion
        elapsed = time.time() - self._start_time
        logger.info(f"RAG stream completed in {elapsed:.2f} seconds")

    # -------------------------------------------------------------------------
    # Pipeline Modes
    # -------------------------------------------------------------------------

    def _handle_no_search_mode(self) -> Iterator[AIMessageChunk]:
        """Handle RAG when document search is disabled."""
        logger.info("Document search disabled, using no-search mode")
        yield None

        prompt = self._get_universal_prompt()
        agent = AgentBase(prompt=prompt, llm_model=self.llm_model)
        yield from agent.stream(self.query, self.chat_history, document_snippets="")

    def _execute_universal_rag(self) -> Iterator[str]:
        """Execute universal RAG mode pipeline."""
        logger.info("Using universal RAG mode")

        yield self._progress_msg("embedding_query")
        query_embedded = self._doc_embedder.embed_query(self.query)

        yield self._progress_msg("searching_docs")
        self._docs = self._doc_embedder.doc_search_by_vector(query_embedded, limit=DEFAULT_SEARCH_LIMIT)
        logger.info(f"Universal RAG: found {len(self._docs)} documents")

    def _execute_oceanbase_rag(self) -> Iterator[Union[str, None]]:
        """
        Execute OceanBase-specific RAG mode pipeline.

        Yields:
            Progress messages.

        Returns:
            "early_return" if query is not OceanBase-related, None otherwise.
        """
        logger.info("Using OceanBase-specific RAG mode")

        # Step 1: Analyze query intent
        yield self._progress_msg("analyzing_intent")
        query_type = self._analyze_query_intent()

        # Initialize RAG agent
        prompt = rag_prompt if self.lang == "zh" else rag_prompt_en
        self._rag_agent = AgentBase(prompt=prompt, llm_model=self.llm_model)

        # Handle non-OceanBase queries
        if query_type == "Chat":
            yield from self._handle_non_oceanbase_query()
            return "early_return"

        # Step 2: Analyze related components
        yield self._progress_msg("analyzing_components")
        related_comps = self._analyze_related_components()

        yield t(
            "list_related_components",
            self.lang,
            ", ".join(related_comps),
        ) + self._elapsed_tips()

        # Step 3: Search documents
        yield from self._search_documents_by_components(related_comps)

        return None

    def _handle_non_oceanbase_query(self) -> Iterator[Union[str, AIMessageChunk]]:
        """Handle queries that are not OceanBase-related."""
        logger.info("Query type is Chat, not OceanBase-related")
        yield self._progress_msg("no_oceanbase")

        yield from self._rag_agent.stream(
            self.query, self.chat_history, document_snippets=""
        )

    # -------------------------------------------------------------------------
    # Query Analysis
    # -------------------------------------------------------------------------

    def _analyze_query_intent(self) -> str:
        """
        Analyze the intent of the query.

        Returns:
            Query type ("Chat" or "Features").
        """
        logger.info(f"Analyzing query intent: query_len={len(self.query)}")

        agent = AgentBase(prompt=guard_prompt, llm_model=self.llm_model)
        result = agent.invoke_json(self.query)

        intent_type = (
            result.get("type", "Features")
            if hasattr(result, "get")
            else "Features"
        )
        logger.info(f"Query intent: {intent_type}")
        return intent_type

    def _analyze_related_components(self) -> list[str]:
        """
        Analyze which OceanBase components are related to the query.

        Returns:
            List of validated component names.
        """
        logger.info(
            f"Analyzing components: query_len={len(self.query)}, "
            f"history_len={len(self.chat_history)}"
        )

        history_text = self._extract_user_input()
        combined_query = "\n".join([history_text, self.query])

        agent = AgentBase(prompt=caa_prompt, llm_model=self.llm_model)
        result = agent.invoke_json(
            query=combined_query,
            background_history=[],
            supported_components=supported_components,
        )

        raw_components = (
            result.get("components", [DEFAULT_COMPONENT])
            if hasattr(result, "get")
            else [DEFAULT_COMPONENT]
        )

        components = self._validate_components(raw_components)
        logger.info(f"Related components: {components}")
        return components

    def _validate_components(self, raw_components: list[str]) -> list[str]:
        """
        Validate and deduplicate component names.

        Args:
            raw_components: Raw component names from analysis.

        Returns:
            Validated and deduplicated component list.
        """
        visited = set()
        valid = []

        for comp in raw_components:
            if comp in supported_components and comp not in visited:
                visited.add(comp)
                valid.append(comp)

        # Ensure observer is always included
        if DEFAULT_COMPONENT not in valid:
            valid.append(DEFAULT_COMPONENT)

        return valid

    # -------------------------------------------------------------------------
    # Document Search
    # -------------------------------------------------------------------------

    def _search_documents_by_components(
        self,
        components: list[str],
    ) -> Iterator[str]:
        """
        Search for documents across related components.

        Args:
            components: List of component names to search.

        Yields:
            Progress messages.
        """
        rerankable = self._is_rerankable()
        limit = (
            DEFAULT_SEARCH_LIMIT
            if rerankable
            else max(3, 13 - 3 * len(components))
        )

        logger.info(
            f"Searching documents: components={components}, "
            f"limit={limit}, rerankable={rerankable}"
        )

        yield self._progress_msg("embedding_query")
        query_embedded = self._doc_embedder.embed_query(self.query)

        # Search each component
        all_docs = []
        for comp in components:
            yield t("searching_docs_for", self.lang, comp) + self._elapsed_tips()

            comp_docs = self._doc_embedder.doc_search_by_vector(
                query_embedded,
                partition_names=[comp],
                limit=limit,
            )
            all_docs.extend(comp_docs)
            logger.debug(f"Found {len(comp_docs)} docs for {comp}")

        # Rerank if applicable
        if rerankable and len(components) > 1:
            yield self._progress_msg("reranking_docs")
            all_docs = self._doc_embedder.rerank(self.query, all_docs)
            self._docs = all_docs[:DEFAULT_RERANK_LIMIT]
            logger.debug(f"Reranked to {len(self._docs)} documents")
        else:
            self._docs = all_docs
            logger.debug(f"Using {len(self._docs)} documents without reranking")

    def _is_rerankable(self) -> bool:
        """Check if reranking is available and enabled."""
        return (
            (self.rerank or get_enable_rerank())
            and self._doc_embedder.has_rerank()
        )

    # -------------------------------------------------------------------------
    # Response Generation
    # -------------------------------------------------------------------------

    def _generate_response(self) -> Iterator[Union[str, AIMessageChunk]]:
        """Generate LLM response with citations and references."""
        yield self._progress_msg("llm_thinking")
        logger.info(f"Generating response with {len(self._docs)} documents")

        # Prepare document context
        docs_content = self._format_documents_content()

        # Get response iterator
        if self.universal_rag:
            prompt = self._get_universal_prompt()
            agent = AgentBase(prompt=prompt, llm_model=self.llm_model)
            ans_iter = agent.stream(
                self.query, self.chat_history, document_snippets=docs_content
            )
        else:
            ans_iter = self._rag_agent.stream(
                self.query, self.chat_history, document_snippets=docs_content
            )

        # Process stream and extract citations
        pruned_refs = yield from self._process_response_stream(ans_iter)

        # Generate references
        if self.show_refs:
            yield from self._generate_references(pruned_refs)

        # Append suffixes
        for suffix in self.suffixes:
            yield AIMessageChunk(content=suffix + "\n")

    def _format_documents_content(self) -> str:
        """Format documents into a single content string."""
        return "\n=====\n".join(
            f"文档片段:\n\n{doc.page_content}" for doc in self._docs
        )

    def _process_response_stream(
        self,
        ans_iter: Iterator[AIMessageChunk],
    ) -> Iterator[Union[AIMessageChunk, None]]:
        """
        Process response stream and extract document citations.

        Args:
            ans_iter: Iterator of response chunks.

        Yields:
            Processed response chunks.

        Returns:
            List of pruned references.
        """
        logger.debug("Processing response stream")

        visited: dict[str, int] = {}
        pruned_refs: list[str] = []
        buffer = ""
        first_token_sent = False

        for chunk in ans_iter:
            buffer += chunk.content

            # Process citations in buffer
            if "[" in buffer and len(buffer) < BUFFER_SIZE_THRESHOLD:
                buffer, new_refs = self._process_citations(buffer, visited)
                pruned_refs.extend(new_refs)

            # Signal first token
            if not first_token_sent:
                first_token_sent = True
                yield None

            yield AIMessageChunk(content=buffer)
            buffer = ""

        # Flush remaining buffer
        if buffer:
            yield AIMessageChunk(content=buffer)

        logger.debug(f"Extracted {len(pruned_refs)} references")
        return pruned_refs

    def _process_citations(
        self,
        buffer: str,
        visited: dict[str, int],
    ) -> tuple[str, list[str]]:
        """
        Process and replace citation markers in buffer.

        Args:
            buffer: Text buffer to process.
            visited: Dictionary of visited URLs to their indices.

        Returns:
            Tuple of (processed buffer, new references).
        """
        new_refs = []
        matches = re.findall(DOC_CITE_PATTERN, buffer)

        if not matches:
            return buffer, new_refs

        # Sort by original text (reverse to handle overlapping replacements)
        sorted_matches = sorted(matches, key=lambda x: x[0], reverse=True)

        for original, order in sorted_matches:
            doc = self._docs[int(order) - 1]
            meta = DocumentMeta.model_validate(doc.metadata)
            doc_url = replace_doc_url(meta.doc_url)

            # Get or assign reference index
            if doc_url in visited:
                idx = visited[doc_url]
            else:
                idx = len(visited) + 1
                visited[doc_url] = idx
                ref_text = f"{idx}. [{meta.doc_name}]({doc_url})"
                new_refs.append(ref_text)

            # Replace citation marker
            ref_link = f"[[{idx}]]({doc_url})"
            buffer = buffer.replace(original, ref_link)

        return buffer, new_refs

    def _generate_references(
        self,
        pruned_refs: list[str],
    ) -> Iterator[AIMessageChunk]:
        """
        Generate reference list.

        Args:
            pruned_refs: Pre-processed references from citation extraction.

        Yields:
            Reference content chunks.
        """
        logger.debug(
            f"Generating references: pruned={len(pruned_refs)}, docs={len(self._docs)}"
        )

        ref_tip = t("ref_tips", self.lang)

        if pruned_refs:
            yield AIMessageChunk(content="\n\n" + ref_tip)
            for ref in pruned_refs:
                yield AIMessageChunk(content="\n" + ref)

        elif self._docs:
            yield AIMessageChunk(content="\n\n" + ref_tip)
            visited: dict[str, bool] = {}

            for doc in self._docs:
                meta = DocumentMeta.model_validate(doc.metadata)
                doc_url = replace_doc_url(meta.doc_url)

                if doc_url in visited:
                    continue
                visited[doc_url] = True

                count = len(visited)
                ref_text = f"{count}. [{meta.doc_name}]({doc_url})"
                yield AIMessageChunk(content="\n" + ref_text)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _get_universal_prompt(self) -> str:
        """Get the appropriate universal RAG prompt for the language."""
        return universal_rag_prompt if self.lang == "zh" else universal_rag_prompt_en

    def _extract_user_input(self) -> str:
        """Extract all user messages from chat history."""
        return "\n".join(
            msg["content"] for msg in self.chat_history if msg["role"] == "user"
        )

    def _progress_msg(self, key: str) -> str:
        """Generate a progress message with elapsed time."""
        return t(key, self.lang) + self._elapsed_tips()

    def _elapsed_tips(self) -> str:
        """Get elapsed time tips string."""
        elapsed = time.time() - self._start_time
        return t("time_elapse", self.lang, elapsed)


# =============================================================================
# Public API Functions
# =============================================================================

def get_elapsed_tips(
    start_time: float,
    end_time: Optional[float] = None,
    /,
    lang: str = "zh",
) -> str:
    """
    Get elapsed time message.

    Args:
        start_time: Start timestamp.
        end_time: End timestamp (defaults to current time).
        lang: Language code.

    Returns:
        Formatted elapsed time message.
    """
    end_time = end_time or time.time()
    elapsed_time = end_time - start_time
    return t("time_elapse", lang, elapsed_time)


def extract_users_input(history: list[dict]) -> str:
    """
    Extract all user messages from chat history.

    Args:
        history: List of message dictionaries.

    Returns:
        Concatenated user input text.
    """
    return "\n".join(msg["content"] for msg in history if msg["role"] == "user")


def doc_rag_stream(
    query: str,
    chat_history: list[dict],
    llm_model: str,
    embedding_config: EmbeddingConfig,
    suffixes: Optional[list[str]] = None,
    universal_rag: bool = False,
    rerank: bool = False,
    search_docs: bool = True,
    lang: str = "zh",
    show_refs: bool = True,
    **kwargs,
) -> Iterator[Union[str, AIMessageChunk]]:
    """
    Stream the response from the RAG model.

    This function provides a convenient interface to the RAGStreamHandler class.
    It orchestrates the complete RAG pipeline including query analysis,
    document retrieval, and response generation.

    Args:
        query: User query string.
        chat_history: List of previous messages.
        llm_model: LLM model name.
        embedding_config: Embedding model configuration.
        suffixes: Additional suffixes to append.
        universal_rag: Whether to use universal RAG mode.
        rerank: Whether to rerank documents.
        search_docs: Whether to search documents.
        lang: Language code ("zh" or "en").
        show_refs: Whether to show references.
        **kwargs: Additional parameters (reserved for future use).

    Yields:
        Progress messages (str) and response chunks (AIMessageChunk).

    Example:
        for chunk in doc_rag_stream("What is OceanBase?", [], "gpt-4", embedding_config):
            if isinstance(chunk, str):
                print(f"Progress: {chunk}")
            elif chunk is not None:
                print(chunk.content, end="")
    """
    handler = RAGStreamHandler(
        query=query,
        chat_history=chat_history,
        llm_model=llm_model,
        embedding_config=embedding_config,
        suffixes=suffixes,
        universal_rag=universal_rag,
        rerank=rerank,
        search_docs=search_docs,
        lang=lang,
        show_refs=show_refs,
    )
    yield from handler.stream()
