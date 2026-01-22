"""
Document embedding utilities for vector database operations.

This module provides the DocumentEmbedder class for embedding documents
into a vector database with support for batched insertion and component-based
partitioning.
"""

import uuid
from typing import Optional

from langchain_core.documents import Document
from langchain_oceanbase.vectorstores import OceanbaseVectorStore
from pyobvector import ObListPartition
from sqlalchemy import Column, Integer

from src.common.config import EmbeddingConfig, get_table_name
from src.common.db import ConnectionParams
from src.common.logger import get_logger
from src.rag.embedding import get_embedding
from src.rag.ob import (
    DEFAULT_SEARCH_LIMIT,
    component_mapping,
    get_part_list,
)

logger = get_logger(__name__)


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
        from src.rag.doc_processing import MarkdownDocumentsLoader

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
