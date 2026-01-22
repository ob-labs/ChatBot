"""
RAG (Retrieval-Augmented Generation) module for document processing and chat.

This module provides the public API for RAG functionality. The implementation
has been split into multiple modules for better organization:

- doc_processing: Markdown parsing and loading utilities (includes DocumentMeta)
- doc_embedder: Document embedding into vector database
- rag_graph: LangGraph pipeline implementation (includes RAGState and RAGStreamHandler)

This module maintains backward compatibility by re-exporting all public APIs.
"""

from typing import Iterator, Optional, Union

from langchain_core.messages import AIMessageChunk

from src.common.config import EmbeddingConfig

# Re-export all public APIs for backward compatibility
from src.rag.doc_embedder import DocumentEmbedder
from src.rag.doc_processing import (
    DEFAULT_MAX_CHUNK_SIZE,
    DocumentMeta,
    MarkdownDocumentsLoader,
    parse_md,
)
from src.rag.rag_graph import RAGState, RAGStreamHandler, create_rag_graph

__all__ = [
    # Models
    "DocumentMeta",
    "RAGState",
    # Document processing
    "parse_md",
    "MarkdownDocumentsLoader",
    "DEFAULT_MAX_CHUNK_SIZE",
    # Document embedding
    "DocumentEmbedder",
    # RAG handler
    "RAGStreamHandler",
    # Public API functions
    "doc_rag_stream",
    "extract_users_input",
]


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
