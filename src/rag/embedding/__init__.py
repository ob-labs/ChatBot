from typing import Optional

from langchain_core.embeddings import Embeddings

from src.common.logger import get_logger

from .bge import BGEEmbedding
from .ollama import DEFAULT_OLLAMA_MODEL, OllamaEmbedding
from .remote_openai import DEFAULT_DIMENSIONS, RemoteOpenAI

logger = get_logger(__name__)

# Global embedding instance (singleton pattern)
__embedding = None


def get_embedding(
    ollama_url: Optional[str] = None,
    ollama_token: Optional[str] = None,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Embeddings:
    """
    Get or create an embedding instance (singleton pattern).

    Priority order:
    1. OllamaEmbedding (if ollama_url and ollama_token provided)
    2. RemoteOpenAI (if base_url, api_key, and model provided)
    3. BGEEmbedding (default fallback)

    Args:
        ollama_url: Ollama API URL
        ollama_token: Ollama API token
        ollama_model: Ollama model name
        base_url: Remote OpenAI-compatible API base URL
        api_key: API key for remote service
        model: Model name for remote service

    Returns:
        Embedding instance
    """
    global __embedding
    if __embedding is not None:
        logger.debug("Returning existing embedding instance")
        return __embedding
    if all([ollama_url, ollama_token]):
        logger.info("Using OllamaEmbedding")
        print("Using OllamaEmbedding")
        __embedding = OllamaEmbedding(
            ollama_url,
            ollama_token,
            ollama_model,
        )
    elif all([base_url, api_key, model]):
        logger.info("Using RemoteOpenAI")
        print("Using RemoteOpenAI")
        __embedding = RemoteOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
    else:
        logger.info("Using BGEEmbedding")
        print("Using BGEEmbedding")
        __embedding = BGEEmbedding()
    logger.debug("Embedding instance created and cached")
    return __embedding


# Export all classes and functions
__all__ = [
    "BGEEmbedding",
    "OllamaEmbedding",
    "RemoteOpenAI",
    "get_embedding",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_DIMENSIONS",
]
