from typing import Dict

from langchain_core.embeddings import Embeddings
from langchain_oceanbase.embedding_utils import DefaultEmbeddingFunctionAdapter

from src.common.config import (
    EmbeddedType,
    EmbeddingConfig,
)
from src.common.logger import get_logger

from .ollama import create_ollama_embedding
from .openai import create_openai_embedding

logger = get_logger(__name__)

# Global embeddings cache (key: EmbeddingConfig, value: embedding instance)
__embeddings: Dict[EmbeddingConfig, Embeddings] = {}


def get_embedding(config: EmbeddingConfig) -> Embeddings:
    """
    Get or create embedding instance based on EmbeddingConfig.

    Uses a global cache to store embedding instances by config.
    If an embedding for the given config doesn't exist, creates it.

    Args:
        config: Embedding model configuration

    Returns:
        Embedding instance
    """
    global __embeddings

    # Return cached embedding if exists
    if config in __embeddings:
        logger.debug(f"Returning cached embedding for: {config}")
        return __embeddings[config]

    # Create new embedding based on config type
    embedding: Embeddings
    if config.embedded_type == EmbeddedType.DEFAULT:
        logger.info(f"Use default embedded model")
        embedding = DefaultEmbeddingFunctionAdapter()
    elif config.embedded_type == EmbeddedType.OLLAMA:
        logger.info(f"Creating Ollama embedding: model={config.model}")
        embedding = create_ollama_embedding(
            base_url=config.base_url,
            model=config.model,
        )
    elif config.embedded_type == EmbeddedType.OPENAI_EMBEDDING:
        logger.info(f"Creating OpenAI embedding: model={config.model}")
        embedding = create_openai_embedding(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
        )
    else:
        logger.info(f"Use default embedded model, but input parameter {config.embedded_type} is not valid")
        return None

    # Cache and return
    __embeddings[config] = embedding
    logger.info(f"Embedding instance created and cached for: {config}")
    return embedding
