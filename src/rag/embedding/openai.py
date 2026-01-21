from typing import Optional

from langchain_openai import OpenAIEmbeddings

from src.common.config import DEFAULT_EMBEDDING_DIMENSIONS
from src.common.logger import get_logger

logger = get_logger(__name__)

# Alias for backward compatibility
DEFAULT_DIMENSIONS = DEFAULT_EMBEDDING_DIMENSIONS


def create_openai_embedding(
    base_url: str,
    api_key: str,
    model: str,
    dimensions: Optional[int] = None,
    **kwargs,
) -> OpenAIEmbeddings:
    """
    Create OpenAI-compatible embedding instance.

    Supports services like Tongyi, Baichuan, Doubao, ZhipuAI, etc.

    Args:
        base_url: API base URL
        api_key: API key
        model: Model name
        dimensions: Embedding dimensions (optional)
        **kwargs: Additional parameters passed to OpenAIEmbeddings

    Returns:
        OpenAIEmbeddings instance
    """
    logger.info(f"Creating OpenAI embedding: base_url={base_url}, model={model}, dimensions={dimensions}")

    embedding_kwargs = {
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
        **kwargs,
    }

    if dimensions is not None:
        embedding_kwargs["dimensions"] = dimensions

    embedding = OpenAIEmbeddings(**embedding_kwargs)
    logger.debug("OpenAI embedding instance created successfully")
    return embedding
