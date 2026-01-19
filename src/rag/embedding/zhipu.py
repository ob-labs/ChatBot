from langchain_openai import OpenAIEmbeddings

from src.common.config import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    get_api_key,
    get_embedding_base_url,
    get_embedding_model,
)
from src.common.logger import get_logger

logger = get_logger(__name__)


def get_embedding() -> OpenAIEmbeddings:
    """
    Get ZhipuAI embedding instance.

    Returns:
        OpenAIEmbeddings instance configured for ZhipuAI
    """
    model = get_embedding_model()
    base_url = get_embedding_base_url()
    logger.info(f"Creating ZhipuAI embedding instance: model={model}, base_url={base_url}")
    embedding = OpenAIEmbeddings(
        api_key=get_api_key(),
        model=model,
        base_url=base_url,
    )
    logger.debug("ZhipuAI embedding instance created successfully")
    return embedding
