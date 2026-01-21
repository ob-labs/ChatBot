from langchain_ollama import OllamaEmbeddings

from src.common.config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL
from src.common.logger import get_logger

logger = get_logger(__name__)


def create_ollama_embedding(
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
) -> OllamaEmbeddings:
    """
    Create an OllamaEmbeddings instance.

    Args:
        base_url: Ollama server base URL (default: http://localhost:11434)
        model: Model name (default: bge-m3)

    Returns:
        OllamaEmbeddings instance
    """
    logger.info(f"Creating OllamaEmbeddings with base_url: {base_url}, model: {model}")
    return OllamaEmbeddings(
        base_url=base_url,
        model=model,
    )
