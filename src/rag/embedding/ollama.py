from typing import List, Union

import requests
from langchain_core.embeddings import Embeddings

from src.common.logger import get_logger

logger = get_logger(__name__)

# Default configuration
DEFAULT_OLLAMA_MODEL = "bge-m3"


class OllamaEmbedding(Embeddings):
    """
    Embedding class for Ollama API.
    """

    def __init__(self, url: str, token: str, model: str = DEFAULT_OLLAMA_MODEL):
        """
        Initialize Ollama embedding.

        Args:
            url: Ollama API URL
            token: API token
            model: Model name
        """
        logger.info(f"Initializing OllamaEmbedding with url: {url}, model: {model}")
        self.url = url
        self.model = model
        self._token = token
        logger.debug("OllamaEmbedding initialized successfully")

    def embed_documents(
        self,
        texts: List[str],
    ) -> Union[List[List[float]], List[dict[int, float]]]:
        logger.debug(f"Embedding {len(texts)} documents using Ollama")
        res = requests.post(
            self.url,
            json={"model": self.model, "input": texts},
            headers={
                "X-Token": self._token or "token",
            },
        )
        if res.status_code != 200:
            logger.error(f"Ollama embedding request failed with status {res.status_code}: {res.text}")
            res.raise_for_status()
        data = res.json()
        logger.debug(f"Successfully embedded {len(texts)} documents")
        return data["embeddings"]

    def embed_query(self, text: str, **kwargs) -> Union[List[float], dict[int, float]]:
        logger.debug(f"Embedding query text using Ollama, length: {len(text)}")
        return self.embed_documents([text])[0]
