from typing import List

import requests
from langchain_core.embeddings import Embeddings

from src.common.logger import get_logger

logger = get_logger(__name__)

# Default configuration
DEFAULT_DIMENSIONS = 1024


class RemoteOpenAI(Embeddings):
    """
    Embedding class for remote OpenAI-compatible APIs.

    Supports services like Tongyi, Baichuan, Doubao, etc.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int = DEFAULT_DIMENSIONS,
        **kwargs,
    ):
        """
        Initialize RemoteOpenAI embedding.

        Args:
            base_url: API base URL
            api_key: API key
            model: Model name
            dimensions: Embedding dimensions
            **kwargs: Additional parameters
        """
        logger.info(f"Initializing RemoteOpenAI with base_url: {base_url}, model: {model}, dimensions: {dimensions}")
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        logger.debug("RemoteOpenAI initialized successfully")

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        logger.debug(f"Embedding {len(texts)} documents using RemoteOpenAI")
        res = requests.post(
            f"{self._base_url}",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Charset": "UTF-8",
            },
            json={
                "input": texts,
                "model": self._model,
                "encoding_format": "float",
                "dimensions": self._dimensions,
            },
        )
        embeddings = []
        try:
            if res.status_code != 200:
                logger.error(f"RemoteOpenAI embedding request failed with status {res.status_code}: {res.text}")
                res.raise_for_status()
            data = res.json()
            for d in data["data"]:
                embeddings.append(d["embedding"][: self._dimensions])
            logger.debug(f"Successfully embedded {len(texts)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"Invalid response from RemoteOpenAI: {res.text}, Error: {e}")
            print("Invalid response:", res.text)
            print("Error", e)
            raise e

    def embed_query(self, text: str, **kwargs) -> List[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        logger.debug(f"Embedding query text using RemoteOpenAI, length: {len(text)}")
        return self.embed_documents([text])[0]
