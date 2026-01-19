import enum
from typing import List, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.common.config import DEFAULT_BGE_MODEL_PATH, get_bge_model_path
from src.common.logger import get_logger

logger = get_logger(__name__)


class BGEEmbedding(Embeddings):
    """
    Embedding class for BGE-M3 model.

    Supports dense, sparse, and hybrid embeddings.
    """

    # Weight configuration for hybrid scoring
    __dense_weight = 0.3
    __sparse_weight = 0.2
    __colbert_weight = 0.5

    class EmbeddingType(enum.Enum):
        Dense = "dense"
        Sparse = "sparse"
        Both = "both"

    def __init__(self, default_embedding_type: EmbeddingType = EmbeddingType.Dense):
        """
        Initialize BGE embedding model.

        Args:
            default_embedding_type: Default embedding type to use
        """
        logger.info(f"Initializing BGEEmbedding with type: {default_embedding_type.value}")
        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception as e:
            logger.error("Module FlagEmbedding not found, please execute `uv add flagembedding` first")
            print("Module FlagEmbedding not found, please execute `uv add flagembedding` first")
            exit(1)
        model_path = get_bge_model_path()
        logger.info(f"Loading BGE model from path: {model_path}")
        self.__model = BGEM3FlagModel(
            model_name_or_path=model_path,
            pooling_method="cls",
            normalize_embeddings=True,
            use_fp16=True,
        )
        self.__default_embedding_type = default_embedding_type
        logger.info("BGEEmbedding initialized successfully")

    def embed_documents(
        self,
        texts: List[str],
        *,
        embedding_type: Union[EmbeddingType, None] = None,
    ) -> Union[List[List[float]], List[dict[int, float]]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.
            embedding_type: Type of embedding to return. Defaults to EmbeddingType.Dense.

        Returns:
            List of embeddings.
        """
        embedding_type = embedding_type or self.__default_embedding_type
        logger.debug(f"Embedding {len(texts)} documents with type: {embedding_type.value}")

        do_dense = embedding_type in [
            self.EmbeddingType.Dense,
            self.EmbeddingType.Both,
        ]
        do_sparse = embedding_type in [
            self.EmbeddingType.Sparse,
            self.EmbeddingType.Both,
        ]

        embed_res = self.__model.encode(
            texts,
            batch_size=1,
            max_length=512,
            return_dense=do_dense,
            return_sparse=do_sparse,
            return_colbert_vecs=False,
        )
        if do_sparse and do_dense:
            dense = [embedding.tolist() for embedding in embed_res["dense_vecs"]]
            sparse = embed_res["lexical_weights"]
            logger.debug(f"Returning both dense and sparse embeddings for {len(texts)} documents")
            return dense, sparse
        elif do_dense:
            logger.debug(f"Returning dense embeddings for {len(texts)} documents")
            return [embedding.tolist() for embedding in embed_res["dense_vecs"]]
        else:
            logger.debug(f"Returning sparse embeddings for {len(texts)} documents")
            return embed_res["lexical_weights"]

    def embed_query(self, text: str, **kwargs) -> Union[List[float], dict[int, float]]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        logger.debug(f"Embedding query text, length: {len(text)}")
        embed_res = self.embed_documents([text], **kwargs)
        return embed_res[0]

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents.

        Args:
            query: Query text.
            documents: List of documents to rerank.

        Returns:
            Reranked documents.
        """
        if len(documents) == 0:
            logger.debug("No documents to rerank")
            return documents
        logger.info(f"Reranking {len(documents)} documents")
        pairs = list(zip([query] * len(documents), [doc.page_content for doc in documents]))
        score_res = self.__model.compute_score(
            pairs,
            batch_size=1,
            max_query_length=512,
            max_passage_length=8192,
            weights_for_different_modes=[
                self.__dense_weight,
                self.__sparse_weight,
                self.__colbert_weight,
            ],
        )
        scores = score_res["colbert+sparse+dense"]
        docs_with_scores = list(zip(scores, documents))
        combined_sorted = sorted(docs_with_scores, key=lambda x: x[0], reverse=True)
        logger.debug(f"Reranking completed, top score: {combined_sorted[0][0] if combined_sorted else 'N/A'}")
        return [doc for _, doc in combined_sorted]
