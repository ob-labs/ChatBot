import enum
import os
from typing import List, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.common.config import (
    DEFAULT_BGE_HF_REPO_ID,
    DEFAULT_BGE_MODEL_PATH,
    get_bge_model_path,
    get_hf_endpoint,
)
from src.common.logger import get_logger

logger = get_logger(__name__)

# Maximum length for embedding
MAX_LENGTH = 4096


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

    # Add a function to ensure the model gets downloaded if the model_path is empty.
    # Note: There is currently a known bug: Directly calling this will result in the error shown in code block (1010-1011)
    def ensure_model_downloaded(self, model_path):
        """
        Ensures that the BGE-M3 model is downloaded if model_path is empty.

        If model_path is empty, attempts to download the model via BGEM3FlagModel.
        If download fails or an exception is raised, logs the error with details.

        Warning:
            THERE IS A BUG: Direct call to BGEM3FlagModel when downloading
            may produce an error as described in code block (1010-1011).
        """
        if not model_path and os.path.exists(model_path) == False:
            logger.info(
                "model_path is empty: attempting to download with BGEM3FlagModel"
            )
            try:
                from FlagEmbedding import BGEM3FlagModel

                BGEM3FlagModel(
                    model_name_or_path=DEFAULT_BGE_HF_REPO_ID,
                    pooling_method="cls",
                    normalize_embeddings=True,
                    use_fp16=True,
                )
                logger.info(
                    "BGEM3FlagModel download attempted (success assumed if no error)."
                )
            except Exception as e:
                logger.error(
                    f"Failed to download BGE-M3 model with BGEM3FlagModel: {e}"
                )
                

    def resolve_model_name_or_path(self, model_path: str) -> str:
        """
        Resolve the best model_name_or_path from a given model_path.

        If the local path exists, attempts to find a usable snapshot directory.
        If not found or invalid, fall back to default Hugging Face repo ID.

        Returns:
            str: path or repo id to use for model loading
        """
        if os.path.exists(model_path):
            # Check if the second-to-last path component is 'snapshots'
            path_parts = model_path.rstrip(os.sep).split(os.sep)
            if len(path_parts) >= 2 and path_parts[-2] == "snapshots":
                # Already contains snapshots in the path, use it directly
                return model_path
            else:
                # Check if model_path contains a 'snapshots' subdirectory
                snapshots_dir = os.path.join(model_path, "snapshots")
                if os.path.exists(snapshots_dir) and os.path.isdir(snapshots_dir):
                    # Find the first version subdirectory in snapshots
                    try:
                        version_dirs = [
                            d
                            for d in os.listdir(snapshots_dir)
                            if os.path.isdir(os.path.join(snapshots_dir, d))
                        ]
                        if version_dirs:
                            version = version_dirs[0]  # Use the first version found
                            return os.path.join(model_path, "snapshots", version)
                        else:
                            # snapshots directory exists but is empty, use default repo ID
                            logger.warning(
                                f"snapshots directory exists but is empty: {snapshots_dir}. "
                                f"Using Hugging Face repo ID: {DEFAULT_BGE_HF_REPO_ID}"
                            )
                            return DEFAULT_BGE_HF_REPO_ID
                    except Exception as e:
                        logger.warning(
                            f"Error reading snapshots directory {snapshots_dir}: {e}. "
                            f"Using Hugging Face repo ID: {DEFAULT_BGE_HF_REPO_ID}"
                        )
                        return DEFAULT_BGE_HF_REPO_ID
                else:
                    # No snapshots directory found, use default repo ID
                    return DEFAULT_BGE_HF_REPO_ID
        else:
            logger.warning(
                f"Local model path does not exist: {model_path}. "
                f"Using Hugging Face repo ID: {DEFAULT_BGE_HF_REPO_ID}"
            )
            return DEFAULT_BGE_HF_REPO_ID
        
    def __init__(self, default_embedding_type: EmbeddingType = EmbeddingType.Dense):
        """
        Initialize BGE embedding model.

        Args:
            default_embedding_type: Default embedding type to use
        """
        logger.info(
            f"Initializing BGEEmbedding with type: {default_embedding_type.value}"
        )
        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception as e:
            logger.error(
                "Module FlagEmbedding not found, please execute `uv add flagembedding` first"
            )
            print(
                "Module FlagEmbedding not found, please execute `uv add flagembedding` first"
            )
            exit(1)

        # Set HF_ENDPOINT environment variable for HuggingFace mirror support
        # This must be set before importing huggingface_hub or initializing models
        hf_endpoint = get_hf_endpoint()
        if hf_endpoint and "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = hf_endpoint
            logger.info(f"Set HF_ENDPOINT to: {hf_endpoint}")

        model_path = get_bge_model_path()
        
        self.ensure_model_downloaded(model_path)
        
        model_name_or_path = self.resolve_model_name_or_path(model_path)

        logger.info(f"Loading BGE model from: {model_name_or_path}")
        self.__model = BGEM3FlagModel(
            model_name_or_path=model_name_or_path,
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
        logger.debug(
            f"Embedding {len(texts)} documents with type: {embedding_type.value}"
        )

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
            max_length=MAX_LENGTH,
            return_dense=do_dense,
            return_sparse=do_sparse,
            return_colbert_vecs=False,
        )
        if do_sparse and do_dense:
            dense = [embedding.tolist() for embedding in embed_res["dense_vecs"]]
            sparse = embed_res["lexical_weights"]
            logger.debug(
                f"Returning both dense and sparse embeddings for {len(texts)} documents"
            )
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
        pairs = list(
            zip([query] * len(documents), [doc.page_content for doc in documents])
        )
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
        logger.debug(
            f"Reranking completed, top score: {combined_sorted[0][0] if combined_sorted else 'N/A'}"
        )
        return [doc for _, doc in combined_sorted]
