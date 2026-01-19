import os
import dotenv

dotenv.load_dotenv()

from FlagEmbedding import BGEM3FlagModel

BGEM3FlagModel(
    model_name_or_path=os.getenv("BGE_MODEL_PATH", "BAAI/bge-m3"),
    pooling_method="cls",
    normalize_embeddings=True,
    use_fp16=True,
)

print(
    """
===================================
BGEM3FlagModel loaded successfully！
===================================
"""
)
