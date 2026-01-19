from FlagEmbedding import BGEM3FlagModel

from src.common.config import DEFAULT_BGE_MODEL_PATH, get_bge_model_path
from src.common.logger import get_logger

logger = get_logger(__name__)

# Load and initialize BGE-M3 model
# This script is used to pre-download and prepare the model
model_path = get_bge_model_path()
logger.info(f"Loading BGE-M3 model from path: {model_path}")
BGEM3FlagModel(
    model_name_or_path=model_path,
    pooling_method="cls",
    normalize_embeddings=True,
    use_fp16=True,
)
logger.info("BGEM3FlagModel loaded successfully")

print(
    """
===================================
BGEM3FlagModel loaded successfully！
===================================
"""
)
