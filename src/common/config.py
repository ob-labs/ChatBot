"""
Configuration management module.

Centralized management of all environment variables and default values.
All os.getenv operations should be performed through this module.
"""
import getpass
import os
from pathlib import Path
from enum import Enum

import dotenv
from pydantic import BaseModel

from src.common.logger import get_logger

logger = get_logger(__name__)

# Load .env file
dotenv.load_dotenv()
logger.info("Configuration module loaded, .env file loaded")

# ==================== LLM Related Configuration ====================
DEFAULT_LLM_MODEL = "qwen-plus"
DEFAULT_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 2000
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_RETRY_COUNT = 1

# ==================== Database Related Configuration ====================
DEFAULT_DB_HOST = "127.0.0.1"
DEFAULT_DB_PORT = "2881"
DEFAULT_DB_USER = "root@test"
DEFAULT_DB_NAME = "test"
DEFAULT_TABLE_NAME = "corpus"

# ==================== Embedding Model Related Configuration ====================
# Use local snapshot path to avoid re-downloading attempts
# The model files are cached at ~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/<version>/
DEFAULT_BGE_MODEL_PATH = str(Path.home() / ".cache/huggingface/hub/models--BAAI--bge-m3")
DEFAULT_BGE_HF_REPO_ID = "BAAI/bge-m3"
DEFAULT_EMBEDDING_MODEL = "embedding-2"
DEFAULT_EMBEDDING_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_TOKEN = "test"
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Ollama embedding defaults
DEFAULT_OLLAMA_MODEL = "bge-m3"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# OpenAI embedding defaults
DEFAULT_EMBEDDING_DIMENSIONS = 1024

# ==================== UI Related Configuration ====================
DEFAULT_UI_LANG = "zh"
SUPPORTED_LANGUAGES = ["zh", "en"]
DEFAULT_LLM_BASE_URL_UI = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_HISTORY_LEN = 3
MAX_HISTORY_LEN = 25
DEFAULT_LLM_MODELS = ["glm-4-flash", "glm-4-air", "glm-4-plus", "glm-4-long"]
DEFAULT_CHAT_HISTORY_LEN = 4

# UI Asset Paths
UI_PAGE_ICON = "images/ob-icon.png"
UI_LOGO_PATH = "images/logo.png"

# UI Text
UI_REF_TIP = "\n\n检索到的相关文档如下:"
UI_WELCOME_MESSAGE = "Hello! How can I help you today?"

# Avatar Mapping
UI_AVATAR_MAP_CHAT = {
    "assistant": "images/ob-icon.png",
    "user": "🧑‍💻",
}
UI_AVATAR_MAP_FLOW = {
    "assistant": "🤖",
    "user": "👨🏻‍💻",
}

# Flow UI Specific Configuration
FLOW_UI_PAGE_TITLE = "Flow UI"
FLOW_UI_BATCH_SIZE = 10

# ==================== Other Configuration ====================
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def get_env(key: str, default: str = None) -> str:
    """
    Get environment variable value.

    Args:
        key: Environment variable name
        default: Default value, returned if environment variable does not exist

    Returns:
        Environment variable value, or default value if not exists
    """
    return os.getenv(key, default)


def get_bool_env(key: str, default: bool = False) -> bool:
    """
    Get boolean type environment variable value.

    Args:
        key: Environment variable name
        default: Default value

    Returns:
        True if environment variable value is "true" (case-insensitive),
        otherwise returns False or default value
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() == "true"


def get_int_env(key: str, default: int = None) -> int:
    """
    Get integer type environment variable value.

    Args:
        key: Environment variable name
        default: Default value

    Returns:
        Integer value of environment variable, or default value if not exists or cannot be converted
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


# ==================== LLM Configuration ====================
def get_llm_model() -> str:
    """
    Get LLM model name.
    """
    return get_env("LLM_MODEL", DEFAULT_LLM_MODEL)


def get_llm_base_url() -> str:
    """
    Get LLM API base URL.
    """
    return get_env("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)


def get_api_key() -> str:
    """
    Get API Key.
    If not exists in environment variables, will prompt user to input.
    """
    api_key = os.getenv("API_KEY")
    if not api_key:
        logger.warning("API_KEY not found in environment variables, prompting user")
        api_key = getpass.getpass("API_KEY: ")
        os.environ["API_KEY"] = api_key
        logger.info("API_KEY set from user input")
    else:
        logger.debug("API_KEY retrieved from environment variables")
    return api_key


# ==================== Database Configuration ====================
def get_db_host() -> str:
    """
    Get database host address.
    """
    return get_env("DB_HOST", DEFAULT_DB_HOST)


def get_db_port() -> str:
    """
    Get database port.
    """
    return get_env("DB_PORT", DEFAULT_DB_PORT)


def get_db_user() -> str:
    """
    Get database username.
    """
    return get_env("DB_USER", DEFAULT_DB_USER)


def get_db_password() -> str:
    """
    Get database password (automatically handles URL encoding of @ symbol for connection strings).
    """
    password = get_env("DB_PASSWORD", DEFAULT_DB_PASSWORD)
    if password:
        return password.replace("@", "%40")
    return ""


def get_db_password_raw() -> str:
    """
    Get raw database password (without URL encoding).
    """
    return get_env("DB_PASSWORD") or ""


def get_db_name() -> str:
    """
    Get database name.
    """
    return get_env("DB_NAME", DEFAULT_DB_NAME)


def get_table_name() -> str:
    """
    Get vector table name.
    """
    return get_env("TABLE_NAME", DEFAULT_TABLE_NAME)


# ==================== Embedding Model Configuration ====================
def get_bge_model_path() -> str:
    """
    Get BGE model path.
    """
    return get_env("BGE_MODEL_PATH", DEFAULT_BGE_MODEL_PATH)


# ==================== UI Configuration ====================
def get_ui_lang() -> str:
    """
    Get UI language.
    """
    lang = get_env("UI_LANG", DEFAULT_UI_LANG)
    if lang not in SUPPORTED_LANGUAGES:
        return DEFAULT_UI_LANG
    return lang


# ==================== Other Configuration ====================
def get_echo() -> bool:
    """
    Get whether to echo SQL queries.
    """
    return get_bool_env("ECHO", False)


def get_enable_rerank() -> bool:
    """
    Get whether to enable reranking.
    """
    return get_bool_env("ENABLE_RERANK", False)


def get_hf_endpoint() -> str:
    """
    Get HuggingFace mirror endpoint.
    """
    return get_env("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)


# ==================== Configuration Classes ====================


class EmbeddedType(str, Enum):
    """Embedding model type enumeration."""
    DEFAULT = "default"
    LOCAL_MODEL = "local_model"
    OLLAMA = "ollama"
    OPENAI_EMBEDDING = "openai_embedding"


class LLMConfig(BaseModel):
    """LLM configuration settings."""
    api_key: str = ""
    model: str = ""
    base_url: str = ""

    def is_valid(self) -> bool:
        """Check if all fields are non-empty."""
        return bool(self.api_key and self.model and self.base_url)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load LLM configuration from environment variables."""
        return cls(
            api_key=get_env("API_KEY", ""),
            model=get_env("LLM_MODEL", ""),
            base_url=get_env("LLM_BASE_URL", ""),
        )


class RAGParserConfig(BaseModel):
    """RAG parser configuration settings."""
    max_chunk_size: int = 4096
    limit: int = 0  # 0 means no limit
    skip_patterns: str = ""  # Comma-separated patterns

    @classmethod
    def from_env(cls) -> "RAGParserConfig":
        """Load RAG parser configuration from environment variables."""
        return cls(
            max_chunk_size=get_int_env("MAX_CHUNK_SIZE", 4096),
            limit=get_int_env("LIMIT", 0),
            skip_patterns=get_env("SKIP_PATTERNS", ""),
        )


class EmbeddingConfig(BaseModel):
    """Embedding model configuration settings."""
    embedded_type: EmbeddedType = EmbeddedType.DEFAULT
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    dimension: int = 1024

    def is_valid(self) -> bool:
        """
        Check if the embedding configuration is valid.
        Validation rules depend on embedded_type.
        """
        if self.embedded_type == EmbeddedType.DEFAULT:
            return True
        elif self.embedded_type == EmbeddedType.LOCAL_MODEL:
            return bool(self.model)
        elif self.embedded_type == EmbeddedType.OLLAMA:
            # Ollama requires model and base_url
            return bool(self.model and self.base_url)
        else:  # OPENAI_EMBEDDING
            return bool(self.api_key and self.model and self.base_url)

    def __eq__(self, other: object) -> bool:
        """
        Check equality based on embedded_type and relevant fields.
        """
        if not isinstance(other, EmbeddingConfig):
            return False
        if self.embedded_type != other.embedded_type:
            return False
        if self.embedded_type == EmbeddedType.DEFAULT:
            return True
        elif self.embedded_type == EmbeddedType.LOCAL_MODEL:
            return self.model == other.model
        elif self.embedded_type == EmbeddedType.OLLAMA:
            return self.model == other.model and self.base_url == other.base_url
        else:  # OPENAI_EMBEDDING
            return (
                self.api_key == other.api_key
                and self.model == other.model
                and self.base_url == other.base_url
            )

    def __hash__(self) -> int:
        """
        Generate hash based on embedded_type and relevant fields.
        """
        if self.embedded_type == EmbeddedType.DEFAULT:
            return hash(self.embedded_type)
        elif self.embedded_type == EmbeddedType.LOCAL_MODEL:
            return hash((self.embedded_type, self.model))
        elif self.embedded_type == EmbeddedType.OLLAMA:
            return hash((self.embedded_type, self.model, self.base_url))
        else:  # OPENAI_EMBEDDING
            return hash((self.embedded_type, self.api_key, self.model, self.base_url))

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Load embedding configuration from environment variables."""
        embedded_type_str = get_env("EMBEDDED_TYPE", "default")
        try:
            embedded_type = EmbeddedType(embedded_type_str)
        except ValueError:
            embedded_type = EmbeddedType.DEFAULT

        return cls(
            embedded_type=embedded_type,
            api_key=get_env("EMBEDDED_API_KEY", ""),
            model=get_env("EMBEDDED_LLM_MODEL", ""),
            base_url=get_env("EMBEDDED_LLM_BASE_URL", ""),
            dimension=get_int_env("EMBEDDED_DIMENSION", 1024),
        )
