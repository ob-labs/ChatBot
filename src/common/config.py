"""
Configuration management module.

Centralized management of all environment variables and default values.
All os.getenv operations should be performed through this module.
"""
import getpass
import os

import dotenv

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
DEFAULT_DB_PORT = "2881"
DEFAULT_TABLE_NAME = "corpus"

# ==================== Embedding Model Related Configuration ====================
DEFAULT_BGE_MODEL_PATH = "BAAI/bge-m3"
DEFAULT_EMBEDDING_MODEL = "embedding-2"
DEFAULT_EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ==================== UI Related Configuration ====================
DEFAULT_UI_LANG = "zh"
SUPPORTED_LANGUAGES = ["zh", "en"]
DEFAULT_LLM_BASE_URL_UI = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_HISTORY_LEN = 3
MAX_HISTORY_LEN = 25
DEFAULT_LLM_MODELS = ["glm-4-flash", "glm-4-air", "glm-4-plus", "glm-4-long"]

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
    return get_env("DB_HOST")


def get_db_port() -> str:
    """
    Get database port.
    """
    return get_env("DB_PORT", DEFAULT_DB_PORT)


def get_db_user() -> str:
    """
    Get database username.
    """
    return get_env("DB_USER")


def get_db_password() -> str:
    """
    Get database password (automatically handles URL encoding of @ symbol for connection strings).
    """
    password = get_env("DB_PASSWORD")
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
    return get_env("DB_NAME")


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


def get_ollama_url() -> str:
    """
    Get Ollama URL.
    """
    return get_env("OLLAMA_URL") or None


def get_ollama_token() -> str:
    """
    Get Ollama token.
    """
    return get_env("OLLAMA_TOKEN") or None


def get_openai_embedding_base_url() -> str:
    """
    Get OpenAI-compatible embedding model base URL.
    """
    return get_env("OPENAI_EMBEDDING_BASE_URL") or None


def get_openai_embedding_api_key() -> str:
    """
    Get OpenAI-compatible embedding model API Key.
    If not set, will fallback to API_KEY.
    """
    return (
        get_env("OPENAI_EMBEDDING_API_KEY")
        or get_env("API_KEY")
        or None
    )


def get_openai_embedding_model() -> str:
    """
    Get OpenAI-compatible embedding model name.
    """
    return get_env("OPENAI_EMBEDDING_MODEL") or None


def get_embedding_model() -> str:
    """
    Get default embedding model name (for zhipu).
    """
    return DEFAULT_EMBEDDING_MODEL


def get_embedding_base_url() -> str:
    """
    Get default embedding model base URL (for zhipu).
    """
    return get_env("LLM_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)


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
