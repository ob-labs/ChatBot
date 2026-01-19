import dotenv

from src.common.logger import get_logger

logger = get_logger(__name__)

dotenv.load_dotenv()
logger.debug("Environment variables loaded")
