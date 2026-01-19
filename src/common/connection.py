from src.common.config import (
    get_db_host,
    get_db_name,
    get_db_password,
    get_db_port,
    get_db_user,
)
from src.common.logger import get_logger

logger = get_logger(__name__)

connection_args = {
    "host": get_db_host(),
    "port": get_db_port(),
    "user": get_db_user(),
    "password": get_db_password(),
    "db_name": get_db_name(),
}

logger.info(f"Database connection configured: host={get_db_host()}, port={get_db_port()}, db_name={get_db_name()}")
