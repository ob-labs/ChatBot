"""
Database management module.

Provides command-line interface for database operations:
- create-table: Create the vector table
- check-connection: Check database connection
"""
import sys

from src.common.config import (
    get_db_host,
    get_db_name,
    get_db_password_raw,
    get_db_port,
    get_db_user,
    get_int_env,
    get_table_name,
)
from src.common.logger import get_logger

logger = get_logger(__name__)


def check_connection():
    """
    Check database connection.
    """
    try:
        import pyseekdb

        logger.info("Connecting to database...")
        client = pyseekdb.Client(
            host=get_db_host(),
            port=get_int_env("DB_PORT", 2881),
            database=get_db_name(),
            user=get_db_user(),
            password=get_db_password_raw(),
        )
        logger.info("Database connection established successfully")
        
        # Test query
        client.execute("SELECT 1")
        logger.info("Database connection test passed")
        print("Database connection check passed!")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        print(f"Database connection check failed: {e}")
        return False


def create_table():
    """
    Create the vector table.
    """
    try:
        import pyseekdb

        table_name = get_table_name()
        logger.info(f"Creating table: {table_name}")
        
        client = pyseekdb.Client(
            host=get_db_host(),
            port=get_int_env("DB_PORT", 2881),
            database=get_db_name(),
            user=get_db_user(),
            password=get_db_password_raw(),
        )
        logger.info("Database connection established")
        
        # Check if table exists
        table_exist = client.has_collection(table_name)
        if table_exist:
            logger.warning(f"Table {table_name} already exists")
            print(f"Table {table_name} already exists")
            return True
        
        # Create table
        create_table_sql = f"""
        CREATE TABLE `{table_name}` (
        `id` varchar(4096) NOT NULL,
        `embedding` VECTOR(1024) DEFAULT NULL,
        `document` longtext DEFAULT NULL,
        `metadata` json DEFAULT NULL,
        `component_code` int(11) NOT NULL,
        PRIMARY KEY (`id`, `component_code`),
        VECTOR KEY `vidx` (`embedding`) WITH (DISTANCE=L2,M=16,EF_CONSTRUCTION=256,LIB=VSAG,TYPE=HNSW, EF_SEARCH=64) BLOCK_SIZE 16384
        ) DEFAULT CHARSET = utf8mb4 ROW_FORMAT = DYNAMIC COMPRESSION = 'zstd_1.3.8' REPLICA_NUM = 1 BLOCK_SIZE = 16384 USE_BLOOM_FILTER = FALSE TABLET_SIZE = 134217728 PCTFREE = 0
        partition by list(component_code)
        (partition `observer` values in (1),
        partition `ocp` values in (2),
        partition `oms` values in (3),
        partition `obd` values in (4),
        partition `operator` values in (5),
        partition `odp` values in (6),
        partition `odc` values in (7),
        partition `p10` values in (DEFAULT));
        """
        
        client.execute(create_table_sql)
        logger.info(f"Table {table_name} created successfully")
        print(f"Table {table_name} created successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to create table: {e}")
        print(f"Failed to create table: {e}")
        return False


def main():
    """
    Main entry point for command-line interface.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m src.common.db <command>")
        print("Commands:")
        print("  create-table      Create the vector table")
        print("  check-connection Check database connection")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create-table":
        success = create_table()
        sys.exit(0 if success else 1)
    elif command == "check-connection":
        success = check_connection()
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: create-table, check-connection")
        sys.exit(1)


if __name__ == "__main__":
    main()
