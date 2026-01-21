"""
Database management module.

Provides:
- ConnectionParams: Database connection parameters dataclass
- DatabaseClient: Unified database client with connection management
- PySeekDBClient: PySeekDB client wrapper for OceanBase operations
- create_pyseekdb_client: Convenience function to create pyseekdb client
- Command-line interface for database operations
"""
import re
import sys
import warnings
from dataclasses import dataclass
from typing import Optional

import pyseekdb
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.common.config import (
    DEFAULT_DB_HOST,
    DEFAULT_DB_NAME,
    DEFAULT_DB_PORT,
    DEFAULT_DB_USER,
    get_db_host,
    get_db_name,
    get_db_password_raw,
    get_db_port,
    get_db_user,
    get_table_name,
)
from src.common.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ConnectionParams:
    """Database connection parameters."""

    host: str = DEFAULT_DB_HOST
    port: str = DEFAULT_DB_PORT
    user: str = DEFAULT_DB_USER
    password: str = ""
    db_name: str = DEFAULT_DB_NAME

    def to_dict(self) -> dict:
        """
        Convert to dictionary for storage or external use.

        Returns:
            Dictionary with connection parameters
        """
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "db_name": self.db_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConnectionParams":
        """
        Create ConnectionParams from dictionary.

        Args:
            data: Dictionary with connection parameters

        Returns:
            ConnectionParams instance
        """
        return cls(
            host=data.get("host", DEFAULT_DB_HOST),
            port=data.get("port", DEFAULT_DB_PORT),
            user=data.get("user", DEFAULT_DB_USER),
            password=data.get("password", ""),
            db_name=data.get("db_name", DEFAULT_DB_NAME),
        )

    @classmethod
    def from_env(cls) -> "ConnectionParams":
        """
        Create ConnectionParams from environment variables.

        Returns:
            ConnectionParams instance with values from environment
        """
        return cls(
            host=get_db_host(),
            port=get_db_port(),
            user=get_db_user(),
            password=get_db_password_raw(),
            db_name=get_db_name(),
        )

    def get_connection_url(self, url_encode_password: bool = True) -> str:
        """
        Get SQLAlchemy connection URL.

        Args:
            url_encode_password: Whether to URL-encode the password

        Returns:
            Connection URL string
        """
        password = self.password
        if url_encode_password and password:
            password = password.replace("@", "%40")
        return (
            f"mysql+pymysql://{self.user}:{password}"
            f"@{self.host}:{self.port}/{self.db_name}"
        )


# =============================================================================
# Database Client
# =============================================================================


class DatabaseClient:
    """
    Unified database client with connection management.

    Provides methods for common database operations and connection testing.
    """

    def __init__(self, params: Optional[ConnectionParams] = None):
        """
        Initialize database client.

        Args:
            params: Connection parameters. If None, loads from environment.
        """
        self._params = params or ConnectionParams.from_env()
        self._engine: Optional[Engine] = None
        logger.debug(
            f"DatabaseClient initialized: host={self._params.host}, "
            f"port={self._params.port}, db_name={self._params.db_name}"
        )

    @property
    def params(self) -> ConnectionParams:
        """Get connection parameters."""
        return self._params

    def get_engine(self) -> Engine:
        """
        Get or create SQLAlchemy engine.

        Returns:
            SQLAlchemy Engine instance
        """
        if self._engine is None:
            self._engine = create_engine(self._params.get_connection_url())
            logger.debug("SQLAlchemy engine created")
        return self._engine

    def test_connection(self) -> bool:
        """
        Test database connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                success = result.scalar() == 1
                if success:
                    logger.info("Database connection test passed")
                return success
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def get_tables(self) -> list[str]:
        """
        Get list of tables in the database.

        Returns:
            List of table names
        """
        engine = self.get_engine()
        with engine.connect() as conn:
            return [row[0] for row in conn.execute(text("SHOW TABLES"))]

    def get_table_info(self, table_name: str) -> tuple:
        """
        Get table row count and structure.

        Args:
            table_name: Name of the table

        Returns:
            Tuple of (row count, table structure list)
        """
        engine = self.get_engine()
        with engine.connect() as conn:
            count = conn.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            ).scalar()
            structure = conn.execute(text(f"DESC {table_name}")).fetchall()
            return count or 0, list(structure)

    def execute(self, sql: str) -> None:
        """
        Execute a SQL statement.

        Args:
            sql: SQL statement to execute
        """
        engine = self.get_engine()
        with engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.debug("Database connection closed")


# =============================================================================
# PySeekDB Client (for specific OceanBase operations)
# =============================================================================


class PySeekDBClient:
    """
    PySeekDB client wrapper for OceanBase-specific operations.

    Uses pyseekdb library for collection operations and SQLAlchemy for raw SQL.
    Note: pyseekdb.Client() returns a _ClientProxy that only exposes collection
    operations (create_collection, has_collection, etc.), not raw SQL execution.
    Raw SQL must be executed via SQLAlchemy.
    """

    def __init__(self, params: Optional[ConnectionParams] = None):
        """
        Initialize PySeekDB client.

        Args:
            params: Connection parameters. If None, loads from environment.
        """
        self._params = params or ConnectionParams.from_env()
        self._client = None
        self._db_client: Optional[DatabaseClient] = None

    def _get_client(self):
        """Get or create pyseekdb client for collection operations."""
        if self._client is None:
            import pyseekdb

            self._client = pyseekdb.Client(
                host=self._params.host,
                port=int(self._params.port),
                database=self._params.db_name,
                user=self._params.user,
                password=self._params.password,
            )
            logger.info("PySeekDB client connected")
        return self._client

    def _get_db_client(self) -> DatabaseClient:
        """Get or create DatabaseClient for raw SQL execution."""
        if self._db_client is None:
            self._db_client = DatabaseClient(self._params)
        return self._db_client

    def check_connection(self) -> bool:
        """
        Check database connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Checking database connection...")
            db_client = self._get_db_client()
            success = db_client.test_connection()
            if success:
                logger.info("Database connection test passed")
            return success
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False

    def has_collection(self, table_name: str) -> bool:
        """
        Check if a collection/table exists.

        Args:
            table_name: Name of the table to check

        Returns:
            True if table exists, False otherwise
        """
        client = self._get_client()
        return client.has_collection(table_name)

    def execute(self, sql: str):
        """
        Execute a SQL statement using SQLAlchemy.

        Args:
            sql: SQL statement to execute
        """
        db_client = self._get_db_client()
        db_client.execute(sql)

# =============================================================================
# Convenience Functions
# =============================================================================


def create_pyseekdb_client(params: Optional[ConnectionParams] = None):  # type: ignore[return]
    """
    Create a pyseekdb client with the given connection parameters.

    This is a convenience function that creates a pyseekdb client directly
    without using the PySeekDBClient wrapper class.

    Args:
        params: Connection parameters. If None, loads from environment.

    Returns:
        pyseekdb.Client instance
    """
    if params is None:
        params = ConnectionParams.from_env()

    client = pyseekdb.Client(
        host=params.host,
        port=int(params.port),
        database=params.db_name,
        user=params.user,
        password=params.password,
    )
    logger.debug(
        f"Created pyseekdb client: host={params.host}, "
        f"port={params.port}, db_name={params.db_name}"
    )
    return client


# Default values for deprecated vector memory parameters
_DEFAULT_VECTOR_MEMORY_LIMIT_PERCENTAGE = 30
_DEFAULT_QUERY_TIMEOUT = 100000000


def ensure_vector_memory_params(
    client,  # pyseekdb.Client - type ignored due to library typing issues
    memory_limit_percentage: int = _DEFAULT_VECTOR_MEMORY_LIMIT_PERCENTAGE,
    query_timeout: int = _DEFAULT_QUERY_TIMEOUT,
) -> None:
    """
    Ensure vector memory parameters are properly configured.

    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Database parameters should be configured at the infrastructure level,
        not in application code.

    Checks and sets the ob_vector_memory_limit_percentage parameter if needed,
    and sets the ob_query_timeout for the session.

    Args:
        client: pyseekdb client instance
        memory_limit_percentage: Target value for ob_vector_memory_limit_percentage
        query_timeout: Query timeout in microseconds

    Raises:
        RuntimeError: If parameter check or setting fails
    """
    warnings.warn(
        "ensure_vector_memory_params is deprecated and will be removed in a future version. "
        "Database parameters should be configured at the infrastructure level.",
        DeprecationWarning,
        stacklevel=2,
    )

    logger.info("Checking and setting database parameters")

    # Check ob_vector_memory_limit_percentage
    vals = []
    params = client.execute(
        "SHOW PARAMETERS LIKE '%ob_vector_memory_limit_percentage%'"
    )
    for row in params:
        val = int(row[6])
        vals.append(val)

    if len(vals) == 0:
        logger.error("ob_vector_memory_limit_percentage not found in parameters.")
        raise RuntimeError("ob_vector_memory_limit_percentage not found in parameters.")

    # Set ob_vector_memory_limit_percentage if any value is 0
    if any(val == 0 for val in vals):
        try:
            logger.info(
                f"Setting ob_vector_memory_limit_percentage to {memory_limit_percentage}"
            )
            client.execute(
                f"ALTER SYSTEM SET ob_vector_memory_limit_percentage = {memory_limit_percentage}"
            )
            logger.info(
                f"Successfully set ob_vector_memory_limit_percentage to {memory_limit_percentage}"
            )
        except Exception as e:
            logger.error(
                f"Failed to set ob_vector_memory_limit_percentage to {memory_limit_percentage}: {e}"
            )
            raise RuntimeError(
                f"Failed to set ob_vector_memory_limit_percentage: {e}"
            ) from e

    # Set query timeout
    logger.info(f"Setting ob_query_timeout to {query_timeout}")
    client.execute(f"SET ob_query_timeout={query_timeout}")
    logger.debug("Database parameters configured successfully")


# =============================================================================
# CLI Functions
# =============================================================================


def check_connection() -> bool:
    """
    Check database connection (CLI function).

    Returns:
        True if successful, False otherwise
    """
    client = PySeekDBClient()
    success = client.check_connection()
    if success:
        print("Database connection check passed!")
    else:
        print("Database connection check failed!")
    return success


def main():
    """Main entry point for command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.common.db <command>")
        print("Commands:")
        print("  check-connection   Check database connection")
        sys.exit(1)

    command = sys.argv[1]

    if command == "check-connection":
        success = check_connection()
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: check-connection")
        sys.exit(1)


# =============================================================================
# Partition Management Functions
# =============================================================================


def get_partition_map(
    table_name: Optional[str] = None,
    conn_params: Optional[ConnectionParams] = None,
) -> dict[str, int]:
    """
    Get partition mapping from table partition information.

    Queries the table's partition information and returns a mapping dictionary
    similar to component_mapping in ob.py. If the table doesn't exist, returns
    the default component_mapping from ob.py. If successful, updates the
    component_mapping in ob.py.

    Args:
        table_name: Name of the table to query. If None, uses default from config.
        conn_params: Database connection parameters. If None, uses environment.

    Returns:
        Dictionary mapping partition names to component codes
    """
    # Import here to avoid circular dependency
    from src.rag.ob import component_mapping

    if table_name is None:
        table_name = get_table_name()

    if conn_params is None:
        conn_params = ConnectionParams.from_env()

    client = PySeekDBClient(conn_params)

    # Check if table exists
    if not client.has_collection(table_name):
        logger.warning(
            f"Table {table_name} does not exist, returning default component_mapping"
        )
        return component_mapping.copy()

    try:
        # Query partition information using SHOW CREATE TABLE
        db_client = client._get_db_client()
        engine = db_client.get_engine()

        with engine.connect() as conn:
            result = conn.execute(text(f"SHOW CREATE TABLE `{table_name}`"))
            row = result.fetchone()
            if row is None:
                raise RuntimeError(f"Failed to get CREATE TABLE statement for {table_name}")
            create_table_sql = row[1]

        # Parse partition information from CREATE TABLE SQL
        # Pattern: partition `name` values in (value)
        partition_pattern = r"partition\s+`([^`]+)`\s+values\s+in\s+\(([^)]+)\)"
        matches = re.findall(partition_pattern, create_table_sql, re.IGNORECASE)

        partition_map: dict[str, int] = {}

        for partition_name, values_str in matches:
            # Handle DEFAULT case
            if "DEFAULT" in values_str.upper():
                partition_map[partition_name] = 0
            else:
                # Extract numeric value
                value_match = re.search(r"(\d+)", values_str)
                if value_match:
                    partition_map[partition_name] = int(value_match.group(1))

        if partition_map:
            logger.info(
                f"Successfully retrieved partition map from table {table_name}: {partition_map}"
            )
            # Update component_mapping in ob.py
            component_mapping.update(partition_map)
            logger.info(f"Updated component_mapping: {component_mapping}")
            return component_mapping.copy()
        else:
            logger.warning(
                f"No partition information found in table {table_name}, "
                "returning default component_mapping"
            )
            return component_mapping.copy()

    except Exception as e:
        logger.error(f"Failed to get partition map from table {table_name}: {e}")
        logger.warning("Returning default component_mapping")
        return component_mapping.copy()


def append_partition(
    new_partition_name: str,
    table_name: Optional[str] = None,
    conn_params: Optional[ConnectionParams] = None,
) -> int:
    """
    Append a new partition for the table.

    Adds a new partition with a value that is the current maximum value + 1,
    then updates component_mapping in ob.py. If the table doesn't exist,
    only updates component_mapping without creating the partition.

    Args:
        new_partition_name: Name of the new partition to create
        table_name: Name of the table. If None, uses default from config.
        conn_params: Database connection parameters. If None, uses environment.

    Returns:
        The component code assigned to the new partition

    Raises:
        RuntimeError: If partition creation fails
    """
    # Import here to avoid circular dependency
    from src.rag.ob import component_mapping

    if table_name is None:
        table_name = get_table_name()

    if conn_params is None:
        conn_params = ConnectionParams.from_env()

    client = PySeekDBClient(conn_params)

    # Check if table exists
    if not client.has_collection(table_name):
        # If table doesn't exist, just update component_mapping
        current_map = component_mapping.copy()
        max_value = max(current_map.values()) if current_map else 0
        new_value = max_value + 1
        
        # Check if partition name already exists
        if new_partition_name in current_map:
            # Ensure component_mapping is updated
            logger.info(f"{new_partition_name} is already exists.")
            return max_value
        
        component_mapping[new_partition_name] = new_value
        logger.info(
            f"Table {table_name} does not exist, updated component_mapping only: "
            f"{component_mapping}"
        )
        return new_value

    # Get current partition map to find maximum value
    current_map = get_partition_map(table_name, conn_params)

    # Find maximum component code value
    max_value = max(current_map.values()) if current_map else 0
    new_value = max_value + 1

    # Check if partition name already exists
    if new_partition_name in current_map:
        # Ensure component_mapping is updated
        logger.warning(
            f"Partition {new_partition_name} already exists with value {current_map[new_partition_name]}"
        )
        return current_map[new_partition_name]

    try:
        # Create new partition using ALTER TABLE
        db_client = client._get_db_client()
        alter_sql = (
            f"ALTER TABLE `{table_name}` "
            f"ADD PARTITION (PARTITION `{new_partition_name}` VALUES IN ({new_value}))"
        )

        logger.info(f"Creating partition {new_partition_name} with value {new_value}")
        db_client.execute(alter_sql)
        logger.info(f"Successfully created partition {new_partition_name}")

        # Update component_mapping
        component_mapping[new_partition_name] = new_value
        logger.info(f"Updated component_mapping: {component_mapping}")

        return new_value

    except Exception as e:
        logger.error(f"Failed to create partition {new_partition_name}: {e}")
        raise RuntimeError(f"Failed to create partition: {e}") from e


