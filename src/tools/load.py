"""
Data loading tool.

Provides functionality to load data from JSON files into a vector database.
"""
import argparse
import json
from typing import Any, Optional

from tqdm import tqdm

from src.common.db import ConnectionParams, create_pyseekdb_client
from src.common.logger import get_logger

logger = get_logger(__name__)

# SQL template for creating the vector table
CREATE_TABLE_SQL = """
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
partition `default_modules` values in (DEFAULT));
"""


class DataLoader:
    """
    Load data into a vector database.

    This class provides methods to load document data from JSON files into
    an OceanBase vector store, with support for table creation and batched
    insertion.
    """

    def __init__(
        self,
        table_name: str = config.get_table_name(),
        conn_params: Optional[ConnectionParams] = None,
    ):
        """
        Initialize the data loader.

        Args:
            table_name: Name of the table to load data into
            conn_params: Database connection parameters. If None, uses environment.
        """
        self._table_name = table_name
        self._conn_params = conn_params or ConnectionParams.from_env()
        self._client = create_pyseekdb_client(self._conn_params)

        logger.info(f"DataLoader initialized: table={table_name}")

    def table_exists(self) -> bool:
        """
        Check if the table exists in the database.

        Returns:
            True if the table exists, False otherwise
        """
        exists = self._client.has_collection(self._table_name)
        logger.debug(f"Table {self._table_name} exists: {exists}")
        return exists

    def create_table(self) -> None:
        """
        Create the vector table.

        Raises:
            RuntimeError: If table creation fails
        """
        logger.info(f"Creating table: {self._table_name}")
        try:
            self._client.execute(CREATE_TABLE_SQL.format(table_name=self._table_name))
            logger.info(f"Table {self._table_name} created successfully")
        except Exception as e:
            logger.error(f"Failed to create table {self._table_name}: {e}")
            raise RuntimeError(f"Failed to create table: {e}") from e

    def ensure_table_exists(self, skip_create: bool = False) -> bool:
        """
        Ensure the table exists or create it.

        Args:
            skip_create: If True, expect table to exist. If False, expect it not to.

        Returns:
            True if table is ready for use

        Raises:
            RuntimeError: If table state doesn't match skip_create expectation
        """
        exists = self.table_exists()

        if exists and not skip_create:
            logger.error(f"Table {self._table_name} already exists.")
            raise RuntimeError(f"Table {self._table_name} already exists.")

        if not exists and skip_create:
            logger.error(f"Table {self._table_name} does not exist.")
            raise RuntimeError(f"Table {self._table_name} does not exist.")

        if not skip_create:
            self.create_table()

        return True

    def _build_insert_sql(self, item: dict[str, Any]) -> str:
        """
        Build an INSERT SQL statement for a single item.

        Args:
            item: Dictionary containing the record data

        Returns:
            SQL INSERT statement string
        """
        id_val = item.get("id", "")
        embedding_val = json.dumps(item.get("embedding", []))
        document_val = json.dumps(item.get("document", ""))
        metadata_val = json.dumps(item.get("metadata", {}))
        component_code_val = item.get("component_code", 0)

        return f"""
        INSERT INTO `{self._table_name}` (`id`, `embedding`, `document`, `metadata`, `component_code`)
        VALUES (
            {json.dumps(id_val)},
            CAST({json.dumps(embedding_val)} AS VECTOR(1024)),
            {document_val},
            CAST({json.dumps(metadata_val)} AS JSON),
            {component_code_val}
        )
        """

    def _build_fallback_insert_sql(self, item: dict[str, Any]) -> str:
        """
        Build a fallback INSERT SQL statement without CAST.

        Args:
            item: Dictionary containing the record data

        Returns:
            SQL INSERT statement string
        """
        id_val = item.get("id", "")
        embedding_val = json.dumps(item.get("embedding", []))
        document_val = json.dumps(item.get("document", ""))
        metadata_val = json.dumps(item.get("metadata", {}))
        component_code_val = item.get("component_code", 0)

        return f"""
        INSERT INTO `{self._table_name}` (`id`, `embedding`, `document`, `metadata`, `component_code`)
        VALUES (
            {json.dumps(id_val)},
            {json.dumps(embedding_val)},
            {document_val},
            {json.dumps(metadata_val)},
            {component_code_val}
        )
        """

    def _insert_item(self, item: dict[str, Any]) -> None:
        """
        Insert a single item into the database.

        Args:
            item: Dictionary containing the record data
        """
        try:
            self._client.execute(self._build_insert_sql(item))
        except Exception as e:
            logger.warning(f"CAST insert failed, using fallback: {e}")
            self._client.execute(self._build_fallback_insert_sql(item))

    def load_from_file(
        self,
        source_file: str,
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> int:
        """
        Load data from a JSON file into the database.

        Args:
            source_file: Path to the source JSON file
            batch_size: Number of records to process in a batch (for progress display)
            show_progress: Whether to show a progress bar

        Returns:
            Number of records loaded
        """
        logger.info(f"Loading data from source file: {source_file}")

        with open(source_file, "r", encoding="utf-8") as f:
            values = json.load(f)

        logger.info(f"Loaded {len(values)} records from source file")

        progress = tqdm(total=len(values)) if show_progress else None

        for i in range(0, len(values), batch_size):
            batch = values[i : i + batch_size]
            logger.debug(
                f"Inserting batch {i // batch_size + 1}: {len(batch)} records"
            )

            for item in batch:
                self._insert_item(item)

            if progress:
                progress.update(len(batch))

        if progress:
            progress.close()

        logger.info(
            f"Successfully loaded {len(values)} records into table {self._table_name}"
        )
        return len(values)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Load data from JSON into a vector database."
    )
    parser.add_argument(
        "--table_name",
        type=str,
        help="Name of the table to load data into.",
        default=config.DEFAULT_TABLE_NAME,
    )
    parser.add_argument(
        "--source_file",
        type=str,
        help="Path to the source JSON file.",
        required=True,
    )
    parser.add_argument(
        "--skip_create",
        action="store_true",
        help="Skip creating the table (expect it to exist).",
        default=False,
    )
    parser.add_argument(
        "--insert_batch",
        type=int,
        help="Number of records to insert in a batch.",
        default=100,
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the data loading tool."""
    args = parse_args()
    logger.info(f"Command line arguments: {args}")
    print("args", args)

    loader = DataLoader(table_name=args.table_name)

    try:
        loader.ensure_table_exists(skip_create=args.skip_create)
    except RuntimeError as e:
        print(str(e))
        exit(1)

    count = loader.load_from_file(
        source_file=args.source_file,
        batch_size=args.insert_batch,
    )

    print(f"Successfully loaded {count} records into table {args.table_name}")


if __name__ == "__main__":
    main()
