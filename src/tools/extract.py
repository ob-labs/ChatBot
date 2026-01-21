"""
Data extraction tool.

Provides functionality to extract data from a vector database to JSON files.
"""
import argparse
import json
from typing import Any, Optional

from tqdm import tqdm

from src.common.db import ConnectionParams, create_pyseekdb_client
from src.common.config import config
from src.common.logger import get_logger

logger = get_logger(__name__)

# Default output fields for extraction
DEFAULT_OUTPUT_FIELDS = ["id", "embedding", "document", "metadata", "component_code"]


class DataExtractor:
    """
    Extract data from a vector database.

    This class provides methods to extract document data from an OceanBase
    vector store and save it to JSON files.
    """

    def __init__(
        self,
        table_name: str = config.get_table_name(),
        conn_params: Optional[ConnectionParams] = None,
    ):
        """
        Initialize the data extractor.

        Args:
            table_name: Name of the table to extract data from
            conn_params: Database connection parameters. If None, uses environment.
        """
        self._table_name = table_name
        self._conn_params = conn_params or ConnectionParams.from_env()
        self._client = create_pyseekdb_client(self._conn_params)

        logger.info(f"DataExtractor initialized: table={table_name}")

    def get_record_count(self) -> int:
        """
        Get the total number of records in the table.

        Returns:
            Number of records in the table
        """
        result = self._client.execute(
            f"SELECT COUNT(*) as cnt FROM {self._table_name}"
        )
        count = result[0][0] if result else 0
        logger.debug(f"Table {self._table_name} has {count} records")
        return count

    def _process_row(self, row: tuple) -> dict[str, Any]:
        """
        Process a single row from the database.

        Args:
            row: Database row tuple (id, embedding, document, metadata, component_code)

        Returns:
            Dictionary with processed row data
        """
        id_val, embedding, document, metadata, comp_code = row
        return {
            "id": id_val,
            "embedding": (
                json.loads(embedding.decode())
                if isinstance(embedding, bytes)
                else embedding
            ),
            "document": document,
            "metadata": (
                json.loads(metadata) if isinstance(metadata, str) else metadata
            ),
            "component_code": comp_code,
        }

    def extract(
        self,
        output_file: str,
        batch_size: int = 500,
        total: int = -1,
        show_progress: bool = True,
        output_fields: Optional[list[str]] = None,
    ) -> int:
        """
        Extract data from the database to a JSON file.

        Args:
            output_file: Path to the output JSON file
            batch_size: Number of records to fetch in each batch
            total: Maximum number of records to extract (-1 for all)
            show_progress: Whether to show a progress bar
            output_fields: List of fields to extract

        Returns:
            Number of records extracted
        """
        if output_fields is None:
            output_fields = DEFAULT_OUTPUT_FIELDS

        logger.info(f"Extracting data from table: {self._table_name}")

        # Get total count for progress bar
        record_count = self.get_record_count()
        logger.info(f"Total records in table: {record_count}")

        # Adjust batch size if total is specified
        effective_batch_size = (
            min(batch_size, total) if total > 0 else batch_size
        )

        # Initialize progress tracking
        progress = tqdm(total=record_count) if show_progress else None

        values = []
        offset = 0
        current_count = effective_batch_size

        while current_count == effective_batch_size:
            logger.debug(
                f"Extracting batch: offset={offset}, batch_size={effective_batch_size}"
            )

            rows = self._client.execute(
                f"SELECT {', '.join(output_fields)} "
                f"FROM {self._table_name} "
                f"LIMIT {effective_batch_size} OFFSET {offset}"
            )

            current_count = len(rows)

            for row in rows:
                values.append(self._process_row(row))
                if progress:
                    progress.update(1)

            offset += current_count

            # Check if we've reached the total limit
            if total > 0 and len(values) >= total:
                logger.info(f"Reached total limit: {total}")
                break

        if progress:
            progress.close()

        # Write to output file
        if values:
            logger.info(f"Writing {len(values)} records to output file: {output_file}")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(values, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully wrote data to {output_file}")
        else:
            logger.warning("No values to write")

        return len(values)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Extract data from a vector database to JSON."
    )
    parser.add_argument(
        "--table_name",
        type=str,
        help="Name of the table to extract data from.",
        default=config.DEFAULT_TABLE_NAME,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the output file.",
        default=config.DEFAULT_TABLE_NAME + ".json",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of records to fetch in a batch.",
        default=500,
    )
    parser.add_argument(
        "--total",
        type=int,
        help="Total number of records to extract (-1 for all).",
        default=-1,
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the data extraction tool.
        Extract data from a vector database.

        This class provides methods to extract document data from an OceanBase
        vector store and save it to JSON files.
    """
    args = parse_args()
    logger.info(f"Command line arguments: {args}")
    print("args", args)

    extractor = DataExtractor(table_name=args.table_name)

    count = extractor.extract(
        output_file=args.output_file,
        batch_size=args.batch_size,
        total=args.total,
    )

    print(f"Successfully extracted {count} records to {args.output_file}")


if __name__ == "__main__":
    main()
