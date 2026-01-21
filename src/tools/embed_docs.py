"""
Document embedding tool.

Provides functionality to embed markdown documents into a vector database.
"""
import argparse

from src.common.config import EmbeddingConfig
from src.common.logger import get_logger
from src.rag.rag import DocumentEmbedder

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Embed markdown documents into a vector database."
    )
    parser.add_argument(
        "--doc_base",
        type=str,
        help="Path to the directory containing markdown documents.",
    )
    parser.add_argument(
        "--table_name",
        type=str,
        help="Name of the table to insert documents into.",
        default=config.DEFAULT_TABLE_NAME,
    )
    parser.add_argument(
        "--skip_patterns",
        type=str,
        nargs="+",
        help="List of regex patterns to skip.",
        default=["oracle"],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of documents to insert in a batch.",
        default=4,
    )
    parser.add_argument(
        "--component",
        type=str,
        default="observer",
        help="Component to assign to the documents.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Maximum number of documents to insert.",
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Echo SQL queries.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the document embedding tool."""
    args = parse_args()
    logger.info(f"Command line arguments: {args}")
    print("args", args)

    if args.doc_base is None:
        logger.warning("No doc_base provided, nothing to embed.")
        print("No doc_base provided. Use --doc_base to specify document directory.")
        return

    embedder = DocumentEmbedder(
        embedding_config=EmbeddingConfig.from_env(),
        table_name=args.table_name,
        echo=args.echo,
    )

    total = embedder.embed_from_directory(
        doc_base=args.doc_base,
        component=args.component,
        skip_patterns=args.skip_patterns,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    print(f"Successfully embedded {total} documents.")


if __name__ == "__main__":
    main()
