import argparse
import uuid

import pyseekdb
from langchain_core.documents import Document
from langchain_oceanbase.vectorstores import OceanbaseVectorStore
from pyobvector import ObListPartition, RangeListPartInfo
from sqlalchemy import Column, Integer

from src.common.config import (
    get_db_password_raw,
    get_ollama_token,
    get_ollama_url,
    get_openai_embedding_api_key,
    get_openai_embedding_base_url,
    get_openai_embedding_model,
)
from src.common.connection import connection_args
from src.common.logger import get_logger
from src.rag.documents import MarkdownDocumentsLoader
from src.rag.documents import component_mapping as cm
from src.rag.embedding import get_embedding

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--doc_base",
    type=str,
    help="Path to the directory containing markdown documents. Documents inside will be inserted into the database if the path is given.",
)
parser.add_argument(
    "--table_name",
    type=str,
    help="Name of the table to insert documents into.",
    default="corpus",
)
parser.add_argument(
    "--skip_patterns",
    type=list,
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

args = parser.parse_args()
logger.info(f"Command line arguments: {args}")
print("args", args)

embeddings = get_embedding(
    ollama_url=get_ollama_url(),
    ollama_token=get_ollama_token(),
    base_url=get_openai_embedding_base_url(),
    api_key=get_openai_embedding_api_key(),
    model=get_openai_embedding_model(),
)

vs = OceanbaseVectorStore(
    embedding_function=embeddings,
    table_name=args.table_name,
    connection_args=connection_args,
    metadata_field="metadata",
    extra_columns=[Column("component_code", Integer, primary_key=True)],
    partitions=ObListPartition(
        is_list_columns=False,
        list_part_infos=[RangeListPartInfo(k, v) for k, v in cm.items()]
        + [RangeListPartInfo("p10", "DEFAULT")],
        list_expr="component_code",
    ),
    echo=args.echo,
)

# Use pyseekdb client for SQL operations
sql_client = pyseekdb.Client(
    host=connection_args["host"],
    port=int(connection_args.get("port", "2881")),
    database=connection_args["db_name"],
    user=connection_args["user"],
    password=get_db_password_raw(),
)

logger.info("Checking and setting database parameters")
vals = []
params = sql_client.execute("SHOW PARAMETERS LIKE '%ob_vector_memory_limit_percentage%'")
for row in params:
    val = int(row[6])
    vals.append(val)
if len(vals) == 0:
    logger.error("ob_vector_memory_limit_percentage not found in parameters.")
    print("ob_vector_memory_limit_percentage not found in parameters.")
    exit(1)
if any(val == 0 for val in vals):
    try:
        logger.info("Setting ob_vector_memory_limit_percentage to 30")
        sql_client.execute("ALTER SYSTEM SET ob_vector_memory_limit_percentage = 30")
        logger.info("Successfully set ob_vector_memory_limit_percentage to 30")
    except Exception as e:
        logger.error(f"Failed to set ob_vector_memory_limit_percentage to 30: {e}")
        print("Failed to set ob_vector_memory_limit_percentage to 30.")
        print("Error message:", e)
        exit(1)
logger.info("Setting ob_query_timeout to 100000000")
sql_client.execute("SET ob_query_timeout=100000000")


def insert_batch(docs: list[Document], comp: str = "observer"):
    code = cm[comp]
    if not code:
        logger.error(f"Component {comp} not found in component_mapping.")
        raise ValueError(f"Component {comp} not found in component_mapping.")
    logger.info(f"Inserting batch of {len(docs)} documents for component {comp}")
    vs.add_documents(
        docs,
        ids=[str(uuid.uuid4()) for _ in range(len(docs))],
        extras=[{"component_code": code} for _ in docs],
        partition_name=comp,
    )
    logger.debug(f"Successfully inserted {len(docs)} documents for component {comp}")


if args.doc_base is not None:
    logger.info(f"Starting document embedding process: doc_base={args.doc_base}, component={args.component}, limit={args.limit}, batch_size={args.batch_size}")
    loader = MarkdownDocumentsLoader(
        doc_base=args.doc_base,
        skip_patterns=args.skip_patterns,
    )
    batch = []
    total_docs = 0
    for doc in loader.load(limit=args.limit):
        if len(batch) == args.batch_size:
            insert_batch(batch, comp=args.component)
            total_docs += len(batch)
            batch = []
        batch.append(doc)

    if len(batch) > 0:
        insert_batch(batch, comp=args.component)
        total_docs += len(batch)
    logger.info(f"Document embedding process completed: total documents embedded={total_docs}")
