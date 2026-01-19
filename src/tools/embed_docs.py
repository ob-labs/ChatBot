import os
import uuid
import argparse
import dotenv

dotenv.load_dotenv()

from langchain_oceanbase.vectorstores import OceanbaseVectorStore
from langchain_core.documents import Document
from src.rag.embeddings import get_embedding
from src.rag.documents import MarkdownDocumentsLoader, component_mapping as cm
from pyobvector import ObListPartition, RangeListPartInfo
from sqlalchemy import Column, Integer
import pyseekdb

from src.common.connection import connection_args

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
print("args", args)

embeddings = get_embedding(
    ollama_url=os.getenv("OLLAMA_URL") or None,
    ollama_token=os.getenv("OLLAMA_TOKEN") or None,
    base_url=os.getenv("OPENAI_EMBEDDING_BASE_URL") or None,
    api_key=os.getenv("OPENAI_EMBEDDING_API_KEY") or None,
    model=os.getenv("OPENAI_EMBEDDING_MODEL") or None,
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
    password=connection_args["password"],
)

vals = []
params = sql_client.execute(
    "SHOW PARAMETERS LIKE '%ob_vector_memory_limit_percentage%'"
)
for row in params:
    val = int(row[6])
    vals.append(val)
if len(vals) == 0:
    print("ob_vector_memory_limit_percentage not found in parameters.")
    exit(1)
if any(val == 0 for val in vals):
    try:
        sql_client.execute(
            "ALTER SYSTEM SET ob_vector_memory_limit_percentage = 30"
        )
    except Exception as e:
        print("Failed to set ob_vector_memory_limit_percentage to 30.")
        print("Error message:", e)
        exit(1)
sql_client.execute("SET ob_query_timeout=100000000")


def insert_batch(docs: list[Document], comp: str = "observer"):
    code = cm[comp]
    if not code:
        raise ValueError(f"Component {comp} not found in component_mapping.")
    vs.add_documents(
        docs,
        ids=[str(uuid.uuid4()) for _ in range(len(docs))],
        extras=[{"component_code": code} for _ in docs],
        partition_name=comp,
    )


if args.doc_base is not None:
    loader = MarkdownDocumentsLoader(
        doc_base=args.doc_base,
        skip_patterns=args.skip_patterns,
    )
    batch = []
    for doc in loader.load(limit=args.limit):
        if len(batch) == args.batch_size:
            insert_batch(batch, comp=args.component)
            batch = []
        batch.append(doc)

    if len(batch) > 0:
        insert_batch(batch, comp=args.component)
