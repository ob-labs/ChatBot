import argparse
import json

import pyseekdb
from tqdm import tqdm

from src.common.config import (
    get_db_host,
    get_db_name,
    get_db_password_raw,
    get_db_port,
    get_db_user,
    get_int_env,
)
from src.common.logger import get_logger

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--table_name",
    type=str,
    help="Name of the table to insert documents into.",
    default="corpus",
)
parser.add_argument(
    "--output_file",
    type=str,
    help="Path to the output file.",
    default="corpus.json",
)
parser.add_argument(
    "--batch_size",
    type=int,
    help="Number of documents to insert in a batch.",
    default=500,
)
parser.add_argument(
    "--total",
    type=int,
    help="Total number of documents to insert.",
    default=-1,
)
args = parser.parse_args()
logger.info(f"Command line arguments: {args}")
print("args", args)

logger.info("Connecting to database")
client = pyseekdb.Client(
    host=get_db_host(),
    port=get_int_env("DB_PORT", 2881),
    database=get_db_name(),
    user=get_db_user(),
    password=get_db_password_raw(),
)
logger.info("Database connection established")

output_fields = ["id", "embedding", "document", "metadata", "component_code"]

offset = 0
batch_size = min(args.batch_size, args.total) if args.total > 0 else args.batch_size
current_count = args.batch_size
output_file = args.output_file

logger.info(f"Extracting data from table: {args.table_name}")
count_result = client.execute(f"SELECT COUNT(*) as cnt FROM {args.table_name}")
count = count_result[0][0] if count_result else 0
logger.info(f"Total records in table: {count}")
progress = tqdm(total=count)

values = []
while current_count == batch_size:
    logger.debug(f"Extracting batch: offset={offset}, batch_size={batch_size}")
    rows = client.execute(
        f"SELECT {', '.join(output_fields)} FROM {args.table_name} LIMIT {batch_size} OFFSET {offset} "
    )
    current_count = len(rows)
    for row in rows:
        id, embedding, document, metadata, comp_code = row
        values.append(
            {
                "id": id,
                "embedding": (
                    json.loads(embedding.decode()) if isinstance(embedding, bytes) else embedding
                ),
                "document": document,
                "metadata": json.loads(metadata) if isinstance(metadata, str) else metadata,
                "component_code": comp_code,
            }
        )
        progress.update(1)
    offset += current_count
    if args.total > 0 and len(values) >= args.total:
        logger.info(f"Reached total limit: {args.total}")
        break

if values:
    logger.info(f"Writing {len(values)} records to output file: {output_file}")
    with open(output_file, "w") as f:
        f.write(json.dumps(values, indent=2))
    logger.info(f"Successfully wrote data to {output_file}")
else:
    logger.warning("No values to write")
