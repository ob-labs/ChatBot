from tqdm import tqdm

import os
import json
import dotenv

dotenv.load_dotenv()
import argparse

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
print("args", args)

from pyobvector import MilvusLikeClient

client = MilvusLikeClient(
    uri=f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}",
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    db_name=os.getenv("DB_NAME"),
)

output_fields = ["id", "embedding", "document", "metadata", "component_code"]

offset = 0
batch_size = min(args.batch_size, args.total) if args.total > 0 else args.batch_size
current_count = args.batch_size
output_file = args.output_file

cur = client.perform_raw_text_sql(f"SELECT COUNT(*) FROM {args.table_name}")
count = cur.fetchone()[0]
progress = tqdm(total=count)

values = []
while current_count == batch_size:
    cur = client.perform_raw_text_sql(
        f"SELECT {' ,'.join(output_fields)} FROM {args.table_name} LIMIT {batch_size} OFFSET {offset} "
    )
    current_count = cur.rowcount
    for id, embedding, document, metadata, comp_code in cur:
        values.append(
            {
                "id": id,
                "embedding": json.loads(embedding.decode()),
                "document": document,
                "metadata": json.loads(metadata),
                "component_code": comp_code,
            }
        )
        progress.update(1)
    offset += current_count
    if args.total > 0 and len(values) >= args.total:
        break

if values:
    with open(output_file, "w") as f:        
        f.write(json.dumps(values, indent=2))
