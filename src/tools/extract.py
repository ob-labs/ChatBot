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

import pyseekdb

client = pyseekdb.Client(
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", "2881")),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

output_fields = ["id", "embedding", "document", "metadata", "component_code"]

offset = 0
batch_size = min(args.batch_size, args.total) if args.total > 0 else args.batch_size
current_count = args.batch_size
output_file = args.output_file

count_result = client.execute(f"SELECT COUNT(*) as cnt FROM {args.table_name}")
count = count_result[0][0] if count_result else 0
progress = tqdm(total=count)

values = []
while current_count == batch_size:
    rows = client.execute(
        f"SELECT {', '.join(output_fields)} FROM {args.table_name} LIMIT {batch_size} OFFSET {offset} "
    )
    current_count = len(rows)
    for row in rows:
        id, embedding, document, metadata, comp_code = row
        values.append(
            {
                "id": id,
                "embedding": json.loads(embedding.decode()) if isinstance(embedding, bytes) else embedding,
                "document": document,
                "metadata": json.loads(metadata) if isinstance(metadata, str) else metadata,
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
