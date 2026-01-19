from tqdm import tqdm

import os
import json
import dotenv
import argparse

dotenv.load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--table_name",
    type=str,
    help="Name of the table to insert documents into.",
    default="corpus",
)
parser.add_argument(
    "--source_file",
    type=str,
    help="Path to the source file.",
    required=True,
)
parser.add_argument(
    "--skip_create",
    action="store_true",
    help="Skip creating the table.",
    default=False,
)
parser.add_argument(
    "--insert_batch",
    type=int,
    help="Number of documents to insert in a batch.",
    default=100,
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

vals = []
params = client.perform_raw_text_sql(
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
        client.perform_raw_text_sql(
            "ALTER SYSTEM SET ob_vector_memory_limit_percentage = 30"
        )
    except Exception as e:
        print("Failed to set ob_vector_memory_limit_percentage to 30.")
        print("Error message:", e)
        exit(1)
client.perform_raw_text_sql("SET ob_query_timeout=100000000")

table_exist = client.check_table_exists(args.table_name)

if table_exist and not args.skip_create:
    print(f"Table {args.table_name} already exists.")
    exit(1)
elif not table_exist and args.skip_create:
    print(f"Table {args.table_name} does not exist.")
    exit(1)
elif not args.skip_create:
    create_table_sql = f"""
    CREATE TABLE `{args.table_name}` (
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

    client.perform_raw_text_sql(create_table_sql)

with open(args.source_file, "r") as f:
    values = json.load(f)
    progress = tqdm(total=len(values))
    for i in range(0, len(values), args.insert_batch):
        batch = values[i : i + args.insert_batch]
        client.insert(args.table_name, batch)
        progress.update(len(batch))
