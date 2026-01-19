import argparse
import json

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
logger.info(f"Command line arguments: {args}")
print("args", args)

import pyseekdb

logger.info("Connecting to database")
client = pyseekdb.Client(
    host=get_db_host(),
    port=get_int_env("DB_PORT", 2881),
    database=get_db_name(),
    user=get_db_user(),
    password=get_db_password_raw(),
)
logger.info("Database connection established")

logger.info("Checking and setting database parameters")
vals = []
params = client.execute("SHOW PARAMETERS LIKE '%ob_vector_memory_limit_percentage%'")
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
        client.execute("ALTER SYSTEM SET ob_vector_memory_limit_percentage = 30")
        logger.info("Successfully set ob_vector_memory_limit_percentage to 30")
    except Exception as e:
        logger.error(f"Failed to set ob_vector_memory_limit_percentage to 30: {e}")
        print("Failed to set ob_vector_memory_limit_percentage to 30.")
        print("Error message:", e)
        exit(1)
logger.info("Setting ob_query_timeout to 100000000")
client.execute("SET ob_query_timeout=100000000")

table_exist = client.has_collection(args.table_name)
logger.info(f"Table {args.table_name} exists: {table_exist}")

if table_exist and not args.skip_create:
    logger.error(f"Table {args.table_name} already exists.")
    print(f"Table {args.table_name} already exists.")
    exit(1)
elif not table_exist and args.skip_create:
    logger.error(f"Table {args.table_name} does not exist.")
    print(f"Table {args.table_name} does not exist.")
    exit(1)
elif not args.skip_create:
    logger.info(f"Creating table: {args.table_name}")
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

    client.execute(create_table_sql)
    logger.info(f"Table {args.table_name} created successfully")

logger.info(f"Loading data from source file: {args.source_file}")
with open(args.source_file, "r") as f:
    values = json.load(f)
    logger.info(f"Loaded {len(values)} records from source file")
    progress = tqdm(total=len(values))
    for i in range(0, len(values), args.insert_batch):
        batch = values[i : i + args.insert_batch]
        logger.debug(f"Inserting batch {i // args.insert_batch + 1}: {len(batch)} records")
        # Use SQL INSERT with proper escaping
        for item in batch:
            id_val = item.get("id", "")
            embedding_val = json.dumps(item.get("embedding", []))
            document_val = json.dumps(item.get("document", ""))  # Escape as JSON string
            metadata_val = json.dumps(item.get("metadata", {}))
            component_code_val = item.get("component_code", 0)
            # Use JSON functions for safe insertion
            insert_sql = f"""
            INSERT INTO `{args.table_name}` (`id`, `embedding`, `document`, `metadata`, `component_code`)
            VALUES (
                {json.dumps(id_val)},
                CAST({json.dumps(embedding_val)} AS VECTOR(1024)),
                {json.dumps(item.get("document", ""))},
                CAST({json.dumps(metadata_val)} AS JSON),
                {component_code_val}
            )
            """
            try:
                client.execute(insert_sql)
            except Exception as e:
                logger.warning(f"CAST insert failed, using fallback: {e}")
                # Fallback to simpler insert if CAST fails
                insert_sql = f"""
                INSERT INTO `{args.table_name}` (`id`, `embedding`, `document`, `metadata`, `component_code`)
                VALUES (
                    {json.dumps(id_val)},
                    {json.dumps(embedding_val)},
                    {json.dumps(item.get("document", ""))},
                    {json.dumps(metadata_val)},
                    {component_code_val}
                )
                """
                client.execute(insert_sql)
        progress.update(len(batch))
    logger.info(f"Successfully loaded {len(values)} records into table {args.table_name}")
