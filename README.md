# OceanBase AI Workshop

[中文版](./README_zh.md)

## Introduction

In this workshop, we will build a RAG chatbot that answers questions related to OceanBase documentation. It uses open-source OceanBase documentation repositories as multi-modal data sources, converting documents into vectors and structured data stored in OceanBase. When users ask questions, the chatbot converts their questions into vectors and performs vector retrieval in the database. By combining the retrieved document content with the user's questions, it leverages Tongyi Qianwen's large language model capabilities to provide more accurate answers.

### Components

The chatbot consists of the following components:

1. A text embedding service that converts documents into vectors, using Tongyi Qianwen's embedding API
2. A database that provides storage and query capabilities for document vectors and other structured data, using OceanBase 4.3.5
3. Several LLM agents that analyze user questions and generate answers based on retrieved documents and questions, built with Tongyi Qianwen's large model capabilities
4. A chat interface for user interaction, built with Streamlit

### Interaction Flow

![RAG Flow](./images/rag-flow.png)

1. User inputs a question in the Web interface and sends it to the chatbot
2. The chatbot converts the user's question into a vector using a text embedding model
3. Uses the vector converted from the user's question as input to retrieve the most similar vectors in OceanBase
4. OceanBase returns the most similar vectors and their corresponding document content
5. The chatbot sends the user's question and retrieved documents to the large language model and requests it to generate an answer
6. The large language model returns the answer in chunks, streaming fashion
7. The chatbot displays the received answer in chunks, streaming fashion on the Web interface, completing one round of Q&A

## Concepts

### What is Text Embedding?

Text embedding is a technique that converts text into numerical vectors. These vectors can capture the semantic information of text, enabling computers to "understand" and process the meaning of text. Specifically:

- Text embedding maps words or sentences to points in a high-dimensional vector space
- In this vector space, semantically similar texts are mapped to nearby locations
- Vectors typically consist of hundreds of numbers (e.g., 512 dimensions, 1024 dimensions)
- Vector similarity can be calculated using mathematical methods (e.g., cosine similarity)
- Common text embedding models include Word2Vec, BERT, BGE, etc.

In this project, we use Tongyi Qianwen's text embedding model to generate vector representations of documents, which will be stored in OceanBase database for subsequent similarity retrieval.

For example, when using an embedding model to convert "apple", "banana", and "orange" into 4-dimensional vectors, their vector representations might look like the diagram below. Note that we reduced the vector dimensions to 4 for easier visualization - in practice, text embedding vectors usually have hundreds or thousands of dimensions. For instance, the text-embedding-v3 model we use from Tongyi Qianwen produces 1024-dimensional vectors.

![Embedding Example](./images/embedding-example.png)

### What is Vector Retrieval?

Vector retrieval is a technique for quickly finding the most similar vectors to a query vector in a vector database. Its key features include:

- Search based on vector distance (e.g., Euclidean distance) or similarity (e.g., cosine similarity)
- Typically uses Approximate Nearest Neighbor (ANN) algorithms to improve retrieval efficiency
- OceanBase 4.3.5 supports the HNSW algorithm, which is a high-performance ANN algorithm
- ANN can quickly find the most similar results approximately from millions or even billions of vectors
- Compared to traditional keyword search, vector retrieval better understands semantic similarity

OceanBase has added excellent support for "vector" as a data type in its relational database model, enabling efficient storage and retrieval of both vector data and conventional structured data in a single database. In this project, we use OceanBase's HNSW (Hierarchical Navigable Small World) vector index to implement efficient vector retrieval, helping us quickly find the most relevant document fragments for user questions.

If we use "Fuji" as a query text in an OceanBase database that already has embeddings for "apple", "banana", and "orange", we might get results like the diagram below, where the similarity between "apple" and "Fuji" is highest. (Assuming we use cosine similarity as the similarity measure)

![Vector Search Example](./images/vector-search-example.png)

### What is RAG?

RAG (Retrieval-Augmented Generation) is a hybrid architecture that combines retrieval systems with generative AI to improve the accuracy and reliability of AI responses. The workflow consists of:

1. Retrieval Phase:

- Convert user questions into vectors
- Retrieve relevant documents from the knowledge base
- Select the most relevant document fragments

2. Generation Phase:

- Provide retrieved documents as context to the large language model
- Generate answers based on questions and context
- Ensure answer sources are traceable

Key advantages of RAG:

- Reduces hallucination problems in large language models
- Can utilize latest knowledge and domain-specific information
- Provides verifiable and traceable answers
- Suitable for building domain-specific Q&A systems

Training and releasing large language models takes considerable time, and training data stops updating once training begins. While the amount of information in the real world continues to increase constantly, it's unrealistic to expect language models to spontaneously master the latest information after being "unconscious" for several months. RAG essentially gives large models a "search engine", allowing them to acquire new knowledge input before answering questions, which typically significantly improves the accuracy of generated responses.

## Prerequisites

Notes: If you are participating in the OceanBase AI Workshop, you can skip steps 1 ~ 4 below. All required software is already prepared on the machine. :)

1. Install [Python 3.11+](https://www.python.org/downloads/) and [pip](https://pip.pypa.io/en/stable/installation/). 

2. Install [uv](https://github.com/astral-sh/uv) with command `curl -LsSf https://astral.sh/uv/install.sh | sh` or `pip install uv`

3. Install [Docker](https://docs.docker.com/engine/install/) (Optional, only required if you want to run OceanBase in a Docker container)

4. Install MySQL client, using `yum install -y mysql` or `apt-get install -y mysql-client` (Optional, only required if you want to test database connection with MySQL client)

5. Ensure your project code is up to date, recommended to run `git pull` in the project directory

6. Register an [Alibaba Cloud Bailian](https://bailian.console.aliyun.com/) account, activate model service and obtain API Key

![Activate Model Service](./images/activate-models.png)

![Confirm to Activate Model Service](./images/confirm-to-activate-models.png)

![Alibaba Cloud Bailian](./images/dashboard.png)

![Get Alibaba Cloud Bailian API Key](./images/get-api-key.png)

## Building the Chatbot

### 1. Get an OceanBase Database

First, we need to obtain an OceanBase database version 4.3.5 or above to store our vector data. You can get an OceanBase database through either of these two methods:

1. Use the OB Cloud database free trial instances. For platform registration and instance creation, please refer to [OB Cloud Database 365-Day Free Trial](https://www.oceanbase.com/free-trial); (Recommended)
2. Use Docker to start a standalone OceanBase database. (Alternative option, requires Docker environment, consumes more local resources)

#### 1.1 Using OB Cloud Database Free Trial Version

##### Register and Create an Instance

Visit the [OB Cloud Database 365-Day Free Trial](https://www.oceanbase.com/free-trial) web page, click the "Try Now" button, register and log in to your account, fill in the relevant information, create an instance, and wait for creation to complete.

##### Get Database Instance Connection Information

Go to the "Instance Workbench" on the instance details page, click the "Connect"-"Get Connection String" button to obtain the database connection information. Fill the connection information into the .env file that will be created in subsequent steps.

![Get Database Connection Information](./images/obcloud-get-connection.png)


#### 1.2 Deploy an OceanBase Database with Docker

##### Start an OceanBase Container

If this is your first time logging into the machine provided by the workshop, you need to start the Docker service with:

```bash
systemctl start docker
```




### 2. Set up environment variables

Next, you need to switch to the workshop project directory:

```bash
git clone https://github.com/ob-labs/ChatBot
cd ChatBot
```

We prepare a `.env.example` file that contains the environment variables required for the chatbot. You can copy the `.env.example` file to `.env` and update the values in the `.env` file.

```bash
cp .env.example .env
# Update the .env file with the correct values, especially the API_KEY and database information
vi .env
```

The content of `.env.example` is as follows. If you are following the workshop steps (using LLM capabilities from Tongyi Qianwen), you need to update `API_KEY` with the API KEY values you obtained from the Alibaba Cloud Bailian console. If you are using an OB Cloud database instance, update the variables starting with `DB_` with your database connection information, then save the file.

```bash

UI_LANG="zh"


#######################################################################
###################                               #####################
###################          Model Setting        #####################
###################                               #####################
#######################################################################
HF_ENDPOINT=https://hf-mirror.com

### Chat model
API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx             -----> Fill in the API Key you just applied for
LLM_MODEL="qwen3-coder-plus"
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# =============================================================================
# Embedding Model Configuration
# =============================================================================
# EMBEDDED_TYPE: Embedding model type, available options:
#   - default: Use built-in sentence-transformers/all-MiniLM-L6-v2 model (no additional config needed)
#   - local_model: Use local embedding model (requires EMBEDDED_LLM_MODEL and EMBEDDED_LLM_BASE_URL)
#   - ollama: Use Ollama embedding service (requires all three params below)
#   - openai_embedding: Use OpenAI embedding API (requires all three params below)
EMBEDDED_TYPE=default

# Vector embedding dimension (must match your embedding model's output dimension)
EMBEDDED_DIMENSION=384

# EMBEDDED_API_KEY: API key for embedding service
#   - Required for: ollama, openai_embedding
#   - Not required for: default, local_model
EMBEDDED_API_KEY=

# EMBEDDED_LLM_MODEL: Embedding model name
#   - For local_model: model name (e.g., BAAI/bge-m3)
#   - For ollama: model name (e.g., nomic-embed-text)
#   - For openai_embedding: model name (e.g., tongyi text-embedding-3-small)
EMBEDDED_LLM_MODEL=

# EMBEDDED_LLM_BASE_URL: Base URL or model path
#   - For local_model: local model path (e.g., /path/to/model), if this is empty, it will be automatically downloaded
#   - For ollama: Ollama server URL (e.g., http://localhost:11434)
#   - For openai_embedding: OpenAI API base URL (e.g., https://api.openai.com/v1)
EMBEDDED_LLM_BASE_URL=


#######################################################################
###################                               #####################
###################          Database Setting     #####################
###################                               #####################
#######################################################################

# Whether to reuse the current database
# When set to true, will reuse existing database connection
# When set to false, will start download dockers, at this time:
#                   if DB_STORE is seekdb, DB_USER must be root
#                   if DB_STORE is oceanbase, DB_USER must be root@test
REUSE_CURRENT_DB=true                ---> If using local docker, change to false; if using the cloud OceanBase you just created, set to true


# Use what kind of docker, seekdb's docker or oceanbase-ce's docker
# Options: seekdb, oceanbase
# seekdb: If REUSE_CURRENT_DB is false, download seekdb docker
# oceanbase: If REUSE_CURRENT_DB is false, download oceanbase-ce docker
DB_STORE=seekdb            ---> If using the cloud OceanBase you just created, set to oceanbase

----> If using the cloud OceanBase you just created, fill in the correct database address
# Database Setting, please change as your environment. 
DB_HOST="127.0.0.1"
DB_PORT="2881"
#if REUSE_CURRENT_DB=false and DB_STORE=seekdb, DB_USER must be root
#if REUSE_CURRENT_DB=false and DB_STORE=oceanbase, DB_USER must be root@test
DB_USER="root"
# If database use OceanBase, the DB_USER will contain tenant's name
# DB_USER="root@test"
DB_PASSWORD="root@test"
DB_NAME="test"


#######################################################################
###################                               #####################
###################         RAG Parser Setting    #####################
###################                               #####################
#######################################################################
# Maximum chunk size for text splitting (in characters)
MAX_CHUNK_SIZE=4096

# Limit the number of documents to process (0 means no limit)
LIMIT=0

# Patterns to skip when processing documents (comma-separated, e.g., "*.log,*.tmp")
SKIP_PATTERNS=""


```

### 3. Install dependencies
```bash
make init
```
This command will take much time to download all dependency, if use docker to run OceanBase seekdb or OceanBase, it will increase time usage.

### 4. Prepare Document Data

In this step, we will clone the open-source documentation repositories of OceanBase components and process them to generate document vectors and other structured data, which will then be inserted into the OceanBase database we deployed in step 1.

```bash
make start
```

```bash
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.xxx.xxx.xxx:8501
  External URL: http://xxx.xxx.xxx.xxx:8501 # This is the URL you can access from your browser
```

Visit the URL displayed in the terminal to open the chatbot application UI.

After starting the program, go to the "Load Documents" page and load documents step by step. It supports compressed files, multiple markdown files, and GitHub addresses.

### 5. Start the Chat UI

Click the "Chat" button in the menu.

![Chat UI](./images/chatbot-ui.png)

## FAQ

### 1. How to change the LLM model used for generating responses?

You can change the LLM model by updating the `LLM_MODEL` environment variable in the `.env` file, or by modifying the "Large Language Model" in the left sidebar of the chat interface. The default value is `qwen-turbo-2024-11-01`, which is a recently released model from Tongyi Qianwen with a relatively high free quota. Other available models include `qwen-plus`, `qwen-max`, `qwen-long`, etc. You can find the complete list of available models in the [Alibaba Cloud Bailian website](https://bailian.console.aliyun.com/)'s model marketplace. Please note the free quotas and pricing standards.

### 2. Can I update document data after initial loading?

Of course. You can insert new document data by running the `src/tools/embed_docs.py` script. For example:

```bash
# This will embed all markdown files in the current directory, including README.md and LEGAL.md
uv run python src/tools/embed_docs.py --doc_base .

# Or you can specify the table to insert data into
uv run python src/tools/embed_docs.py --doc_base . --table_name my_table
```

### 3. How to see the SQL statements executed by the database during embedding and retrieval?

When inserting documents yourself, you can set the `--echo` flag to see the SQL statements executed by the script:

```bash
uv run python src/tools/embed_docs.py --doc_base . --table_name my_table --echo
```

You will see output like this:

```bash
2024-10-16 03:17:13,439 INFO sqlalchemy.engine.Engine
CREATE TABLE my_table (
        id VARCHAR(4096) NOT NULL,
        embedding VECTOR(1024),
        document LONGTEXT,
        metadata JSON,
        component_code INTEGER NOT NULL,
        PRIMARY KEY (id, component_code)
)
...
```

You can also set `ECHO=true` before launching the chat UI to see the SQL statements executed by the chat UI.

```bash
ECHO=true TABLE_NAME=my_table uv run streamlit run --server.runOnSave false src/frontend/chat_ui.py
```

### 4. Why don't changes to the .env file take effect after starting the UI service?

If you edit the .env file or code files, you need to restart the UI service for the changes to take effect. You can terminate the service with `Ctrl + C`, then run `uv run streamlit run --server.runOnSave false src/frontend/chat_ui.py` again to restart the service.

### 5. How to change the language of the chat UI?

You can change the language of the chat interface by updating the `UI_LANG` environment variable in the `.env` file. The default value is `zh`, which means Chinese. You can change it to `en` to switch to English. You need to restart the UI after updating for the changes to take effect.
