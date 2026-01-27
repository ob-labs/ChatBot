# OceanBase AI 动手实战营

[英文版](./README.md)

## 项目介绍

在这个动手实战营中，我们会构建一个 RAG 聊天机器人，用来回答与 OceanBase 文档相关的问题。它将采用开源的 OceanBase 文档仓库作为多模数据源，把文档转换为向量和结构化数据存储在 OceanBase 中。用户提问时，该机器人将用户的问题同样转换为向量之后在数据库中进行向量检索，结合向量检索得到的文档内容，借助通义千问提供的大语言模型能力来为用户提供更加准确的回答。

### 项目组成

该机器人将由以下几个组件组成：

1. 将文档转换为向量的文本嵌入服务，在这里我们使用通义千问的嵌入 API
2. 提供存储和查询文档向量和其他结构化数据能力的数据库，我们使用 OceanBase 4.3.5 版本
3. 若干分析用户问题、基于检索到的文档和用户问题生成回答的 LLM 智能体，利用通义千问的大模型能力构建
4. 机器人与用户交互的聊天界面，采用 Streamlit 搭建

### 交互流程

![RAG 流程](./images/rag-flow.png)

1. 用户在 Web 界面中输入想要咨询的问题并发送给机器人
2. 机器人将用户提出的问题使用文本嵌入模型转换为向量
3. 将用户提问转换而来的向量作为输入在 OceanBase 中检索最相似的向量
4. OceanBase 返回最相似的一些向量和对应的文档内容
5. 机器人将用户的问题和查询到的文档一起发送给大语言模型并请它生成问题的回答
6. 大语言模型分片地、流式地将答案返回给机器人
7. 机器人将接收到的答案也分片地、流式地显示在 Web 界面中，完成一轮问答

## 概念解析

### 什么是文本嵌入?

文本嵌入是一种将文本转换为数值向量的技术。这些向量能够捕捉文本的语义信息，使计算机可以"理解"和处理文本的含义。具体来说:

- 文本嵌入将词语或句子映射到高维向量空间中的点
- 在这个向量空间中，语义相似的文本会被映射到相近的位置
- 向量通常由数百个数字组成(如 512 维、1024 维等)
- 可以用数学方法(如余弦相似度)计算向量之间的相似度
- 常见的文本嵌入模型包括 Word2Vec、BERT、BGE 等

在本项目中，我们使用通义千问的文本嵌入模型来生成文档的向量表示，这些向量将被存储在 OceanBase 数据库中用于后续的相似度检索。

例如使用嵌入模型将“苹果”、“香蕉”和“橘子”分别转换为 4 维的向量，它们的向量表示可能如下图所示，需要注意的是我们为了方便表示，将向量的维度降低到了 4 维，实际上文本嵌入产生的向量维数通常是几百或者几千维，例如我们使用的通义千问 text-embedding-v3 产生的向量维度是 1024 维。

![Embedding Example](./images/embedding-example.png)

### 什么是向量检索?

向量检索是在向量数据库中快速找到与查询向量最相似的向量的技术。其核心特点包括:

- 基于向量间的距离（如欧氏距离）或相似度（如余弦相似度）进行搜索
- 通常使用近似最近邻（Approximate Nearest Neighbor, ANN）算法来提高检索效率
- OceanBase 4.3.5 支持 HNSW 算法，这是一种高效的 ANN 算法
- 使用 ANN 可以快速从百万甚至亿级别的向量中找到近似最相似的结果
- 相比传统关键词搜索，向量检索能更好地理解语义相似性

OceanBase 在关系型数据库模型基础上将“向量”作为一种数据类型进行了完好的支持，使得在 OceanBase 一款数据库中能够同时针对向量数据和常规的结构化数据进行高效的存储和检索。在本项目中，我们会使用 OceanBase 建立 HNSW (Hierarchical Navigable Small World) 向量索引来实现高效的向量检索，帮助我们快速找到与用户问题最相关的文档片段。

如果我们在已经嵌入“苹果”、“香蕉”和“橘子”的 OceanBase 数据库中使用“红富士”作为查询文本，那么我们可能会得到如下的结果，其中“苹果”和“红富士”之间的相似度最高。（假设我们使用余弦相似度作为相似度度量）

![Vector Search Example](./images/vector-search-example.png)

### 什么是 RAG?

RAG (Retrieval-Augmented Generation，检索增强生成) 是一种结合检索系统和生成式 AI 的混合架构，用于提高 AI 回答的准确性和可靠性。其工作流程为:

1. 检索阶段:

- 将用户问题转换为向量
- 在知识库中检索相关文档
- 选择最相关的文档片段

2. 生成阶段:

- 将检索到的文档作为上下文提供给大语言模型
- 大语言模型基于问题和上下文生成回答
- 确保回答的内容来源可追溯

RAG 的主要优势有：

- 降低大语言模型的幻觉问题
- 能够利用最新的知识和专业领域信息
- 提供可验证和可追溯的答案
- 适合构建特定领域的问答系统

大语言模型的训练和发布需要耗费较长的时间，且训练数据在开启训练之后便停止了更新。而现实世界的信息熵增无时无刻不在持续，要让大语言模型在“昏迷”几个月之后还能自发地掌握当下最新的信息显然是不现实的。而 RAG 就是让大模型用上了“搜索引擎”，在回答问题前先获取新的知识输入，这样通常能较大幅度地提高生成回答的准确性。

## 准备工作

注意：如果您正在参加 OceanBase AI 动手实战营，您可以跳过以下步骤 1 ~ 4。所有所需的软件都已经在机器上准备好了。:)

1. 安装 [Python 3.11+](https://www.python.org/downloads/) 和 [pip](https://pip.pypa.io/en/stable/installation/)。如果您的机器上 Python 版本较低，可以使用 Miniconda 来创建新的 Python 3.11 及以上的环境，具体可参考 [Miniconda 安装指南](https://docs.anaconda.com/miniconda/install/)。

2. 安装 [uv](https://github.com/astral-sh/uv)，可参考命令 `curl -LsSf https://astral.sh/uv/install.sh | sh` 或 `pip install uv`

3. 安装 [Docker](https://docs.docker.com/engine/install/)（可选，如果您计划使用 Docker 在本地部署 OceanBase 数据库则必须安装）

4. 安装 MySQL 客户端，可参考 `yum install -y mysql` 或者 `apt-get install -y mysql-client`（可选，如果您需要使用 MySQL 客户端检验数据库连接则必须安装）

5. 确保您机器上该项目的代码是最新的状态，建议进入项目目录执行 `git pull`

6. 注册[阿里云百炼](https://bailian.console.aliyun.com/)账号，开通模型服务并获取 API Key

![点击开通模型服务](./images/activate-models.png)

![确认开通模型服务](./images/confirm-to-activate-models.png)

![阿里云百炼](./images/dashboard.png)

![获取阿里云百炼 API Key](./images/get-api-key.png)

## 构建聊天机器人

### Docker 快速启动（推荐）

如果您想使用 Docker 快速部署聊天机器人，请参见 [Docker 部署指南](./docker/README.md)。

```bash
# 1. 配置环境变量
cd docker
cp .env.example .env
vim .env  # 设置您的 API_KEY

# 2. 使用 Docker Compose 启动
docker compose up -d

# 3. 访问 http://localhost:8501
```

### 手动部署

### 1. 获取 OceanBase 数据库

我们首先要获取 OceanBase 4.3.5 版本及以上的数据库来存储我们的向量数据。您可以通过以下两种方式获取 OceanBase 数据库：

1. 使用 OB Cloud 云数据库免费试用版，平台注册和实例开通请参考[OB Cloud 云数据库 365 天免费试用](https://www.oceanbase.com/free-trial)；（推荐）
2. 使用 Docker 启动单机版 OceanBase 数据库。（备选，需要有 Docker 环境，消耗较多本地资源）

#### 1.1 使用 OB Cloud 云数据库免费试用版

##### 注册并开通实例

进入[OB Cloud 云数据库 365 天免费试用](https://www.oceanbase.com/free-trial)页面，点击“立即试用”按钮，注册并登录账号，填写相关信息，开通实例，等待创建完成。

##### 获取数据库实例连接串

进入实例详情页的“实例工作台”，点击“连接”-“获取连接串”按钮来获取数据库连接串，将其中的连接信息填入后续步骤中创建的 .env 文件内。

![获取数据库连接串](./images/obcloud-get-connection.png)


#### 1.2 使用 Docker 启动单机版 OceanBase 数据库

##### 启动 OceanBase 容器

如果你是第一次登录动手实战营提供的机器，你需要通过以下命令启动 Docker 服务：

```bash
systemctl start docker
```


### 2. 设置环境变量
接下来，您需要切换到动手实战营的项目目录：

```bash
git clone https://github.com/ob-labs/ChatBot
cd ChatBot
```

我们准备了一个 `.env.example` 文件，其中包含了聊天机器人所需的环境变量。您可以将 `.env.example` 文件复制到 `.env` 并更新 `.env` 文件中的值。

```bash
cp .env.example .env
# 更新 .env 文件中的值，特别是 API_KEY 和数据库连接信息
vi .env
```

`.env.example` 文件的内容如下，如果您正在按照动手实战营的步骤进行操作（使用通义千问提供的 LLM 能力），您需要把 `API_KEY` 更新为您从阿里云百炼控制台获取的 API KEY 值，如果您使用 OB Cloud 的数据库实例，请将 `DB_` 开头的变量更新为您的数据库连接信息，然后保存文件。

```bash

UI_LANG="zh"


#######################################################################
###################                               #####################
###################          Model Setting        #####################
###################                               #####################
#######################################################################
HF_ENDPOINT=https://hf-mirror.com

### Chat model
API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx             -----> 此处填写 刚才申请的 API Key
LLM_MODEL="qwen3-coder-plus"
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# =============================================================================
# Embedding Model Configuration
# =============================================================================
# EMBEDDED_TYPE: Embedding model type, available options:
#   - default: Use built-in sentence-transformers/all-MiniLM-L6-v2 model (no additional config needed)
#   - ollama: Use Ollama embedding service (requires all three params below)
#   - openai_embedding: Use OpenAI embedding API (requires all three params below)
EMBEDDED_TYPE=default

# Vector embedding dimension (must match your embedding model's output dimension)
EMBEDDED_DIMENSION=384

# EMBEDDED_API_KEY: API key for embedding service
#   - Required for: ollama, openai_embedding
#   - Not required for: default
EMBEDDED_API_KEY=

# EMBEDDED_LLM_MODEL: Embedding model name
#   - For ollama: model name (e.g., nomic-embed-text)
#   - For openai_embedding: model name (e.g., tongyi text-embedding-3-small)
EMBEDDED_LLM_MODEL=

# EMBEDDED_LLM_BASE_URL: Base URL or model path
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
REUSE_CURRENT_DB=true                ---> 如果使用本地docker, 此处改为false, 如果使用刚才云山的 OceanBase, 此处填写true


# Use what kind of docker, seekdb's docker or oceanbase-ce's docker
# Options: seekdb, oceanbase
# seekdb: If REUSE_CURRENT_DB is false, download seekdb docker
# oceanbase: If REUSE_CURRENT_DB is false, download oceanbase-ce docker
DB_STORE=seekdb            ---> 如果使用刚才云上的 OceanBase,  此处填写 oceanbase, 

----> 如果使用刚才云上的 OceanBase, 此处填写 正确的数据库地址
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

### 3. 安装依赖
```bash
make init
```
可能会花一些时间去下载所有的依赖, 如果使用docker 来运行 OceanBase seekdb 或 OceanBase 的话, 会花费更长的时间, 并可能需要设置国内的 docker 镜像地址. 

### 4. 准备文档数据

在该步骤中，我们将克隆 OceanBase 相关组件的开源文档仓库并处理它们，生成文档的向量数据和其他结构化数据后将数据插入到我们在步骤 1 中部署好的 OceanBase 数据库。

```bash
make start
```

```bash
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.xxx.xxx.xxx:8501
  External URL: http://xxx.xxx.xxx.xxx:8501 # 这是您可以从浏览器访问的 URL
```
访问终端中显示的 URL 来打开聊天机器人应用界面。

启动程序后, 进入 “加载文档” 页面, 一步一步加载文档即可, 支持 压缩包的形式, 多个markdown 文件, github 地址。 

### 5. 启动聊天界面

点击菜单的对话按钮, 


![Chat UI](./images/chatbot-ui.png)

## FAQ

### 1. 如何更改用于生成回答的 LLM 模型？

您可以通过更新 `.env` 文件中的 `LLM_MODEL` 环境变量来更改 LLM 模型，或者是在启动的对话界面左侧修改“大语言模型”。默认值是 `qwen-turbo-2024-11-01`，这是通义千问近期推出的具有较高免费额度的模型。还有其他可用的模型，如 `qwen-plus`、`qwen-max`、`qwen-long` 等。您可以在[阿里云百炼网站](https://bailian.console.aliyun.com/)的模型广场中找到完整的可用模型列表。请注意免费额度及计费标准。

### 2. 是否可以在初始加载后更新文档数据？

当然可以。您可以通过运行 `src/tools/embed_docs.py` 脚本插入新的文档数据。例如：

```bash
# 这将在当前目录中嵌入所有 markdown 文件，其中包含 README.md 和 LEGAL.md
uv run python src/tools/embed_docs.py --doc_base .

# 或者您可以指定要插入数据的表
uv run python src/tools/embed_docs.py --doc_base . --table_name my_table
```


### 3. 如何查看嵌入和检索过程中数据库执行的操作？

当您自己插入文档时，可以设置 `--echo` 标志来查看脚本执行的 SQL 语句，如下所示：

```bash
uv run python src/tools/embed_docs.py --doc_base . --table_name my_table --echo
```

您将看到以下输出：

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

您还可以在启动聊天界面之前设置 `ECHO=true`，以查看聊天界面执行的 SQL 语句。

```bash
ECHO=true TABLE_NAME=my_table uv run streamlit run --server.runOnSave false src/frontend/chat_ui.py
```

### 4. 为什么我在启动 UI 服务后再编辑 .env 文件不再生效？

如果你编辑了 .env 文件或者是代码文件，需要重启 UI 服务才能生效。你可以通过 `Ctrl + C` 终止服务，然后重新运行 `uv run streamlit run --server.runOnSave false src/frontend/chat_ui.py` 来重启服务。

### 5. 如何更改聊天界面的语言？

你可以通过更新 `.env` 文件中的 `UI_LANG` 环境变量来更改聊天界面的语言。默认值是 `zh`，表示中文。你可以将其更改为 `en` 来切换到英文。更新完成后需要重启服务才能生效。
