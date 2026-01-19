import json
import os
import time
import uuid
from typing import Iterator, Optional, Union

import dotenv
from langchain_core.messages import BaseMessageChunk
from langchain_oceanbase.vectorstores import OceanbaseVectorStore
from pydantic import BaseModel
from sqlalchemy import create_engine, text

dotenv.load_dotenv()

import streamlit as st

from src.agents.base import AgentBase
from src.agents.universe_rag_agent import prompt as universal_rag_prompt
from src.common.logger import get_logger
from src.rag.documents import parse_md
from src.rag.embedding import OllamaEmbedding

logger = get_logger(__name__)

# Configuration constants
STATE_FILE_PATH = "./data/uploaded/state.json"
DOCS_DIR = "./data/uploaded/docs"
DEFAULT_BATCH_SIZE = 10
REF_TIP = "\n\n检索到的相关文档如下:"
DEFAULT_LLM_MODEL = "glm-4-flash"
DEFAULT_LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_CHAT_HISTORY_LEN = 4


def init_state():
    """
    Initialize the session state with default values.
    """
    logger.debug("Initializing session state")
    st.session_state.step = 0
    st.session_state.table = ""
    st.session_state.connection = {}


def save_state():
    """
    Save the current session state to a JSON file.
    """
    logger.debug(f"Saving session state to {STATE_FILE_PATH}")
    try:
        with open(STATE_FILE_PATH, "w") as f:
            d = {
                "step": st.session_state.step,
                "table": st.session_state.table,
                "connection": st.session_state.connection,
            }
            json.dump(d, f)
        logger.debug("Session state saved successfully")
    except Exception as e:
        logger.error(f"Failed to save session state: {e}")


def load_state():
    """
    Load session state from a JSON file if it exists.
    """
    if os.path.exists(STATE_FILE_PATH):
        logger.debug(f"Loading session state from {STATE_FILE_PATH}")
        try:
            with open(STATE_FILE_PATH, "r") as f:
                d = json.load(f)
                st.session_state.update(d)
            logger.debug("Session state loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
    else:
        logger.debug("No saved session state found")


# Initialize session state
if "step" not in st.session_state:
    logger.info("Initializing new session state")
    init_state()

# Load saved state
load_state()
logger.info("Flow UI module loaded")


class StreamResponse:
    """
    Helper class for streaming chatbot responses.

    Accumulates message chunks and provides methods to generate
    the response stream with optional prefix/suffix iterators.
    """

    def __init__(self, chunks: Iterator[BaseMessageChunk] = []):
        """
        Initialize StreamResponse with message chunks.

        Args:
            chunks: Iterator of BaseMessageChunk objects
        """
        self.chunks = chunks
        self.__whole_msg = ""

    def generate(
        self,
        *,
        prefix: Union[Iterator, None] = None,
        suffix: Union[Iterator, None] = None,
    ) -> Iterator[str]:
        """
        Generate response stream with optional prefix and suffix iterators.

        Args:
            prefix: Optional iterator to yield before chunks
            suffix: Optional iterator to yield after chunks

        Yields:
            Response text chunks as strings
        """
        if prefix:
            for pre in prefix:
                yield pre
        for chunk in self.chunks:
            self.__whole_msg += chunk.content
            yield chunk.content
        if suffix:
            for suf in suffix:
                yield suf

    def get_whole(self) -> str:
        """
        Get the complete accumulated message.

        Returns:
            Complete message as string
        """
        return self.__whole_msg


def remove_refs(history: list[dict]) -> list[dict]:
    """
    Remove reference sections from chat history.

    This prevents the model from generating its own reference list
    by removing content after the reference marker.

    Args:
        history: List of message dictionaries

    Returns:
        List of messages with references removed
    """
    return [
        {
            "role": msg["role"],
            "content": msg["content"].split(REF_TIP)[0],
        }
        for msg in history
    ]


def get_engine(**connection_params):
    """
    Create a SQLAlchemy engine from connection parameters.

    Args:
        **connection_params: Database connection parameters including:
            - user: Database username
            - password: Database password
            - host: Database host
            - port: Database port
            - db_name: Database name

    Returns:
        SQLAlchemy engine instance
    """
    return create_engine(
        f"mysql+pymysql://{connection_params['user']}:{connection_params['password']}"
        f"@{connection_params['host']}:{connection_params['port']}"
        f"/{connection_params['db_name']}"
    )


# Initialize embeddings (using hardcoded values - consider moving to config)
embeddings = OllamaEmbedding(
    url="http://30.249.224.105:8080/api/embed",
    token="test",
)


def step_forward():
    """
    Move to the next step in the flow and save state.
    """
    st.session_state.step += 1
    save_state()
    st.rerun()


def step_back():
    """
    Move to the previous step in the flow and save state.
    """
    st.session_state.step -= 1
    save_state()
    st.rerun()


class Step(BaseModel):
    """
    Represents a step in the flow UI.

    Attributes:
        name: Step name
        desc: Step description
        form: Optional form configuration
    """

    name: str
    desc: Optional[str] = None
    form: Optional[dict] = None


# Define flow steps
steps: list[Step] = [
    Step(
        name="Database Connection",
        desc="Fill in the database connection information.",
    ),
    Step(
        name="Table Selection",
        desc="Select the table you want to chat with.",
    ),
    Step(
        name="Upload Data",
        desc="Upload the data you want to retrieve with. Should be in format of .md",
    ),
    Step(
        name="Start Chatting",
        desc="Now you can start chatting with the chatbot.",
    ),
]


def render_sidebar():
    """
    Render the sidebar with navigation controls.
    """
    st.title("Flow UI")
    st.markdown(
        """
        This is a simple flow UI for the chatbot. You can follow the steps to chat with the chatbot.
        """
    )
    st.markdown(
        """
        **Note:** The chatbot will only work with the selected table.
        """
    )
    if st.button("Reset", use_container_width=True):
        init_state()
        if os.path.exists(STATE_FILE_PATH):
            os.remove(STATE_FILE_PATH)
        if os.path.exists(DOCS_DIR):
            for root, _, files in os.walk(DOCS_DIR):
                for file in files:
                    os.remove(os.path.join(root, file))
    if st.session_state.get("step", 0) > 0:
        if st.button(
            "Back",
            key="sidebar_back",
            icon="👈🏻",
            use_container_width=True,
        ):
            step_back()


def render_progress():
    """
    Render the progress bar and current step information.
    """
    current_step = steps[st.session_state.step]
    progress_text = current_step.name
    st.progress(
        (st.session_state.step + 1) / len(steps),
        text=f"Step {st.session_state.step + 1} / {len(steps)}: {progress_text}",
    )
    st.info(current_step.desc)


# Initialize page
st.set_page_config(
    page_title="Flow UI",
    page_icon="./demo/ob-icon.png",
)
st.logo("./demo/logo.png")

# Render sidebar
with st.sidebar:
    render_sidebar()

# Render progress
render_progress()


def render_database_connection_step():
    """
    Render step 0: Database connection configuration.
    """
    st.header("Database Connection")
    connection = st.session_state.get("connection", {})
    host = st.text_input(
        label="Host",
        value=connection.get("host", "127.0.0.1"),
        placeholder="Database host, e.g. 127.0.0.1",
    )
    port = st.text_input(
        label="Port",
        value=connection.get("port", "2881"),
        placeholder="Database port, e.g. 2881",
    )
    user = st.text_input(
        label="User",
        value=connection.get("user", "root@test"),
        placeholder="Database user, e.g. root@test",
    )
    db_password = st.text_input(
        label="Password",
        type="password",
        value=connection.get("password", ""),
        placeholder="Database password, empty if no password.",
    )
    db_name = st.text_input(
        label="Database",
        value=connection.get("database", "test"),
        placeholder="Database name, e.g. test",
    )
    connection_params = {
        "host": host,
        "port": port,
        "user": user,
        "password": db_password,
        "db_name": db_name,
    }
    required_values = [host, port, user, db_name]
    if st.button("Submit", type="primary"):
        if not all(required_values):
            st.error("Please fill in all the fields except password.")
            st.stop()
        try:
            engine = get_engine(**connection_params)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                if result.scalar() == 1:
                    st.session_state.connection = connection_params
                    step_forward()
                else:
                    st.error("Connection failed.")
        except Exception as e:
            st.error(f"Connection failed: {e}")


def render_table_selection_step():
    """
    Render step 1: Table selection or creation.
    """
    st.header("Table Selection")
    connection = st.session_state.connection
    engine = get_engine(**connection)

    with engine.connect() as conn:
        tables = []
        for row in conn.execute(text("SHOW TABLES")):
            tables.append(row[0])

        selecting = st.toggle("Select Table")
        if selecting:
            table = st.selectbox("Table", tables)
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            st.caption(f"Number of rows in table {table}: {count}")
            st.caption(f"Structure of table {table}:")
            st.table(conn.execute(text(f"DESC {table}")).fetchall())

        table = st.text_input(
            "Create Table",
            value=st.session_state.table,
            placeholder="Input table name to create the table if not exists.",
            disabled=selecting,
        )
    col1, col2 = st.columns(2)
    if col1.button(
        "Back",
        icon="👈🏻",
        use_container_width=True,
    ):
        step_back()
    if col2.button(
        "Submit",
        type="primary",
        icon="📤",
        use_container_width=True,
    ):
        if not table:
            st.error("Please input or select a table.")
            st.stop()
        st.session_state.table = table
        step_forward()


def render_upload_data_step():
    """
    Render step 2: Upload and process markdown documents.
    """
    st.header("Upload Data")
    connection = st.session_state.connection
    uploaded_file = st.file_uploader(
        "Choose a file",
        accept_multiple_files=True,
        type=["md"],
    )
    vs = OceanbaseVectorStore(
        embedding_function=embeddings,
        table_name=st.session_state.table,
        connection_args=connection,
        metadata_field="metadata",
    )
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    files = list(filter(lambda x: x.endswith(".md"), os.listdir(DOCS_DIR)))
    if len(files) > 0:
        st.caption(f"Uploaded {len(files)} files")
    col1, col2, col3, col4 = st.columns(4)
    if col1.button(
        "Back",
        icon="👈🏻",
        use_container_width=True,
    ):
        step_back()
    if uploaded_file is not None and col2.button(
        "Submit",
        icon="📤",
        type="primary",
        use_container_width=True,
    ):
        for file in uploaded_file:
            with open(
                os.path.join(DOCS_DIR, file.name),
                "wb",
            ) as f:
                f.write(file.getvalue())
        st.success("Files uploaded successfully.")
        st.rerun()
    if col3.button("Process", icon="⚙️", type="primary", use_container_width=True):
        total = len(files)
        batch = []
        bar = st.progress(0, text="Processing files")
        for i, file in enumerate(files):
            bar.progress((i + 1) / total, text=f"Processing {file}")
            for doc in parse_md(os.path.join(DOCS_DIR, file)):
                batch.append(doc)
                if len(batch) == DEFAULT_BATCH_SIZE:
                    vs.add_documents(
                        batch,
                        ids=[str(uuid.uuid4()) for _ in range(len(batch))],
                    )
                    batch = []
        if batch:
            vs.add_documents(
                batch,
                ids=[str(uuid.uuid4()) for _ in range(len(batch))],
            )
        st.success("Files processed successfully.")
    if col4.button(
        "Next",
        icon="👉🏻",
        use_container_width=True,
    ):
        step_forward()


if st.session_state.step == 0:
    render_database_connection_step()
elif st.session_state.step == 1:
    render_table_selection_step()
elif st.session_state.step == 2:
    render_upload_data_step()
elif st.session_state.step == 3:
    print(st.session_state)
    st.header("Upload Data")
    c = st.session_state.connection
    uploaded_file = st.file_uploader(
        "Choose a file",
        accept_multiple_files=True,
        type=["md"],
    )
    vs = OceanbaseVectorStore(
        embedding_function=embeddings,
        table_name=st.session_state.table,
        connection_args=c,
        metadata_field="metadata",
    )
    if not os.path.exists("data/uploaded/docs"):
        os.makedirs("data/uploaded/docs")
    files = list(filter(lambda x: x.endswith(".md"), os.listdir("data/uploaded/docs")))
    if len(files) > 0:
        st.caption(f"Uploaded {len(files)} files")
    col1, col2, col3, col4 = st.columns(4)
    if col1.button(
        "Back",
        icon="👈🏻",
        use_container_width=True,
    ):
        step_back()
    if uploaded_file is not None and col2.button(
        "Submit",
        icon="📤",
        type="primary",
        use_container_width=True,
    ):
        for file in uploaded_file:
            with open(
                os.path.join(
                    "data/uploaded",
                    "docs",
                    file.name,
                ),
                "wb",
            ) as f:
                f.write(file.getvalue())
        st.success("Files uploaded successfully.")
        st.rerun()
    if col3.button("Process", icon="⚙️", type="primary", use_container_width=True):
        total = len(files)
        batch = []
        bar = st.progress(0, text="Processing files")
        for i, file in enumerate(files):
            bar.progress((i + 1) / total, text=f"Processing {file}")
            for doc in parse_md(os.path.join("data/uploaded", "docs", file)):
                batch.append(doc)
                if len(batch) == 10:
                    vs.add_documents(
                        batch,
                        ids=[str(uuid.uuid4()) for _ in range(len(batch))],
                    )
                    batch = []
        if batch:
            vs.add_documents(
                batch,
                ids=[str(uuid.uuid4()) for _ in range(len(batch))],
            )
        st.success("Files processed successfully.")
    if col4.button(
        "Next",
        icon="👉🏻",
        use_container_width=True,
    ):
        step_forward()


def render_chat_step():
    """
    Render step 3: Chat interface with the chatbot.
    """
    with st.container(border=1):
        if st.button(
            "Back",
            icon="👈🏻",
            use_container_width=True,
        ):
            step_back()
        st.caption(
            "If you want to use other LLM vendors compatible with OpenAI API, "
            "please modify the following fields. The default settings are for "
            "[ZhipuAI](https://bigmodel.cn)."
        )
        llm_model = st.text_input("Model", value=DEFAULT_LLM_MODEL)
        llm_base_url = st.text_input(
            "Base URL",
            value=DEFAULT_LLM_BASE_URL,
        )
        llm_api_key = st.text_input("API Key", type="password")

    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]

    # Display chat history
    AVATAR_MAP = {
        "assistant": "🤖",
        "user": "👨🏻‍💻",
    }
    for msg in st.session_state.messages:
        avatar = AVATAR_MAP.get(msg["role"], "🤖")
        st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    # Initialize vector store
    vs = OceanbaseVectorStore(
        embedding_function=embeddings,
        table_name=st.session_state.table,
        connection_args=st.session_state.connection,
        metadata_field="metadata",
    )

    # Handle user input
    if prompt := st.chat_input("Ask something..."):
        st.chat_message("user", avatar=AVATAR_MAP["user"]).write(prompt)

        # Get recent chat history
        history = st.session_state["messages"][-DEFAULT_CHAT_HISTORY_LEN:]

        # Search for relevant documents
        docs = vs.similarity_search(prompt)
        docs_content = "\n=====\n".join([f"文档片段:\n\n" + chunk.page_content for chunk in docs])

        # Create RAG agent and stream response
        universal_rag_agent = AgentBase(
            prompt=universal_rag_prompt,
            llm_model=llm_model or DEFAULT_LLM_MODEL,
        )
        ans_itr = universal_rag_agent.stream(
            prompt,
            history,
            document_snippets=docs_content,
        )
        res = StreamResponse(ans_itr)

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Build reference list
        def ref_generator():
            yield REF_TIP
            visited = set()
            for chunk in docs:
                if chunk.metadata["doc_url"] not in visited:
                    visited.add(chunk.metadata["doc_url"])
                    yield f"\n* [{chunk.metadata['doc_name']}]({chunk.metadata['doc_url']})"

        # Display assistant response
        st.chat_message("assistant", avatar=AVATAR_MAP["assistant"]).write_stream(
            res.generate(suffix=ref_generator())
        )

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": res.get_whole()})


if st.session_state.step == 3:
    render_chat_step()
