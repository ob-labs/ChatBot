import os
import time
import json
import uuid
import glob
from typing import Iterator, Union

import dotenv
from langchain_oceanbase.vectorstores import OceanbaseVectorStore
from langchain_core.messages import BaseMessageChunk
from pydantic import BaseModel
from sqlalchemy import create_engine, text

dotenv.load_dotenv()

import streamlit as st

from src.agents.base import AgentBase
from src.agents.universe_rag_agent import prompt as universal_rag_prompt
from src.rag.documents import parse_md
from src.rag.embeddings import OllamaEmbedding


def init_state():
    st.session_state.step = 0
    st.session_state.table = ""
    st.session_state.connection = {}


if "step" not in st.session_state:
    init_state()


def save_state():
    with open("./data/uploaded/state.json", "w") as f:
        d = {
            "step": st.session_state.step,
            "table": st.session_state.table,
            "connection": st.session_state.connection,
        }
        json.dump(d, f)


def load_state():
    if os.path.exists("./data/uploaded/state.json"):
        with open("./uploaded/state.json", "r") as f:
            d = json.load(f)
            st.session_state.update(d)


load_state()


class StreamResponse:
    """
    StreamResponse is a class that helps to stream the response from the chatbot.
    """

    def __init__(self, chunks: Iterator[BaseMessageChunk] = []):
        self.chunks = chunks
        self.__whole_msg = ""

    def generate(
        self,
        *,
        prefix: Union[Iterator, None] = None,
        suffix: Union[Iterator, None] = None,
    ) -> Iterator[str]:
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
        return self.__whole_msg


ref_tip = "\n\n检索到的相关文档如下:"


def remove_refs(history: list[dict]) -> list[dict]:
    """
    Remove the references from the chat history.
    This prevents the model from generating its own reference list.
    """
    return [
        {
            "role": msg["role"],
            "content": msg["content"].split(ref_tip)[0],
        }
        for msg in history
    ]


def get_engine(**c):
    return create_engine(
        f"mysql+pymysql://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['db_name']}"
    )


embeddings = OllamaEmbedding(
    url="http://30.249.224.105:8080/api/embed",
    token="test",
)


def step_forward():
    st.session_state.step += 1
    save_state()
    st.rerun()


def step_back():
    st.session_state.step -= 1
    save_state()
    st.rerun()


class Step(BaseModel):
    name: str
    desc: str | None = None
    form: dict | None = None


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

st.set_page_config(
    page_title="Flow UI",
    page_icon="./demo/ob-icon.png",
)
st.logo("./demo/logo.png")

with st.sidebar:
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
        if os.path.exists("./data/uploaded/state.json"):
            os.remove("./data/uploaded/state.json")
        if os.path.exists("./data/uploaded/docs"):
            for root, _, f in os.walk("./data/uploaded/docs"):
                for file in f:
                    os.remove(os.path.join(root, file))
    if st.session_state.get("step", 0) > 0:
        if st.button(
            "Back",
            key="sidebar_back",
            icon="👈🏻",
            use_container_width=True,
        ):
            step_back()

progress_text = steps[st.session_state.step].name
my_bar = st.progress(
    (st.session_state.step + 1) / len(steps),
    text=f"Step {st.session_state.step + 1} / {len(steps)}: {progress_text}",
)
st.info(steps[st.session_state.step].desc)

if st.session_state.step == 0:
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
    c = {
        "host": host,
        "port": port,
        "user": user,
        "password": db_password,
        "db_name": db_name,
    }
    values = [host, port, user, db_name]
    if st.button("Submit", type="primary"):
        if not all(values):
            st.error("Please fill in all the fields except password.")
            st.stop()
        try:
            engine = get_engine(**c)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                if result.scalar() == 1:
                    st.session_state.connection = c
                    step_forward()
                else:
                    st.error("Connection failed.")
        except Exception as e:
            st.error(f"Connection failed: {e}")

elif st.session_state.step == 1:
    st.header("Table Selection")
    print(st.session_state)
    c = st.session_state.connection
    engine = get_engine(**c)

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
elif st.session_state.step == 2:
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

elif st.session_state.step == 3:
    print(st.session_state)
    with st.container(border=1):
        if st.button(
            "Back",
            icon="👈🏻",
            use_container_width=True,
        ):
            step_back()
        st.caption(
            "If you want to use other LLM vendors compatible with OpenAI API, please modify the following fields. The default settings are for [ZhipuAI](https://bigmodel.cn)."
        )
        llm_model = st.text_input("Model", value="glm-4-flash")
        llm_base_url = st.text_input(
            "Base URL",
            value="https://open.bigmodel.cn/api/paas/v4/",
        )
        llm_api_key = st.text_input("API Key", type="password")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]

    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👨🏻‍💻"
        st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    vs = OceanbaseVectorStore(
        embedding_function=embeddings,
        table_name=st.session_state.table,
        connection_args=st.session_state.connection,
        metadata_field="metadata",
    )
    if prompt := st.chat_input("Ask something..."):
        st.chat_message("user", avatar="👨🏻‍💻").write(prompt)

        history = st.session_state["messages"][-4:]

        docs = vs.similarity_search(prompt)
        docs_content = "\n=====\n".join(
            [f"文档片段:\n\n" + chunk.page_content for i, chunk in enumerate(docs)]
        )
        universal_prompt = universal_rag_prompt
        universal_rag_agent = AgentBase(
            prompt=universal_prompt,
            llm_model="glm-4-flash",
        )
        ans_itr = universal_rag_agent.stream(
            prompt,
            history,
            document_snippets=docs_content,
        )
        res = StreamResponse(ans_itr)

        st.session_state.messages.append({"role": "user", "content": prompt})

        refs = [ref_tip]
        visited = set()
        for i, chunk in enumerate(docs):
            if chunk.metadata["doc_url"] not in visited:
                visited.add(chunk.metadata["doc_url"])
                refs.append(
                    f"\n* [{chunk.metadata['doc_name']}]({chunk.metadata['doc_url']})"
                )

        st.chat_message("assistant", avatar="🤖").write_stream(
            res.generate(suffix=refs)
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": res.get_whole()}
        )
