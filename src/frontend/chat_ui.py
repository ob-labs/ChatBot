from typing import Iterator, Union

import streamlit as st
from langchain_core.messages import BaseMessageChunk

from src.common.config import (
    DEFAULT_HISTORY_LEN,
    DEFAULT_LLM_BASE_URL_UI,
    DEFAULT_LLM_MODELS,
    DEFAULT_TABLE_NAME,
    DEFAULT_UI_LANG,
    MAX_HISTORY_LEN,
    SUPPORTED_LANGUAGES,
    get_llm_base_url,
    get_llm_model,
    get_table_name,
    get_ui_lang,
)
from src.common.logger import get_logger
from src.frontend.i18n import t
from src.rag.doc_rag import doc_rag_stream

logger = get_logger(__name__)


class StreamResponse:
    """
    Helper class for streaming chatbot responses.

    Accumulates message chunks and provides methods to generate
    the response stream with optional prefix/suffix.
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
        prefix: Union[str, None] = None,
        suffix: Union[str, None] = None,
    ) -> Iterator[str]:
        """
        Generate response stream with optional prefix and suffix.

        Args:
            prefix: Optional string to yield before chunks
            suffix: Optional string to yield after chunks

        Yields:
            Response text chunks as strings
        """
        if prefix:
            yield prefix
        for chunk in self.chunks:
            self.__whole_msg += chunk.content
            yield chunk.content
        if suffix:
            yield suffix

    def get_whole(self) -> str:
        """
        Get the complete accumulated message.

        Returns:
            Complete message as string
        """
        return self.__whole_msg


def get_language() -> str:
    """
    Get the UI language from environment variable.

    Returns:
        Language code ("zh" or "en"), defaults to "zh"
    """
    return get_ui_lang()


def remove_refs(history: list[dict], lang: str) -> list[dict]:
    """
    Remove reference sections from chat history.

    This prevents the model from generating its own reference list
    by removing content after the reference marker.

    Args:
        history: List of message dictionaries
        lang: Language code for reference marker text

    Returns:
        List of messages with references removed
    """
    return [
        {
            "role": msg["role"],
            "content": msg["content"].split(t("ref_tips", lang))[0],
        }
        for msg in history
    ]


# Initialize page configuration
lang = get_language()
logger.info(f"Initializing chat UI with language: {lang}")
st.set_page_config(
    page_title=t("title", lang),
    page_icon="demo/ob-icon.png",
)
st.title(t("title", lang))
st.caption(t("caption", lang))
st.logo("demo/logo.png")

# Get environment configuration
env_table_name = get_table_name()
env_llm_base_url = get_llm_base_url()
logger.debug(f"Chat UI configuration: table_name={env_table_name}, llm_base_url={env_llm_base_url}")

with st.sidebar:
    st.subheader(t("setting", lang))
    st.text_input(
        t("lang_input", lang),
        value=lang,
        disabled=True,
        help=t("lang_help", lang),
    )
    st.text_input(
        t("table_name_input", lang),
        value=env_table_name,
        disabled=True,
        help=t("table_name_help", lang),
    )
    # LLM model selection
    if env_llm_base_url == DEFAULT_LLM_BASE_URL_UI:
        llm_model = st.selectbox(
            t("llm_model", lang),
            DEFAULT_LLM_MODELS,
            index=0,
            help=t("llm_model_help", lang),
        )
    else:
        llm_model = st.text_input(
            t("llm_model", lang),
            value=get_llm_model(),
        )

    # Chat history length configuration
    history_len = st.slider(
        t("chat_history_len", lang),
        min_value=0,
        max_value=MAX_HISTORY_LEN,
        value=DEFAULT_HISTORY_LEN,
        help=t("chat_history_len_help", lang),
    )
    search_docs = st.checkbox(
        t("search_docs", lang),
        True,
        help=t("search_docs_help", lang),
    )
    show_refs = st.checkbox(
        t("show_refs", lang),
        True,
        help=t("show_refs_help", lang),
    )
    oceanbase_only = st.checkbox(
        t("oceanbase_only", lang),
        True,
        help=t("oceanbase_only_help", lang),
    )
    rerank = st.checkbox(
        t("rerank", lang),
        False,
        help=t("rerank_help", lang),
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": t("hello", lang)}]

# Avatar configuration for chat messages
AVATAR_MAP = {
    "assistant": "demo/ob-icon.png",
    "user": "🧑‍💻",
}

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"], avatar=AVATAR_MAP[msg["role"]]).write(msg["content"])


# Handle user input
if prompt := st.chat_input(t("chat_placeholder", lang=lang)):
    logger.info(f"User input received: prompt length={len(prompt)}, lang={lang}")
    st.chat_message("user", avatar=AVATAR_MAP["user"]).write(prompt)

    # Get recent chat history
    history = st.session_state["messages"][-history_len:] if history_len > 0 else []
    logger.debug(f"Using chat history: length={len(history)}, history_len={history_len}")

    # Stream RAG response
    logger.info(f"Starting RAG stream: oceanbase_only={oceanbase_only}, rerank={rerank}, search_docs={search_docs}, llm_model={llm_model}")
    it = doc_rag_stream(
        query=prompt,
        chat_history=remove_refs(history, lang),
        universal_rag=not oceanbase_only,
        rerank=rerank,
        llm_model=llm_model,
        search_docs=search_docs,
        lang=lang,
        show_refs=show_refs,
    )

    # Display processing status
    with st.status(t("processing", lang), expanded=True) as status:
        for msg in it:
            if not isinstance(msg, str):
                status.update(label=t("finish_thinking", lang))
                logger.debug("RAG stream processing completed")
                break
            st.write(msg)

    # Create response stream handler
    res = StreamResponse(it)

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    logger.debug("User message added to chat history")

    # Display assistant response
    st.chat_message("assistant", avatar=AVATAR_MAP["assistant"]).write_stream(res.generate())

    # Add assistant response to history
    response_content = res.get_whole()
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    logger.info(f"Assistant response added to chat history, response length: {len(response_content)}")
