"""
Chat UI module for the document RAG chatbot.

This module provides a Streamlit-based chat interface with configurable
settings for LLM model selection, chat history management, and document search.
It also supports navigation between Chat and Load Docs pages.
"""

from dataclasses import dataclass
from typing import Iterator, Union

import streamlit as st
from langchain_core.messages import BaseMessageChunk

from src.common.config import (
    DEFAULT_HISTORY_LEN,
    DEFAULT_LLM_BASE_URL_UI,
    DEFAULT_LLM_MODELS,
    EmbeddingConfig,
    MAX_HISTORY_LEN,
    UI_AVATAR_MAP_CHAT,
    UI_LOGO_PATH,
    UI_PAGE_ICON,
    get_llm_base_url,
    get_llm_model,
    get_table_name,
    get_ui_lang,
)
from src.common.logger import get_logger
from src.frontend.i18n import t
from src.frontend.menu import MENU_CHAT, MENU_LOAD_DOCS, MenuNavigator
from src.rag.rag import doc_rag_stream

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ChatSettings:
    """Configuration settings for chat behavior."""

    llm_model: str
    history_len: int
    search_docs: bool
    show_refs: bool
    oceanbase_only: bool
    rerank: bool


# =============================================================================
# Helper Classes
# =============================================================================


class StreamResponse:
    """
    Helper class for streaming chatbot responses.

    Accumulates message chunks and provides methods to generate
    the response stream with optional prefix/suffix.
    """

    def __init__(
        self, chunks: Union[Iterator[Union[str, BaseMessageChunk]], None] = None
    ):
        """
        Initialize StreamResponse with message chunks.

        Args:
            chunks: Iterator of BaseMessageChunk or string objects
        """
        self.chunks: Iterator[Union[str, BaseMessageChunk]] = (
            chunks if chunks is not None else iter([])
        )
        self._whole_msg = ""

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
            # Handle both string and BaseMessageChunk types
            if isinstance(chunk, str):
                self._whole_msg += chunk
                yield chunk
            elif hasattr(chunk, "content"):
                content = chunk.content
                if isinstance(content, str):
                    self._whole_msg += content
                    yield content
        if suffix:
            yield suffix

    def get_whole(self) -> str:
        """
        Get the complete accumulated message.

        Returns:
            Complete message as string
        """
        return self._whole_msg


# =============================================================================
# UI Components
# =============================================================================


class PageConfigurator:
    """Handles page-level configuration and setup."""

    def __init__(self, lang: str):
        """
        Initialize page configurator.

        Args:
            lang: Language code for UI text
        """
        self.lang = lang

    def setup(self) -> None:
        """Configure Streamlit page settings and display title."""
        st.set_page_config(
            page_title=t("title", self.lang),
            page_icon=UI_PAGE_ICON,
        )
        st.logo(UI_LOGO_PATH)


class SidebarConfigurator:
    """Handles sidebar configuration and settings input."""

    def __init__(self, lang: str):
        """
        Initialize sidebar configurator.

        Args:
            lang: Language code for UI text
        """
        self.lang = lang
        self._table_name = get_table_name()
        self._llm_base_url = get_llm_base_url()
        logger.debug(
            f"Sidebar initialized: table_name={self._table_name}, "
            f"llm_base_url={self._llm_base_url}"
        )

    def render(self) -> ChatSettings:
        """
        Render sidebar and collect user settings.

        Returns:
            ChatSettings containing all user-configured options
        """
        with st.sidebar:
            st.subheader(t("setting", self.lang))
            self._render_info_fields()
            llm_model = self._render_llm_selector()
            return self._render_chat_options(llm_model)

    def _render_info_fields(self) -> None:
        """Render read-only information fields."""
        st.text_input(
            t("lang_input", self.lang),
            value=self.lang,
            disabled=True,
            help=t("lang_help", self.lang),
        )
        st.text_input(
            t("table_name_input", self.lang),
            value=self._table_name,
            disabled=True,
            help=t("table_name_help", self.lang),
        )

    def _render_llm_selector(self) -> str:
        """
        Render LLM model selector based on configuration.

        Returns:
            Selected LLM model name
        """
        if self._llm_base_url == DEFAULT_LLM_BASE_URL_UI:
            return st.selectbox(
                t("llm_model", self.lang),
                DEFAULT_LLM_MODELS,
                index=0,
                help=t("llm_model_help", self.lang),
            )
        else:
            return st.text_input(
                t("llm_model", self.lang),
                value=get_llm_model(),
            )

    def _render_chat_options(self, llm_model: str) -> ChatSettings:
        """
        Render chat configuration options.

        Args:
            llm_model: Selected LLM model name

        Returns:
            ChatSettings with all configured options
        """
        history_len = st.slider(
            t("chat_history_len", self.lang),
            min_value=0,
            max_value=MAX_HISTORY_LEN,
            value=DEFAULT_HISTORY_LEN,
            help=t("chat_history_len_help", self.lang),
        )
        search_docs = st.checkbox(
            t("search_docs", self.lang),
            True,
            help=t("search_docs_help", self.lang),
        )
        show_refs = st.checkbox(
            t("show_refs", self.lang),
            True,
            help=t("show_refs_help", self.lang),
        )
        oceanbase_only = st.checkbox(
            t("oceanbase_only", self.lang),
            True,
            help=t("oceanbase_only_help", self.lang),
        )
        rerank = st.checkbox(
            t("rerank", self.lang),
            False,
            help=t("rerank_help", self.lang),
        )

        return ChatSettings(
            llm_model=llm_model,
            history_len=history_len,
            search_docs=search_docs,
            show_refs=show_refs,
            oceanbase_only=oceanbase_only,
            rerank=rerank,
        )


class ChatMessageHandler:
    """Handles chat message display and processing."""

    def __init__(self, lang: str, settings: ChatSettings):
        """
        Initialize chat message handler.

        Args:
            lang: Language code for UI text
            settings: Chat configuration settings
        """
        self.lang = lang
        self.settings = settings
        self._init_session_state()

    def _init_session_state(self) -> None:
        """Initialize session state for chat messages."""
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": t("hello", self.lang)}
            ]

    def display_history(self) -> None:
        """Display all messages in chat history."""
        for msg in st.session_state.messages:
            st.chat_message(msg["role"], avatar=UI_AVATAR_MAP_CHAT[msg["role"]]).write(
                msg["content"]
            )

    def process_user_input(self) -> None:
        """Process user input and generate response."""
        prompt = st.chat_input(t("chat_placeholder", lang=self.lang))
        if not prompt:
            return

        logger.info(f"User input received: prompt length={len(prompt)}, lang={self.lang}")
        self._display_user_message(prompt)
        self._generate_and_display_response(prompt)

    def _display_user_message(self, prompt: str) -> None:
        """Display user message in chat."""
        st.chat_message("user", avatar=UI_AVATAR_MAP_CHAT["user"]).write(prompt)

    def _generate_and_display_response(self, prompt: str) -> None:
        """Generate RAG response and display it."""
        history = self._get_chat_history()
        response_iterator = self._create_rag_stream(prompt, history)

        self._display_processing_status(response_iterator)

        response_handler = StreamResponse(response_iterator)
        self._add_message("user", prompt)
        self._display_assistant_response(response_handler)
        self._add_message("assistant", response_handler.get_whole())

    def _get_chat_history(self) -> list[dict]:
        """
        Get recent chat history based on settings.

        Returns:
            List of recent messages with references removed
        """
        history_len = self.settings.history_len
        if history_len > 0:
            history = st.session_state["messages"][-history_len:]
        else:
            history = []

        logger.debug(f"Using chat history: length={len(history)}, history_len={history_len}")
        return self._remove_refs(history)

    def _remove_refs(self, history: list[dict]) -> list[dict]:
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
                "content": msg["content"].split(t("ref_tips", self.lang))[0],
            }
            for msg in history
        ]

    def _create_rag_stream(
        self, prompt: str, history: list[dict]
    ) -> Iterator[Union[str, BaseMessageChunk]]:
        """
        Create RAG response stream.

        Args:
            prompt: User query
            history: Chat history

        Returns:
            Iterator yielding status messages and response chunks
        """
        logger.info(
            f"Starting RAG stream: oceanbase_only={self.settings.oceanbase_only}, "
            f"rerank={self.settings.rerank}, search_docs={self.settings.search_docs}, "
            f"llm_model={self.settings.llm_model}"
        )
        return doc_rag_stream(
            query=prompt,
            chat_history=history,
            llm_model=self.settings.llm_model,
            embedding_config=EmbeddingConfig.from_env(),
            universal_rag=not self.settings.oceanbase_only,
            rerank=self.settings.rerank,
            search_docs=self.settings.search_docs,
            lang=self.lang,
            show_refs=self.settings.show_refs,
        )

    def _display_processing_status(
        self, iterator: Iterator[Union[str, BaseMessageChunk]]
    ) -> None:
        """Display processing status while generating response."""
        with st.status(t("processing", self.lang), expanded=True) as status:
            for msg in iterator:
                if not isinstance(msg, str):
                    status.update(label=t("finish_thinking", self.lang))
                    logger.debug("RAG stream processing completed")
                    break
                st.write(msg)

    def _display_assistant_response(self, response_handler: StreamResponse) -> None:
        """Display streamed assistant response."""
        st.chat_message("assistant", avatar=UI_AVATAR_MAP_CHAT["assistant"]).write_stream(
            response_handler.generate()
        )

    def _add_message(self, role: str, content: str) -> None:
        """
        Add message to session state.

        Args:
            role: Message role ("user" or "assistant")
            content: Message content
        """
        st.session_state.messages.append({"role": role, "content": content})
        logger.debug(f"{role.capitalize()} message added to chat history, length: {len(content)}")


# =============================================================================
# Main Application
# =============================================================================


def run_chat_page(lang: str) -> None:
    """
    Run the chat page.

    Args:
        lang: Language code for UI text
    """
    # Show title and caption
    st.title(t("title", lang))
    st.caption(t("caption", lang))

    # Render sidebar settings and get configuration
    sidebar = SidebarConfigurator(lang)
    settings = sidebar.render()

    # Handle chat messages
    chat_handler = ChatMessageHandler(lang, settings)
    chat_handler.display_history()
    chat_handler.process_user_input()


def run_load_docs_page() -> None:
    """Run the load docs page (flow_ui)."""
    # Import flow_ui module and run its content
    # show_menu=False because menu is already rendered in main()
    from src.frontend.flow_ui import FlowController
    controller = FlowController()
    controller.run_content(show_menu=False)


def main() -> None:
    """Main entry point for the chat UI application."""
    # Initialize language setting
    lang = get_ui_lang()
    logger.info(f"""
                --------------------------------
                Initializing chat UI with language: {lang}
                --------------------------------
                """)

    # Setup base page configuration (only once)
    page_config = PageConfigurator(lang)
    page_config.setup()

    # Render menu buttons in top right corner
    menu_nav = MenuNavigator(lang)
    menu_nav.apply_css()
    selected_page = menu_nav.render_menu()

    # Route to selected page
    if selected_page == MENU_CHAT:
        run_chat_page(lang)
    elif selected_page == MENU_LOAD_DOCS:
        run_load_docs_page()


# Run the application only when executed directly
if __name__ == "__main__":
    main()
