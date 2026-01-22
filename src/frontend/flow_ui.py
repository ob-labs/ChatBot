"""
Flow UI module for the document RAG chatbot.

This module provides a step-by-step wizard interface for:
1. View current configuration status
2. Configure Chat LLM settings
3. Configure Embedding LLM settings
4. Configure RAG parser settings
5. Upload and load documents
"""

import os
import shutil
from typing import Callable, List, Optional

import dotenv
import streamlit as st
from pydantic import BaseModel

from src.common.compress import extract_archive, is_archive_file
from src.common.file_path import is_markdown_file
from src.common.download import clone_github_repo
from src.common.config import (
    DEFAULT_TABLE_NAME,
    FLOW_UI_PAGE_TITLE,
    UI_LOGO_PATH,
    UI_PAGE_ICON,
    EmbeddedType,
    EmbeddingConfig,
    LLMConfig,
    RAGParserConfig,
    get_db_host,
    get_db_name,
    get_db_password_raw,
    get_db_port,
    get_db_user,
    get_env,
    get_table_name,
    get_ui_lang,
)
from src.common.db import (
    ConnectionParams,
    DatabaseClient,
    append_partition,
    get_partition_map,
)
from src.common.logger import get_logger
from src.frontend.i18n import t
from src.frontend.menu import MenuNavigator
from src.rag.rag import DocumentEmbedder
from src.rag.ob import component_mapping, supported_components
from src.common.config import DEFAULT_EMBEDDING_MODEL_NAME, DEFAULT_BGE_HF_REPO_ID

dotenv.load_dotenv()

logger = get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Directory paths
DATA_DIR = "./data"
UPLOADED_DIR = os.path.join(DATA_DIR, "uploaded")
LOADED_DIR = os.path.join(DATA_DIR, "loaded")
TMP_DIR = os.path.join(DATA_DIR, "tmp")

# Flow steps
FLOW_STEP_STATUS = 0
FLOW_STEP_CHAT_LLM = 1
FLOW_STEP_EMBED_LLM = 2
FLOW_STEP_RAG_PARSER = 3
FLOW_STEP_UPLOAD = 4


# =============================================================================
# Data Classes
# =============================================================================


class StepDefinition(BaseModel):
    """Represents a step in the flow UI."""
    name_key: str
    desc_key: Optional[str] = None

    def get_name(self, lang: str) -> str:
        """Get translated step name."""
        return t(self.name_key, lang)

    def get_desc(self, lang: str) -> str:
        """Get translated step description."""
        return t(self.desc_key, lang) if self.desc_key else ""


# Define all flow steps with translation keys
FLOW_STEPS: list[StepDefinition] = [
    StepDefinition(
        name_key="flow_step_status",
        desc_key="flow_step_status_desc",
    ),
    StepDefinition(
        name_key="flow_step_chat_llm",
        desc_key="flow_step_chat_llm_desc",
    ),
    StepDefinition(
        name_key="flow_step_embedding",
        desc_key="flow_step_embedding_desc",
    ),
    StepDefinition(
        name_key="flow_step_rag_parser",
        desc_key="flow_step_rag_parser_desc",
    ),
    StepDefinition(
        name_key="flow_step_upload",
        desc_key="flow_step_upload_desc",
    ),
]


# =============================================================================
# State Management
# =============================================================================


class StateManager:
    """
    Manages session state persistence and operations.

    Configuration classes (LLMConfig, EmbeddingConfig, RAGParserConfig)
    automatically load from .env file via their from_env() methods.
    """

    def __init__(self):
        """Initialize state manager."""
        self._ensure_session_state()

    def _ensure_session_state(self) -> None:
        """Ensure session state has required keys."""
        if "step" not in st.session_state:
            logger.info("Initializing new session state")
            self.reset()
        if "state_manager" not in st.session_state:
            self._init_state_from_env()

    def _init_state_from_env(self) -> None:
        """Initialize state using config classes that load from .env."""
        logger.info("Initializing state from config classes")

        # Config classes load from .env via their from_env() methods
        chat_llm_config = LLMConfig.from_env()
        embedding_config = EmbeddingConfig.from_env()
        rag_parser_config = RAGParserConfig.from_env()

        # Store in session state
        st.session_state.state_manager = {
            "chat_llm_config": chat_llm_config.model_dump(),
            "embedding_config": embedding_config.model_dump(),
            "rag_parser_config": rag_parser_config.model_dump(),
        }

        # Database connection
        st.session_state.connection = {
            "host": get_db_host(),
            "port": get_db_port(),
            "user": get_db_user(),
            "password": get_db_password_raw(),
            "db_name": get_db_name(),
        }
        st.session_state.table = get_table_name()

    def reset(self) -> None:
        """Reset session state to default values."""
        logger.debug("Resetting session state")
        st.session_state.step = FLOW_STEP_STATUS
        if "state_manager" in st.session_state:
            del st.session_state.state_manager
        self._init_state_from_env()

    def step_forward(self) -> None:
        """Move to the next step."""
        st.session_state.step += 1
        st.rerun()

    def step_back(self) -> None:
        """Move to the previous step."""
        st.session_state.step -= 1
        st.rerun()

    def go_to_step(self, step: int) -> None:
        """Go to a specific step."""
        st.session_state.step = step
        st.rerun()

    @property
    def current_step(self) -> int:
        """Get current step index."""
        return st.session_state.get("step", FLOW_STEP_STATUS)

    @property
    def chat_llm_config(self) -> LLMConfig:
        """Get chat LLM configuration."""
        data = st.session_state.state_manager.get("chat_llm_config", {})
        return LLMConfig.model_validate(data)

    @chat_llm_config.setter
    def chat_llm_config(self, config: LLMConfig) -> None:
        """Set chat LLM configuration."""
        st.session_state.state_manager["chat_llm_config"] = config.model_dump()

    @property
    def embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration."""
        data = st.session_state.state_manager.get("embedding_config", {})
        return EmbeddingConfig.model_validate(data)

    @embedding_config.setter
    def embedding_config(self, config: EmbeddingConfig) -> None:
        """Set embedding configuration."""
        st.session_state.state_manager["embedding_config"] = config.model_dump()

    @property
    def rag_parser_config(self) -> RAGParserConfig:
        """Get RAG parser configuration."""
        data = st.session_state.state_manager.get("rag_parser_config", {})
        return RAGParserConfig.model_validate(data)

    @rag_parser_config.setter
    def rag_parser_config(self, config: RAGParserConfig) -> None:
        """Set RAG parser configuration."""
        st.session_state.state_manager["rag_parser_config"] = config.model_dump()

    @property
    def table_name(self) -> str:
        """Get selected table name."""
        return st.session_state.get("table", "")

    def get_connection_params(self) -> ConnectionParams:
        """Get connection parameters as ConnectionParams object."""
        conn = st.session_state.get("connection", {})
        return ConnectionParams.from_dict(conn)

    def get_db_client(self) -> DatabaseClient:
        """Get database client with current connection params."""
        return DatabaseClient(self.get_connection_params())


# =============================================================================
# Helper Functions
# =============================================================================


def get_lang() -> str:
    """Get current UI language."""
    return get_ui_lang()


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    for dir_path in [UPLOADED_DIR, LOADED_DIR, TMP_DIR]:
        os.makedirs(dir_path, exist_ok=True)


def get_loaded_modules() -> List[dict]:
    """
    Get list of loaded modules from loaded directory.

    Returns:
        List of dicts with module info (name, path, file_count)
    """
    modules = []
    if not os.path.exists(LOADED_DIR):
        return modules

    for name in os.listdir(LOADED_DIR):
        module_path = os.path.join(LOADED_DIR, name)
        if os.path.isdir(module_path):
            # Count markdown files
            file_count = 0
            for root, _, files in os.walk(module_path):
                file_count += sum(1 for f in files if is_markdown_file(f))
            modules.append({
                "name": name,
                "path": module_path,
                "file_count": file_count,
            })
    return modules




# =============================================================================
# UI Components
# =============================================================================


class PageConfigurator:
    """Handles page-level configuration."""

    @staticmethod
    def setup() -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(page_title=FLOW_UI_PAGE_TITLE, page_icon=UI_PAGE_ICON)
        st.logo(UI_LOGO_PATH)
        # Apply menu CSS
        menu_nav = MenuNavigator()
        menu_nav.apply_css()


class SidebarRenderer:
    """Renders the sidebar with navigation controls."""

    def __init__(self, state_manager: StateManager):
        """Initialize sidebar renderer."""
        self._state = state_manager
        self._lang = get_lang()

    def render(self) -> None:
        """Render the sidebar."""
        with st.sidebar:
            self._render_header()
            # Use a unique key to avoid duplicate key errors
            if st.button(t("flow_reset_config", self._lang), key="flow_ui_sidebar_reset_config", use_container_width=True):
                self._state.reset()
                st.rerun()

    def _render_header(self) -> None:
        """Render sidebar header."""
        st.title(t("flow_sidebar_title", self._lang))
        st.markdown(t("flow_sidebar_desc", self._lang))


class ProgressRenderer:
    """Renders the progress indicator."""

    def __init__(self, state_manager: StateManager):
        """Initialize progress renderer."""
        self._state = state_manager
        self._lang = get_lang()

    def render(self) -> None:
        """Render progress bar and current step info."""
        current_step = FLOW_STEPS[self._state.current_step]
        total_steps = len(FLOW_STEPS)

        progress_text = t("flow_step_progress", self._lang).format(
            self._state.current_step + 1, total_steps, current_step.get_name(self._lang)
        )
        st.progress(
            (self._state.current_step + 1) / total_steps,
            text=progress_text,
        )
        st.info(current_step.get_desc(self._lang))


# =============================================================================
# Step Renderers
# =============================================================================


class StepRendererBase:
    """Base class for step renderers."""

    def __init__(self, state_manager: StateManager):
        """Initialize step renderer."""
        self._state = state_manager
        self._lang = get_lang()

    def render(self) -> None:
        """Render the step. Override in subclasses."""
        raise NotImplementedError

    def _render_navigation_buttons(
        self,
        show_back: bool = True,
        show_next: bool = True,
        next_label: Optional[str] = None,
        next_callback: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Render navigation buttons."""
        if next_label is None:
            next_label = t("flow_next_step", self._lang)
        cols = st.columns(2 if show_back and show_next else 1)

        if show_back and self._state.current_step > 0:
            if cols[0].button(t("flow_prev_step", self._lang), use_container_width=True):
                self._state.step_back()

        if show_next:
            col_idx = 1 if show_back else 0
            if cols[col_idx].button(
                f"➡️ {next_label}",
                type="primary",
                use_container_width=True,
            ):
                if next_callback is None or next_callback():
                    self._state.step_forward()


class StatusStep(StepRendererBase):
    """Renders the status overview step."""

    def render(self) -> None:
        """Render status overview."""
        st.header(t("flow_status_header", self._lang))

        # Chat LLM Settings
        st.subheader(t("flow_chat_llm_settings", self._lang))
        chat_config = self._state.chat_llm_config
        col1, col2, col3 = st.columns(3)
        col1.write(f"**API Key:** {t('flow_status_set', self._lang) if chat_config.api_key else t('flow_status_not_set', self._lang)}")
        col2.write(f"**Model:** {chat_config.model or t('flow_status_not_configured', self._lang)}")
        base_url_display = chat_config.base_url[:30] + "..." if len(chat_config.base_url) > 30 else chat_config.base_url or t("flow_status_not_configured", self._lang)
        col3.write(f"**Base URL:** {base_url_display}")

        # Embedding LLM Settings
        st.subheader(t("flow_embedding_settings", self._lang))
        embed_config = self._state.embedding_config

        col1, col2, col3 = st.columns(3)
        col1.write(f"**Embedding Type:** {embed_config.embedded_type.value}")

        if embed_config.embedded_type == EmbeddedType.DEFAULT:
            col2.write(f"**{t('flow_default_model', self._lang)}:** {DEFAULT_EMBEDDING_MODEL_NAME}")
        elif embed_config.embedded_type == EmbeddedType.LOCAL_MODEL:
            col2.write(f"**{t('flow_local_model', self._lang)}:** {embed_config.model or t('flow_status_not_configured', self._lang)}")
        elif embed_config.embedded_type == EmbeddedType.OLLAMA:
            # Ollama only needs model and base_url, no api_key
            col2.write(f"**Model:** {embed_config.model or t('flow_status_not_configured', self._lang)}")
        else:  # OpenAI_Embedding
            col2.write(f"**API Key:** {t('flow_status_set', self._lang) if embed_config.api_key else t('flow_status_not_set', self._lang)}")

        col3.write(f"**Dimension:** {embed_config.dimension}")

        # RAG Parser Settings
        st.subheader(t("flow_rag_settings", self._lang))
        rag_config = self._state.rag_parser_config
        col1, col2, col3 = st.columns(3)
        col1.write(f"**Max Chunk Size:** {rag_config.max_chunk_size}")
        col2.write(f"**Limit:** {rag_config.limit if rag_config.limit > 0 else t('flow_unlimited', self._lang)}")
        col3.write(f"**Skip Patterns:** {rag_config.skip_patterns or t('flow_none', self._lang)}")

        # Loaded Modules
        st.subheader(t("flow_loaded_modules", self._lang))
        modules = get_loaded_modules()
        if modules:
            for module in modules:
                with st.expander(f"📁 {module['name']}"):
                    st.write(f"**{t('flow_dir_path', self._lang)}** {module['path']}")
                    st.write(f"**{t('flow_file_count', self._lang)}** {module['file_count']}")
        else:
            st.info(t("flow_no_modules", self._lang))

        # Navigation
        st.markdown("---")
        self._render_navigation_buttons(show_back=False)


class ChatLLMStep(StepRendererBase):
    """Renders Chat LLM configuration step."""

    def render(self) -> None:
        """Render Chat LLM configuration."""
        st.header(t("flow_chat_llm_header", self._lang))

        config = self._state.chat_llm_config

        st.info(t("flow_chat_llm_info", self._lang))

        api_key = st.text_input(
            "API Key",
            value=config.api_key,
            type="password",
            help=t("flow_api_key_help", self._lang),
        )

        model = st.text_input(
            "Model",
            value=config.model,
            help=t("flow_model_help", self._lang),
        )

        base_url = st.text_input(
            "Base URL",
            value=config.base_url,
            help=t("flow_base_url_help", self._lang),
        )

        # Update config in session state
        new_config = LLMConfig(api_key=api_key, model=model, base_url=base_url)
        self._state.chat_llm_config = new_config

        # Navigation
        st.markdown("---")

        def validate() -> bool:
            if not new_config.is_valid():
                st.error(t("flow_fill_all_fields", self._lang))
                return False
            return True

        self._render_navigation_buttons(next_callback=validate)


class EmbeddingLLMStep(StepRendererBase):
    """Renders Embedding LLM configuration step."""

    def render(self) -> None:
        """Render Embedding LLM configuration."""
        st.header(t("flow_embedding_header", self._lang))

        embed_config = self._state.embedding_config

        # Embedding type selection
        type_options = [e.value for e in EmbeddedType]
        selected_type = st.selectbox(
            "Embedding Type",
            options=type_options,
            index=type_options.index(embed_config.embedded_type.value),
            help=t("flow_embedding_type_help", self._lang),
        )
        new_embed_type = EmbeddedType(selected_type)

        st.markdown("---")

        # Initialize variables for config creation
        api_key = ""
        model = ""
        base_url = ""

        # Configuration based on type
        if new_embed_type == EmbeddedType.DEFAULT:
            st.info(t("flow_default_embedding_model", self._lang).format(DEFAULT_EMBEDDING_MODEL_NAME))
            st.caption(t("flow_default_embedding_note", self._lang))

            # Show disabled inputs
            st.text_input("API Key", value="", disabled=True)
            st.text_input("Model", value=DEFAULT_EMBEDDING_MODEL_NAME, disabled=True)
            st.text_input("Base URL", value="", disabled=True)
            # Dimension setting (common for all types)
            st.markdown("---")
            dimension = 384 
            embed_config.dimension = dimension
            st.text_input("Dimension", value=embed_config.dimension, disabled=True)

        elif new_embed_type == EmbeddedType.LOCAL_MODEL:
            st.info(t("flow_local_embedding_info", self._lang))
            
            # Show disabled inputs
            st.text_input("API Key", value="", disabled=True)
            st.text_input("Model", value=DEFAULT_BGE_HF_REPO_ID + "(performance is lower than default, no recommend to use)", disabled=True)
            st.text_input("Base URL", value="", disabled=True)
            model = DEFAULT_BGE_HF_REPO_ID
            base_url = ""
            
            # Dimension setting (common for all types)
            st.markdown("---")
            embed_config.dimension = 1024
            dimension = st.number_input(
                "Dimension",
                value=embed_config.dimension,
                min_value=128,
                max_value=4096,
                step=128,
                help=t("flow_dimension_help", self._lang),
            )

        elif new_embed_type == EmbeddedType.OLLAMA:
            # Ollama only requires model and base_url, no api_key needed
            st.info(t("flow_ollama_info", self._lang))

            model = st.text_input(
                "Model",
                value=embed_config.model,
                help=t("flow_embedding_model_help", self._lang),
            )

            base_url = st.text_input(
                "Base URL",
                value=embed_config.base_url,
                help=t("flow_embedding_base_url_help", self._lang),
            )
            
            # Dimension setting (common for all types)
            st.markdown("---")
            dimension = st.number_input(
                "Dimension",
                value=embed_config.dimension,
                min_value=128,
                max_value=4096,
                step=128,
                help=t("flow_dimension_help", self._lang),
            )

        else:  # OpenAI_Embedding
            st.info(t("flow_openai_embedding_info", self._lang))
            if embed_config.is_valid() == False:
                embed_config.api_key = self._state.chat_llm_config.api_key
                embed_config.base_url = self._state.chat_llm_config.base_url

            api_key = st.text_input(
                "API Key",
                value=embed_config.api_key,
                type="password",
                help=t("flow_embedding_api_key_help", self._lang),
            )

            model = st.text_input(
                "Model",
                value=embed_config.model,
                help=t("flow_embedding_model_help", self._lang),
            )

            base_url = st.text_input(
                "Base URL",
                value=embed_config.base_url,
                help=t("flow_embedding_base_url_help", self._lang),
            )
            
            # Dimension setting (common for all types)
            st.markdown("---")
            dimension = st.number_input(
                "Dimension",
                value=embed_config.dimension,
                min_value=128,
                max_value=4096,
                step=128,
                help=t("flow_dimension_help", self._lang),
            )

        

        # Create new EmbeddingConfig
        new_config = EmbeddingConfig(
            embedded_type=new_embed_type,
            api_key=api_key,
            model=model,
            base_url=base_url,
            dimension=dimension,
        )
        self._state.embedding_config = new_config

        # Navigation
        st.markdown("---")

        def validate() -> bool:
            if not new_config.is_valid():
                if new_embed_type == EmbeddedType.LOCAL_MODEL:
                    st.error(t("flow_fill_model_name", self._lang))
                else:
                    st.error(t("flow_fill_all_fields", self._lang))
                return False
            return True

        self._render_navigation_buttons(next_callback=validate)


class RAGParserStep(StepRendererBase):
    """Renders RAG parser configuration step."""

    def render(self) -> None:
        """Render RAG parser configuration."""
        st.header(t("flow_rag_header", self._lang))

        config = self._state.rag_parser_config

        st.info(t("flow_rag_info", self._lang))

        max_chunk_size = st.number_input(
            "Max Chunk Size",
            value=config.max_chunk_size,
            min_value=256,
            max_value=16384,
            step=256,
            help=t("flow_max_chunk_help", self._lang),
        )

        limit = st.number_input(
            "Limit",
            value=config.limit,
            min_value=0,
            step=10,
            help=t("flow_limit_help", self._lang),
        )

        skip_patterns = st.text_input(
            "Skip Patterns",
            value=config.skip_patterns,
            help=t("flow_skip_patterns_help", self._lang),
        )

        # Update config
        new_config = RAGParserConfig(
            max_chunk_size=max_chunk_size,
            limit=limit,
            skip_patterns=skip_patterns,
        )
        self._state.rag_parser_config = new_config

        # Navigation
        st.markdown("---")
        self._render_navigation_buttons()


class UploadStep(StepRendererBase):
    """Renders document upload step."""

    def render(self) -> None:
        """Render document upload interface."""
        st.header(t("flow_upload_header", self._lang))

        ensure_directories()

        # Component/Partition selection
        partition_map = get_partition_map()
        component_options = list(partition_map.keys())
        new_module_option = t("new_module", self._lang)
        component_options.append(new_module_option)
        
        selected_component = st.selectbox(
            t("flow_select_partition", self._lang),
            options=component_options,
            index=0,
            help=t("flow_select_partition_help", self._lang),
            key="component_select",
        )
        st.session_state.selected_component = selected_component

        # Handle new module creation or use existing partition
        if selected_component == new_module_option:
            # User selected "New", ask for new module name
            module_name = st.text_input(
                t("flow_module_name", self._lang),
                value=st.session_state.get("upload_module_name", ""),
                help=t("flow_module_name_help", self._lang),
                key="module_name_input",
            )
            module_name = module_name.strip()
            st.session_state.upload_module_name = module_name
            
            # Append new partition if module_name is provided
            if module_name:
                try:
                    if module_name not in partition_map.keys():
                        append_partition(module_name)
                        st.success(f"Partition '{module_name}' created successfully")
                except Exception as e:
                    st.error(f"Failed to create partition: {e}")
                    logger.error(f"Failed to create partition {module_name}: {e}")
        else:
            # User selected existing partition, use it as module_name
            module_name = selected_component
            st.session_state.upload_module_name = module_name

        st.markdown("---")

        # Upload options
        st.subheader(t("flow_select_upload_method", self._lang))

        # Option 1: Archive upload
        with st.expander(t("flow_upload_archive", self._lang), expanded=True):
            st.caption(t("flow_archive_formats", self._lang))
            archive_files = st.file_uploader(
                t("flow_select_archive", self._lang),
                type=["zip", "tar", "gz", "bz2", "xz", "tgz"],
                key="archive_upload",
            )

        # Option 2: Markdown files upload
        with st.expander(t("flow_upload_markdown", self._lang)):
            st.caption(t("flow_markdown_formats", self._lang))
            md_files = st.file_uploader(
                t("flow_select_markdown", self._lang),
                type=["md", "mdx"],
                accept_multiple_files=True,
                key="md_upload",
            )

        # Option 3: GitHub URL
        with st.expander(t("flow_github_clone", self._lang)):
            github_url = st.text_input(
                "GitHub URL",
                placeholder="https://github.com/username/repo",
                key="github_url_input",
            )

        # Navigation and actions
        st.markdown("---")

        col1, col2 = st.columns(2)

        if col1.button(t("flow_prev_step", self._lang), use_container_width=True):
            self._state.step_back()

        if col2.button(t("flow_load_docs", self._lang), type="primary", use_container_width=True):
            self._process_upload(
                module_name=module_name,
                archive_files=archive_files,
                md_files=md_files,
                github_url=github_url,
            )

    def _process_upload(
        self,
        module_name: str,
        archive_files,
        md_files,
        github_url: str,
    ) -> None:
        """Process the upload based on the selected method."""
        # Validate module name
        if not module_name or not module_name.strip():
            st.error(t("flow_enter_module_name", self._lang))
            return

        module_name = module_name.strip()

        # Check if module already exists
        module_upload_path = os.path.join(UPLOADED_DIR, module_name)
        module_loaded_path = os.path.join(LOADED_DIR, module_name)

        if os.path.exists(module_loaded_path):
            logger.warning(t("flow_module_exists", self._lang).format(module_name))
            shutil.rmtree(module_loaded_path)

        # Determine upload method and process
        try:
            if archive_files:
                self._process_archive_upload(module_name, archive_files, module_upload_path)
            elif md_files:
                self._process_md_upload(module_name, md_files, module_upload_path)
            elif github_url:
                self._process_github_clone(module_name, github_url, module_upload_path)
            else:
                st.error(t("flow_select_method", self._lang))
                return

            # Embed documents (use module_name as component/partition name)
            self._embed_documents(module_name, module_upload_path, module_loaded_path)

        except Exception as e:
            logger.error(f"Upload processing failed: {e}")
            st.error(t("flow_upload_error", self._lang).format(e))
            # Cleanup on error
            if os.path.exists(module_upload_path):
                shutil.rmtree(module_upload_path)

    def _process_archive_upload(self, module_name: str, archive_file, dest_path: str) -> None:
        """Process archive file upload."""
        os.makedirs(dest_path, exist_ok=True)

        # Save archive to temp
        tmp_path = os.path.join(TMP_DIR, archive_file.name)
        os.makedirs(TMP_DIR, exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(archive_file.getvalue())

        # Extract
        with st.spinner(t("flow_extracting", self._lang)):
            if not extract_archive(tmp_path, dest_path):
                raise Exception(t("flow_extract_failed", self._lang))

        # Cleanup temp
        os.remove(tmp_path)
        st.success(t("flow_archive_extracted", self._lang).format(dest_path))

    def _process_md_upload(self, module_name: str, md_files, dest_path: str) -> None:
        """Process one or more markdown files upload from user's browser."""
        if not md_files:
            raise Exception(t("flow_no_files_selected", self._lang))

        os.makedirs(dest_path, exist_ok=True)

        with st.spinner(t("flow_saving_files", self._lang)):
            for md_file in md_files:
                file_path = os.path.join(dest_path, md_file.name)
                with open(file_path, "wb") as f:
                    f.write(md_file.getvalue())

        st.success(t("flow_files_saved", self._lang).format(len(md_files), dest_path))

    def _process_github_clone(self, module_name: str, github_url: str, dest_path: str) -> None:
        """Process GitHub repository clone."""
        if not github_url.strip():
            raise Exception(t("flow_enter_github_url", self._lang))

        with st.spinner(t("flow_cloning_repo", self._lang)):
            if not clone_github_repo(github_url.strip(), dest_path):
                raise Exception(t("flow_clone_failed", self._lang))

        st.success(t("flow_repo_cloned", self._lang).format(dest_path))

    def _embed_documents(
        self, module_name: str, upload_path: str, loaded_path: str
    ) -> None:
        """Embed documents into the vector database."""
        rag_config = self._state.rag_parser_config

        # Parse skip patterns
        skip_patterns = []
        if rag_config.skip_patterns:
            skip_patterns = [p.strip() for p in rag_config.skip_patterns.split(",") if p.strip()]

        st.markdown("---")
        st.subheader(t("flow_load_progress", self._lang))

        progress_bar = st.progress(0, text=t("flow_preparing", self._lang))
        status_text = st.empty()

        try:
            # Count total files first
            total_files = 0
            for root, _, files in os.walk(upload_path):
                total_files += sum(1 for f in files if is_markdown_file(f))

            if total_files == 0:
                raise Exception(t("flow_no_markdown_to_load", self._lang))

            status_text.text(t("flow_found_files", self._lang).format(total_files, module_name))

            # Initialize embedder
            embedder = DocumentEmbedder(
                embedding_config=self._state.embedding_config,
                table_name=self._state.table_name or DEFAULT_TABLE_NAME,
            )

            # Process with progress
            progress_bar.progress(0.1, text=t("flow_init_embedding", self._lang))

            # Embed documents
            progress_bar.progress(0.2, text=t("flow_loading_docs", self._lang))

            total_docs = embedder.embed_from_directory(
                doc_base=upload_path,
                component=module_name,
                skip_patterns=skip_patterns,
                batch_size=64,
                limit=rag_config.limit if rag_config.limit > 0 else 0,
            )

            progress_bar.progress(0.9, text=t("flow_moving_files", self._lang))

            # Move to loaded directory
            shutil.move(upload_path, loaded_path)

            progress_bar.progress(1.0, text=t("flow_complete", self._lang))
            status_text.text(t("flow_docs_loaded", self._lang).format(total_docs))

            st.success(t("flow_module_loaded", self._lang).format(module_name))

            # Clear module name
            st.session_state.upload_module_name = ""

            # Show button to go back to status
            if st.button(t("flow_back_to_status", self._lang), use_container_width=True):
                self._state.go_to_step(FLOW_STEP_STATUS)

        except Exception as e:
            progress_bar.progress(0, text=t("flow_load_failed", self._lang))
            raise


# =============================================================================
# Flow Controller
# =============================================================================


class FlowController:
    """Controls the flow UI application."""

    def __init__(self):
        """Initialize flow controller."""
        self._state = StateManager()
        self._step_renderers: dict[int, StepRendererBase] = {
            FLOW_STEP_STATUS: StatusStep(self._state),
            FLOW_STEP_CHAT_LLM: ChatLLMStep(self._state),
            FLOW_STEP_EMBED_LLM: EmbeddingLLMStep(self._state),
            FLOW_STEP_RAG_PARSER: RAGParserStep(self._state),
            FLOW_STEP_UPLOAD: UploadStep(self._state),
        }

    def run(self) -> None:
        """Run the flow UI application with page setup."""
        # Setup page
        PageConfigurator.setup()

        # Run content
        self.run_content()

    def run_content(self, show_menu: bool = True) -> None:
        """
        Run the flow UI content without page setup.

        Args:
            show_menu: Whether to show the top-right menu buttons
        """
        # Render menu buttons in top right corner
        if show_menu:
            menu_nav = MenuNavigator()
            menu_nav.render_menu()

        # Render sidebar and progress
        SidebarRenderer(self._state).render()
        ProgressRenderer(self._state).render()

        # Render current step
        self._render_current_step()

    def _render_current_step(self) -> None:
        """Render the current step."""
        step = self._state.current_step
        if step in self._step_renderers:
            self._step_renderers[step].render()
        else:
            st.error(t("flow_unknown_step", get_lang()).format(step))


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the flow UI application."""
    logger.info("""
                --------------------------------
                Flow UI module loaded
                --------------------------------
                """)
    controller = FlowController()
    controller.run()


# Run the application only when executed directly
if __name__ == "__main__":
    main()
