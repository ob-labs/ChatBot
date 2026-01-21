"""
Shared menu navigation module for the frontend.

This module provides a consistent navigation menu that can be used
across different pages (chat_ui and flow_ui).
"""

import streamlit as st

from src.common.config import get_ui_lang
from src.frontend.i18n import t


# =============================================================================
# Constants
# =============================================================================

# Menu options
MENU_CHAT = "chat"
MENU_LOAD_DOCS = "load_docs"

# CSS to hide Streamlit deploy button and add custom menu
MENU_CSS = """
<style>
    /* Hide default deploy button and toolbar */
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {visibility: hidden;}
    
    /* Custom top-right menu container */
    .top-right-menu {
        position: fixed;
        top: 14px;
        right: 14px;
        z-index: 999999;
        display: flex;
        gap: 8px;
    }
    
    .top-right-menu a {
        padding: 6px 16px;
        border-radius: 6px;
        border: 1px solid #e0e0e0;
        background-color: #ffffff;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.2s;
        text-decoration: none;
        color: #333;
    }
    
    .top-right-menu a:hover {
        background-color: #f0f0f0;
        border-color: #ccc;
    }
    
    .top-right-menu a.active {
        background-color: #1976d2;
        color: white;
        border-color: #1976d2;
    }
</style>
"""


# =============================================================================
# Menu Navigator
# =============================================================================


class MenuNavigator:
    """Handles menu navigation between pages."""

    def __init__(self, lang: str = None):
        """
        Initialize menu navigator.

        Args:
            lang: Language code for UI text. If None, will use get_ui_lang().
        """
        self.lang = lang if lang else get_ui_lang()
        # Initialize current page in session state
        if "current_page" not in st.session_state:
            st.session_state.current_page = MENU_CHAT

    def apply_css(self) -> None:
        """Apply the menu CSS styles."""
        st.markdown(MENU_CSS, unsafe_allow_html=True)

    def render_menu(self) -> str:
        """
        Render navigation menu buttons in top right corner (replacing deploy button).

        Returns:
            Selected menu option key
        """
        # Check URL query params FIRST to determine current page
        query_params = st.query_params
        if "page" in query_params:
            page = query_params["page"]
            if page in [MENU_CHAT, MENU_LOAD_DOCS]:
                st.session_state.current_page = page

        chat_label = t("menu_chat", self.lang)
        load_docs_label = t("menu_load_docs", self.lang)
        current = st.session_state.current_page

        # Determine active state for styling
        chat_active = "active" if current == MENU_CHAT else ""
        docs_active = "active" if current == MENU_LOAD_DOCS else ""

        # Render menu links in fixed position (using <a> tags for reliable navigation)
        menu_html = f"""
        <div class="top-right-menu">
            <a href="?page=chat" class="{chat_active}" target="_self">{chat_label}</a>
            <a href="?page=load_docs" class="{docs_active}" target="_self">{load_docs_label}</a>
        </div>
        """
        st.markdown(menu_html, unsafe_allow_html=True)

        return st.session_state.current_page
