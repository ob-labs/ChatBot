#!/bin/bash

# Start the ChatBot UI
# Usage: ./scripts/start.sh [chat|flow]

cd "$(dirname "$0")/.."

uv run streamlit run --server.runOnSave false src/frontend/chat_ui.py
