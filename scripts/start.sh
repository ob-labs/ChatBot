#!/bin/bash

# Start the ChatBot UI
# Usage: ./scripts/start.sh [chat|flow]

cd "$(dirname "$0")/.."

UI_TYPE=${1:-chat}

if [ "$UI_TYPE" = "chat" ]; then
    uv run streamlit run --server.runOnSave false src/frontend/chat_ui.py
elif [ "$UI_TYPE" = "flow" ]; then
    uv run streamlit run --server.runOnSave false src/frontend/flow_ui.py
else
    echo "Unknown UI type: $UI_TYPE"
    echo "Usage: ./scripts/start.sh [chat|flow]"
    exit 1
fi
