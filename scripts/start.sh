#!/usr/bin/env bash

# Start the ChatBot UI
# Usage: ./scripts/start.sh [chat|flow]

set -e

cd "$(dirname "$0")/.."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please run 'make init' first."
    exit 1
fi

# Source .env file
source .env

UI_TYPE=${1:-chat}

if [ "$UI_TYPE" = "chat" ]; then
    echo "Starting ChatBot UI (chat mode)..."
    uv run streamlit run --server.runOnSave false src/frontend/chat_ui.py
elif [ "$UI_TYPE" = "flow" ]; then
    echo "Starting ChatBot UI (flow mode)..."
    uv run streamlit run --server.runOnSave false src/frontend/flow_ui.py
else
    echo "Unknown UI type: $UI_TYPE"
    echo "Usage: ./scripts/start.sh [chat|flow]"
    exit 1
fi
