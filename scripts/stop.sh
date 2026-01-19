#!/usr/bin/env bash

# Stop the ChatBot UI
# This script stops any running Streamlit processes

set -e

cd "$(dirname "$0")/.."

echo "Stopping ChatBot UI..."

# Find and kill Streamlit processes
STREAMLIT_PIDS=$(pgrep -f "streamlit run" || true)

if [ -z "$STREAMLIT_PIDS" ]; then
    echo "No running Streamlit processes found"
else
    echo "Found Streamlit processes: $STREAMLIT_PIDS"
    for PID in $STREAMLIT_PIDS; do
        echo "Stopping process $PID..."
        kill $PID 2>/dev/null || true
    done
    echo "ChatBot UI stopped"
fi

# Also check for processes on common Streamlit ports
for PORT in 8501 8502 8503; do
    PORT_PID=$(lsof -ti:$PORT 2>/dev/null || true)
    if [ -n "$PORT_PID" ]; then
        echo "Stopping process on port $PORT (PID: $PORT_PID)..."
        kill $PORT_PID 2>/dev/null || true
    fi
done

echo "Stop complete"
