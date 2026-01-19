#!/usr/bin/env bash

# Stop the ChatBot UI service
# This script finds and kills any running streamlit processes related to the ChatBot

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "Stopping ChatBot UI service..."

# Find and kill streamlit processes running chat_ui.py or flow_ui.py
PIDS=$(ps aux | grep -E "streamlit.*chat_ui\.py|streamlit.*flow_ui\.py" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No ChatBot UI service is running."
    exit 0
fi

for PID in $PIDS; do
    echo "Stopping process $PID..."
    kill "$PID" 2>/dev/null || true
done

# Wait a bit for processes to terminate
sleep 2

# Force kill if still running
PIDS=$(ps aux | grep -E "streamlit.*chat_ui\.py|streamlit.*flow_ui\.py" | grep -v grep | awk '{print $2}')
if [ -n "$PIDS" ]; then
    for PID in $PIDS; do
        echo "Force stopping process $PID..."
        kill -9 "$PID" 2>/dev/null || true
    done
fi

echo "ChatBot UI service stopped."
