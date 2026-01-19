#!/usr/bin/env bash

# Initialize the ChatBot project
# This script sets up the project environment, installs dependencies, and prepares configuration files

set -e

cd "$(dirname "$0")/.."

echo "Initializing ChatBot project..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo ".env file not found. Creating from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Please update the .env file with your configuration (especially API_KEY and database settings)"
    else
        echo "Warning: .env.example not found. Please create .env file manually."
    fi
else
    echo ".env file already exists"
fi

# Install dependencies
echo "Installing dependencies with uv..."
uv sync

echo "Initialization complete!"
echo ""
echo "Next steps:"
echo "1. Update .env file with your API keys and database configuration"
echo "2. Run 'make start' to start the chatbot"
