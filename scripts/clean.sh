#!/usr/bin/env bash

# Clean the ChatBot project
# This script removes temporary files, cache, and build artifacts

set -e

cd "$(dirname "$0")/.."

echo "Cleaning ChatBot project..."

# Remove Python cache files
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove .pyc files in __pycache__ directories
find . -type d -name "*.pyc" -exec rm -r {} + 2>/dev/null || true

# Remove Streamlit cache
echo "Removing Streamlit cache..."
rm -rf .streamlit/cache 2>/dev/null || true

# Remove virtual environment (if using uv's default .venv)
if [ -d ".venv" ]; then
    echo "Removing virtual environment..."
    rm -rf .venv
fi

# Remove build artifacts
echo "Removing build artifacts..."
rm -rf build/ dist/ *.egg-info 2>/dev/null || true

# Remove temporary files
echo "Removing temporary files..."
rm -rf .pytest_cache .mypy_cache .ruff_cache 2>/dev/null || true

# Remove log files
echo "Removing log files..."
find . -type f -name "*.log" -delete 2>/dev/null || true

echo "Clean complete!"
