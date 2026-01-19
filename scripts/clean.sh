#!/usr/bin/env bash

# Clean temporary files and caches
# This script removes Python cache files, __pycache__ directories, and other temporary files

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "Cleaning temporary files and caches..."

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove .pyc files in __pycache__ directories
find . -type f -path "*/__pycache__/*.pyc" -delete 2>/dev/null || true

# Remove .DS_Store files (macOS)
find . -type f -name ".DS_Store" -delete 2>/dev/null || true

# Remove Streamlit cache
if [ -d ".streamlit" ]; then
    find .streamlit -type f -name "*.cache" -delete 2>/dev/null || true
fi

# Remove pytest cache
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Remove mypy cache
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

# Remove coverage files
find . -type f -name ".coverage" -delete 2>/dev/null || true
find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true

# Remove .egg-info directories
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Remove dist and build directories
[ -d "dist" ] && rm -rf dist
[ -d "build" ] && rm -rf build

echo "Cleanup completed successfully!"
