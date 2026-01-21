#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment variables
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "Error: .env file not found. Please create .env file first."
    exit 1
fi

source "$PROJECT_ROOT/.env"

if [[ -z "$DB_NAME" ]]; then
    echo "Error: Please provide a database name in the .env file"
    exit 1
fi

if [[ -z "$DB_USER" ]]; then
    echo "Error: Please provide a database user in the .env file"
    exit 1
fi

if [[ -z "$DB_HOST" ]]; then
    echo "Error: Please provide a database host in the .env file"
    exit 1
fi

if [[ -z "$DB_PORT" ]]; then
    echo "Error: Please provide a database port in the .env file"
    exit 1
fi

echo "Environment variables check passed."
