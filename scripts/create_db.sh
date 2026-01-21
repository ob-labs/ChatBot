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

# Create the database using IF NOT EXISTS to avoid errors if database already exists
echo "Creating database $DB_NAME if it does not exist..."

if [[ -z "$DB_PASSWORD" ]]; then
    # No password
    if ! mysql -u "$DB_USER" -h "$DB_HOST" -P "$DB_PORT" -e "CREATE DATABASE IF NOT EXISTS \`$DB_NAME\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;" 2>&1; then
        echo "Error: Failed to create database $DB_NAME"
        exit 1
    fi
else
    # With password (use -p without space to avoid password prompt)
    if ! mysql -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" -P "$DB_PORT" -e "CREATE DATABASE IF NOT EXISTS \`$DB_NAME\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;" 2>&1; then
        echo "Error: Failed to create database $DB_NAME"
        exit 1
    fi
fi

echo "Database $DB_NAME created successfully (or already exists)"

# Create the vector table
echo "Creating vector table..."
cd "$PROJECT_ROOT"
# if ! uv run python -m src.common.db create-table; then
#     echo "Error: Failed to create vector table"
#     exit 1
# fi

echo "All database setup completed successfully!"