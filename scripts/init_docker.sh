#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "Error: .env file not found. Please create .env file first."
    exit 1
fi

echo "Loading environment variables from .env file..."
set -a
source "$PROJECT_ROOT/.env"
set +a
echo "Environment variables loaded successfully."

# Ensure uv is available when running under sudo (PATH may be reset)
if ! command -v uv &> /dev/null; then
    if [ -n "${SUDO_USER:-}" ]; then
        ORIGINAL_HOME="$(eval echo "~${SUDO_USER}")"
        if [ -x "${ORIGINAL_HOME}/.local/bin/uv" ]; then
            export PATH="${ORIGINAL_HOME}/.local/bin:${PATH}"
        fi
    fi
fi
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found in PATH. If you installed uv for your user, run:"
    echo "  sudo -E bash scripts/init_docker.sh"
    echo "or add uv to PATH (e.g., ~/.cargo/bin)."
    exit 1
fi

# Check if Docker is available and running
if command -v docker &> /dev/null; then
    if sudo docker info &> /dev/null; then
        echo "Docker is available and running."
    else
        echo "Warning: Docker is installed but not running. You may need to start Docker service:"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "  Open Docker Desktop application"
        else
            echo "  sudo systemctl start docker"
            exit 1
        fi
    fi
else
    echo "Warning: Docker is not installed. You may need Docker to run OceanBase database."
fi

# Download Docker images based on DB_STORE
if [ "${DB_STORE}" = "seekdb" ]; then
    docker_name="seekdb"
    # Check if container already exists
    if sudo docker ps -a --format "{{.Names}}" | grep -q "^${docker_name}$"; then
        echo "Container '${docker_name}' already exists."
        if sudo docker ps --format "{{.Names}}" | grep -q "^${docker_name}$"; then
            echo "Container is already running."
        else
            echo "Starting existing container..."
            sudo docker start "${docker_name}"
        fi
    else
        echo "Downloading latest seekdb docker image..."
        export ROOT_PASSWORD=${DB_PASSWORD}
        sudo docker run --name seekdb -e ROOT_PASSWORD=${DB_PASSWORD} -d -p 2881:2881 -p 2886:2886 oceanbase/seekdb
    fi
elif [ "${DB_STORE}" = "oceanbase" ]; then
    docker_name="oceanbase-ce"
    # Check if container already exists
    if sudo docker ps -a --format "{{.Names}}" | grep -q "^${docker_name}$"; then
        echo "Container '${docker_name}' already exists."
        if sudo docker ps --format "{{.Names}}" | grep -q "^${docker_name}$"; then
            echo "Container is already running."
        else
            echo "Starting existing container..."
            sudo docker start "${docker_name}"
        fi
    else
        echo "Downloading latest oceanbase-ce docker image..."
        export OB_TENANT_PASSWORD=${DB_PASSWORD}
        sudo docker run -p 2881:2881 --name oceanbase-ce -e OB_TENANT_PASSWORD=${DB_PASSWORD} -e datafile_size=10G -d oceanbase/oceanbase-ce
    fi
else
    echo "Warning: Unknown DB_STORE value: ${DB_STORE}. Skipping docker image download."
    exit 0
fi

# Wait for download to complete
echo "Waiting for docker image download to complete..."
total=60
for ((i=1; i<=total; i++)); do
    percent=$((i * 100 / total))
    filled=$((i * 50 / total))
    empty=$((50 - filled))
    bar=$(printf "%${filled}s" | tr ' ' '█')$(printf "%${empty}s" | tr ' ' '░')
    printf "\r[${bar}] ${percent}%% (${i}/${total}s)"
    sleep 1
done
echo ""

# Check database connection
echo "Checking database connection..."
cd "$PROJECT_ROOT"

# Run the database connection check using mysql command
source .env
if [[ -z "$DB_PASSWORD" ]]; then
    # No password
    if ! mysql -u "$DB_USER" -h "$DB_HOST" -P "$DB_PORT" -e "SELECT 1;" 2>&1; then
        connection_failed=true
    else
        connection_failed=false
    fi
else
    # With password (use -p without space to avoid password prompt)
    if ! mysql -u "$DB_USER" -p"$DB_PASSWORD" -h "$DB_HOST" -P "$DB_PORT" -e "SELECT 1;" 2>&1; then
        connection_failed=true
    else
        connection_failed=false
    fi
fi

if [ "$connection_failed" = true ]; then
    echo "Database connection check failed!"
    echo "Please ensure that Docker is running properly."
    echo "Please wait for 60 seconds and rerun this script with sudo bash scripts/init_docker.sh."
    
    # Determine docker_name based on DB_STORE
    if [ "${DB_STORE}" = "seekdb" ]; then
        docker_name="seekdb"
    elif [ "${DB_STORE}" = "oceanbase" ]; then
        docker_name="oceanbase-ce"
    else
        echo "Warning: Unknown DB_STORE value: ${DB_STORE}. Cannot determine docker name."
        exit 1
    fi
    
    # Check if docker container exists and is running
    if ! sudo docker ps --format "{{.Names}}" | grep -q "^${docker_name}$"; then
        echo "Docker container '${docker_name}' is not running."
        echo "Checking if container exists..."
        if sudo docker ps -a --format "{{.Names}}" | grep -q "^${docker_name}$"; then
            echo "Container exists but is not running. Attempting to start it..."
            sudo docker start "${docker_name}" || true
            echo "Waiting for container to be ready..."
            sleep 5
        else
            echo "Container '${docker_name}' does not exist."
            echo "Please run 'make init' first to create the container."
            exit 1
        fi
    fi
    
    echo "Checking Docker logs for errors..."
    echo "=========================================="
    sudo docker logs --tail 50 "${docker_name}" 2>&1 || true
    echo "=========================================="
    echo ""
    echo "If you see errors above, please fix them before continuing."
    echo "You can monitor the logs with: sudo docker logs -f ${docker_name}"
    echo ""
    echo "After fixing the issues, please run this script again."
    exit 1
    
fi
