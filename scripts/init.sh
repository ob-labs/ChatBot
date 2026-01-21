#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Starting project initialization..."

# Check .env configuration
echo "Checking .env configuration..."
if ! bash "$SCRIPT_DIR/check_env.sh" 2>&1; then
    echo "Failed to check .env configuration!"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Check Python version
echo "Checking Python version..."
REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=11

# Get current Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    echo "Current Python version: $PYTHON_VERSION"
else
    PYTHON_MAJOR=0
    PYTHON_MINOR=0
    echo "Python 3 is not installed."
fi

# Check if Python version is less than 3.10
if [ "$PYTHON_MAJOR" -lt "$REQUIRED_PYTHON_MAJOR" ] || \
   { [ "$PYTHON_MAJOR" -eq "$REQUIRED_PYTHON_MAJOR" ] && [ "$PYTHON_MINOR" -lt "$REQUIRED_PYTHON_MINOR" ]; }; then
    echo "Python version is below 3.10. Installing Python 3.12 using uv..."
    
    # Install Python 3.12 using uv
    uv python install 3.12
    
    # Create virtual environment with Python 3.12
    echo "Creating virtual environment with Python 3.12..."
    cd "$PROJECT_ROOT"
    uv venv --python 3.12
    
    # Activate the virtual environment
    if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
        echo "Virtual environment activated with Python 3.12"
    else
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
    
    # Verify Python version in venv
    VENV_PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "Virtual environment Python version: $VENV_PYTHON_VERSION"
else
    echo "Python version check passed: $PYTHON_VERSION"
fi

# Check system memory (at least 8GB)
echo "Checking system memory..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    TOTAL_MEM_BYTES=$(sysctl -n hw.memsize)
    TOTAL_MEM_GB=$((TOTAL_MEM_BYTES / 1024 / 1024 / 1024))
else
    # Linux
    TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
fi
if [ -z "$TOTAL_MEM_GB" ] || [ "$TOTAL_MEM_GB" -lt 8 ]; then
    echo "Error: System memory is insufficient. Required: at least 8GB, Current: ${TOTAL_MEM_GB}GB"
    exit 1
fi
echo "System memory check passed: ${TOTAL_MEM_GB}GB"

# Check disk space (at least 10GB available)
echo "Checking disk space..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: df -g returns space in 1GB blocks
    AVAILABLE_SPACE_GB=$(df -g "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
else
    # Linux: df -BG returns space in GB
    AVAILABLE_SPACE_GB=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
fi
if [ -z "$AVAILABLE_SPACE_GB" ] || [ "$AVAILABLE_SPACE_GB" -lt 10 ]; then
    echo "Error: Disk space is insufficient. Required: at least 10GB, Current: ${AVAILABLE_SPACE_GB}GB"
    exit 1
fi
echo "Disk space check passed: ${AVAILABLE_SPACE_GB}GB available"

# Check if mysql command is available
echo "Checking MySQL client..."
if ! command -v mysql &> /dev/null; then
    echo "MySQL client is not installed. Installing MySQL client..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS: Use Homebrew to install MySQL client
        if ! command -v brew &> /dev/null; then
            echo "Error: Homebrew is not installed. Please install Homebrew first:"
            echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
        
        if ! brew install mysql-client 2>&1; then
            echo "Error: Failed to install MySQL client using Homebrew."
            echo "Please install MySQL client manually:"
            echo "  brew install mysql-client"
            exit 1
        fi
        
        # Add MySQL client to PATH
        if [ -d "/usr/local/opt/mysql-client/bin" ]; then
            export PATH="/usr/local/opt/mysql-client/bin:$PATH"
        elif [ -d "/opt/homebrew/opt/mysql-client/bin" ]; then
            export PATH="/opt/homebrew/opt/mysql-client/bin:$PATH"
        fi
    else
        # Linux: Use package manager
        if command -v apt-get &> /dev/null; then
            # Debian/Ubuntu
            if ! sudo apt-get update && sudo apt-get install -y mysql-client 2>&1; then
                echo "Error: Failed to install MySQL client using apt-get."
                echo "Please install MySQL client manually:"
                echo "  sudo apt-get update && sudo apt-get install -y mysql-client"
                exit 1
            fi
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL
            if ! sudo yum install -y mysql 2>&1; then
                echo "Error: Failed to install MySQL client using yum."
                echo "Please install MySQL client manually:"
                echo "  sudo yum install -y mysql"
                exit 1
            fi
        elif command -v dnf &> /dev/null; then
            # Fedora
            if ! sudo dnf install -y mysql 2>&1; then
                echo "Error: Failed to install MySQL client using dnf."
                echo "Please install MySQL client manually:"
                echo "  sudo dnf install -y mysql"
                exit 1
            fi
        else
            echo "Error: Cannot determine package manager. Please install MySQL client manually."
            exit 1
        fi
    fi
    
    # Verify installation
    if ! command -v mysql &> /dev/null; then
        echo "Error: MySQL client installation completed but mysql command is still not found."
        echo "Please ensure MySQL client is installed and available in PATH."
        exit 1
    fi
    
    echo "MySQL client installed successfully."
else
    echo "MySQL client check passed."
fi

# Install dependencies
echo "Installing dependencies with uv..."
cd "$PROJECT_ROOT"
uv sync

# Check if .env file exists and export environment variables
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "Error: .env file not found. Please create .env file first."
    exit 1
fi

echo "Loading environment variables from .env file..."
set -a
source "$PROJECT_ROOT/.env"
set +a
echo "Environment variables loaded successfully."

# Initialize Docker container if not reusing current DB
if [ "${REUSE_CURRENT_DB}" != "true" ]; then
    bash "$SCRIPT_DIR/init_docker.sh"
fi

# Create database
echo "Creating database..."
if ! bash "$SCRIPT_DIR/create_db.sh" 2>&1; then
    echo "Failed to create database!"
    exit 1
fi

echo "Initialization completed!"
