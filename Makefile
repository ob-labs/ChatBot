.PHONY: init start stop clean help

# Default target
help:
	@echo "ChatBot Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make init   - Initialize the project (install dependencies, setup .env)"
	@echo "  make start  - Start the ChatBot UI (default: chat mode)"
	@echo "  make stop   - Stop the ChatBot UI"
	@echo "  make clean  - Clean temporary files and cache"
	@echo ""
	@echo "Examples:"
	@echo "  make init"
	@echo "  make start"
	@echo "  make stop"
	@echo "  make clean"

# Initialize the project
init:
	@bash scripts/init.sh

# Start the ChatBot UI
start:
	@bash scripts/start.sh chat

# Stop the ChatBot UI
stop:
	@bash scripts/stop.sh

# Clean temporary files and cache
clean:
	@bash scripts/clean.sh
