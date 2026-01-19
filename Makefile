.PHONY: init start stop clean help

help:
	@echo "Available targets:"
	@echo "  make init   - Initialize the project (install dependencies)"
	@echo "  make start  - Start the ChatBot UI service"
	@echo "  make stop   - Stop the ChatBot UI service"
	@echo "  make clean  - Clean temporary files and caches"

init:
	@bash scripts/init.sh

start:
	@bash scripts/start.sh

stop:
	@bash scripts/stop.sh

clean:
	@bash scripts/clean.sh
