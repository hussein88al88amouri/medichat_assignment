SHELL := /bin/bash
# Makefile for Llama3.1:8B Project

# Variables
PYTHON = python
PIP = pip
VENV_DIR = ./env
VENV_PYTHON = $(VENV_DIR)/bin/python
VENV_PIP = $(VENV_DIR)/bin/pip
REQUIREMENTS = requirements.txt

# Default target
.DEFAULT_GOAL := help

# Help target
help:
	@echo "Makefile for Llama3.1:8B Project"
	@echo ""
	@echo "Targets:"
	@echo "  help            - Show this help message"
	@echo "  setup           - Create virtual environment and install dependencies"
	@echo "  run             - Run the main application"
	@echo "  test            - Run unit tests"
	@echo "  lint            - Run linters"
	@echo "  clean           - Remove temporary files and directories"
	@echo "  clean-venv      - Remove virtual environment"
	@echo "  purge           - Clean and reinstall everything"
	@echo "  install         - Install or update dependencies"

# Check for Python and pip
check-deps:
	@echo "Checking for Python and pip..."
	@if ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "Python is not installed. Please install Python3."; \
		exit 1; \
	fi
	@echo "Python is installed."
	@if ! command -v $(PIP) >/dev/null 2>&1; then \
		echo "pip is not installed. Installing pip..."; \
		sudo apt update && sudo apt install -y python3-pip; \
	fi
	@echo "pip is installed."

# Create virtual environment and install dependencies
setup: check-deps
	@echo "Setting up virtual environment..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "Virtual environment created."; \
	fi
	@echo "Installing dependencies..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r $(REQUIREMENTS)
	@echo "Setup completed."

# Run the main application
run:
	@echo "Running the application..."
	$(VENV_PYTHON) main.py

# Run tests
test:
	@echo "Running tests..."
	$(VENV_PYTHON) -m unittest discover tests

# Run linters
lint:
	@echo "Running linters..."
	$(VENV_PYTHON) -m flake8 src/ tests/

# Clean temporary files and directories
clean:
	@echo "Cleaning temporary files and directories..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -r {} +
	@echo "Cleanup completed."

# Clean virtual environment
clean-venv:
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "Virtual environment removed."

# Purge: remove all and reinstall environment
purge: clean clean-venv setup

# Install or update dependencies
install:
	@echo "Installing or updating dependencies..."
	$(VENV_PIP) install -r $(REQUIREMENTS)
	@echo "Dependencies installed or updated."
