#!/bin/bash
# Test runner script for AI Algorithms Demo backend

set -e

echo "Running pytest tests..."
echo "======================"

# Check if we're in a virtual environment or Docker
if [ -n "$VIRTUAL_ENV" ] || [ -f "/.dockerenv" ]; then
    # Running in venv or Docker, run tests directly
    python -m pytest "$@"
else
    # Not in venv/Docker, try to use Docker
    echo "Not in virtual environment. Attempting to run via Docker..."
    if command -v docker &> /dev/null; then
        docker build -t ai-algorithms-backend . && \
        docker run --rm ai-algorithms-backend pytest "$@"
    else
        echo "Error: Neither virtual environment nor Docker is available."
        echo "Please either:"
        echo "  1. Create and activate a virtual environment:"
        echo "     python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        echo "  2. Install Docker and run tests via Docker"
        exit 1
    fi
fi
