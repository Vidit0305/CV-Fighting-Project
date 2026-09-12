#!/usr/bin/env bash
# Runner script for CV Fighter on Linux / macOS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d ".venv" ]; then
    echo "Starting CV Fighter using virtual environment (.venv)..."
    ./.venv/bin/python main.py "$@"
elif [ -d "venv" ]; then
    echo "Starting CV Fighter using virtual environment (venv)..."
    ./venv/bin/python main.py "$@"
else
    echo "No virtual environment found. Running with system python3..."
    python3 main.py "$@"
fi
