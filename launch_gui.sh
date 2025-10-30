#!/bin/bash

# CryptoBot Matrix Edition Launcher
# Automatically starts the Matrix-themed web GUI

echo "========================================="
echo "  CryptoBot Matrix Edition"
echo "  Version 2.0.0"
echo "========================================="
echo ""
echo "Starting Matrix-themed Web GUI..."
echo ""

# Default port
PORT=8080

# Check if port argument provided
if [ ! -z "$1" ]; then
    PORT=$1
fi

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port $PORT is already in use!"
    echo "   Trying alternative port 8081..."
    PORT=8081
fi

# Launch the GUI
cd "$(dirname "$0")"
python3 gui_web_matrix.py --port $PORT

echo ""
echo "GUI server stopped."
