#!/bin/bash

# Use PORT from environment or default to 8000
export PORT="${PORT:-8000}"

# Activate virtual environment and start the application
source /opt/venv/bin/activate
exec python api.py 