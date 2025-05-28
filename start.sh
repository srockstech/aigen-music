#!/bin/bash

# Use PORT from environment or default to 8000
export PORT="${PORT:-8000}"

# Activate venv_audiocraft and start the application
source venv_audiocraft/bin/activate
exec python api.py 