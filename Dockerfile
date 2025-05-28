FROM python:3.9.18-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create venv_audiocraft
RUN python -m venv venv_audiocraft

# Copy requirements first
COPY requirements.txt .

# Install dependencies in virtual environment
SHELL ["/bin/bash", "-c"]
RUN source venv_audiocraft/bin/activate && \
    pip install --no-cache-dir torch==2.1.0+cpu torchaudio==2.1.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    git clone --depth 1 https://github.com/facebookresearch/audiocraft.git && \
    cd audiocraft && \
    pip install --no-cache-dir -e . && \
    cd .. && \
    rm -rf audiocraft

# Copy application files
COPY api.py .
COPY start.py .
COPY README.md .
COPY LICENSE .

# Create outputs directory
RUN mkdir -p outputs

# Expose default port
EXPOSE 8000

# Create a wrapper script to activate venv and run the app
RUN echo '#!/bin/bash\nsource /app/venv_audiocraft/bin/activate\nexec python start.py' > /app/run.sh && \
    chmod +x /app/run.sh

# Start the application
CMD ["/app/run.sh"] 