FROM python:3.9.18-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create venv_audiocraft
RUN python -m venv venv_audiocraft
ENV PATH="/app/venv_audiocraft/bin:$PATH"

# Install PyTorch CPU first
RUN pip install --no-cache-dir torch==2.1.0+cpu torchaudio==2.1.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install AudioCraft from source with specific torch version
RUN git clone --depth 1 https://github.com/facebookresearch/audiocraft.git && \
    cd audiocraft && \
    pip install --no-cache-dir -e .

# Copy application files
COPY api.py .
COPY README.md .
COPY LICENSE .

# Create outputs directory
RUN mkdir -p outputs

# Expose default port
EXPOSE 8000

# Start the application directly with shell command to handle PORT
CMD /bin/bash -c "source venv_audiocraft/bin/activate && python -c \"import uvicorn; import os; port = int(os.getenv('PORT', '8000')); uvicorn.run('api:app', host='0.0.0.0', port=port)\"" 