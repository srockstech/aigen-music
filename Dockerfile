FROM python:3.9.18-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch CPU first
RUN pip install --no-cache-dir torch==2.1.0+cpu torchaudio==2.1.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install AudioCraft from source with specific torch version
RUN git clone --depth 1 https://github.com/facebookresearch/audiocraft.git && \
    cd audiocraft && \
    pip install --no-cache-dir -e .

# Second stage
FROM python:3.9.18-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only necessary files
COPY api.py .
COPY README.md .
COPY LICENSE .

# Create outputs directory
RUN mkdir -p outputs

# Expose port
ENV PORT=8000
EXPOSE 8000

# Start the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "${PORT}"] 