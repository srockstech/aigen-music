FROM python:3.9.18-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install AudioCraft from source
RUN git clone https://github.com/facebookresearch/audiocraft.git && \
    cd audiocraft && \
    pip install --no-cache-dir -e .

# Copy application code
COPY . .

# Expose port
ENV PORT=8000
EXPOSE 8000

# Start the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"] 