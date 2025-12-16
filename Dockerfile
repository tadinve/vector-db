# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create data directory
RUN mkdir -p /app/lance_data

# Set environment variables
ENV LANCE_DB_PATH=/app/lance_data
ENV EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV CHUNK_SIZE=500
ENV CHUNK_OVERLAP=50
ENV PORT=8080

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/')" || exit 1

# Run the application
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
