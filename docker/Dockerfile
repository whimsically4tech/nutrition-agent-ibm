FROM python:3.11-slim

LABEL maintainer="IBM Nutrition Agent"
LABEL description="AI-powered Multi-Agent Nutrition Platform — IBM Granite 3.0"

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY data/ ./data/

# Create directories for ChromaDB persistence and uploads
RUN mkdir -p ./data/chroma_db ./data/uploads

# Default environment (override via --env-file or env vars)
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the RAG ingest on first start if DB is empty, then launch API
CMD ["sh", "-c", \
    "python -m src.rag.ingest 2>/dev/null || true && \
     uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2"]
