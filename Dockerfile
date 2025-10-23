# Multi-stage Docker build for SOP QA Tool
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy requirements first for better caching
COPY --chown=app:app requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Create data directories
RUN mkdir -p data/chunks data/raw_docs data/faiss_index data/logs data/evaluation

# Set Python path
ENV PATH="/home/app/.local/bin:$PATH"
ENV PYTHONPATH="/home/app:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "sop_qa_tool.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --user --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    jupyter

# Production stage
FROM base as production

# Remove unnecessary files
RUN rm -rf tests/ examples/ notebooks/ docs/ .git/

# Use production settings
ENV MODE=aws
ENV LOG_LEVEL=INFO
ENV ENABLE_DEBUG=false

# Run as non-root
USER app