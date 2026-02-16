# Use Python 3.11 slim image for better PyYAML wheel compatibility
## 1
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1 DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    postgresql-client \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libdbus-1-3 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libxshmfence1 libgbm1 libgtk-3-0 libx11-6 libxext6 libxrender1 \
    libxcb1 libcairo2 libpango-1.0-0 libasound2 fonts-liberation libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Upgrade pip and install build tools, then install requirements
# Using pip resolver v2 with timeout to speed up dependency resolution
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    PIP_DEFAULT_TIMEOUT=100 pip install --no-cache-dir \
    --use-deprecated=legacy-resolver -r requirements.txt

# Install Playwright browsers (for web scraping)
RUN playwright install chromium  && \
    playwright install-deps chromium

# Verify critical dependencies are installed correctly
# RUN echo "🔍 Verifying dependencies..." && \
#     python -c "import grpcio; print(f'✅ grpcio version: {grpcio.__version__}')" && \
#     python -c "import google.generativeai; print('✅ google-generativeai imported successfully')" && \
#     (python -c "from langchain_google_genai import ChatGoogleGenerativeAI; print('✅ ChatGoogleGenerativeAI imported successfully')" || \
#      (echo "⚠️  ChatGoogleGenerativeAI import failed - this may be fixed by the patch in agents.py" && \
#       python -c "import traceback; import sys; sys.path.insert(0, '/app'); from app.services.agents.agents import GEMINI_AVAILABLE; print(f'GEMINI_AVAILABLE will be set during app startup: {GEMINI_AVAILABLE}')" || true))

# Copy .env file
COPY .env .env

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/backups

# Expose port 8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI application with uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
