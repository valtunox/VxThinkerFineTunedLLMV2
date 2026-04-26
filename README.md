# VA LLM Specialist Model

> **IT Specialist AI for Cloud, DevOps, Deployment, Support, and Computer Skills**

An open-source, production-grade AI specialist model for IT operations, cloud engineering, DevOps, deployments, infrastructure support, troubleshooting, networking, security, and technical customer service. VA LLM Specialist Model is grounded in your organization's IT datasets and operational knowledge so it answers as an IT specialist instead of a generalist healthcare or finance model.

**Author**: Joel Otepa Wembo - [joelwembo.com](https://joelwembo.com)

---

## Table of Contents

- [Use Cases](#use-cases)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
  - [Step 1: Environment Setup](#step-1-environment-setup)
  - [Step 2: Confirm Specialist Scope](#step-2-confirm-specialist-scope)
  - [Step 3: Prepare Datasets](#step-3-prepare-datasets)
  - [Step 4: Precompute Embeddings](#step-4-precompute-embeddings)
  - [Step 5: Train the Model](#step-5-train-the-model)
  - [Step 6: Serve the API](#step-6-serve-the-api)
  - [Step 7: Query Your Model](#step-7-query-your-model)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Data Management](#data-management)
- [Troubleshooting](#troubleshooting)

---

## Architecture

VA LLM Specialist Model uses a hybrid **RAG (Retrieval-Augmented Generation) + Deterministic Reasoning** architecture.

```
┌─────────────────────────────────────────────────────────────┐
│                    User / Application                        │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP Request
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Gateway (app.py)                        │
│  ├─ Rate Limiting        ├─ Metrics Collection              │
│  ├─ Request Logging      └─ Circuit Breaker                 │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐    ┌──────────┐
    │ /search │    │ /api/v1  │    │ /api/v2  │
    └────┬────┘    └────┬─────┘    └────┬─────┘
         │              │               │
         ▼              ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                    VectorStore (FAISS)                       │
│  ├─ Embedding Model: all-MiniLM-L6-v2 (384-dim)             │
│  ├─ L1 Cache: Embeddings (TTL 1h)                           │
│  └─ L2 Cache: Search Results (TTL 30m)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              ReasoningEngine (reasoning.py)                  │
│  Search → Analyze → Synthesize → Decide                      │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Description |
|-----------|-------------|
| **VectorStore** | FAISS-based semantic search with sentence-transformers embeddings |
| **ReasoningEngine** | Multi-step chain-of-thought reasoning for cloud operations |
| **Cache** | Multi-level TTL caching (embeddings L1, search results L2) |
| **Metrics** | Prometheus metrics for observability |
| **Health** | Kubernetes-compatible readiness/liveness probes |

---

## Features

- **Semantic Search**: Query infrastructure knowledge using natural language
- **Chain-of-Thought Reasoning**: Multi-step analysis with confidence scoring
- **Multi-Cloud Support**: AWS, Azure, GCP, and Kubernetes intelligence
- **100% Offline**: No external API calls required - full data sovereignty
- **Production Ready**: Rate limiting, circuit breaker, structured logging
- **Observable**: Prometheus metrics, Grafana dashboards, health probes
- **Scalable**: Docker Compose, Kubernetes (AKS), horizontal scaling

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (recommended)
- 4GB+ RAM

### Option 1: Docker Compose (Recommended)

```bash
# Clone and navigate to project
cd va_llm_v1

# Start full stack (API + Redis + Prometheus + Grafana)
docker-compose up -d

# Check status
docker-compose ps
```

Services will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

### Option 2: Local Development

```bash
# Clone the repository
git clone https://github.com/valtunox/va_llm_specialist_model.git
cd va_llm_specialist_model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env  # Edit with your API keys
```

**Key environment variables:**

```env
# Specialist Scope
ACTIVE_INDUSTRY=cloud               # legacy env; runtime scope is IT-specialist only

# LLM Provider (for reasoning API)
MODEL_PROVIDER=gemini               # ollama | openai | gemini | anthropic | huggingface
GOOGLE_API_KEY=your-key-here        # Or OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.

# Optional live web search enrichment
VALLM_ENABLE_WEB_SEARCH=true
VALLM_WEB_SEARCH_PROVIDER=auto      # auto | tavily | serpapi | duckduckgo
TAVILY_API_KEY=your-key-here

# Database (optional)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/vallm

# Redis (optional, for caching)
REDIS_URL=redis://localhost:6379/0
```

---

### Step 2: Confirm Specialist Scope

This repository is now intentionally IT-only. The default training and indexing
pipeline loads only these dataset folders:

- `app/data/datasets/cloud/`
- `app/data/datasets/automation/`
- `app/data/datasets/customer_service/`
- `app/data/datasets/skills/`
- `app/data/datasets/uploaded/`

Healthcare, finance, and generic business datasets are excluded from the
specialist pipeline by default.

Typical commands:

```bash
# Precompute IT-only retrieval data
python -m app.services.ai.ml.precompute

# Train the IT specialist model
python -m app.services.ai.ml.train

# Query the IT specialist API with optional web enrichment
curl -X POST http://localhost:8000/api/models/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Why are IAM access keys not rotated in 90 days?",
    "use_web_search": true,
    "web_search_max_results": 5
  }'
```

The service will:
1. Download embedding model (first run only, ~80MB)
2. Auto-build FAISS index if not present (when `VALLM_AUTO_PRECOMPUTE=true`)
3. Start API server on http://localhost:8000

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Query endpoint
curl -X POST http://localhost:8000/api/models/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Why are IAM access keys not rotated in 90 days?",
    "top_k": 5,
    "include_reasoning": true
  }'

# Developer endpoint (Terraform generation)
curl -X POST http://localhost:8000/api/models/v1/developer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Create Terraform config for EKS cluster",
    "include_code": true
  }'
```

---

### Step 4: Precompute Embeddings

Build a FAISS vector index from your datasets for semantic search:

```bash
# Precompute the IT specialist corpus
python -m app.services.ai.ml.precompute

# CSV-only precompute
python -m app.services.ai.ml.precompute --no-documents

# Output:
#   app/data/vectorstore/index.faiss
#   app/data/vectorstore/documents.pkl
```

**What happens:**
1. Loads only the curated IT dataset folders under `app/data/datasets/`
2. Converts rows to text for cloud, DevOps, deployment, support, and skills retrieval
3. Generates embeddings with sentence-transformers
4. Builds FAISS IndexFlatIP (cosine similarity)
5. Saves index + documents to `app/data/vectorstore/`

---

### Step 5: Train the Model

Fine-tune a causal language model on your IT specialist data:

```bash
# Build image
docker build -t vallm:latest .

# Run container
docker run -p 8000:8000 -v $(pwd)/app/data:/app/data vallm:latest
```

### Docker Compose (Full Stack)

```bash
# Start the FastAPI server
python -m app.app

# Or with uvicorn directly
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload

# Or with Docker
docker-compose up -d

# View logs
docker-compose logs -f vallm

# Stop all services
docker-compose down
```

**Included Services**:
- `vallm`: FastAPI application (port 8000)
- `redis`: Caching layer (port 6379)
- `prometheus`: Metrics collection (port 9090)
- `grafana`: Metrics visualization (port 3000)

### Kubernetes (AKS)

Manifests are in `deployment/kubernetes/`:

```bash
# Apply manifests
kubectl apply -f deployment/kubernetes/

# Check deployment
kubectl get pods -l app=vallm
```

See `deployment.md` for detailed instructions.

### CI/CD

**GitHub Actions** (`.github/workflows/data-pipeline.yml`):
- Linting and testing
- Docker build and push
- Deploy to VM or Kubernetes

**Azure Pipelines** (`azure-pipelines.yml`):
- Build and push to Azure Container Registry
- Deploy to VM via SSH
- Deploy to AKS

---

## API Reference

### V1 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | HTML status page |
| GET | `/health` | Basic health check |
| GET | `/health/ready` | Readiness probe (K8s) |
| GET | `/health/live` | Liveness probe (K8s) |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | OpenAPI documentation |
| POST | `/search` | Vector similarity search |
| POST | `/generate` | Text generation (if model loaded) |

### V2 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/model/v1/query` | Main RAG query with reasoning |
| POST | `/api/model/v1/developer` | Developer/Terraform assistance |
| POST | `/api/model/v1/terminal` | CLI command assistance |

**Example Request**:
```json
{
  "query": "How do I provision an EC2 instance?",
  "top_k": 5,
  "include_reasoning": true,
  "filter_type": "resource"
}
```

### V2 Endpoints - NLP + Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/model/v2/query` | NLP-enhanced query with entity extraction |
| POST | `/api/model/v2/upload` | Document/image upload for analysis |
| POST | `/api/model/v2/extract` | Entity extraction from text |
| GET | `/api/model/v2/status` | NLP capability status |

### V3 Endpoints - Incident Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/model/v3/query` | Incident pattern detection |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VALLM_AUTO_PRECOMPUTE` | `true` | Auto-build FAISS index if missing |
| `VALLM_AUTO_TRAIN` | `false` | Auto-train LLM model if missing |
| `USE_CUDA` | `false` | Enable GPU acceleration |
| `VALLM_JSON_LOGGING` | `false` | Enable structured JSON logging |
| `VALLM_RATE_LIMIT_ENABLED` | `false` | Enable rate limiting |
| `VALLM_RATE_LIMIT_PER_MINUTE` | `60` | Max requests per minute per client |
| `VALLM_CACHE_EMBEDDINGS` | `true` | Enable embedding cache (L1) |
| `VALLM_CACHE_SEARCH` | `true` | Enable search result cache (L2) |
| `ENVIRONMENT` | `production` | Deployment environment |
| `PORT` | `8000` | API server port |
| `REDIS_HOST` | `redis` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |

### GPU Support

```bash
# Enable CUDA
export USE_CUDA=true

# Install GPU-enabled FAISS
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Changing the Embedding Model

Edit `app/embeddings.py`:
```python
vector_store = VectorStore(
    data_dir=data_dir,
    model_name="all-mpnet-base-v2"  # Higher quality, larger model
)
```

---

## Monitoring

### Prometheus Metrics

Available at `/metrics`:

- `http_requests_total` - Request count by method/endpoint/status
- `http_request_duration_seconds` - Request latency histogram
- `vector_search_requests_total` - Vector search operations
- `vector_search_duration_seconds` - Search latency
- `cache_hits_total` / `cache_misses_total` - Cache effectiveness
- `llm_generation_requests_total` - LLM generation operations

### Grafana Dashboards

Pre-configured dashboards in `monitoring/grafana/dashboards/`:
- VA LLM Specialist Model Overview
- API Performance
- Cache Metrics
- System Health

Access Grafana at http://localhost:3000 (default: admin/admin)

### Health Checks

```bash
# Basic health
curl http://localhost:8000/health

# Readiness (checks vector store, FAISS, memory)
curl http://localhost:8000/health/ready

# Liveness
curl http://localhost:8000/health/live
```

### Kubernetes

```bash
# View recent logs via API
curl http://localhost:8000/logs

# Log statistics
curl http://localhost:8000/logs/stats

# Docker logs
docker-compose logs -f vallm
```

---

## Project Structure

```
va_llm_v1/
├── app/                          # Main application
│   ├── app.py                    # FastAPI application
│   ├── embeddings.py             # VectorStore (FAISS)
│   ├── reasoning.py              # ReasoningEngine
│   ├── routes.py                 # All API endpoints (v1, v2, v3)
│   ├── precompute.py             # Build FAISS index
│   ├── train.py                  # LLM fine-tuning
│   ├── cache.py                  # Multi-level caching
│   ├── circuit_breaker.py        # Resilience pattern
│   ├── exceptions.py             # Custom exceptions
│   ├── health.py                 # Health check endpoints
│   ├── logging_config.py         # Structured logging
│   ├── metrics.py                # Prometheus metrics
│   ├── rate_limit.py             # Rate limiting
│   ├── data/                     # Data directory
│   │   ├── *.csv                 # Knowledge base files
│   │   ├── *.pdf                 # PDF documents
│   │   ├── vectorstore/          # FAISS index artifacts
│   │   └── model/                # Trained model artifacts
│   └── tests/                    # Test files
├── deployment/                   # Deployment configs
│   ├── kubernetes/               # K8s manifests
│   └── vm/                       # VM deployment
├── monitoring/                   # Monitoring stack
│   ├── prometheus.yml            # Prometheus config
│   └── grafana/                  # Grafana dashboards
├── scripts/                      # Utility scripts
├── Dockerfile                    # Container image
├── docker-compose.yml            # Full stack setup
├── requirements.txt              # Python dependencies
├── azure-pipelines.yml           # Azure DevOps CI/CD
└── .github/workflows/            # GitHub Actions CI/CD
```

---

## Data Management

### Knowledge Base Location

All data is in `app/data/`:
- `*.csv` - Structured knowledge (resources, incidents, recommendations)
- `*.pdf` - DevOps documentation
- `*.json` - Configuration data
- `vectorstore/` - FAISS index and document metadata

### Building the Index

```bash
# Standard precompute
python -m app.precompute

# Check index status
python -m app.precompute --action check
```

### Expanding the Dataset

```bash
# Generate synthetic data
python scripts/massive_data_expansion.py

# Rebuild index after adding data
python -m app.precompute
```

### LLM Fine-Tuning (Optional)

```bash
# Train on CSV data
python -m app.train --num-train-epochs 1

# Model saved to app/data/model/
```

---

## Troubleshooting

### Common Issues

**FAISS index not found**
```bash
# Rebuild the index
python -m app.precompute
```

**Index has 0 vectors / generic responses**
- Ensure CSV files exist in `app/data/`
- Run `python -m app.precompute` to populate the index

**Model download failed**
- Check internet connection (first run only)
- Model is cached after first download (~80MB)

**Out of memory**
- Use smaller model: `all-MiniLM-L6-v2` (default, 80MB)
- Reduce batch size in `embeddings.py`
- Increase container memory limits

**Connection refused on health check**
- Wait for startup to complete
- Check logs: `docker-compose logs vallm`

### Debug Mode

```bash
# Enable verbose logging
export VALLM_JSON_LOGGING=true
python -m app.app
```

---

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **Framework** | FastAPI, Uvicorn |
| **AI/ML** | sentence-transformers, FAISS, PyTorch, spaCy |
| **Data** | Pandas, NumPy, PyPDF2 |
| **Caching** | Redis, in-memory TTL cache |
| **Monitoring** | Prometheus, Grafana |
| **Container** | Docker, Docker Compose |
| **Orchestration** | Kubernetes (AKS) |
| **CI/CD** | GitHub Actions, Azure Pipelines |

---

## License

BSD 3-Clause License — see [LICENSE](./LICENSE) for details.

© 2026 Joel Otepa Wembo

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository and create a feature branch
2. Make changes and add/update tests
3. Run linting: `flake8 app/`
4. Submit a pull request describing your changes

For detailed deployment instructions, see `deployment.md`.

