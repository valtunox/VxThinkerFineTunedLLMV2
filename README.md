# VaLLM: Sovereign AI for Cloud Operations

> **Mission**: A private, production-grade AI reasoning engine that understands *your* specific cloud infrastructure. Unlike generic LLMs, VaLLM is grounded in your actual production data (logs, resources, configurations) and provides DevOps intelligence without data leaving your network.

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Project Structure](#project-structure)
- [Data Management](#data-management)
- [Troubleshooting](#troubleshooting)

---

## Architecture

VaLLM uses a hybrid **RAG (Retrieval-Augmented Generation) + Deterministic Reasoning** architecture.

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
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the service
python -m app.app
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
curl -X POST http://localhost:8000/api/model/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Why are IAM access keys not rotated in 90 days?",
    "top_k": 5,
    "include_reasoning": true
  }'

# Developer endpoint (Terraform generation)
curl -X POST http://localhost:8000/api/model/v1/developer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Create Terraform config for EKS cluster",
    "include_code": true
  }'
```

---

## Deployment

### Docker

```bash
# Build image
docker build -t vallm:latest .

# Run container
docker run -p 8000:8000 -v $(pwd)/app/data:/app/data vallm:latest
```

### Docker Compose (Full Stack)

```bash
# Start all services
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

### Core Endpoints

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

### V1 Endpoints - RAG + Reasoning

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
- VaLLM Overview
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

### Logs

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

Private - Internal use only.

---

## Contributing

1. Create a feature branch
2. Make changes and add tests
3. Run linting: `flake8 app/`
4. Submit pull request

For detailed deployment instructions, see `deployment.md`.

Summary of what’s in place:
1. va_llm_v1 – Provision intent API
New: va_llm_v1/app/services/ai/ml/cloud_routes.py
POST /api/cloud/provision-intent
Body: {"query": "user message"}.
Response:
Provisioning: query_type: "provisioning", intent (e.g. provision_vm, provision_docker, provision_database, …), payload (Golang-ready dict built from cloud_deployments.csv), confidence, match_prompt.
Non‑provisioning: query_type one of "incident", "cost", "billing", "security", "recommendation", "other"; intent and payload are null so the agent does not call Golang.
Uses the existing vector store over cloud_deployments (and any other CSVs). Rows with an intent column are treated as deployment rows; best match is turned into intent + payload via _raw_to_golang_payload() for VM, Kubernetes, Docker, FastAPI, static website, and database.
app.py
Registers the cloud router and documents the new endpoint in the API docstring.
So va_llm_v1 is the place that decides intent and how Golang wants the request; it returns that in the payload for the agent.
2. Precompute – Tag deployment rows
app/services/ai/ml/precompute.py
For each row, if the CSV has an intent column and it’s set, metadata is set with "type": "deployment".
Lets you later restrict search to deployment rows (e.g. with filter_type="deployment") if you want.
Re-run precompute after pulling so the FAISS index includes this metadata (and cloud_deployments.csv):
python -m app.services.ai.ml.precompute from the va_llm_v1 root (with --dataset-dir app/data if you use it).
3. InfinityAI cloud agent – Call va_llm, validate, then Golang
InfinityAI/app/services/ai/agents/cloud/agent.py
fetch_provision_intent(query)
Calls va_llm_v1 POST /api/cloud/provision-intent.
URL: env VA_LLM_V1_URL or VALLM_URL (default http://localhost:8002).
Confidence threshold: PROVISION_INTENT_CONFIDENCE_MIN (default 0.2).
Flow in handle_deployment_operation:
LLM (OpenAI/Claude/etc.) still runs first to produce the user-facing reply.
Call va_llm_v1 provision-intent.
If query_type == "provisioning" and intent is set and confidence >= PROVISION_INTENT_CONFIDENCE_MIN:
Use that intent and payload; merge in user_id, workspace_id, session_id, details.
Trigger Golang via execute_provisioning_service(intent, payload) (unchanged).
If query_type != "provisioning" (incidents, cost, billing, security, recommendations):
Do not set intent or call Golang; the LLM reply is the full answer (optionally formatted by the same or another LLM).
If va_llm_v1 is down or returns no provisioning match:
Fallback to the existing keyword-based intent + extractors and, when a provisioning intent is found, still call the Golang API.
So the agent scores/double-checks by only accepting provisioning when va_llm returns query_type=provisioning with sufficient confidence; then it triggers the Golang provisioner. For non‑provisioning, it only uses the LLM to format the answer.
Flow summary
User: “Deploy a t2.micro EC2” or “What were last week’s incidents?” or “Any cost recommendations?”
va_llm_v1 (provision-intent):
For “Deploy…”: returns query_type: "provisioning", intent, Golang payload.
For incidents/cost/recommendations: returns query_type: "incident" / "cost" / "recommendation" and no intent/payload.
Agent:
Provisioning → merge session/user/workspace into payload → call Golang → then LLM can format the provisioning result for the user.
Non‑provisioning → no Golang; LLM (OpenAI/Claude/etc.) formats the answer from context (e.g. incidents, cost, security, recommendations).
Ensure va_llm_v1 is running (e.g. port 8002) and VA_LLM_V1_URL points to it when running the InfinityAI agent. After changing precompute, re-run it and restart va_llm_v1 so the new index is loaded.