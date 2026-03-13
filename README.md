# VaLLM Specialist Model

> **Multi-Purpose AI for Document Intelligence & Business Analytics**

A private, production-grade AI specialist model for document verification, financial analysis, billing/invoice processing, accounting automation, and business recommendations. VaLLM is grounded in your organization's actual documents, transactions, and business data to deliver precise, domain-specific intelligence without data leaving your network.

**Author**: Joel Otepa Wembo - [joelwembo.com](https://joelwembo.com)

---

## Table of Contents

- [Use Cases](#use-cases)
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

## Use Cases

### Document Verification & Analysis

| Question | What VaLLM Does |
|----------|-----------------|
| Is this invoice authentic? Check for tampering. | Runs OCR extraction, cross-references entity data (vendor, amounts, dates), checks structural consistency, returns confidence score with risk flags. |
| Extract all line items, tax amounts, and payment terms from this PDF. | Document parser extracts structured fields and classifies document type (invoice, receipt, contract, statement). |
| Compare these two contracts and highlight differences. | Document matching computes content similarity, metadata match, structural similarity, and entity overlap. |
| Find duplicate or near-duplicate invoices. | Deduplication pipeline uses content hashing and vector similarity to flag potential duplicates. |

### Financial Analysis & Accounting

| Question | What VaLLM Does |
|----------|-----------------|
| Reconcile this bank statement against accounts payable. | Matches transactions by amount, date proximity, and reference number; flags unmatched items. |
| Categorize these 500 transactions into expense categories. | Entity scoring model classifies each transaction with confidence scores and SHAP explanations. |
| What's our spending trend Q1 vs Q2, and forecast Q3? | Financial prediction model analyzes historical patterns, seasonal trends, and market conditions. |
| Flag transactions above $10K lacking proper documentation. | Cross-references transaction records against document store for compliance gaps. |

### Billing & Invoice Processing

| Question | What VaLLM Does |
|----------|-----------------|
| Process this batch of 200 scanned receipts. | OCR pipeline (pdfplumber -> pypdf -> Tesseract) extracts amounts, dates, vendors into structured records. |
| Which invoices are past due and total outstanding? | Queries transaction store by due date and payment status, aggregates by vendor, age, priority. |
| Match incoming payments to open invoices. | Amount matching, reference number lookup, and fuzzy vendor matching for auto-reconciliation. |

### Business Recommendations

| Question | What VaLLM Does |
|----------|-----------------|
| Where can we reduce costs based on spending data? | Analyzes spending patterns, vendor concentration, and category trends; ranks optimization opportunities by impact. |
| Which vendors should we consolidate for better pricing? | Clusters vendor data by category, analyzes spend distribution, identifies consolidation opportunities. |
| What compliance risks exist in our document workflow? | Scans verification records for missing approvals, expired documents, incomplete audit trails. |

### Entity Scoring & Risk Assessment

| Question | What VaLLM Does |
|----------|-----------------|
| Score this vendor application for risk level. | XGBoost model evaluates industry, company size, risk indicators, region; returns score with SHAP breakdown. |
| Rate creditworthiness of this business entity. | Combines financial statement analysis, transaction history, and industry benchmarks into composite score. |

### Semantic Search & Retrieval

| Question | What VaLLM Does |
|----------|-----------------|
| Find all documents related to "tax withholding compliance 2025". | Hybrid BM25 + dense vector search with cross-encoder reranking. |
| Search for contracts similar to this lease agreement. | Encodes query document and performs nearest-neighbor search across the vector store. |

---

## Architecture

VaLLM uses a hybrid **RAG + Multi-Agent + Scoring** architecture.

```
+-------------------------------------------------------------+
|                    User / Application                        |
+----------------------------+--------------------------------+
                             | HTTP Request
                             v
+-------------------------------------------------------------+
|              FastAPI Gateway (app.py)                        |
|  +- Rate Limiting        +- Metrics Collection              |
|  +- Request Logging      +- CORS / Auth                     |
+----------------------------+--------------------------------+
                             |
         +-------------------+-------------------+
         v                   v                   v
    +---------+        +-----------+       +-----------+
    | /search |        | /api/v1   |       | /api/v2   |
    +----+----+        +-----+-----+       +-----+-----+
         |                   |                   |
         v                   v                   v
+-------------------------------------------------------------+
|                 VectorStore (FAISS)                          |
|  +- Embedding: BGE / all-MiniLM-L6-v2 (384-dim)            |
|  +- L1 Cache: Embeddings (TTL 1h)                           |
|  +- L2 Cache: Search Results (TTL 30m)                      |
+----------------------------+--------------------------------+
                             |
         +-------------------+-------------------+
         v                   v                   v
+----------------+  +-----------------+  +------------------+
| Document OCR   |  | XGBoost Scoring |  | Multi-LLM Router |
| pdfplumber     |  | SHAP Explain    |  | OpenAI/Anthropic |
| pypdf/Tesseract|  | Risk Assessment |  | Google/Ollama    |
+----------------+  +-----------------+  +------------------+
                             |
                             v
+-------------------------------------------------------------+
|              Multi-Agent Orchestration                       |
|  DocumentAnalysis -> Verification -> Financial -> Recommend  |
+-------------------------------------------------------------+
```

### Core Components

| Component | Description |
|-----------|-------------|
| **VectorStore** | FAISS-based semantic search with sentence-transformers embeddings |
| **Document OCR** | Multi-fallback text extraction (pdfplumber -> pypdf -> Tesseract) |
| **Multi-LLM Router** | Intelligent routing between OpenAI, Anthropic, Google, Ollama, and local models |
| **XGBoost Scoring** | Entity and risk scoring with SHAP explainability |
| **Document Matching** | Content similarity, deduplication, and cross-referencing |
| **Financial Prediction** | Trend analysis and forecasting with confidence intervals |
| **Multi-Agent Pipeline** | Document analysis, verification, financial analysis, and recommendation agents |
| **Kafka Events** | Async event streaming for document processing and data imports |
| **Celery Tasks** | Background processing for OCR, imports, and batch operations |
| **Monitoring** | Prometheus metrics, structured logging, health probes |

---

## Features

- **Document Verification**: OCR extraction, authenticity checks, confidence scoring with risk flags
- **Financial Analysis**: Transaction reconciliation, categorization, trend forecasting
- **Billing Processing**: Batch invoice/receipt processing, payment matching, aging reports
- **Business Recommendations**: Cost optimization, vendor consolidation, compliance risk detection
- **Entity Scoring**: XGBoost risk assessment with SHAP explainability
- **Semantic Search**: Hybrid BM25 + dense vector search with cross-encoder reranking
- **Multi-LLM Support**: OpenAI, Anthropic Claude, Google Gemini, Qwen, DeepSeek, HuggingFace, Ollama
- **Multi-Industry**: Configurable for HEALTHCARE, FINANCE, CLOUD, AUTOMATION, CUSTOMER_SERVICE
- **Multi-Tenant**: Organization-scoped data isolation with PostgreSQL
- **100% On-Premise Option**: Full data sovereignty with local models via Ollama
- **Production Ready**: Rate limiting, structured logging, Celery task queues, Kafka events
- **Observable**: Prometheus metrics, Grafana dashboards, health probes

---

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ (for ORM models)
- Redis (for Celery broker and caching)
- Docker and Docker Compose (recommended)
- 4GB+ RAM

### Option 1: Docker Compose (Recommended)

```bash
# Clone and navigate to project
cd va_llm_specialist_model

# Start full stack (API + Redis + PostgreSQL + Prometheus + Grafana)
docker-compose up -d

# Check status
docker-compose ps
```

Services will be available at:
- **API**: http://localhost:8746
- **API Docs**: http://localhost:8746/docs
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
3. Auto-train LLM if not present (when `VALLM_AUTO_TRAIN=true`)
4. Start API server on http://localhost:8746

### Test the API

```bash
# Health check
curl http://localhost:8746/health

# Semantic search
curl -X POST http://localhost:8746/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "invoices with missing tax documentation",
    "top_k": 5
  }'

# Text generation
curl -X POST http://localhost:8746/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Analyze this invoice for compliance issues:",
    "max_new_tokens": 200
  }'

# RAG query (V1)
curl -X POST http://localhost:8746/api/models/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the top overdue invoices this quarter?",
    "top_k": 5,
    "include_reasoning": true
  }'
```

---

## Deployment

### Docker

```bash
# Build image
docker build -t vallm:latest .

# Run container
docker run -p 8746:8746 -v $(pwd)/app/data:/app/data vallm:latest
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
- `vallm`: FastAPI application (port 8746)
- `redis`: Celery broker and caching (port 6379)
- `postgres`: Document and transaction storage (port 5432)
- `prometheus`: Metrics collection (port 9090)
- `grafana`: Metrics visualization (port 3000)

### Kubernetes

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

**GitLab CI** (`.gitlab-ci.yml`):
- Multi-stage pipeline
- Container registry push
- Kubernetes deployment

**Azure Pipelines** (`azure-pipelines.yml`):
- Build and push to Azure Container Registry
- Deploy to VM via SSH or AKS

---

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | HTML status page |
| GET | `/health` | Basic health check |
| GET | `/docs` | OpenAPI documentation |
| GET | `/stats` | Vector store and cache statistics |
| GET | `/logs` | View recent logs |
| GET | `/logs/stats` | Log statistics |
| DELETE | `/logs/clear` | Clear logs (with backup) |
| POST | `/search` | Vector similarity search |
| POST | `/generate` | Text generation (if model loaded) |

### V1 Endpoints - RAG + Reasoning

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/models/v1/query` | RAG query with chain-of-thought reasoning |
| POST | `/api/models/v1/developer` | Developer assistance |
| POST | `/api/models/v1/terminal` | CLI/terminal assistance |

**Example Request**:
```json
{
  "query": "Which vendors have the most overdue invoices?",
  "top_k": 5,
  "include_reasoning": true
}
```

### V2 Endpoints - NLP + Document Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/models/v2/query` | NLP-enhanced query with entity extraction |
| POST | `/api/models/v2/upload` | Document/image upload for OCR analysis |
| POST | `/api/models/v2/extract` | Entity extraction from text |
| GET | `/api/models/v2/status` | NLP capability status |

### V3 Endpoints - Analytics & Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/models/v3/query` | Financial analytics and pattern detection |

### Monitoring Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/monitoring/health` | Detailed health with dependency checks |
| GET | `/metrics` | Prometheus metrics |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VALLM_AUTO_PRECOMPUTE` | `true` | Auto-build FAISS index if missing |
| `VALLM_AUTO_TRAIN` | `true` | Auto-train LLM model if missing |
| `USE_CUDA` | `false` | Enable GPU acceleration |
| `VALLM_JSON_LOGGING` | `false` | Enable structured JSON logging |
| `VALLM_RATE_LIMIT_ENABLED` | `false` | Enable rate limiting |
| `VALLM_RATE_LIMIT_PER_MINUTE` | `60` | Max requests per minute per client |
| `VALLM_CACHE_EMBEDDINGS` | `true` | Enable embedding cache (L1) |
| `VALLM_CACHE_SEARCH` | `true` | Enable search result cache (L2) |
| `DATABASE_URL` | - | PostgreSQL connection string |
| `REDIS_HOST` | `redis` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `OPENAI_API_KEY` | - | OpenAI API key (for multi-LLM routing) |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka brokers |
| `ENVIRONMENT` | `production` | Deployment environment |

### GPU Support

```bash
# Enable CUDA
export USE_CUDA=true

# Install GPU-enabled FAISS
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Multi-Industry Configuration

VaLLM supports industry-specific configurations:

```python
# Set via environment or settings
VALLM_INDUSTRY = "FINANCE"  # HEALTHCARE, FINANCE, CLOUD, AUTOMATION, CUSTOMER_SERVICE
```

Each industry mode adjusts entity extraction rules, classification categories, compliance checks, and scoring weights.

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
- `document_ocr_requests_total` - Document processing operations
- `scoring_requests_total` - Entity scoring operations

### Health Checks

```bash
# Basic health
curl http://localhost:8746/health

# Detailed health (monitoring router)
curl http://localhost:8746/monitoring/health
```

### Logs

```bash
# View recent logs via API
curl http://localhost:8746/logs

# Log statistics
curl http://localhost:8746/logs/stats

# Docker logs
docker-compose logs -f vallm
```

---

## Project Structure

```
va_llm_specialist_model/
+-- app/                              # Main application
|   +-- app.py                        # FastAPI application entry point
|   +-- __init__.py
|   +-- auth/                         # Authentication & rate limiting
|   +-- core/                         # Settings, logging, model registry
|   +-- orm/                          # SQLAlchemy models (Document, Transaction, etc.)
|   +-- schemas/                      # Pydantic request/response schemas
|   +-- services/
|   |   +-- ai/
|   |   |   +-- agents/              # Multi-agent orchestration
|   |   |   |   +-- base_agent.py    # Document, Verification, Financial, Recommendation agents
|   |   |   +-- ml/
|   |   |       +-- embedding.py     # Embedding service (BGE / MiniLM)
|   |   |       +-- search.py        # Hybrid BM25 + dense search
|   |   |       +-- scoring.py       # XGBoost entity scoring
|   |   |       +-- matching.py      # Document matching & deduplication
|   |   |       +-- explainability.py # SHAP feature importance
|   |   |       +-- ocr.py           # Document OCR pipeline
|   |   |       +-- train.py         # LLM fine-tuning
|   |   |       +-- precompute.py    # FAISS index builder
|   |   |       +-- llm_router.py    # Multi-LLM routing (OpenAI/Anthropic/Ollama)
|   |   |       +-- models/          # ML model implementations
|   |   +-- celery/                   # Celery task queue
|   |   +-- kafka/                    # Kafka event streaming
|   |   +-- rabbitmq/                 # RabbitMQ messaging
|   |   +-- redis/                    # Redis caching service
|   |   +-- monitoring/               # Prometheus metrics & health
|   |   +-- queue/                    # Background tasks (OCR, data import)
|   |   +-- sse/                      # Server-Sent Events
|   |   +-- tenants/                  # Multi-tenant service
|   +-- data/                         # Data directory
|   |   +-- datasets/                 # CSV/PDF/JSON source files
|   |   +-- vectorstore/              # FAISS index artifacts
|   |   +-- models/                   # Trained model artifacts
|   +-- tests/                        # Test files
+-- scripts/                          # Utility scripts (data fetching, preprocessing)
+-- deployment/                       # Deployment configs
+-- .github/workflows/                # GitHub Actions CI/CD
+-- .gitlab-ci.yml                    # GitLab CI/CD
+-- azure-pipelines.yml               # Azure Pipelines CI/CD
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
+-- requirements_analysis.txt         # Extended analysis dependencies
```

---

## Data Management

### Knowledge Base

All data is in `app/data/`:
- `datasets/*.csv` - Structured document and transaction data
- `datasets/*.pdf` - Financial documents, invoices, contracts
- `datasets/*.json` - Configuration and reference data
- `vectorstore/` - FAISS index and document metadata
- `models/` - Trained model weights

### Building the Index

```bash
# Build FAISS index from datasets
python -m app.services.ai.ml.precompute

# Auto-build is enabled by default (VALLM_AUTO_PRECOMPUTE=true)
```

### LLM Fine-Tuning

```bash
# Train on domain-specific data
python -m app.services.ai.ml.train --num-train-epochs 1

# Model saved to app/data/models/
```

### Adding New Data

1. Place CSV/PDF/JSON files in `app/data/datasets/`
2. Run `python -m app.services.ai.ml.precompute` to rebuild the index
3. Restart the service to load the new index

---

## Troubleshooting

### Common Issues

**FAISS index not found**
```bash
python -m app.services.ai.ml.precompute
```

**Index has 0 vectors / generic responses**
- Ensure data files exist in `app/data/datasets/`
- Run precompute to populate the index

**Model download failed**
- Check internet connection (first run only)
- Embedding model is cached after first download (~80MB)

**Out of memory**
- Use smaller embedding model: `all-MiniLM-L6-v2` (default, 80MB)
- Reduce batch size in embedding configuration
- Increase container memory limits

**Database connection failed**
- Ensure PostgreSQL is running and `DATABASE_URL` is set
- Run migrations if needed

### Debug Mode

```bash
export VALLM_JSON_LOGGING=true
python -m app.app
```

---

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **Framework** | FastAPI, Uvicorn, Pydantic v2 |
| **AI/ML** | sentence-transformers, FAISS, PyTorch, XGBoost, SHAP, spaCy |
| **LLM** | OpenAI, Anthropic Claude, Google Gemini, Qwen, DeepSeek, Ollama |
| **Agents** | LangChain, LangGraph |
| **OCR** | pdfplumber, pypdf, Tesseract |
| **Database** | PostgreSQL, SQLAlchemy |
| **Caching** | Redis |
| **Task Queue** | Celery |
| **Messaging** | Kafka, RabbitMQ |
| **Monitoring** | Prometheus, Grafana |
| **Container** | Docker, Docker Compose |
| **Orchestration** | Kubernetes (AKS/EKS) |
| **CI/CD** | GitHub Actions, GitLab CI, Azure Pipelines |

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
