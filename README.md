# VaLLM Specialist Model — Multi-Industry AI Engine

> **Build domain-specific AI models for any industry.** One codebase, five industries (and counting). Fine-tune, embed, and serve models tailored to healthcare, finance, cloud, automation, or customer service — just pick your industry and go.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Supported Industries](#supported-industries)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
  - [Step 1: Environment Setup](#step-1-environment-setup)
  - [Step 2: Choose Your Industry](#step-2-choose-your-industry)
  - [Step 3: Prepare Datasets](#step-3-prepare-datasets)
  - [Step 4: Precompute Embeddings](#step-4-precompute-embeddings)
  - [Step 5: Train the Model](#step-5-train-the-model)
  - [Step 6: Serve the API](#step-6-serve-the-api)
  - [Step 7: Query Your Model](#step-7-query-your-model)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Dataset Reference](#dataset-reference)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## Overview

VaLLM Specialist Model is a **multi-industry AI engine** that lets you build domain-specific AI models from a single codebase. Instead of hardcoding for one domain, you configure an industry — and the entire pipeline (datasets, prompts, training, embeddings, API) adapts automatically.

**How it works:**

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│  Select      │────▶│  Prepare     │────▶│  Train /    │────▶│  Serve   │
│  Industry    │     │  Datasets    │     │  Embed      │     │  API     │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────┘
   config.py          prepare_data       train.py /            FastAPI
   .env               industry-aware     precompute.py         routes.py
```

---

## Supported Industries

| Industry | Use Cases | Starter Datasets |
|----------|-----------|-----------------|
| 🏥 **Healthcare** | Medical Q&A, clinical notes, diagnosis assistance, ICD coding | `medical_qa.csv`, `clinical_notes.csv`, `icd_codes.csv` |
| 💰 **Finance** | Financial analysis, transaction monitoring, compliance, accounting Q&A | `financial_qa.csv`, `transactions.csv`, `financial_terms.csv` |
| ☁️ **Cloud** | Cloud provisioning, DevOps automation, infrastructure Q&A | `cloud_deployments.csv`, `devops_qa.csv` |
| ⚙️ **Automation** | Workflow design, process optimization, RPA, CI/CD | `automation_workflows.csv`, `process_optimization.csv` |
| 🎧 **Customer Service** | Ticket resolution, FAQ bots, escalation routing | `support_tickets.csv`, `faq.csv`, `escalation_rules.csv` |

---

## Architecture

```
va_llm_specialist_model/
│
├── app/
│   ├── core/
│   │   ├── settings.py          # Global config + industry selection
│   │   ├── industry.py          # Industry registry (prompts, datasets, eval tasks)
│   │   ├── model_loader.py      # Model loading utilities
│   │   └── db.py                # Database configuration
│   │
│   ├── services/ai/ml/
│   │   ├── train.py             # Fine-tuning (industry-aware)
│   │   ├── precompute.py        # Embedding generation (industry-aware)
│   │   ├── embeddings.py        # VectorStore + FAISS search
│   │   ├── routes.py            # API endpoints (v1/v2/v3)
│   │   └── reasoning.py         # LLM reasoning engine
│   │
│   └── data/
│       ├── datasets/
│       │   ├── healthcare/      # Healthcare CSVs, JSONs, TXTs
│       │   ├── finance/         # Finance datasets
│       │   ├── cloud/           # Cloud/DevOps datasets
│       │   ├── automation/      # Automation datasets
│       │   └── customer_service/# Customer service datasets
│       ├── models/              # Trained model weights (per industry)
│       └── vectorstore/         # FAISS indexes (per industry)
│
├── deployment/                  # Kubernetes, Prometheus configs
├── scripts/                     # Data fetchers (S3, Azure, URL)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/valtunox/va_llm_specialist_model.git
cd va_llm_specialist_model

# Install
pip install -r requirements.txt

# Set industry (default: cloud)
export ACTIVE_INDUSTRY=healthcare

# Run the full pipeline
python app/services/ai/ml/precompute.py --industry healthcare
python app/services/ai/ml/train.py --industry healthcare --num-train-epochs 1

# Start the API
python -m app.app
```

---

## Step-by-Step Guide

### Step 1: Environment Setup

**Prerequisites:**
- Python 3.10+
- CUDA GPU (recommended) or CPU
- 8GB+ RAM (16GB+ for larger models)

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
# Industry Selection
ACTIVE_INDUSTRY=healthcare          # healthcare | finance | cloud | automation | customer_service

# LLM Provider (for reasoning API)
MODEL_PROVIDER=gemini               # ollama | openai | gemini | anthropic | huggingface
GOOGLE_API_KEY=your-key-here        # Or OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.

# Database (optional)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/vallm

# Redis (optional, for caching)
REDIS_URL=redis://localhost:6379/0
```

---

### Step 2: Choose Your Industry

Set your target industry in `.env` or via environment variable:

```bash
# Option A: Environment variable
export ACTIVE_INDUSTRY=finance

# Option B: CLI flag (overrides .env)
python app/services/ai/ml/train.py --industry finance

# Option C: API query parameter (runtime switching)
curl "http://localhost:8000/api/v1/query?industry=healthcare"
```

The industry selection controls:
- **System prompts** — Industry-specific instruction prefixes
- **Dataset paths** — `app/data/datasets/{industry}/`
- **Model output** — `app/data/models/{industry}/`
- **Vector index** — `app/data/vectorstore/{industry}/`
- **Evaluation tasks** — Industry-appropriate benchmarks

---

### Step 3: Prepare Datasets

Each industry has a dataset directory with starter CSVs included:

```
app/data/datasets/
├── healthcare/
│   ├── medical_qa.csv          # 80 medical Q&A pairs
│   ├── clinical_notes.csv      # 60 clinical records
│   └── icd_codes.csv           # 50 ICD-10 codes
├── finance/
│   ├── financial_qa.csv        # 80 finance Q&A pairs
│   ├── transactions.csv        # 70 transaction records
│   └── financial_terms.csv     # 60 financial terms
├── cloud/
│   ├── cloud_deployments.csv   # 80 deployment configs
│   └── devops_qa.csv           # 70 DevOps Q&A pairs
├── automation/
│   ├── automation_workflows.csv # 60 workflow definitions
│   └── process_optimization.csv # 50 optimization records
└── customer_service/
    ├── support_tickets.csv     # 80 support tickets
    ├── faq.csv                 # 60 FAQ entries
    └── escalation_rules.csv    # 40 escalation rules
```

**Add your own data:**

Drop CSV, JSON, TXT, or PDF files into the appropriate industry folder. Supported formats:

| Format | How It's Processed |
|--------|-------------------|
| **CSV** | Each row → training text (key-value pairs) |
| **JSON** | Nested objects flattened, arrays iterated |
| **TXT** | Chunked by paragraphs/sections |
| **PDF** | Page-by-page text extraction |

**Fetch from cloud storage:**

```bash
# From S3
python scripts/database/s3_fetcher.py --bucket my-data --prefix healthcare/

# From Azure Blob
python scripts/database/azure_blob_fetcher.py --container datasets --prefix finance/

# From URL
python scripts/database/url_fetcher.py --url https://example.com/data.csv --output app/data/datasets/healthcare/
```

---

### Step 4: Precompute Embeddings

Build a FAISS vector index from your datasets for semantic search:

```bash
# Precompute for a specific industry
python app/services/ai/ml/precompute.py --industry healthcare

# With custom embedding model
python app/services/ai/ml/precompute.py \
  --industry finance \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 128

# Output:
#   app/data/vectorstore/healthcare/faiss_index.bin
#   app/data/vectorstore/healthcare/documents.pkl
```

**What happens:**
1. Loads all CSVs/JSONs from `app/data/datasets/{industry}/`
2. Converts rows to text using industry-specific templates
3. Generates embeddings with sentence-transformers
4. Builds FAISS IndexFlatIP (cosine similarity)
5. Saves index + documents to `app/data/vectorstore/{industry}/`

---

### Step 5: Train the Model

Fine-tune a causal language model on your industry data:

```bash
# Basic training (CPU-friendly, uses distilgpt2 by default)
python app/services/ai/ml/train.py --industry healthcare --num-train-epochs 1

# With a larger model (needs GPU)
python app/services/ai/ml/train.py \
  --industry finance \
  --model-name-or-path microsoft/phi-2 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 4

# Train on specific file types only
python app/services/ai/ml/train.py \
  --industry cloud \
  --file-types csv,json

# Train on a single file
python app/services/ai/ml/train.py \
  --industry healthcare \
  --dataset app/data/datasets/healthcare/medical_qa.csv
```

**Model options by hardware:**

| Hardware | Model | VRAM | Notes |
|----------|-------|------|-------|
| CPU only | `sshleifer/tiny-gpt2` | — | ~2MB, instant, for testing |
| CPU / Low RAM | `distilgpt2` | — | ~350MB, decent quality |
| 8GB GPU | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~3GB | Good starter |
| 12GB GPU | `microsoft/phi-2` | ~6GB | Strong reasoning |
| 16GB+ GPU | `mistralai/Mistral-7B-Instruct-v0.2` | ~14GB | Production quality |
| 24GB+ GPU | `Qwen/Qwen2.5-Coder-7B-Instruct` | ~15GB | Best for code/cloud |

**Output:**
```
app/data/models/{industry}/
├── config.json
├── tokenizer.json
└── pytorch_model.bin
```

---

### Step 6: Serve the API

```bash
# Start the FastAPI server
python -m app.app

# Or with uvicorn directly
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload

# Or with Docker
docker-compose up -d
```

The server auto-loads:
- Trained model from `app/data/models/{active_industry}/`
- FAISS index from `app/data/vectorstore/{active_industry}/`
- Industry-specific prompts and routing

---

### Step 7: Query Your Model

**V1 — Simple query:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of Type 2 diabetes?"}'
```

**V2 — With entity extraction:**
```bash
curl -X POST http://localhost:8000/api/v2/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze this transaction for fraud risk", "industry": "finance"}'
```

**V3 — Advanced reasoning:**
```bash
curl -X POST http://localhost:8000/api/v3/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Design a CI/CD pipeline for a microservices architecture", "industry": "cloud"}'
```

**Runtime industry switching:**
```bash
# Override industry per request
curl "http://localhost:8000/api/v1/query?industry=customer_service" \
  -d '{"query": "How do I handle an angry customer about a billing issue?"}'
```

---

## Configuration

### Industry Registry (`app/core/industry.py`)

Each industry is defined in `INDUSTRY_REGISTRY` with:

```python
INDUSTRY_REGISTRY = {
    Industry.HEALTHCARE: {
        "system_prompt": "You are an AI assistant for healthcare...",
        "dataset_sources": ["medmcqa", "pubmedqa", "medical_meadow"],
        "eval_tasks": ["medqa", "pubmedqa"],
        "domain_keywords": ["patient", "diagnosis", "treatment", ...],
        "embedding_text_template": "Medical context: {content}",
        ...
    },
    # ... finance, cloud, automation, customer_service
}
```

### Settings (`app/core/settings.py`)

Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `ACTIVE_INDUSTRY` | `cloud` | Target industry |
| `MODEL_PROVIDER` | `gemini` | LLM provider for reasoning |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `VECTOR_DB_TYPE` | `faiss` | Vector database backend |
| `ENABLE_TRAINING` | `true` | Enable training endpoints |
| `ENABLE_EMBEDDINGS` | `true` | Enable embedding generation |

---

## API Reference

### V1 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/query` | Industry-aware semantic query |
| POST | `/api/v1/developer` | Developer-focused analysis |
| POST | `/api/v1/terminal` | Terminal command generation |

### V2 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v2/query` | Query with entity extraction |
| POST | `/api/v2/extract` | Named entity extraction |
| POST | `/api/v2/upload` | Upload and process documents |
| GET | `/api/v2/status` | System status and health |

### V3 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v3/query` | Advanced multi-step reasoning |

### Common Parameters

```json
{
  "query": "Your question here",
  "industry": "healthcare",       // Optional: override active industry
  "top_k": 5,                     // Number of similar documents to retrieve
  "include_sources": true          // Include source documents in response
}
```

---

## Deployment

### Docker

```bash
# Build and run
docker-compose up -d

# With specific industry
ACTIVE_INDUSTRY=finance docker-compose up -d
```

### Kubernetes

```bash
# AWS EKS
kubectl apply -f deployment/kubernetes/eks/deploy_kubernetes_eks.yml

# Azure AKS
kubectl apply -f deployment/kubernetes/aks/deploy_kubernetes_aks.yml

# Generic
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
```

### CI/CD

- **GitHub Actions**: `.github/workflows/LLM_PROD_VM_CICD.yml`
- **GitLab CI**: `.gitlab-ci.yml`
- **Azure Pipelines**: `azure-pipelines.yml`

---

## Monitoring

Prometheus + Grafana stack included:

```bash
# Prometheus config
deployment/monitoring/prometheus.yml

# Metrics endpoint
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/health
```

---

## Roadmap

### Phase 1 ✅ — Foundation
- [x] Multi-industry engine architecture
- [x] Industry registry with 5 industries
- [x] Industry-aware training pipeline
- [x] Industry-aware embedding pipeline
- [x] Starter datasets for all industries
- [x] Runtime industry switching via API

### Phase 2 🔄 — Scale
- [ ] Add more industries (legal, education, real estate, manufacturing)
- [ ] HuggingFace dataset auto-download per industry
- [ ] QLoRA / LoRA fine-tuning for large models (70B+)
- [ ] Multi-industry model (single model, multiple domains)
- [ ] Automated evaluation benchmarks per industry

### Phase 3 🔮 — Production
- [ ] Model versioning and A/B testing
- [ ] Auto-scaling inference (vLLM / TGI)
- [ ] RAG pipeline with industry-specific chunking
- [ ] Feedback loop (user corrections → retraining)
- [ ] Enterprise SSO + RBAC per industry

### Phase 4 🚀 — Platform
- [ ] Web UI for dataset management
- [ ] One-click industry deployment
- [ ] Marketplace for pre-trained industry models
- [ ] Multi-tenant SaaS mode
- [ ] Custom industry creation wizard

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Add datasets to the appropriate `app/data/datasets/{industry}/` folder
4. Run training + precompute to verify
5. Submit a PR

**Adding a new industry:**

1. Add enum value to `Industry` in `app/core/industry.py`
2. Add entry to `INDUSTRY_REGISTRY` with prompts, datasets, eval tasks
3. Create `app/data/datasets/{new_industry}/` with starter CSVs
4. Test: `python app/services/ai/ml/precompute.py --industry new_industry`

---

## License

MIT — see [LICENSE](LICENSE)

---

**Built by [Valtunox](https://github.com/valtunox)** | Questions? Open an issue.
