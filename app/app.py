"""
VaLLM - Vector-based Local LLM for Cloud Operations
=====================================================
FastAPI application with embeddings, FAISS, and chain-of-thoughts reasoning.

A sovereign, private AI reasoning engine for cloud infrastructure operations.
Unlike generic LLMs, VaLLM is grounded in your actual production data (logs,
resources, configurations) and provides precise DevOps intelligence.

ARCHITECTURE:
=============
    ┌─────────────────────────────────────────────────────────────┐
    │                      VaLLM CORE                             │
    ├─────────────────────────────────────────────────────────────┤
    │  Embeddings (sentence-transformers/all-MiniLM-L6-v2)        │
    │  Vector Store (FAISS) ─────► Semantic Search                │
    │  LLM Model (distilgpt2/Mistral) ─────► Text Generation      │
    │  Reasoning Engine ─────► Chain-of-Thought Analysis          │
    │  NLP (spaCy) ─────► Entity Extraction (optional)            │
    └─────────────────────────────────────────────────────────────┘

MEMORY & CONTEXT:
=================
    - Session Memory: In-memory conversation history per request
    - Vector Memory: FAISS index for semantic retrieval (long-term)
    - Context Window: Managed by LLM token limits
    - For multi-tenant: Add PostgreSQL for persistent sessions

DATA FLOW:
==========
    1. precompute.py → Embeds CSVs → FAISS index (vectorstore/)
    2. train.py → Fine-tunes LLM on CSVs → Model weights (model/)
    3. app.py → Loads both → Serves API endpoints

USAGE:
======
    # Quick start
    python -m app.app

    # Auto-train is on by default; set VALLM_AUTO_TRAIN=false to disable

ENVIRONMENT VARIABLES:
======================
    VALLM_AUTO_PRECOMPUTE=true  (default: true)  - Auto-build FAISS if missing
    VALLM_AUTO_TRAIN=true       (default: true)  - Auto-train LLM if missing
    USE_CUDA=true               (default: false) - Use GPU for inference
    VALLM_CACHE_EMBEDDINGS=true (default: true)  - Cache query embeddings (L1, TTL 1h)
    VALLM_CACHE_SEARCH=true     (default: true)  - Cache search results (L2, TTL 30m)
    VALLM_CACHE_EMBEDDINGS_MAXSIZE (default: 2000)
    VALLM_CACHE_SEARCH_MAXSIZE     (default: 1000)

API ENDPOINTS:
==============
    Core:
        GET  /health              - Health check
        GET  /logs                - View recent logs
        POST /search              - Vector similarity search
        POST /generate            - LLM text generation
    
    V1 (RAG + Reasoning):
        POST /api/models/v1/query     - RAG query with reasoning
        POST /api/models/v1/developer - Developer assistance
        POST /api/models/v1/terminal  - Terminal/CLI assistance

    V2 (NLP + Document Analysis):
        POST /api/models/v2/query     - NLP-enhanced query
        POST /api/models/v2/extract   - Entity extraction
        POST /api/models/v2/upload    - Document/image upload
        GET  /api/models/v2/status    - NLP capabilities status

    V3 (Cloud/DevOps Incident Patterns):
        POST /api/models/v3/query     - Unusual incident patterns + metrics

    Cloud provisioning (intent + Golang payload for agent):
        POST /api/cloud/provision-intent - Intent + payload for provisioning; query_type for incidents/cost/billing/security/recommendations

DEPLOYMENT:
===========
    Single Node:  python -m app.app
    Production:   uvicorn app.app:app --host 0.0.0.0 --port 8745 --workers 4
    Docker:       docker run -p 8745:8745 vallm:latest
    Kubernetes:   See README.md for deployment manifests
"""

import os
import sys
import json
import logging
import time
import uuid
import subprocess
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

import faiss
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from starlette.middleware.base import BaseHTTPMiddleware

# =============================================================================
# AUTO-BUILD CONFIGURATION
# =============================================================================
# Set to True to automatically run precompute.py when FAISS index is missing
AUTO_PRECOMPUTE = os.getenv("VALLM_AUTO_PRECOMPUTE", "true").lower() == "true"
# Set to True to automatically run train.py when model is missing (takes longer!)
AUTO_TRAIN = os.getenv("VALLM_AUTO_TRAIN", "true").lower() == "true"

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Create logs directory
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "logs.txt"

# Custom formatter for beautiful logs
class VaLLMFormatter(logging.Formatter):
    """Custom formatter with colors and structured output"""
    
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[95m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        # Add color for console
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Build the log message
        if hasattr(record, 'request_id'):
            prefix = f"[{timestamp}] [{record.levelname:8}] [REQ:{record.request_id[:8]}]"
        else:
            prefix = f"[{timestamp}] [{record.levelname:8}]"
        
        # Add color for terminal output
        if hasattr(record, 'use_color') and record.use_color:
            return f"{color}{prefix}{reset} {record.getMessage()}"
        return f"{prefix} {record.getMessage()}"


# Configure logger
logger = logging.getLogger("vallm")
logger.setLevel(logging.DEBUG)

# Console handler (with colors)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = VaLLMFormatter()
console_handler.setFormatter(console_formatter)

# File handler (no colors, detailed)
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def log_request(request_id: str, method: str, path: str, client: str, body: dict = None):
    """Log incoming request"""
    log_entry = {
        "type": "REQUEST",
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "method": method,
        "path": path,
        "client_ip": client,
        "body_preview": str(body)[:200] if body else None
    }
    
    # Console log
    logger.info(f"→ {method} {path} | Client: {client}")
    
    # Detailed file log
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"REQUEST | {datetime.utcnow().isoformat()}\n")
        f.write(f"{'='*80}\n")
        f.write(json.dumps(log_entry, indent=2))
        f.write("\n")


def log_response(request_id: str, status_code: int, duration_ms: float, response_preview: str = None):
    """Log outgoing response"""
    log_entry = {
        "type": "RESPONSE",
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        "response_preview": response_preview[:300] if response_preview else None
    }
    
    # Status emoji
    if status_code < 300:
        status_icon = "✓"
    elif status_code < 400:
        status_icon = "→"
    elif status_code < 500:
        status_icon = "⚠"
    else:
        status_icon = "✗"
    
    # Console log
    logger.info(f"← {status_icon} {status_code} | {duration_ms:.2f}ms")
    
    # Detailed file log
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\nRESPONSE | {datetime.utcnow().isoformat()}\n")
        f.write(f"{'-'*40}\n")
        f.write(json.dumps(log_entry, indent=2))
        f.write(f"\n{'='*80}\n\n")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests and responses"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Try to get request body for POST requests
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body_bytes = await request.body()
                if body_bytes:
                    body = json.loads(body_bytes.decode())
                # Reset body for downstream processing
                async def receive():
                    return {"type": "http.request", "body": body_bytes}
                request._receive = receive
            except:
                pass
        
        # Log request
        log_request(
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            client=client_ip,
            body=body
        )
        
        # Process request and measure time
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000
            duration_seconds = duration_ms / 1000
            
            # Try to get response body preview
            response_preview = None
            
            # Log response
            log_response(
                request_id=request_id,
                status_code=response.status_code,
                duration_ms=duration_ms,
                response_preview=response_preview
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            duration_seconds = duration_ms / 1000
            
            logger.error(f"✗ Error: {str(e)} | {duration_ms:.2f}ms")
            raise

# Support both direct execution and module execution
try:
    from .services.ai.ml.embeddings import VectorStore
    from .services.ai.ml.reasoning import ReasoningEngine
    from .services.ai.ml.routes import router, router_v2, router_v3
    from .services.ai.ml.cloud_routes import router as cloud_router
except ImportError as e:
    # Only fall back when running without a package context.
    if "attempted relative import with no known parent package" in str(e):
        from services.ai.ml.embeddings import VectorStore
        from services.ai.ml.reasoning import ReasoningEngine
        from services.ai.ml.routes import router, router_v2, router_v3
        from services.ai.ml.cloud_routes import router as cloud_router
    else:
        raise

# Global instances
vector_store = None
reasoning_engine = None
tokenizer = None
model = None
faiss_index = None


def _run_subprocess_with_progress(cmd: list, cwd: str, timeout_seconds: int = 3600, label: str = "Process"):
    """Run a subprocess and stream stdout/stderr so progress is visible in real time."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Use UTF-8 for subprocess stdout so emoji/special chars work on Windows (avoid cp1252 UnicodeEncodeError)
    if sys.platform == "win32":
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        for line in proc.stdout or []:
            line = line.rstrip()
            if line:
                print(f"    \033[90m[{label}] {line}\033[0m")
        proc.wait(timeout=timeout_seconds)
        return proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        print(f"\n    \033[93m⚠️  {label} timed out after {timeout_seconds}s\033[0m")
        return -1
    except Exception as e:
        print(f"\n    \033[91m❌ {label} error: {e}\033[0m")
        return -1


def run_precompute():
    """Run precompute.py to build FAISS index; streams progress to console."""
    print("\n" + "=" * 70)
    print("🔧 AUTO-PRECOMPUTE: Building FAISS index (progress below)...")
    print("=" * 70 + "\n")
    precompute_script = Path(__file__).parent / "services" / "ai" / "ml" / "precompute.py"
    if not precompute_script.exists():
        print("    ❌ precompute.py not found!")
        return False
    cwd = str(Path(__file__).parent.parent)
    code = _run_subprocess_with_progress(
        [sys.executable, "-m", "app.services.ai.ml.precompute"],
        cwd=cwd,
        timeout_seconds=600,
        label="precompute",
    )
    if code == 0:
        print("\n    ✅ FAISS index built successfully!")
        return True
    print(f"\n    ❌ Precompute failed with code: {code}")
    return False


def run_train():
    """Run train.py to build LLM model; streams training progress (epochs, loss) to console."""
    print("\n" + "=" * 70)
    print("🔧 AUTO-TRAIN: Training LLM model — watch progress below (epochs, loss)...")
    print("=" * 70 + "\n")
    train_module = Path(__file__).parent / "services" / "ai" / "ml" / "train.py"
    if not train_module.exists():
        print("    ❌ train.py not found!")
        return False
    cwd = str(Path(__file__).parent.parent)
    code = _run_subprocess_with_progress(
        [sys.executable, "-m", "app.services.ai.ml.train", "--num-train-epochs", "1"],
        cwd=cwd,
        timeout_seconds=3600,
        label="train",
    )
    if code == 0:
        print("\n    ✅ LLM model trained successfully!")
        return True
    print(f"\n    ❌ Training failed with code: {code}")
    return False


def check_csv_files(data_dir: Path) -> int:
    """Check how many CSV files exist in data directory"""
    csv_files = list(data_dir.glob("*.csv"))
    return len(csv_files)


def _gather_model_info(model_dir: Path) -> Dict[str, Any]:
    """Read trained model metadata from model_dir/config.json."""
    info: Dict[str, Any] = {"available": False}
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return info
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        info["available"] = True
        info["architecture"] = (cfg.get("architectures") or ["unknown"])[0]
        info["model_type"] = cfg.get("model_type", "unknown")
        info["hidden_size"] = cfg.get("hidden_size", "?")
        info["num_layers"] = cfg.get("num_hidden_layers", "?")
        info["attention_heads"] = cfg.get("num_attention_heads", "?")
        info["vocab_size"] = cfg.get("vocab_size", "?")
        checkpoints = sorted(d.name for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-"))
        info["checkpoints"] = checkpoints
        safetensors = model_dir / "model.safetensors"
        if safetensors.exists():
            info["model_size_mb"] = round(safetensors.stat().st_size / (1024 * 1024), 1)
    except Exception:
        pass
    return info


def _gather_vectorstore_info(vectorstore_dir: Path, vector_count: int, datasets_dir: Path) -> Dict[str, Any]:
    """Gather vectorstore and dataset metadata for display."""
    info: Dict[str, Any] = {"vector_count": vector_count}
    faiss_file = vectorstore_dir / "faiss_index.bin"
    docs_file = vectorstore_dir / "documents.pkl"
    if faiss_file.exists():
        info["index_size_mb"] = round(faiss_file.stat().st_size / (1024 * 1024), 1)
    if docs_file.exists():
        info["docs_size_mb"] = round(docs_file.stat().st_size / (1024 * 1024), 1)
    if datasets_dir.exists():
        dataset_files = sorted(
            p.name for p in datasets_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".sql", ".md", ".pdf", ".json"}
        )
        info["dataset_files"] = dataset_files
    return info


def _pad(text: str, width: int = 35) -> str:
    """Pad text for box alignment."""
    s = str(text)
    return (s[: width - 2] + "..") if len(s) > width else f"{s:<{width}}"


def matrix_print(text: str, style: str = "info"):
    """Matrix-style console output"""
    colors = {
        "header": "\033[92m",  # Green
        "info": "\033[96m",    # Cyan
        "success": "\033[92m", # Green
        "warning": "\033[93m", # Yellow
        "error": "\033[91m",   # Red
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
    }
    c = colors.get(style, colors["info"])
    reset = colors["reset"]
    try:
        print(f"{c}{text}{reset}")
    except UnicodeEncodeError:
        # Fallback for Windows consoles that don't support UTF-8
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(f"{c}{safe_text}{reset}")


def display_matrix_banner():
    """Display Matrix-style startup banner (large, wide logo)"""
    # VaLLM logo - original then doubled for 2x width
    raw_lines = [
        "    ██╗   ██╗ █████╗ ██╗     ██╗     ███╗   ███╗",
        "    ██║   ██║██╔══██╗██║     ██║     ████╗ ████║",
        "    ██║   ██║███████║██║     ██║     ██╔████╔██║",
        "    ╚██╗ ██╔╝██╔══██║██║     ██║     ██║╚██╔╝██║",
        "     ╚████╔╝ ██║  ██║███████╗███████╗██║ ╚═╝ ██║",
        "      ╚═══╝  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚═╝",
    ]
    logo_lines = ["  " + "".join(c * 2 for c in line) for line in raw_lines]
    banner = """
\033[92m

"""
    banner += "\n".join(logo_lines)
    banner += """

\033[96m
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                                                                              │
    │          V E C T O R - A U G M E N T E D   L O C A L   L A N G U A G E        │
    │                              M O D E L                                        │
    │                                                                              │
    │           Cloud Operations  •  DevOps Intelligence  •  Fine Funed Model           │
    │                                                                              │
    └──────────────────────────────────────────────────────────────────────────────┘
\033[0m"""
    try:
        print(banner)
    except UnicodeEncodeError:
        # Simplified banner for consoles with limited character support
        print("\033[92m    VaLLM - Vector-Augmented Local Language Model\033[0m")


def display_loading_bar(text: str, current: int, total: int):
    """Display a loading progress bar"""
    bar_length = 40
    filled = int(bar_length * current / total)
    try:
        bar = "█" * filled + "░" * (bar_length - filled)
        percent = int(100 * current / total)
        print(f"\033[96m    [{bar}] {percent}% - {text}\033[0m")
    except UnicodeEncodeError:
        bar = "#" * filled + "-" * (bar_length - filled)
        percent = int(100 * current / total)
        print(f"\033[96m    [{bar}] {percent}% - {text}\033[0m")


def display_system_info(
    data_dir: Path,
    model_loaded: bool,
    vector_count: int,
    device: str,
    vectorstore_path: Path | None = None,
    model_path: Path | None = None,
    datasets_dir: Path | None = None,
    model_info: Dict[str, Any] | None = None,
    vs_info: Dict[str, Any] | None = None,
):
    """Display system configuration with datasets, model architecture, and vectorstore details."""
    _datasets = datasets_dir if datasets_dir is not None else data_dir / "datasets"
    csv_count = check_csv_files(_datasets)
    vs_path = vectorstore_path or data_dir / "vectorstore"
    mdl_path = model_path or data_dir / "model"
    model_info = model_info or _gather_model_info(Path(mdl_path))
    vs_info = vs_info or _gather_vectorstore_info(Path(vs_path), vector_count, Path(_datasets))
    vs_str = _pad(str(vs_path)[-38:], 38)
    mdl_str = _pad(str(mdl_path)[-38:], 38)

    try:
        print("\033[92m")
        print("    ┌─────────────────────────────────────────────────────────┐")
        print("    │                  SYSTEM CONFIGURATION                   │")
        print("    ├─────────────────────────────────────────────────────────┤")
        print(f"    │  📂 Data Directory    : {_pad(str(data_dir)[-38:], 38)} │")
        print(f"    │  📦 Vectorstore       : {vs_str} │")
        print(f"    │  🤖 Model Path        : {mdl_str} │")
        print(f"    │  🤖 LLM Model         : {_pad('✅ LOADED' if model_loaded else '❌ NOT LOADED')} │")
        print(f"    │  📊 Vector Count      : {_pad(str(vector_count))} │")
        print(f"    │  💻 Compute Device    : {_pad(device.upper())} │")
        print("    ├─────────────────────────────────────────────────────────┤")
        print("    │              EMBEDDING & VECTOR STORE                    │")
        print("    ├─────────────────────────────────────────────────────────┤")
        print(f"    │  🔧 Embedding Model   : {_pad('all-MiniLM-L6-v2')} │")
        idx_mb = vs_info.get("index_size_mb", "?")
        doc_mb = vs_info.get("docs_size_mb", "?")
        print(f"    │  💾 Index / Docs      : {_pad(f'{idx_mb} MB / {doc_mb} MB')} │")
        ds_files = vs_info.get("dataset_files", [])
        if ds_files:
            print(f"    │  📂 Datasets (used)  : {_pad(str(len(ds_files)) + ' files')} │")
            for name in ds_files[:5]:
                print(f"    │     · {_pad(name, 54)} │")
            if len(ds_files) > 5:
                print(f"    │     ... +{len(ds_files) - 5} more{' ' * 46} │")
        else:
            print(f"    │  📂 Datasets         : {_pad('none (add CSV/txt to data/datasets/)')} │")
        print("    ├─────────────────────────────────────────────────────────┤")
        print("    │                  TRAINED MODEL (LLM)                     │")
        print("    ├─────────────────────────────────────────────────────────┤")
        if model_info.get("available"):
            print(f"    │  🏗️  Architecture      : {_pad(model_info.get('architecture', '?'))} │")
            arch = f"{model_info.get('model_type','?')} | {model_info.get('num_layers','?')}L / {model_info.get('attention_heads','?')}H"
            print(f"    │  📏 Structure         : {_pad(arch)} │")
            print(f"    │  📖 Vocab Size        : {_pad(str(model_info.get('vocab_size', '?')))} │")
            sz = model_info.get("model_size_mb", "?")
            print(f"    │  💾 Model Size        : {_pad(f'{sz} MB')} │")
        else:
            print(f"    │  ⚠️  Status            : {_pad('NOT FOUND (run train; auto-train is default)')} │")
        print("    ├─────────────────────────────────────────────────────────┤")
        print("    │  🔄 Auto-Precompute   : " + _pad("ON" if AUTO_PRECOMPUTE else "OFF") + " │")
        print("    │  🔄 Auto-Train        : " + _pad("ON" if AUTO_TRAIN else "OFF") + " │")
        print("    ├─────────────────────────────────────────────────────────┤")
        print("    │  GET  /health          POST /api/models/v1/query (RAG)  │")
        print("    │  GET  /logs            POST /api/models/v2/query (NLP)  │")
        print("    │  POST /api/models/v3/query (Incident patterns)           │")
        print("    └─────────────────────────────────────────────────────────┘")
        print("\033[0m")

        if vector_count == 0 or not model_loaded:
            print("\033[93m")
            print("    ┌─────────────────────────────────────────────────────────┐")
            print("    │                    ⚠️  SETUP REQUIRED                   │")
            print("    ├─────────────────────────────────────────────────────────┤")
            if vector_count == 0:
                print("    │  Vector store missing or empty. To enable semantic search:  │")
                print("    │  1. Add CSV/text files to app/data/datasets/                 │")
                print("    │  2. Run: python -m app.services.ai.ml.precompute            │")
                print("    │  Or set VALLM_AUTO_PRECOMPUTE=true (default) to auto-build.   │")
            if not model_loaded:
                print("    │  LLM model not found. To enable generation:                   │")
                print("    │  Run: python -m app.services.ai.ml.train                     │")
                print("    │  Auto-train is on by default; add data and restart.          │")
            print("    └─────────────────────────────────────────────────────────┘")
            print("\033[0m")
    except UnicodeEncodeError:
        print("\033[92m")
        print("    --- SYSTEM CONFIGURATION ---")
        print(f"    Data Dir: {data_dir} | Vectors: {vector_count} | Model: {'LOADED' if model_loaded else 'NOT LOADED'}")
        if model_info.get("available"):
            print(f"    Model: {model_info.get('architecture', '?')} | {model_info.get('num_layers')}L")
        if vector_count == 0 or not model_loaded:
            print("    SETUP: Add data to data/datasets/, run precompute and/or train.")
        print("\033[0m")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global vector_store, reasoning_engine, tokenizer, model, faiss_index
    
    display_matrix_banner()
    
    matrix_print("\n    ⚡ INITIALIZING NEURAL NETWORK SYSTEMS...\n", "header")
    
    try:
        # Step 1: Data directory and paths
        display_loading_bar("Scanning data directory", 1, 7)
        data_dir = Path(__file__).parent / "data"
        datasets_dir = data_dir / "datasets"
        vectorstore_dir = data_dir / "vectorstore"
        faiss_index_path = vectorstore_dir / "faiss_index.bin"
        documents_file = vectorstore_dir / "documents.pkl"
        model_dir = data_dir / "models"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
        if not datasets_dir.exists():
            datasets_dir.mkdir(parents=True, exist_ok=True)
            matrix_print("    ⚠️  Datasets directory created. Add CSV/dataset files to app/data/datasets/", "warning")
        csv_count = check_csv_files(datasets_dir)

        # Step 2: Auto-build vectorstore and/or model if missing or empty
        display_loading_bar("Checking vectorstore and model", 2, 7)
        vectorstore_missing = not faiss_index_path.exists() or not documents_file.exists()
        model_missing = not (model_dir / "config.json").exists()
        if vectorstore_missing and AUTO_PRECOMPUTE and csv_count > 0:
            matrix_print(f"    🔄 Vectorstore missing or empty — running precompute ({csv_count} dataset files)...", "warning")
            if run_precompute():
                matrix_print("    ✓ Precompute completed", "success")
            else:
                matrix_print("    ⚠️  Precompute failed; continuing without index", "warning")
        if model_missing and AUTO_TRAIN and csv_count > 0:
            matrix_print(f"    🔄 Model missing — running training ({csv_count} dataset files)...", "warning")
            if run_train():
                matrix_print("    ✓ Training completed", "success")
            else:
                matrix_print("    ⚠️  Training failed; continuing without model", "warning")
        if (vectorstore_missing or model_missing) and csv_count == 0:
            matrix_print("    ⚠️  No dataset files in app/data/datasets/ — add CSV/text to enable auto-build", "warning")

        # Step 3: Vector store
        display_loading_bar("Loading vector store", 3, 7)
        vector_store = VectorStore(data_dir=str(data_dir))
        await vector_store.initialize()
        matrix_print("    ✓ Vector store initialized", "success")

        # Step 4: Reasoning engine
        display_loading_bar("Initializing reasoning engine", 4, 7)
        reasoning_engine = ReasoningEngine(vector_store)
        await reasoning_engine.initialize()
        matrix_print("    ✓ Reasoning engine online", "success")

        # Step 5: LLM model
        display_loading_bar("Loading LLM model", 5, 7)
        tokenizer = None
        model = None
        if model_dir.exists() and (model_dir / "config.json").exists():
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                model = AutoModelForCausalLM.from_pretrained(str(model_dir)).to(device)
                matrix_print(f"    ✓ LLM model loaded → {device.upper()}", "success")
            except Exception as e:
                matrix_print(f"    ⚠️  LLM model error: {e}", "warning")
        else:
            matrix_print("    ⚠️  No LLM model found.", "warning")
            matrix_print("    📝 Run: python -m app.services.ai.ml.train  (auto-train is default)", "info")

        # Step 6: FAISS index and vector count
        display_loading_bar("Loading FAISS vector index", 6, 7)
        vector_count = 0
        faiss_index = None
        if faiss_index_path.exists():
            try:
                faiss_index = faiss.read_index(str(faiss_index_path))
                vector_count = faiss_index.ntotal
                matrix_print(f"    ✓ FAISS index loaded → {vector_count} vectors", "success")
            except Exception as e:
                matrix_print(f"    ⚠️  FAISS error: {e}", "warning")
        else:
            matrix_print("    ⚠️  No FAISS index; run precompute or set VALLM_AUTO_PRECOMPUTE=true", "warning")

        # Step 7: App state and system info
        display_loading_bar("Completing initialization", 7, 7)
        app.state.vector_store = vector_store
        app.state.reasoning_engine = reasoning_engine
        app.state.tokenizer = tokenizer
        app.state.model = model
        app.state.faiss_index = faiss_index

        model_info = _gather_model_info(model_dir)
        vs_info = _gather_vectorstore_info(vectorstore_dir, vector_count, datasets_dir)
        display_system_info(
            data_dir=data_dir,
            model_loaded=(model is not None),
            vector_count=vector_count,
            device=device,
            vectorstore_path=vectorstore_dir,
            model_path=model_dir,
            datasets_dir=datasets_dir,
            model_info=model_info,
            vs_info=vs_info,
        )
        
        matrix_print("    ⚡ VALLM NEURAL NETWORK ONLINE ⚡\n", "header")
        
    except Exception as e:
        matrix_print(f"\n    ❌ CRITICAL ERROR: {e}", "error")
        import traceback
        traceback.print_exc()
        raise
    
    yield
    
    # Cleanup
    matrix_print("\n    🔌 SHUTTING DOWN NEURAL SYSTEMS...", "warning")
    if vector_store:
        await vector_store.cleanup()
    if reasoning_engine:
        await reasoning_engine.cleanup()
    
    matrix_print("    ✓ Shutdown complete\n", "success")


# Create FastAPI app
app = FastAPI(
    title="VaLLM - Vector-based Local LLM",
    description="Private cloud operations AI with embeddings and chain-of-thoughts reasoning",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Optional: Legacy rate limiting middleware (if enabled)
try:
    try:
        from .auth.rate_limit import RateLimitMiddleware, RATE_LIMIT_ENABLED
    except ImportError:
        from auth.rate_limit import RateLimitMiddleware, RATE_LIMIT_ENABLED
    if RATE_LIMIT_ENABLED:
        app.add_middleware(RateLimitMiddleware)
        logger.info("✅ Legacy rate limiting enabled")
except ImportError:
    pass

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Setup JSON logging (if enabled)
try:
    try:
        from .core.logging_config import setup_json_logging
    except ImportError:
        from core.logging_config import setup_json_logging
    setup_json_logging()
except ImportError:
    pass

# Log startup message
logger.info(f"📁 Logs will be saved to: {LOG_FILE}")

# Include routes
app.include_router(router, prefix="/api/models/v1")
app.include_router(router_v2, prefix="/api/models/v2")
app.include_router(router_v3, prefix="/api/models/v3")
app.include_router(cloud_router)  # POST /api/cloud/provision-intent (intent + Golang payload)

# Include monitoring routes
try:
    try:
        from .services.monitoring.monitoring_services_router import router as monitoring_router
    except ImportError as e:
        if "attempted relative import with no known parent package" in str(e):
            from services.monitoring.monitoring_services_router import router as monitoring_router
        else:
            raise

    app.include_router(monitoring_router)
    logger.info("✅ Monitoring routes registered")
except ImportError as e:
    logger.warning(f"Monitoring routes not available: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Display a beautiful status page for the VaLLM service."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VaLLM Status</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background: linear-gradient(135deg, #2b5876, #4e4376);
            color: white;
            text-align: center;
        }
        .container {
            padding: 40px;
            border-radius: 15px;
            background: rgba(0, 0, 0, 0.2);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        h1 {
            font-size: 4rem;
            margin-bottom: 10px;
            font-weight: 600;
        }
        p {
            font-size: 1.5rem;
            margin: 5px 0;
        }
        .status {
            display: inline-block;
            padding: 10px 25px;
            border-radius: 25px;
            background-color: #27ae60;
            font-size: 1.5rem;
            font-weight: bold;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>VaLLM</h1>
        <p>Vector-Augmented Local Language Model</p>
        <div class="status">Online</div>
    </div>
</body>
</html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/health")
async def health():
    """Basic health check (for load balancers) - unchanged"""
    return {"status": "healthy"}


@app.get("/logs")
async def get_logs(lines: int = 50):
    """
    View recent logs
    
    Args:
        lines: Number of recent lines to return (default 50)
    """
    try:
        if not LOG_FILE.exists():
            return {"logs": [], "message": "No logs yet"}
        
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "log_file": str(LOG_FILE),
            "total_lines": len(all_lines),
            "showing": len(recent_lines),
            "logs": [line.rstrip() for line in recent_lines]
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/logs/stats")
async def get_log_stats():
    """Get logging statistics"""
    try:
        if not LOG_FILE.exists():
            return {"message": "No logs yet"}
        
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count requests and responses
        request_count = content.count('"type": "REQUEST"')
        response_count = content.count('"type": "RESPONSE"')
        error_count = content.count('ERROR')
        
        # Get file size
        file_size = LOG_FILE.stat().st_size
        
        return {
            "log_file": str(LOG_FILE),
            "file_size_kb": round(file_size / 1024, 2),
            "total_requests": request_count,
            "total_responses": response_count,
            "errors": error_count,
            "created": datetime.fromtimestamp(LOG_FILE.stat().st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(LOG_FILE.stat().st_mtime).isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


@app.delete("/logs/clear")
async def clear_logs():
    """Clear all logs"""
    try:
        if LOG_FILE.exists():
            # Backup before clearing
            backup_file = LOG_DIR / f"logs_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            LOG_FILE.rename(backup_file)
            
            # Create new empty log file
            LOG_FILE.touch()
            
            logger.info("🗑️ Logs cleared (backup created)")
            
            return {
                "message": "Logs cleared",
                "backup": str(backup_file)
            }
        return {"message": "No logs to clear"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/stats")
async def get_stats(request: Request):
    """
    Vector store and cache statistics.
    Includes total_vectors, by_type, and cache (embedding L1, search L2) stats when enabled.
    """
    try:
        vs = getattr(request.app.state, "vector_store", None)
        if vs is None:
            return {"error": "Vector store not initialized"}
        return await vs.get_vector_store_stats()
    except Exception as e:
        return {"error": str(e)}


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9


class GenerateResponse(BaseModel):
    response: str
    text: str  # Alias for LangChain compatibility
    model_loaded: bool
    device: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    text: str
    score: float
    metadata: dict


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text using the trained LLM model"""
    global tokenizer, model
    
    if tokenizer is None or model is None:
        fallback_msg = "Model not loaded. Please run 'python ./app/train.py' first to train and export the model."
        logger.warning("LLM input/output skipped (model not loaded)")
        return GenerateResponse(
            response=fallback_msg,
            text=fallback_msg,
            model_loaded=False,
            device="none"
        )
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"LLM input: {request.prompt}")
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"LLM output: {generated_text}")
        
        return GenerateResponse(
            response=generated_text,
            text=generated_text,  # For LangChain compatibility
            model_loaded=True,
            device=device
        )
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        return GenerateResponse(
            response=error_msg,
            text=error_msg,
            model_loaded=True,
            device="error"
        )


@app.post("/search")
async def search(request: SearchRequest):
    """
    Search the vector store for relevant documents.
    This endpoint is compatible with LangChain retrievers.
    
    Returns:
        {"results": [{"text": "...", "score": 0.12, "metadata": {...}}, ...]}
    """
    global vector_store
    
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        search_results = await vector_store.search(
            query=request.query,
            top_k=request.top_k
        )
        
        # Format for LangChain compatibility
        results = []
        for r in search_results:
            results.append({
                "text": r.get("document", ""),
                "score": r.get("score", 0.0),
                "metadata": r.get("metadata", {})
            })
        
        return {"results": results}
    
    except Exception as e:
        import traceback
        logger.error(f"Search error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Search error: {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import sys
    import os
    
    # Ensure root directory is in sys.path for uvicorn imports
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        
    # Handle Windows encoding issues for special characters
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except (AttributeError, Exception):
            pass

    uvicorn.run(
        app,  # Pass the app object directly to avoid double import issues
        host="0.0.0.0",
        port=8746,
        reload=False
    )

