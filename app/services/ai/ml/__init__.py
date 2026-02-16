"""
VaLLM ML Services - Machine Learning and AI components
"""

__version__ = "1.0.0"

# Core ML components
from .embeddings import VectorStore, embedding_service
from .reasoning import ReasoningEngine
from .cache import TTLCache, get_embedding_cache, get_search_cache, get_all_cache_stats

# Routes
from .routes import router, router_v2, router_v3

__all__ = [
    # Core
    "VectorStore",
    "embedding_service",
    "ReasoningEngine",
    # Cache
    "TTLCache",
    "get_embedding_cache",
    "get_search_cache",
    "get_all_cache_stats",
    # Routes
    "router",
    "router_v2",
    "router_v3",
]
