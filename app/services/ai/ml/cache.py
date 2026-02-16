"""
Multi-level caching for VaLLM.

- Embedding cache (L1): cache generated query embeddings. TTL 1h.
- Search result cache (L2): cache vector search results. TTL 30m.

Config via env:
  VALLM_CACHE_EMBEDDINGS=true|false   (default: true)
  VALLM_CACHE_SEARCH=true|false       (default: true)
  VALLM_CACHE_EMBEDDINGS_MAXSIZE      (default: 2000)
  VALLM_CACHE_SEARCH_MAXSIZE          (default: 1000)
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
CACHE_EMBEDDINGS = os.getenv("VALLM_CACHE_EMBEDDINGS", "true").lower() == "true"
CACHE_SEARCH = os.getenv("VALLM_CACHE_SEARCH", "true").lower() == "true"
EMBEDDINGS_TTL = 60 * 60          # 1 hour
SEARCH_TTL = 30 * 60              # 30 minutes
EMBEDDINGS_MAXSIZE = int(os.getenv("VALLM_CACHE_EMBEDDINGS_MAXSIZE", "2000"))
SEARCH_MAXSIZE = int(os.getenv("VALLM_CACHE_SEARCH_MAXSIZE", "1000"))


def _normalize_query(q: str) -> str:
    return " ".join(q.strip().split()) if q else ""


def _search_key(query: str, top_k: int, filter_type: Optional[str]) -> str:
    nq = _normalize_query(query)
    ft = filter_type or ""
    return f"{nq}|{top_k}|{ft}"


class TTLCache:
    """Thread-safe in-memory TTL cache with max size and LRU-like eviction."""

    def __init__(self, maxsize: int, ttl_seconds: float, name: str = "cache"):
        self._maxsize = max(1, maxsize)
        self._ttl = ttl_seconds
        self._name = name
        self._data: Dict[str, Tuple[Any, float]] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            val, expires = entry
            if time.monotonic() > expires:
                del self._data[key]
                self._order = [k for k in self._order if k != key]
                self.misses += 1
                return None
            self.hits += 1
            return val

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._evict_expired()
            expires = time.monotonic() + self._ttl
            if key in self._data:
                self._data[key] = (value, expires)
                self._order = [k for k in self._order if k != key]
                self._order.append(key)
                return
            while len(self._data) >= self._maxsize and self._order:
                evict = self._order.pop(0)
                self._data.pop(evict, None)
            self._data[key] = (value, expires)
            self._order.append(key)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, e) in self._data.items() if now > e]
        for k in expired:
            self._data.pop(k, None)
            self._order = [x for x in self._order if x != k]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            self._evict_expired()
            total = self.hits + self.misses
            return {
                "name": self._name,
                "size": len(self._data),
                "maxsize": self._maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0,
            }


# -----------------------------------------------------------------------------
# Global caches (lazy-init to avoid import-time side effects)
# -----------------------------------------------------------------------------
_embedding_cache: Optional[TTLCache] = None
_search_cache: Optional[TTLCache] = None


def get_embedding_cache() -> Optional[TTLCache]:
    global _embedding_cache
    if not CACHE_EMBEDDINGS:
        return None
    if _embedding_cache is None:
        _embedding_cache = TTLCache(
            maxsize=EMBEDDINGS_MAXSIZE,
            ttl_seconds=EMBEDDINGS_TTL,
            name="embedding",
        )
    return _embedding_cache


def get_search_cache() -> Optional[TTLCache]:
    global _search_cache
    if not CACHE_SEARCH:
        return None
    if _search_cache is None:
        _search_cache = TTLCache(
            maxsize=SEARCH_MAXSIZE,
            ttl_seconds=SEARCH_TTL,
            name="search",
        )
    return _search_cache


def get_all_cache_stats() -> List[Dict[str, Any]]:
    out = []
    for maybe in (get_embedding_cache(), get_search_cache()):
        if maybe:
            out.append(maybe.stats())
    return out
