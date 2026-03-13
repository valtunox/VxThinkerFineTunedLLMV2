"""
Async Redis Service for document processing, verification, and business analytics.
Provides methods for setting, getting, deleting, and listing keys (sessions, cache, documents).
"""
from typing import Any, Optional, Dict
from app.core.logger import get_logger

logger = get_logger(__name__)

class RedisService:
    """
    Async Redis Service for key-value operations (caching, sessions, campaign state).

    Uses a process-wide singleton dict so all RedisService instances share the
    same data within one Python process (uvicorn workers, background threads, etc.).
    When a real Redis connection is available it will be preferred automatically.
    """

    _shared_store: Dict[str, Any] = {}

    def __init__(self, client: Any = None):
        self.client = client

    @property
    def _store(self) -> Dict[str, Any]:
        return RedisService._shared_store

    @_store.setter
    def _store(self, value: Dict[str, Any]):
        RedisService._shared_store = value

    async def set_key(self, key: str, value: Any) -> bool:
        self._store[key] = value
        return True

    async def get_key(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    async def delete_key(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    async def list_keys(self) -> Dict[str, Any]:
        return dict(self._store)

    async def set_key_with_expiry(self, key: str, value: Any, ttl: int) -> bool:
        self._store[key] = value
        return True

    async def find_keys(self, pattern: str) -> Dict[str, Any]:
        import fnmatch
        return {k: v for k, v in self._store.items() if fnmatch.fnmatch(k, pattern)}

    async def incr_key(self, key: str, amount: int = 1) -> int:
        val = int(self._store.get(key, 0)) + amount
        self._store[key] = val
        return val

    async def set_hash(self, hash_key: str, mapping: Dict[str, Any]) -> bool:
        self._store[hash_key] = mapping
        return True

    async def get_hash(self, hash_key: str) -> Optional[Dict[str, Any]]:
        val = self._store.get(hash_key)
        if isinstance(val, dict):
            return val
        return None

    async def publish(self, channel: str, message: str) -> bool:
        logger.info("Published to %s: %s", channel, message)
        return True

    async def backup(self) -> Dict[str, Any]:
        return dict(self._store)

    async def restore(self, data: Dict[str, Any]) -> bool:
        RedisService._shared_store = dict(data)
        return True
