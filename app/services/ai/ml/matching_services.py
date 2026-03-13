"""
VaLLM Specialist Model - Document Matching Orchestration.

Author: Joel Otepa Wembo
https://joelwembo.com

High-level document matching functions that load documents from the database
and run the MatchingService for similarity, deduplication, and cross-referencing.
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# In-memory cache for match results (key: (org_id, doc_id) -> {expires_at, data})
MATCH_CACHE_TTL_SECONDS = 86400  # 24 hours
_match_cache: dict[tuple[str, str], dict[str, Any]] = {}


def _cache_key(org_id: str, doc_id: str) -> tuple[str, str]:
    return (org_id, doc_id)


def get_cached_matches(doc_id: str, org_id: str = "default") -> Optional[dict]:
    """Return cached match result if present and not expired."""
    key = _cache_key(org_id, doc_id)
    entry = _match_cache.get(key)
    if not entry:
        return None
    if time.time() > entry["expires_at"]:
        del _match_cache[key]
        return None
    return entry["data"]


def _set_cached_matches(doc_id: str, org_id: str, result: dict) -> None:
    """Store match result in cache."""
    key = _cache_key(org_id, doc_id)
    _match_cache[key] = {
        "expires_at": time.time() + MATCH_CACHE_TTL_SECONDS,
        "data": result,
    }


async def match_documents_batch(
    source_doc_id: str,
    target_doc_ids: List[str],
    org_id: str = "default",
    source_data: Optional[Dict[str, Any]] = None,
    targets_data: Optional[List[Dict[str, Any]]] = None,
) -> dict:
    """Match a source document against targets using the ML MatchingService.

    Pass source_data and targets_data directly from ORM/schemas for production use.
    """
    from app.services.ai.ml.matching import matching_service

    raw_matches = await matching_service.match_documents(
        source_doc_id=source_doc_id,
        target_doc_ids=target_doc_ids,
        source_data=source_data,
        targets_data=targets_data,
    )

    formatted = []
    for m in raw_matches:
        score_pct = round(m["match_score"] * 100)
        if score_pct >= 85:
            status = "high_match"
        elif score_pct >= 70:
            status = "good_match"
        elif score_pct >= 55:
            status = "partial_match"
        else:
            status = "low_match"

        formatted.append({
            "document_id": m["document_id"],
            "match_score": score_pct,
            "status": status,
            "detailed_scores": m.get("detailed_scores", {}),
            "confidence": m.get("confidence", 0.8),
        })

    total = len(formatted)
    high = sum(1 for m in formatted if m["status"] == "high_match")
    good = sum(1 for m in formatted if m["status"] == "good_match")
    avg_score = round(sum(m["match_score"] for m in formatted) / total) if total else 0

    result = {
        "success": True,
        "source_document_id": source_doc_id,
        "matches": formatted,
        "stats": {
            "total": total,
            "high_match": high,
            "good_match": good,
            "avg_score": avg_score,
        },
    }
    _set_cached_matches(source_doc_id, org_id, result)
    return result


async def find_document_duplicates(
    documents: List[Dict[str, Any]],
    threshold: float = 0.85,
) -> dict:
    """Find potential duplicates among a list of documents."""
    from app.services.ai.ml.matching import matching_service

    duplicates = await matching_service.find_duplicates(documents, threshold)

    return {
        "success": True,
        "duplicates_found": len(duplicates),
        "duplicates": duplicates,
        "threshold": threshold,
    }
