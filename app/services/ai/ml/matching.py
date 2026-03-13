"""
VaLLM Specialist Model - Document Matching Service.

Author: Joel Otepa Wembo
https://joelwembo.com

Document matching and similarity service for finding duplicate documents,
cross-referencing, and content-based document retrieval.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
from datetime import datetime
import asyncio

from core.logging import performance_logger

# Conditional imports to avoid breaking if optional dependencies are missing
try:
    from .search import search_service
except (ImportError, Exception) as e:
    import logging
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(f"Search service not available: {e}")
    search_service = None

try:
    from . import embedding_service
except (ImportError, Exception) as e:
    import logging
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(f"Embedding service not available: {e}")
    embedding_service = None

logger = logging.getLogger(__name__)


class MatchingService:
    """Document matching and similarity service"""

    def __init__(self):
        if search_service is None:
            logger.warning("Search service not available - matching functionality may be limited")
        if embedding_service is None:
            logger.warning("Embedding service not available - matching functionality may be limited")

        self.search_service = search_service
        self.embedding_service = embedding_service

        # Matching configuration for document similarity
        self.matching_weights = {
            "content": 0.40,
            "metadata": 0.20,
            "structure": 0.15,
            "entities": 0.15,
            "category": 0.10,
        }

        # Performance tracking
        self.matching_count = 0
        self.total_matching_time = 0.0
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize matching service"""
        try:
            self.is_initialized = True
            logger.info("Document Matching Service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Matching Service: {e}")
            return False

    async def match_documents(
        self,
        source_doc_id: str,
        target_doc_ids: List[str],
        criteria: Optional[Dict[str, Any]] = None,
        *,
        source_data: Optional[Dict[str, Any]] = None,
        targets_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Match a source document against target documents for similarity."""
        try:
            start_time = datetime.utcnow()
            self.matching_count += 1

            source = source_data if source_data is not None else await self._get_document_data(source_doc_id)
            targets = (
                targets_data
                if targets_data is not None
                else await self._get_documents_data(target_doc_ids)
            )

            matches = []
            for target in targets:
                match_score = await self._calculate_document_match(source, target)
                match_result = {
                    "document_id": target.get("id", ""),
                    "source_id": source_doc_id,
                    "match_score": match_score["overall_score"],
                    "detailed_scores": match_score,
                    "confidence": match_score.get("confidence", 0.8),
                }
                matches.append(match_result)

            matches.sort(key=lambda x: x["match_score"], reverse=True)

            matching_time = (datetime.utcnow() - start_time).total_seconds()
            self.total_matching_time += matching_time

            return matches

        except Exception as e:
            logger.error(f"Document matching failed: {e}")
            raise

    async def find_duplicates(
        self,
        documents: List[Dict[str, Any]],
        threshold: float = 0.85,
    ) -> List[Dict[str, Any]]:
        """Find potential duplicate documents based on content similarity."""
        duplicates = []
        for i, doc_a in enumerate(documents):
            for j, doc_b in enumerate(documents):
                if j <= i:
                    continue
                score = await self._calculate_document_match(doc_a, doc_b)
                if score["overall_score"] >= threshold:
                    duplicates.append({
                        "document_a": doc_a.get("id", str(i)),
                        "document_b": doc_b.get("id", str(j)),
                        "similarity_score": score["overall_score"],
                        "detailed_scores": score,
                    })
        return duplicates

    async def cross_reference(
        self,
        document_data: Dict[str, Any],
        reference_docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Cross-reference a document against a set of reference documents."""
        results = []
        for ref in reference_docs:
            score = await self._calculate_document_match(document_data, ref)
            results.append({
                "reference_id": ref.get("id", ""),
                "reference_title": ref.get("title", ""),
                "match_score": score["overall_score"],
                "matching_fields": score,
            })
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results

    async def _calculate_document_match(
        self, doc_a: Dict[str, Any], doc_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate detailed match scores between two documents."""
        scores = {}

        # Content similarity
        content_a = doc_a.get("content_text", "") or doc_a.get("content", "") or ""
        content_b = doc_b.get("content_text", "") or doc_b.get("content", "") or ""
        scores["content_score"] = self._text_similarity(content_a, content_b)

        # Metadata similarity (document_type, language, tags)
        scores["metadata_score"] = self._metadata_similarity(doc_a, doc_b)

        # Structure similarity (page_count, file_size)
        scores["structure_score"] = self._structure_similarity(doc_a, doc_b)

        # Entity overlap
        entities_a = doc_a.get("entities", []) or []
        entities_b = doc_b.get("entities", []) or []
        scores["entities_score"] = self._list_overlap(entities_a, entities_b)

        # Category match
        cat_a = (doc_a.get("document_type", "") or doc_a.get("category", "")).lower()
        cat_b = (doc_b.get("document_type", "") or doc_b.get("category", "")).lower()
        scores["category_score"] = 1.0 if cat_a and cat_a == cat_b else 0.3

        # Overall score
        overall_score = sum(
            scores[f"{k}_score"] * self.matching_weights[k]
            for k in self.matching_weights
        )
        scores["overall_score"] = overall_score
        scores["confidence"] = 0.8

        return scores

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Simple word-overlap similarity between two texts."""
        if not text_a or not text_b:
            return 0.0
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0

    def _metadata_similarity(self, doc_a: Dict, doc_b: Dict) -> float:
        """Compare metadata fields."""
        score = 0.0
        total = 0
        for field in ["document_type", "language", "mime_type"]:
            val_a = (doc_a.get(field, "") or "").lower()
            val_b = (doc_b.get(field, "") or "").lower()
            if val_a and val_b:
                total += 1
                if val_a == val_b:
                    score += 1.0
        # Tags overlap
        tags_a = doc_a.get("tags", []) or []
        tags_b = doc_b.get("tags", []) or []
        if tags_a and tags_b:
            total += 1
            score += self._list_overlap(tags_a, tags_b)
        return score / total if total > 0 else 0.5

    def _structure_similarity(self, doc_a: Dict, doc_b: Dict) -> float:
        """Compare structural features."""
        page_a = doc_a.get("page_count", 0) or 0
        page_b = doc_b.get("page_count", 0) or 0
        if page_a and page_b:
            return min(page_a, page_b) / max(page_a, page_b)
        return 0.5

    def _list_overlap(self, list_a: List, list_b: List) -> float:
        """Calculate overlap between two lists."""
        if not list_a or not list_b:
            return 0.0
        set_a = set(str(x).lower() for x in list_a)
        set_b = set(str(x).lower() for x in list_b)
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0

    async def _get_document_data(self, doc_id: str) -> Dict[str, Any]:
        """Get document data (mock). In production, pass data from ORM."""
        return {
            "id": doc_id,
            "title": "Sample Document",
            "document_type": "invoice",
            "content_text": "Sample document content for matching",
            "language": "en",
            "page_count": 2,
            "tags": ["finance", "invoice"],
            "entities": ["Company A", "2024"],
        }

    async def _get_documents_data(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple documents (mock). In production, pass data from ORM."""
        return [await self._get_document_data(did) for did in doc_ids]

    def get_matching_stats(self) -> Dict[str, Any]:
        """Get matching service statistics"""
        avg_matching_time = (
            (self.total_matching_time / self.matching_count)
            if self.matching_count > 0
            else 0
        )
        return {
            "total_matches": self.matching_count,
            "total_matching_time": self.total_matching_time,
            "average_matching_time": avg_matching_time,
            "matching_weights": self.matching_weights,
        }


# Global matching service instance
matching_service = MatchingService()
