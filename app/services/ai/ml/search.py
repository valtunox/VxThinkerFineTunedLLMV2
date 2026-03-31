"""
VaLLM Specialist Model - Search Service.

Author: Joel Otepa Wembo
https://joelwembo.com

Hybrid BM25 + dense vector search with cross-encoder reranking for document and entity search.

Retrieval pipeline:
  Stage 1 - Recall:   BM25 (sparse) + BGE bi-encoder (dense) → candidate set
  Stage 2 - Rerank:   cross-encoder/ms-marco-MiniLM-L6-v2 → precision ordering
"""

from typing import Dict, Any, List, Optional, Union
import logging
import asyncio
import numpy as np
from datetime import datetime
import json

# Conditional imports for optional dependencies
try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None
    RANK_BM25_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None
    SKLEARN_AVAILABLE = False

try:
    from app.core.settings import get_settings
    from app.core.logging import performance_logger, ai_logger
except ImportError:
    from core.settings import get_settings
    from core.logging import performance_logger, ai_logger

# Conditional import for embedding service
try:
    from .embedding import embedding_service
except ImportError:
    embedding_service = None

settings = get_settings()
logger = logging.getLogger(__name__)


class SearchService:
    """Hybrid search service combining BM25 and dense vector search"""
    
    def __init__(self):
        if not RANK_BM25_AVAILABLE:
            logger.warning("rank_bm25 is not installed. BM25 search will not be available.")
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn is not installed. TF-IDF search will not be available.")
        if embedding_service is None:
            logger.warning("Embedding service is not available. Vector search will not be available.")
        
        self.embedding_service = embedding_service
        self.bm25_index = None
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        else:
            self.tfidf_vectorizer = None
        self.documents = []
        self.document_embeddings = None
        
        # Search configuration
        self.hybrid_weights = {
            "bm25_weight": 0.3,
            "vector_weight": 0.7
        }
        
        # Performance tracking
        self.search_count = 0
        self.total_search_time = 0.0
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize search service. Embedding service init is deferred to first use."""
        try:
            if self.embedding_service is None:
                logger.warning(
                    "Search Service starting without embedding service - vector search will be disabled "
                    "(BM25/TF-IDF may still work)."
                )
            else:
                logger.info("Search Service: embedding service will auto-initialize on first use")
            
            self.is_initialized = True
            logger.info("Search Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Search Service: {e}")
            return False
    
    async def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for search"""
        try:
            logger.info(f"Indexing {len(documents)} documents...")
            
            self.documents = documents
            
            # Extract text content for indexing
            texts = []
            for doc in documents:
                # Combine relevant text fields - All lowercase
                text_parts = []
                
                # Title fields - all lowercase
                title = doc.get("title") or doc.get("shiftname") or doc.get("name")
                if title:
                    text_parts.append(str(title))
                
                # Name fields (for candidates, users, etc.) - ORM uses first_name/last_name
                first_name = doc.get("first_name") or doc.get("firstname")
                last_name = doc.get("last_name") or doc.get("lastname")
                if first_name and last_name:
                    text_parts.append(f"{first_name} {last_name}")
                
                # Content fields - all lowercase
                content = doc.get("content") or doc.get("notes")
                if content:
                    text_parts.append(str(content))
                
                # Description fields - all lowercase
                description = doc.get("description")
                if description:
                    text_parts.append(str(description))
                
                # Skills fields - all lowercase
                skills = doc.get("skills") or doc.get("technical_skills")
                if skills:
                    if isinstance(skills, list):
                        text_parts.extend([str(s) for s in skills])
                    else:
                        text_parts.append(str(skills))
                
                # Email fields - all lowercase
                email = doc.get("email")
                if email:
                    text_parts.append(str(email))
                
                combined_text = " ".join(text_parts)
                texts.append(combined_text)
            
            # Build BM25 index
            if RANK_BM25_AVAILABLE:
                tokenized_texts = [text.split() for text in texts]
                self.bm25_index = BM25Okapi(tokenized_texts)
            else:
                logger.warning("BM25 indexing skipped - rank_bm25 not available")
            
            # Generate document embeddings
            if self.embedding_service:
                self.document_embeddings = await self.embedding_service.generate_embeddings(texts)
            
            # Fit TF-IDF vectorizer
            if SKLEARN_AVAILABLE and self.tfidf_vectorizer:
                self.tfidf_vectorizer.fit(texts)
            
            logger.info(f"Successfully indexed {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            raise
    
    async def search(self, query: str, top_k: int = 10, 
                    search_type: str = "hybrid",
                    filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search documents using hybrid approach"""
        try:
            start_time = datetime.utcnow()
            self.search_count += 1
            
            if not self.documents or len(self.documents) == 0:
                return []
            
            if search_type == "bm25":
                results = await self._bm25_search(query, top_k, filters)
            elif search_type == "vector":
                results = await self._vector_search(query, top_k, filters)
            else:  # hybrid
                results = await self._hybrid_search(query, top_k, filters)
            
            # Update metrics
            search_time = (datetime.utcnow() - start_time).total_seconds()
            self.total_search_time += search_time
            
            # Log performance
            ai_logger.log_search_query(
                query=query,
                results_count=len(results),
                search_time_ms=search_time * 1000,
                search_type=search_type
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def _bm25_search(self, query: str, top_k: int, 
                          filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """BM25 keyword search"""
        try:
            if not RANK_BM25_AVAILABLE or self.bm25_index is None:
                logger.warning("BM25 search not available - rank_bm25 not installed")
                return []
            
            # Tokenize query
            query_tokens = query.split()
            
            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # Create results with scores
            results = []
            for i, score in enumerate(bm25_scores):
                if score > 0:  # Only include documents with non-zero scores
                    result = self.documents[i].copy()
                    result["search_score"] = float(score)
                    result["search_type"] = "bm25"
                    results.append(result)
            
            # Sort by score
            results.sort(key=lambda x: x["search_score"], reverse=True)
            
            # Apply filters
            if filters:
                results = self._apply_filters(results, filters)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    async def _vector_search(self, query: str, top_k: int,
                           filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Dense vector similarity search"""
        try:
            if not SKLEARN_AVAILABLE or cosine_similarity is None:
                logger.warning("Vector search not available - scikit-learn not installed")
                return []
            
            if not self.embedding_service:
                logger.warning("Vector search not available - embedding service not available")
                return []
            
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embeddings([query])
            
            if self.document_embeddings is None:
                return []
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
            
            # Create results with scores
            results = []
            for i, similarity in enumerate(similarities):
                result = self.documents[i].copy()
                result["search_score"] = float(similarity)
                result["search_type"] = "vector"
                results.append(result)
            
            # Sort by similarity
            results.sort(key=lambda x: x["search_score"], reverse=True)
            
            # Apply filters
            if filters:
                results = self._apply_filters(results, filters)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _hybrid_search(self, query: str, top_k: int,
                           filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Hybrid BM25 + vector search with cross-encoder reranking.

        Stage 1 (Recall): Fetch top_k*3 candidates from BM25 and dense vector search.
        Stage 2 (Rerank): Score the merged candidate set with cross-encoder for precision.
        """
        try:
            # Stage 1: Recall - get a broad set of candidates
            bm25_results = await self._bm25_search(query, top_k * 3, filters)
            vector_results = await self._vector_search(query, top_k * 3, filters)

            # Normalize scores (use floor so worst candidate is not 0% in summary)
            bm25_results = self._normalize_scores(bm25_results, "search_score")
            vector_results = self._normalize_scores(vector_results, "search_score")

            # Create score maps
            bm25_scores = {result["id"]: result["search_score"] for result in bm25_results}
            vector_scores = {result["id"]: result["search_score"] for result in vector_results}

            # Combine scores
            all_ids = set(bm25_scores.keys()) | set(vector_scores.keys())

            hybrid_results = []
            for doc_id in all_ids:
                # Find original document
                doc = next((d for d in self.documents if d.get("id") == doc_id), None)
                if not doc:
                    continue

                result = doc.copy()
                bm25_score = bm25_scores.get(doc_id, 0)
                vector_score = vector_scores.get(doc_id, 0)

                # Calculate hybrid score
                hybrid_score = (
                    bm25_score * self.hybrid_weights["bm25_weight"] +
                    vector_score * self.hybrid_weights["vector_weight"]
                )

                result["search_score"] = float(hybrid_score)
                result["bm25_score"] = float(bm25_score)
                result["vector_score"] = float(vector_score)
                result["search_type"] = "hybrid"

                hybrid_results.append(result)

            # Sort by hybrid score
            hybrid_results.sort(key=lambda x: x["search_score"], reverse=True)
            # Normalize with floor so worst candidate is not 0% in match summary
            hybrid_results = self._normalize_scores(
                hybrid_results, "search_score", min_floor=0.15
            )

            # Stage 2: Cross-encoder reranking for precision
            reranker_available = (
                self.embedding_service is not None
                and getattr(self.embedding_service, "reranker", None) is not None
            )
            if reranker_available and len(hybrid_results) > 1:
                # Build text representation for each candidate
                candidate_texts = []
                for r in hybrid_results:
                    parts = []
                    title = r.get("title") or r.get("name") or ""
                    if title:
                        parts.append(str(title))
                    first_name = r.get("first_name") or r.get("firstname")
                    last_name = r.get("last_name") or r.get("lastname")
                    if first_name and last_name:
                        parts.append(f"{first_name} {last_name}")
                    desc = r.get("description") or r.get("content") or ""
                    if desc:
                        parts.append(str(desc))
                    skills = r.get("skills") or r.get("technical_skills") or ""
                    if skills:
                        parts.append(str(skills) if isinstance(skills, str) else " ".join(skills))
                    candidate_texts.append(" ".join(parts) if parts else "unknown")

                reranked = await self.embedding_service.rerank(
                    query, candidate_texts, top_k=top_k
                )

                # Map reranked results back
                final_results = []
                for rr in reranked:
                    result = hybrid_results[rr["index"]]
                    result["rerank_score"] = rr["rerank_score"]
                    result["search_type"] = "hybrid+rerank"
                    final_results.append(result)

                return final_results

            return hybrid_results[:top_k]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def _normalize_scores(
        self,
        results: List[Dict[str, Any]],
        score_key: str,
        min_floor: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Normalize scores to [min_floor, 1] range. Use min_floor > 0 so the worst result is not 0% (e.g. 0.15 for 15%)."""
        if not results:
            return results

        scores = [result[score_key] for result in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            for result in results:
                result[score_key] = 1.0
        else:
            # Map [min_score, max_score] -> [min_floor, 1.0]
            span = max_score - min_score
            for result in results:
                raw = (result[score_key] - min_score) / span
                result[score_key] = min_floor + (1.0 - min_floor) * raw

        return results
    
    def _apply_filters(self, results: List[Dict[str, Any]], 
                      filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to search results"""
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            include = True
            
            for filter_key, filter_value in filters.items():
                if filter_key not in result:
                    include = False
                    break
                
                result_value = result[filter_key]
                
                # Handle different filter types
                if isinstance(filter_value, list):
                    # List filter - check if result value is in list
                    if isinstance(result_value, list):
                        # Both are lists - check for intersection
                        if not any(val in filter_value for val in result_value):
                            include = False
                            break
                    else:
                        # Result is single value, filter is list
                        if result_value not in filter_value:
                            include = False
                            break
                
                elif isinstance(filter_value, dict):
                    # Range filter
                    if "min" in filter_value and result_value < filter_value["min"]:
                        include = False
                        break
                    if "max" in filter_value and result_value > filter_value["max"]:
                        include = False
                        break
                
                else:
                    # Exact match filter
                    if result_value != filter_value:
                        include = False
                        break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    async def search_documents(self, query: str, filters: Optional[Dict[str, Any]] = None,
                              top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for documents"""
        try:
            document_filters = filters or {}

            results = await self.search(
                query=query,
                top_k=top_k,
                search_type="hybrid",
                filters=document_filters
            )

            for result in results:
                result["search_context"] = "document"
                result["match_confidence"] = self._calculate_match_confidence(result, query)

            return results

        except Exception as e:
            logger.error(f"Document search failed: {e}")
            raise

    async def search_entities(self, query: str, filters: Optional[Dict[str, Any]] = None,
                             top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for business entities (transactions, organizations, recommendations)"""
        try:
            entity_filters = filters or {}

            results = await self.search(
                query=query,
                top_k=top_k,
                search_type="hybrid",
                filters=entity_filters
            )

            for result in results:
                result["search_context"] = "entity"
                result["match_confidence"] = self._calculate_match_confidence(result, query)

            return results

        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            raise
    
    def _calculate_match_confidence(self, result: Dict[str, Any], query: str) -> float:
        """Calculate match confidence for a result"""
        try:
            base_score = result.get("search_score", 0)
            
            # Boost score based on query terms in title
            title = result.get("title", "").lower()
            query_terms = query.lower().split()
            title_matches = sum(1 for term in query_terms if term in title)
            title_boost = title_matches / len(query_terms) if query_terms else 0
            
            # Calculate final confidence
            confidence = min(base_score + (title_boost * 0.2), 1.0)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search service statistics"""
        avg_search_time = (self.total_search_time / self.search_count) if self.search_count > 0 else 0
        
        return {
            "total_searches": self.search_count,
            "total_search_time": self.total_search_time,
            "average_search_time": avg_search_time,
            "indexed_documents": len(self.documents),
            "hybrid_weights": self.hybrid_weights,
            "embedding_dimension": self.document_embeddings.shape[1] if self.document_embeddings is not None else 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check search service health"""
        try:
            # Test search functionality
            test_results = await self.search("test query", top_k=1)
            
            embedding_ok = None
            if self.embedding_service is None:
                embedding_ok = False
            else:
                embedding_ok = bool(getattr(self.embedding_service, "is_initialized", False))

            return {
                "status": "healthy",
                "is_initialized": self.is_initialized,
                "documents_indexed": len(self.documents),
                "embedding_service_healthy": embedding_ok,
                "stats": self.get_search_stats()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "is_initialized": self.is_initialized,
                "documents_indexed": len(self.documents)
            }


# Global search service instance (only create if dependencies are available)
try:
    search_service = SearchService()
except Exception as e:
    logger.warning(f"Failed to initialize global search service instance: {e}")
    logger.warning("Search service will be unavailable. Some features may not work.")
    search_service = None
