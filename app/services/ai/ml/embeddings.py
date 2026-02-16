"""
Embeddings and Vector Store Management
Uses sentence-transformers, FAISS (Inner Product), and PostgreSQL with pgvector

This module provides runtime embedding generation and vector search.
Uses cosine similarity via normalized Inner Product.
"""

import os
import json
import pickle
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Internal Cache & Metrics imports (maintaining your existing structure)
try:
    from .cache import (
        get_embedding_cache,
        get_search_cache,
        get_all_cache_stats,
        _normalize_query,
        _search_key,
    )
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False
    get_embedding_cache = lambda: None
    get_search_cache = lambda: None
    get_all_cache_stats = lambda: []
    def _normalize_query(q: str) -> str: return " ".join(q.strip().split()) if q else ""
    def _search_key(query: str, top_k: int, filter_type) -> str: return f"{_normalize_query(query)}|{top_k}|{filter_type or ''}"

# Metrics removed; use services/monitoring for observability if needed.
_METRICS_AVAILABLE = False
embedding_cache_hits = embedding_cache_misses = None
search_cache_hits = search_cache_misses = None

# ============================================================================
# EMBEDDING MODEL - Must match precompute.py
# Use a sentence-transformers model (e.g. all-MiniLM-L6-v2). Do NOT use
# causal LLMs like Qwen/Qwen2.5-3B; they are not built for similarity embeddings.
# ============================================================================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# EMBEDDING_MODEL_NAME  = "roneneldan/TinyStories-1M"          # ~4MB, 1M params
EMBEDDING_DIM = None  # Auto-detected at runtime from model (384 for all-MiniLM-L6-v2)
# ============================================================================

class VectorStore:
    """Manages embeddings, FAISS IndexFlatIP, and vector database persistence"""

    def __init__(self, data_dir: str = None, model_name: str = EMBEDDING_MODEL_NAME):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self.model_name = model_name
        self.model = None
        self.faiss_index = None
        self.documents = []
        self.metadata = []
        self.content_ids = []

        # Paths
        self.data_storage_dir = self.data_dir / "vectorstore"
        self.data_storage_dir.mkdir(exist_ok=True, parents=True)
        self.index_path = self.data_storage_dir / "faiss_index.bin"
        self.documents_path = self.data_storage_dir / "documents.pkl"

        self.embedding_dim = None  # Set after model loads
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            print(f"VectorStore initializing on: {self.device}")
        except UnicodeEncodeError:
            print(f"VectorStore initializing on: {self.device}")

    async def initialize(self):
        """Initialize the vector store - load model and handle IndexFlatIP"""
        loop = asyncio.get_event_loop()

        # Load model
        def _load_model():
            m = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)
            if getattr(m.tokenizer, "pad_token", None) is None:
                m.tokenizer.pad_token = m.tokenizer.eos_token
            return m
        self.model = await loop.run_in_executor(self.executor, _load_model)

        # Detect embedding dimension from model
        test_embedding = await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode(["test"], convert_to_numpy=True)
        )
        self.embedding_dim = test_embedding.shape[1]
        print(f"Embedding model: {self.model_name} ({self.embedding_dim} dims)")

        # Load or build index
        if self.index_path.exists() and self.documents_path.exists():
            await self._load_index()
            # Validate: index dimension must match model dimension
            if self.faiss_index and self.faiss_index.d != self.embedding_dim:
                print(
                    f"WARNING: Index dimension ({self.faiss_index.d}) != "
                    f"model dimension ({self.embedding_dim}). "
                    f"Rebuilding empty index. Run precompute.py to re-index."
                )
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                self.documents = []
                self.metadata = []
                self.content_ids = []
            # Validate: documents count must match index vectors
            elif self.faiss_index and self.faiss_index.ntotal != len(self.documents):
                print(
                    f"WARNING: Index has {self.faiss_index.ntotal} vectors but "
                    f"{len(self.documents)} documents. Run precompute.py to rebuild."
                )
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                self.documents = []
                self.metadata = []
                self.content_ids = []
        else:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

        return True

    async def _load_index(self):
        loop = asyncio.get_event_loop()
        self.faiss_index = await loop.run_in_executor(
            self.executor,
            lambda: faiss.read_index(str(self.index_path))
        )
        
        if self.documents_path.exists():
            with open(self.documents_path, 'rb') as f:
                data = await loop.run_in_executor(self.executor, pickle.load, f)
                self.documents = data.get('documents', [])
                self.metadata = data.get('metadata', [])
                self.content_ids = data.get('content_ids', [])

    async def store_embeddings(self, texts: List[str], content_type: str, content_ids: List[str], metadata_list: List[Dict]):
        """Generates, L2-normalizes, and stores embeddings for a batch of texts."""
        if not self.model or self.faiss_index is None:
            await self.initialize()

        if not texts:
            return 0

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        )
        
        # Normalize for Inner Product (Cosine Similarity)
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        await loop.run_in_executor(
            self.executor,
            lambda: self.faiss_index.add(embeddings)
        )
        
        self.documents.extend(texts)
        self.content_ids.extend(content_ids)
        
        for meta in metadata_list:
            meta['type'] = content_type
            self.metadata.append(meta)
            
        await self._save_index()
        return len(texts)

    async def _save_index(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            lambda: faiss.write_index(self.faiss_index, str(self.index_path))
        )
        
        with open(self.documents_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'content_ids': self.content_ids
            }, f)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search similar documents using normalized Inner Product."""
        if self.faiss_index is None or not self.model:
            await self.initialize()

        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []

        sk = _search_key(query, top_k, filter_type)
        s_cache = get_search_cache() if _CACHE_AVAILABLE else None
        
        # ... (Search Cache Logic) ...
        
        loop = asyncio.get_event_loop()
        nq = _normalize_query(query)
        query_embedding = None
        
        # ... (Embedding Cache Logic) ...
        
        if query_embedding is None:
            query_embedding = await loop.run_in_executor(
                self.executor,
                lambda: self.model.encode([query], convert_to_numpy=True).astype("float32"),
            )
            # Normalize query for Cosine Similarity
            faiss.normalize_L2(query_embedding)
            
        # Search the Index
        k_search = top_k * 5 if filter_type else top_k
        distances, indices = await loop.run_in_executor(
            self.executor,
            lambda: self.faiss_index.search(
                query_embedding, min(k_search, self.faiss_index.ntotal)
            ),
        )

        results = []
        for _, (score, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.documents):
                continue
            meta = self.metadata[idx]
            if filter_type and meta.get("type") != filter_type:
                continue
            
            results.append({
                "document": self.documents[idx],
                "metadata": meta,
                "score": float(score), # In IP with normalized vectors, this is the cosine similarity
                "distance": float(1 - score), 
            })
            if len(results) >= top_k:
                break

        return results

    async def get_context(self, query: str, top_k: int = 10) -> str:
        results = await self.search(query, top_k=top_k)
        return "\n".join([f"[{i+1}] {r['metadata'].get('type','').upper()}: {r['document']}" for i, r in enumerate(results)])

    async def cleanup(self):
        if self.executor:
            self.executor.shutdown(wait=True)

# Global instance
embedding_service = VectorStore()