"""
VaLLM Specialist Model - Enhanced Embedding Service.

Author: Joel Otepa Wembo
https://joelwembo.com

GPU-accelerated embedding generation with batching, FAISS vector storage (app/data/vectorstore),
and multi-format document loading (PDF, DOCX, TXT, HTML). Single source of truth for all embedding operations.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import asyncio
import numpy as np
import torch
import pickle
import faiss
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

# Conditional imports for sentence_transformers
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    try:
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except ImportError:
        SentenceTransformer = None
        SENTENCE_TRANSFORMERS_AVAILABLE = False
    CrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False

# Document processing libraries (optional)
try:
    import pypdf
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    HTML_AVAILABLE = True
except ImportError:
    HTML_AVAILABLE = False

try:
    from app.core.settings import get_settings
    from app.core.logging import performance_logger, ai_logger
except ImportError:
    from core.settings import get_settings
    from core.logging import performance_logger, ai_logger
from .vector_adapter import vector_store

settings = get_settings()
logger = logging.getLogger(__name__)


class DocumentLoader:
    """Multi-format document loader for embeddings"""
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Extract text from PDF"""
        if not PDF_AVAILABLE:
            raise ImportError("pypdf/pdfplumber not installed. Install with: pip install pypdf pdfplumber")
        
        try:
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            return text.strip()
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying pypdf: {e}")
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            return text.strip()
    
    @staticmethod
    def load_docx(file_path: str) -> str:
        """Extract text from DOCX"""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not installed. Install with: pip install python-docx")
        
        doc = DocxDocument(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return text.strip()
    
    @staticmethod
    def load_txt(file_path: str) -> str:
        """Load text file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()
    
    @staticmethod
    def load_html(file_path: str) -> str:
        """Extract text from HTML"""
        if not HTML_AVAILABLE:
            raise ImportError("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        return soup.get_text(separator='\n', strip=True)
    
    @staticmethod
    def load_json(file_path: str) -> str:
        """Load JSON and convert to text"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Convert to readable text format
        return json.dumps(data, indent=2)
    
    @staticmethod
    def load_document(file_path: Union[str, Path]) -> Tuple[str, str]:
        """
        Load document from any supported format
        Returns: (text_content, file_type)
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        loaders = {
            '.pdf': DocumentLoader.load_pdf,
            '.docx': DocumentLoader.load_docx,
            '.doc': DocumentLoader.load_docx,
            '.txt': DocumentLoader.load_txt,
            '.md': DocumentLoader.load_txt,
            '.html': DocumentLoader.load_html,
            '.htm': DocumentLoader.load_html,
            '.json': DocumentLoader.load_json,
            '.jsonl': DocumentLoader.load_txt,
        }
        
        loader = loaders.get(suffix)
        if not loader:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        text = loader(str(file_path))
        return text, suffix[1:]  # Remove dot from extension


class EmbeddingService:
    """
    Enhanced GPU-accelerated embedding service
    Combines runtime embedding generation, FAISS index management, and document loading
    """
    
    def __init__(self):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence_transformers is not installed. EmbeddingService will not be functional.")

        self.model_name = settings.embedding_model
        self.device = settings.embedding_device
        self.batch_size = settings.embedding_batch_size
        self.dimension = settings.embedding_dimension

        # BGE models require a query instruction prefix for retrieval tasks
        self.is_bge_model = "bge" in self.model_name.lower()
        self.query_instruction = "Represent this sentence for searching relevant passages: " if self.is_bge_model else ""

        # Reranker (cross-encoder) for second-stage ranking
        self.reranker_model_name = getattr(settings, "reranker_model", "cross-encoder/ms-marco-MiniLM-L6-v2")
        self.reranker = None

        # Model and cache
        self.model = None
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 10000

        # FAISS index management (merged from training/embeddings.py)
        self.faiss_index = None
        self.documents = []  # Store original documents
        self.metadata = []   # Store metadata for each document
        self.content_ids = []  # Store content IDs
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Paths: app root = app/ (embedding.py is in app/services/ai/ml/)
        app_root = Path(__file__).resolve().parents[3]
        self.data_dir = app_root / "data"
        self.cache_dir = self.data_dir / "models" / "cache"
        self.vectorstore_dir = self.data_dir / "vectorstore"
        self.index_path = self.vectorstore_dir / "index.faiss"
        self.metadata_path = self.vectorstore_dir / "documents.pkl"
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.total_embeddings_generated = 0
        self.total_processing_time = 0.0
        self.is_initialized = False
        self._init_lock = asyncio.Lock()
        
        # Document loader
        self.doc_loader = DocumentLoader()
    
    async def initialize(self) -> bool:
        """Initialize embedding service with FAISS index. Idempotent and concurrency-safe."""
        if self.is_initialized:
            return True
        async with self._init_lock:
            if self.is_initialized:
                return True
            return await self._do_initialize()

    async def _do_initialize(self) -> bool:
        """Actual initialization logic — called exactly once, guarded by _init_lock."""
        try:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.error("sentence_transformers is not installed. Please install it to use the embedding service.")
                return False
            
            # Auto-detect device availability
            if self.device == "cuda":
                if not torch.cuda.is_available():
                    logger.warning("⚠️  CUDA requested but not available. Falling back to CPU.")
                    logger.warning("   To use GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                    self.device = "cpu"
                else:
                    logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            
            # Load SentenceTransformer model
            logger.info(f"Loading embedding model: {self.model_name}")
            logger.info(f"Target device: {self.device.upper()}")
            loop = asyncio.get_event_loop()
            # Try local cache first to avoid network HEAD checks every run
            try:
                self.model = await loop.run_in_executor(
                    self.executor,
                    lambda: SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        cache_folder=str(self.cache_dir),
                        local_files_only=True
                    )
                )
                logger.info("Loaded embedding model from local cache (offline)")
            except Exception:
                logger.info("Model not in cache, downloading...")
                self.model = await loop.run_in_executor(
                    self.executor,
                    lambda: SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        cache_folder=str(self.cache_dir)
                    )
                )
            
            # Get actual embedding dimension
            test_embedding = await loop.run_in_executor(
                self.executor,
                lambda: self.model.encode(["test"], convert_to_numpy=True)
            )
            self.dimension = test_embedding.shape[1]
            
            logger.info(f"✅ Model loaded on {self.device.upper()} with dimension {self.dimension}")
            
            # Load or create FAISS index
            if self.index_path.exists() and self.metadata_path.exists():
                await self._load_faiss_index()
                logger.info(f"Loaded existing FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                self.faiss_index = faiss.IndexFlatL2(self.dimension)
                logger.info("Created new FAISS index")
            
            # Initialize cross-encoder reranker
            if CROSS_ENCODER_AVAILABLE:
                try:
                    logger.info(f"Loading reranker model: {self.reranker_model_name}")
                    # Try local cache first to avoid network HEAD checks every run
                    try:
                        import os as _os
                        _prev_hf = _os.environ.get("HF_HUB_OFFLINE")
                        _os.environ["HF_HUB_OFFLINE"] = "1"
                        self.reranker = await loop.run_in_executor(
                            self.executor,
                            lambda: CrossEncoder(
                                self.reranker_model_name,
                                max_length=512,
                                device=self.device
                            )
                        )
                        if _prev_hf is None:
                            _os.environ.pop("HF_HUB_OFFLINE", None)
                        else:
                            _os.environ["HF_HUB_OFFLINE"] = _prev_hf
                        logger.info(f"Reranker loaded from local cache (offline): {self.reranker_model_name}")
                    except Exception:
                        import os as _os
                        _os.environ.pop("HF_HUB_OFFLINE", None)
                        logger.info("Reranker not in cache, downloading...")
                        self.reranker = await loop.run_in_executor(
                            self.executor,
                            lambda: CrossEncoder(
                                self.reranker_model_name,
                                max_length=512,
                                device=self.device
                            )
                        )
                        logger.info(f"Reranker downloaded and loaded: {self.reranker_model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load reranker (search will work without reranking): {e}")
                    self.reranker = None
            else:
                logger.info("CrossEncoder not available - reranking disabled")

            # Initialize vector adapter (for backward compatibility)
            vector_store_success = await vector_store.initialize()
            if not vector_store_success:
                logger.warning("Vector adapter failed to initialize (FAISS index still available)")

            self.is_initialized = True

            # Warm up model (must be after is_initialized=True to avoid deadlock)
            await self._warm_up_model()
            logger.info("✅ Embedding Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Embedding Service: {e}")
            return False
    
    async def _load_faiss_index(self):
        """Load existing FAISS index and metadata"""
        loop = asyncio.get_event_loop()
        
        # Load FAISS index
        self.faiss_index = await loop.run_in_executor(
            self.executor,
            lambda: faiss.read_index(str(self.index_path))
        )
        
        # Load documents and metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, 'rb') as f:
                data = await loop.run_in_executor(self.executor, pickle.load, f)
                self.documents = data.get('documents', [])
                self.metadata = data.get('metadata', [])
                self.content_ids = data.get('content_ids', [])
    
    async def _save_faiss_index(self):
        """Save FAISS index and metadata to disk"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            lambda: faiss.write_index(self.faiss_index, str(self.index_path))
        )
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'content_ids': self.content_ids
            }, f)
    
    async def _warm_up_model(self):
        """Warm up the model with a small batch"""
        try:
            warm_up_texts = ["This is a warm up text for the embedding model."]
            await self.generate_embeddings(warm_up_texts)
            logger.info("Model warm-up completed")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")
    
    async def _ensure_initialized(self):
        """Auto-initialize on first use if background startup hasn't completed yet. Thread-safe via is_initialized flag."""
        if not self.is_initialized:
            await self.initialize()

    async def generate_embeddings(self, texts: Union[str, List[str]], 
                                batch_size: Optional[int] = None,
                                show_progress: bool = False) -> np.ndarray:
        """Generate embeddings for texts with intelligent batching"""
        try:
            await self._ensure_initialized()
            start_time = datetime.now(timezone.utc)
            
            # Normalize input
            if isinstance(texts, str):
                texts = [texts]
            
            if not texts:
                return np.array([])
            
            # Use provided batch size or default
            batch_size = batch_size or self.batch_size
            
            # Check cache first
            embeddings = []
            texts_to_process = []
            cache_indices = []
            
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self.embedding_cache:
                    embeddings.append(self.embedding_cache[cache_key])
                    cache_indices.append(i)
                    self.cache_hits += 1
                else:
                    texts_to_process.append(text)
            
            # Generate embeddings for non-cached texts
            if texts_to_process:
                new_embeddings = await self._generate_batch_embeddings(
                    texts_to_process, batch_size, show_progress
                )
                
                # Cache new embeddings
                for text, embedding in zip(texts_to_process, new_embeddings):
                    cache_key = self._get_cache_key(text)
                    self.embedding_cache[cache_key] = embedding
                    if len(self.embedding_cache) > self.max_cache_size:
                        # Evict oldest entries (first 20%)
                        evict_count = self.max_cache_size // 5
                        keys_to_evict = list(self.embedding_cache.keys())[:evict_count]
                        for k in keys_to_evict:
                            del self.embedding_cache[k]
                    self.cache_misses += 1
                
                # Insert new embeddings at correct positions
                final_embeddings = [None] * len(texts)
                new_idx = 0
                for i, text in enumerate(texts):
                    if i in cache_indices:
                        final_embeddings[i] = embeddings[cache_indices.index(i)]
                    else:
                        final_embeddings[i] = new_embeddings[new_idx]
                        new_idx += 1
                
                embeddings = final_embeddings
            
            embeddings_array = np.array(embeddings)
            
            # Update metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.total_processing_time += processing_time
            self.total_embeddings_generated += len(texts)
            
            # Log performance
            ai_logger.log_embedding_generation(
                text_length=sum(len(text) for text in texts),
                embedding_dim=len(embeddings_array[0]) if len(embeddings_array) > 0 else 0,
                processing_time_ms=processing_time * 1000
            )
            
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def _generate_batch_embeddings(self, texts: List[str], batch_size: int,
                                        show_progress: bool = False,
                                        is_query: bool = False) -> List[np.ndarray]:
        """Generate embeddings in batches. For BGE models, set is_query=True for retrieval queries."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.model is None:
            raise ImportError("sentence_transformers is not installed or model is not initialized")

        # BGE models need an instruction prefix on the query side for better retrieval
        if is_query and self.query_instruction:
            texts = [self.query_instruction + t for t in texts]

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Generate embeddings for batch
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                convert_to_numpy=True,
                normalize_embeddings=True,  # BGE recommends L2-normalized embeddings
                show_progress_bar=show_progress
            )

            embeddings.extend(batch_embeddings)

        return embeddings
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        content = f"{self.model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    # ========================================================================
    # CROSS-ENCODER RERANKING
    # ========================================================================

    # Max chars per segment so [query, doc] stays under model max_length (512 tokens ~ 2000 chars)
    RERANK_QUERY_MAX_CHARS = 1000
    RERANK_DOC_MAX_CHARS = 1200

    async def rerank(self, query: str, documents: List[str],
                     top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank documents using the cross-encoder reranker (ms-marco-MiniLM-L6-v2).

        The cross-encoder scores each (query, document) pair independently,
        producing much more accurate relevance scores than bi-encoder cosine
        similarity alone. Use this as a second-stage ranker after an initial
        FAISS or BM25 retrieval.

        Query and each document are truncated so the combined input fits the
        model's 512-token limit; this avoids meaningless truncation and very
        low scores when long job descriptions and resumes are passed.

        Args:
            query: The search query string.
            documents: List of document texts to rerank.
            top_k: Number of top results to return.

        Returns:
            List of dicts with 'index', 'text', and 'rerank_score', sorted best-first.
        """
        await self._ensure_initialized()
        if not self.reranker:
            logger.warning("Reranker not available - returning documents in original order")
            return [
                {"index": i, "text": doc, "rerank_score": 1.0 - (i * 0.01)}
                for i, doc in enumerate(documents[:top_k])
            ]

        try:
            loop = asyncio.get_event_loop()

            # Truncate query and docs so the model gets meaningful text within 512 tokens.
            # Long JD + full resume often get truncated mid-text and produce near-zero scores.
            q = (query or "").strip()[: self.RERANK_QUERY_MAX_CHARS]
            truncated_docs = [(doc or "").strip()[: self.RERANK_DOC_MAX_CHARS] for doc in documents]

            # Build (query, document) pairs for the cross-encoder
            pairs = [[q, doc] for doc in truncated_docs]

            # Score all pairs
            scores = await loop.run_in_executor(
                self.executor,
                lambda: self.reranker.predict(pairs, show_progress_bar=False)
            )

            # Pair scores with original indices
            scored = [
                {"index": i, "text": documents[i], "rerank_score": float(scores[i])}
                for i in range(len(documents))
            ]
            scored.sort(key=lambda x: x["rerank_score"], reverse=True)

            return scored[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return [
                {"index": i, "text": doc, "rerank_score": 0.0}
                for i, doc in enumerate(documents[:top_k])
            ]

    # ========================================================================
    # FAISS INDEX MANAGEMENT (merged from training/embeddings.py)
    # ========================================================================
    
    async def store_embeddings_in_faiss(self, texts: List[str], content_type: str, 
                                        content_ids: List[str], metadata_list: List[Dict]) -> int:
        """
        Store embeddings directly in FAISS index
        Used by precompute script for bulk indexing
        """
        if not self.model or self.faiss_index is None:
            await self.initialize()
        
        if not texts:
            return 0
        
        # Generate embeddings
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        )
        
        # Add to FAISS index
        await loop.run_in_executor(
            self.executor,
            lambda: self.faiss_index.add(embeddings.astype('float32'))
        )
        
        # Update in-memory storage
        self.documents.extend(texts)
        self.content_ids.extend(content_ids)
        
        # Enhance metadata
        for meta in metadata_list:
            meta['type'] = content_type
            self.metadata.append(meta)
        
        # Save to disk
        await self._save_faiss_index()
        
        return len(texts)
    
    async def search_faiss(self, query: str, top_k: int = 5,
                          filter_type: Optional[str] = None,
                          use_reranker: bool = True) -> List[Dict[str, Any]]:
        """
        Search FAISS index with optional cross-encoder reranking.

        Stage 1: Bi-encoder (BGE) retrieves top candidates via FAISS.
        Stage 2: Cross-encoder (ms-marco-MiniLM-L6-v2) reranks for precision.
        """
        if self.faiss_index is None or not self.model:
            await self.initialize()
            if self.faiss_index is None or self.faiss_index.ntotal == 0:
                return []

        # Generate query embedding (with BGE instruction prefix)
        loop = asyncio.get_event_loop()
        query_text = (self.query_instruction + query) if self.query_instruction else query
        query_embedding = await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode(
                [query_text], convert_to_numpy=True, normalize_embeddings=True
            ).astype('float32')
        )

        # Stage 1: Retrieve a wider set of candidates from FAISS
        retrieve_k = top_k * 5 if (filter_type or (use_reranker and self.reranker)) else top_k
        distances, indices = await loop.run_in_executor(
            self.executor,
            lambda: self.faiss_index.search(query_embedding, min(retrieve_k, self.faiss_index.ntotal))
        )

        # Collect initial results
        candidates = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            meta = self.metadata[idx] if idx < len(self.metadata) else {}

            # Apply content-type filter
            if filter_type and meta.get('type') != filter_type:
                continue

            candidates.append({
                'document': self.documents[idx],
                'metadata': meta,
                'score': float(1 / (1 + distance)),
                'distance': float(distance)
            })

        # Stage 2: Cross-encoder reranking for higher precision
        if use_reranker and self.reranker and len(candidates) > 1:
            candidate_texts = [c['document'] for c in candidates]
            reranked = await self.rerank(query, candidate_texts, top_k=top_k)

            # Map reranked results back to full candidate objects
            results = []
            for r in reranked:
                orig = candidates[r['index']]
                orig['rerank_score'] = r['rerank_score']
                results.append(orig)
            return results

        return candidates[:top_k]
    
    async def get_faiss_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics"""
        if self.faiss_index is None:
            return {"total_vectors": 0}
        
        by_type = {}
        for m in self.metadata:
            t = m.get('type', 'unknown')
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            "total_vectors": self.faiss_index.ntotal,
            "dimension": self.dimension,
            "by_type": by_type,
            "documents_count": len(self.documents)
        }
    
    # ========================================================================
    # DOCUMENT LOADING
    # ========================================================================
    
    async def load_document(self, file_path: Union[str, Path]) -> Tuple[str, str]:
        """
        Load and extract text from any supported document format
        Returns: (text_content, file_type)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.doc_loader.load_document(file_path)
        )
    
    async def load_and_embed_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load document and generate embeddings
        Returns document info with embeddings
        """
        try:
            # Load document
            text, file_type = await self.load_document(file_path)
            
            if not text.strip():
                return {"error": "Empty document", "file_path": str(file_path)}
            
            # Generate embedding
            embeddings = await self.generate_embeddings([text])
            
            return {
                "file_path": str(file_path),
                "file_type": file_type,
                "text": text,
                "text_length": len(text),
                "embedding": embeddings[0],
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to load and embed document {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "error": str(e),
                "success": False
            }
    
    # ========================================================================
    # SIMILARITY & SEARCH
    # ========================================================================
    
    async def get_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        try:
            embeddings = await self.generate_embeddings([text1, text2])
            
            if len(embeddings) != 2:
                raise ValueError("Failed to generate embeddings for both texts")
            
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            raise
    
    async def find_most_similar(self, query_text: str, candidate_texts: List[str], 
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar texts to query"""
        try:
            query_embedding = await self.generate_embeddings([query_text])
            candidate_embeddings = await self.generate_embeddings(candidate_texts)
            
            similarities = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = np.dot(query_embedding[0], candidate_embedding) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(candidate_embedding)
                )
                similarities.append({
                    "index": i,
                    "text": candidate_texts[i],
                    "similarity": float(similarity)
                })
            
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise
    
    # ========================================================================
    # VECTOR STORE OPERATIONS (for backward compatibility)
    # ========================================================================
    
    async def store_embeddings(self, texts: List[str], content_type: str, 
                              content_ids: List[str] = None,
                              metadata_list: List[Dict] = None) -> int:
        """Generate and store embeddings in vector database"""
        try:
            embeddings = await self.generate_embeddings(texts)
            
            embeddings_data = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                content_id = content_ids[i] if content_ids and i < len(content_ids) else str(i)
                metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
                
                embeddings_data.append({
                    'content': text,
                    'content_type': content_type,
                    'content_id': content_id,
                    'embedding': embedding,
                    'metadata': metadata
                })
            
            stored_count = await vector_store.store_embeddings(embeddings_data)
            logger.info(f"Stored {stored_count} embeddings for {content_type}")
            return stored_count
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            return 0
    
    async def similarity_search(self, query_text: str, content_type: str = None, 
                              limit: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform similarity search using vector database"""
        try:
            query_embedding = await self.generate_embeddings([query_text])
            if len(query_embedding) == 0:
                return []
            
            results = await vector_store.similarity_search(
                query_embedding[0], content_type, limit, threshold
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics"""
        cache_size = len(self.embedding_cache)
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            "cache_size": cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "memory_usage_mb": self._estimate_cache_memory_usage()
        }
    
    def _estimate_cache_memory_usage(self) -> float:
        """Estimate cache memory usage in MB"""
        if not self.embedding_cache:
            return 0.0
        
        embedding_size = self.dimension * 4  # 4 bytes per float32
        total_size = len(self.embedding_cache) * embedding_size
        return total_size / (1024 * 1024)
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Embedding cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        avg_processing_time = (self.total_processing_time / self.total_embeddings_generated) if self.total_embeddings_generated > 0 else 0
        
        return {
            "total_embeddings_generated": self.total_embeddings_generated,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            "cache_stats": self.get_cache_stats(),
            "faiss_stats": {"total_vectors": self.faiss_index.ntotal if self.faiss_index else 0}
        }
    
    async def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            return await vector_store.get_stats()
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check embedding service health"""
        try:
            test_text = "Health check test"
            embedding = await self.generate_embeddings([test_text])
            
            if len(embedding) == 0 or len(embedding[0]) != self.dimension:
                raise Exception("Embedding dimension mismatch")
            
            vector_stats = await self.get_vector_store_stats()
            faiss_stats = await self.get_faiss_stats()
            
            return {
                "status": "healthy",
                "is_initialized": self.is_initialized,
                "model_loaded": self.model is not None,
                "embedding_model": self.model_name,
                "reranker_model": self.reranker_model_name,
                "reranker_loaded": self.reranker is not None,
                "device": self.device,
                "stats": self.get_stats(),
                "vector_store_stats": vector_stats,
                "faiss_stats": faiss_stats,
                "supported_formats": [".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm", ".json", ".jsonl"]
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "is_initialized": self.is_initialized,
                "model_loaded": self.model is not None
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        await self._save_faiss_index()


# Global embedding service instance
embedding_service = EmbeddingService()
