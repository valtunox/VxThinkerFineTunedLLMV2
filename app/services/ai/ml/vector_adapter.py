"""
VaLLM Specialist Model - Vector Database Adapter.

Author: Joel Otepa Wembo
https://joelwembo.com

Unified interface for FAISS and pgvector; provides vector store adapters for
embedding storage and similarity search (app/data/vectorstore).
"""

import asyncio
import logging
import numpy as np
import faiss
import pickle
import json
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
import asyncpg
from asyncpg import Connection, Pool

from core.settings import get_settings
from core.logging import performance_logger, ai_logger

settings = get_settings()
logger = logging.getLogger(__name__)


def _default_vectorstore_dir() -> Path:
    """Resolved app/data/vectorstore (vector_adapter.py is in app/services/ai/ml/)."""
    app_root = Path(__file__).resolve().parents[3]
    return app_root / "data" / "vectorstore"


class VectorStoreAdapter(ABC):
    """Abstract base class for vector store adapters"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the vector store"""
        pass
    
    @abstractmethod
    async def store_embeddings(self, embeddings_data: List[Dict]) -> int:
        """Store multiple embeddings"""
        pass
    
    @abstractmethod
    async def similarity_search(self, query_embedding: np.ndarray, 
                              content_type: str = None, 
                              limit: int = 10, 
                              threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the vector store"""
        pass


class FAISSAdapter(VectorStoreAdapter):
    """FAISS vector store adapter. Uses app/data/vectorstore by default."""

    def __init__(self, index_path: str = None):
        if index_path is None:
            index_path = str(_default_vectorstore_dir())
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.dimension = settings.embedding_dimension
        self.index = None
        self.id_to_metadata = {}
        self.metadata_file = self.index_path / "documents.pkl"
        self.index_file = self.index_path / "index.faiss"
        
        # Performance tracking
        self.total_searches = 0
        self.total_search_time = 0.0
        self.is_initialized = False
        self._init_lock = asyncio.Lock()
    
    async def initialize(self) -> bool:
        """Initialize FAISS index. Concurrency-safe — only the first caller does work."""
        if self.is_initialized:
            return True
        async with self._init_lock:
            if self.is_initialized:
                return True
            return await self._do_initialize()

    async def _do_initialize(self) -> bool:
        """Actual FAISS init — called exactly once."""
        try:
            # Load existing index if it exists
            if self.index_file.exists() and self.metadata_file.exists():
                logger.info("Loading existing FAISS index...")
                self.index = faiss.read_index(str(self.index_file))

                with open(self.metadata_file, 'rb') as f:
                    raw_data = pickle.load(f)

                # Handle format from EmbeddingService._save_faiss_index():
                #   {'documents': [...], 'metadata': [...], 'content_ids': [...]}
                # Convert to FAISSAdapter format: {int_idx: {metadata_dict}}
                if isinstance(raw_data, dict) and 'documents' in raw_data:
                    documents = raw_data.get('documents', [])
                    metadata_list = raw_data.get('metadata', [])
                    content_ids = raw_data.get('content_ids', [])
                    self.id_to_metadata = {}
                    for i in range(len(documents)):
                        meta = metadata_list[i] if i < len(metadata_list) else {}
                        self.id_to_metadata[i] = {
                            'content_id': content_ids[i] if i < len(content_ids) else str(i),
                            'content_type': meta.get('type', 'document') if isinstance(meta, dict) else 'document',
                            'content': documents[i] if isinstance(documents[i], str) else str(documents[i]),
                            'metadata': meta if isinstance(meta, dict) else {}
                        }
                    logger.info(f"Converted EmbeddingService format: {len(self.id_to_metadata)} documents")
                else:
                    self.id_to_metadata = raw_data

                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index
                logger.info("Creating new FAISS index...")
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                self.id_to_metadata = {}
            
            self.is_initialized = True
            logger.info("FAISS adapter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS adapter: {e}")
            return False
    
    async def store_embeddings(self, embeddings_data: List[Dict]) -> int:
        """Store embeddings in FAISS index"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            stored_count = 0
            embeddings_to_add = []
            metadata_to_add = []
            
            for data in embeddings_data:
                embedding = data['embedding']
                if embedding is None or len(embedding) != self.dimension:
                    continue
                
                # Normalize embedding for cosine similarity
                embedding_norm = embedding / np.linalg.norm(embedding)
                embeddings_to_add.append(embedding_norm)
                
                metadata = {
                    'content_id': data['content_id'],
                    'content_type': data['content_type'],
                    'content': data['content'],
                    'metadata': data.get('metadata', {}),
                    'stored_at': datetime.utcnow().isoformat()
                }
                metadata_to_add.append(metadata)
                stored_count += 1
            
            if embeddings_to_add:
                # Add to FAISS index
                embeddings_array = np.array(embeddings_to_add).astype('float32')
                start_id = self.index.ntotal
                
                self.index.add(embeddings_array)
                
                # Store metadata
                for i, metadata in enumerate(metadata_to_add):
                    self.id_to_metadata[start_id + i] = metadata
                
                # Save index and metadata
                await self._save_index()
                
                logger.info(f"Stored {stored_count} embeddings in FAISS index")
            
            return stored_count
            
        except Exception as e:
            logger.error(f"Failed to store embeddings in FAISS: {e}")
            return 0
    
    async def similarity_search(self, query_embedding: np.ndarray, 
                              content_type: str = None, 
                              limit: int = 10, 
                              threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform similarity search using FAISS"""
        try:
            if not self.is_initialized or self.index.ntotal == 0:
                return []
            
            start_time = datetime.utcnow()
            
            # Normalize query embedding
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            query_norm = query_norm.reshape(1, -1).astype('float32')
            
            # Search FAISS index
            scores, indices = self.index.search(query_norm, min(limit * 2, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                if idx in self.id_to_metadata:
                    metadata = self.id_to_metadata[idx]
                    
                    # Filter by content type if specified
                    if content_type and metadata['content_type'] != content_type:
                        continue
                    
                    # Apply threshold
                    if score >= threshold:
                        results.append({
                            'content_id': metadata['content_id'],
                            'similarity': float(score),
                            'metadata': metadata['metadata'],
                            'content': metadata['content'],
                            'content_type': metadata['content_type']
                        })
            
            # Sort by similarity and limit results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            results = results[:limit]
            
            # Update performance metrics
            search_time = (datetime.utcnow() - start_time).total_seconds()
            self.total_searches += 1
            self.total_search_time += search_time
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS similarity search failed: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get FAISS statistics"""
        try:
            avg_search_time = self.total_search_time / self.total_searches if self.total_searches > 0 else 0
            
            # Count by content type
            by_type = {}
            for metadata in self.id_to_metadata.values():
                content_type = metadata['content_type']
                by_type[content_type] = by_type.get(content_type, 0) + 1
            
            return {
                "total_vectors": self.index.ntotal if self.index else 0,
                "dimension": self.dimension,
                "by_type": by_type,
                "total_searches": self.total_searches,
                "avg_search_time_ms": avg_search_time * 1000,
                "index_size_mb": self._get_index_size(),
                "is_initialized": self.is_initialized
            }
            
        except Exception as e:
            logger.error(f"Failed to get FAISS stats: {e}")
            return {}
    
    async def _save_index(self):
        """Save FAISS index and metadata"""
        try:
            if self.index:
                faiss.write_index(self.index, str(self.index_file))
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.id_to_metadata, f)
                
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def _get_index_size(self) -> float:
        """Get index file size in MB"""
        try:
            if self.index_file.exists():
                return self.index_file.stat().st_size / (1024 * 1024)
            return 0.0
        except:
            return 0.0
    
    async def close(self):
        """Close FAISS adapter"""
        try:
            if self.is_initialized:
                await self._save_index()
                logger.info("FAISS adapter closed")
        except Exception as e:
            logger.error(f"Failed to close FAISS adapter: {e}")


class PGVectorAdapter(VectorStoreAdapter):
    """PostgreSQL + pgvector adapter"""
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or settings.database_url
        # Convert to asyncpg format
        self.connection_string = self.connection_string.replace("postgresql+asyncpg://", "postgresql://")
        self.pool: Optional[Pool] = None
        self.dimension = settings.embedding_dimension
        
        # Performance tracking
        self.total_searches = 0
        self.total_search_time = 0.0
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize PostgreSQL + pgvector"""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            # Create tables
            await self._create_tables()
            
            self.is_initialized = True
            logger.info("PGVector adapter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PGVector adapter: {e}")
            return False
    
    async def _create_tables(self):
        """Create embedding tables with pgvector support"""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create embeddings table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    content_hash VARCHAR(64) UNIQUE NOT NULL,
                    content_type VARCHAR(50) NOT NULL,
                    content_id VARCHAR(255) NOT NULL,
                    embedding vector({self.dimension}),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_content_hash 
                ON embeddings(content_hash)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_content_type 
                ON embeddings(content_type)
            """)
            
            # Create vector similarity index
            try:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_vector_cosine 
                    ON embeddings USING hnsw (embedding vector_cosine_ops)
                """)
            except Exception as e:
                logger.warning(f"Could not create vector index: {e}")
    
    async def store_embeddings(self, embeddings_data: List[Dict]) -> int:
        """Store embeddings in PostgreSQL"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            stored_count = 0
            
            async with self.pool.acquire() as conn:
                for data in embeddings_data:
                    content_hash = hashlib.md5(data['content'].encode()).hexdigest()
                    metadata_json = json.dumps(data.get('metadata', {}))
                    
                    await conn.execute("""
                        INSERT INTO embeddings 
                        (content_hash, content_type, content_id, embedding, metadata, updated_at)
                        VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP)
                        ON CONFLICT (content_hash) 
                        DO UPDATE SET 
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            updated_at = CURRENT_TIMESTAMP
                    """, content_hash, data['content_type'], data['content_id'], 
                        data['embedding'].tolist(), metadata_json)
                    
                    stored_count += 1
            
            return stored_count
            
        except Exception as e:
            logger.error(f"Failed to store embeddings in PGVector: {e}")
            return 0
    
    async def similarity_search(self, query_embedding: np.ndarray, 
                              content_type: str = None, 
                              limit: int = 10, 
                              threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform similarity search using pgvector"""
        try:
            if not self.is_initialized:
                return []
            
            start_time = datetime.utcnow()
            
            async with self.pool.acquire() as conn:
                if content_type:
                    rows = await conn.fetch("""
                        SELECT content_id, embedding, metadata,
                               1 - (embedding <=> $1) as similarity
                        FROM embeddings 
                        WHERE content_type = $2
                        AND 1 - (embedding <=> $1) > $3
                        ORDER BY embedding <=> $1
                        LIMIT $4
                    """, query_embedding.tolist(), content_type, threshold, limit)
                else:
                    rows = await conn.fetch("""
                        SELECT content_id, embedding, metadata,
                               1 - (embedding <=> $1) as similarity
                        FROM embeddings 
                        WHERE 1 - (embedding <=> $1) > $2
                        ORDER BY embedding <=> $1
                        LIMIT $3
                    """, query_embedding.tolist(), threshold, limit)
                
                results = []
                for row in rows:
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    results.append({
                        'content_id': row['content_id'],
                        'similarity': float(row['similarity']),
                        'metadata': metadata
                    })
            
            # Update performance metrics
            search_time = (datetime.utcnow() - start_time).total_seconds()
            self.total_searches += 1
            self.total_search_time += search_time
            
            return results
            
        except Exception as e:
            logger.error(f"PGVector similarity search failed: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get PostgreSQL statistics"""
        try:
            if not self.pool:
                return {}
            
            async with self.pool.acquire() as conn:
                # Total embeddings
                total_count = await conn.fetchval("SELECT COUNT(*) FROM embeddings")
                
                # By type
                by_type_rows = await conn.fetch("""
                    SELECT content_type, COUNT(*) as count 
                    FROM embeddings 
                    GROUP BY content_type
                """)
                by_type = {row['content_type']: row['count'] for row in by_type_rows}
                
                avg_search_time = self.total_search_time / self.total_searches if self.total_searches > 0 else 0
                
                return {
                    "total_vectors": total_count,
                    "dimension": self.dimension,
                    "by_type": by_type,
                    "total_searches": self.total_searches,
                    "avg_search_time_ms": avg_search_time * 1000,
                    "is_initialized": self.is_initialized
                }
                
        except Exception as e:
            logger.error(f"Failed to get PGVector stats: {e}")
            return {}
    
    async def close(self):
        """Close PostgreSQL adapter"""
        try:
            if self.pool:
                await self.pool.close()
                logger.info("PGVector adapter closed")
        except Exception as e:
            logger.error(f"Failed to close PGVector adapter: {e}")


class VectorStoreFactory:
    """Factory for creating vector store adapters"""
    
    @staticmethod
    def create_adapter(vector_db_type: str = None) -> VectorStoreAdapter:
        """Create vector store adapter based on configuration"""
        vector_db_type = vector_db_type or settings.vector_db_type
        vectorstore_dir = _default_vectorstore_dir()

        if vector_db_type.lower() == "faiss":
            return FAISSAdapter(index_path=str(vectorstore_dir))
        elif vector_db_type.lower() == "pgvector":
            return PGVectorAdapter()
        else:
            logger.warning(f"Unknown vector DB type: {vector_db_type}, defaulting to FAISS")
            return FAISSAdapter(index_path=str(vectorstore_dir))


# Global vector store adapter instance
vector_store = VectorStoreFactory.create_adapter()
