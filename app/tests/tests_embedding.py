"""
VA LLM Specialist Model — Embedding Service Unit Tests
=======================================================

Author: Joel Otepa Wembo
https://joelwembo.com

DESCRIPTION
===========
Unit tests for the EmbeddingService class. Covers initialization,
embedding generation, caching, FAISS index operations, similarity
calculations, document loading, and cross-encoder reranking
availability. No running server is required.

Model loading is mocked where possible for fast execution, but
tests that verify actual embedding quality load the real model.

PREREQUISITES
=============
    pip install pytest pytest-asyncio numpy

USAGE
=====
    python -m pytest app/tests/tests_embedding.py -v -s --tb=short
"""

import sys
import os
import json
import atexit
import asyncio
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_APP_DIR = _PROJECT_ROOT / "app"
for _p in [str(_PROJECT_ROOT), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
_SCORECARD: Dict[str, Dict[str, Any]] = {}
_TEST_COUNTER = 0


def _record_score(test_name: str, score: int, details: str = ""):
    global _TEST_COUNTER
    _TEST_COUNTER += 1
    score = max(1, min(10, score))
    bar = "#" * score + "." * (10 - score)
    label = (
        "PERFECT" if score == 10 else
        "EXCELLENT" if score >= 8 else
        "GOOD" if score >= 6 else
        "FAIR" if score >= 4 else
        "POOR"
    )
    _SCORECARD[f"Test {_TEST_COUNTER}"] = {
        "name": test_name, "score": score, "label": label, "details": details,
    }
    print(f"\n  {'=' * 65}")
    print(f"  [Test {_TEST_COUNTER}] {test_name}")
    print(f"  Score: {score}/10 [{bar}] {label}")
    if details:
        print(f"  Details: {details}")
    print(f"  {'=' * 65}")


def _print_final_scorecard():
    if not _SCORECARD:
        return
    total = sum(v["score"] for v in _SCORECARD.values())
    count = len(_SCORECARD)
    avg = total / count if count else 0
    print("\n\n" + "=" * 70)
    print("  EMBEDDING SERVICE TESTS — FINAL SCORECARD")
    print("=" * 70)
    for key, val in _SCORECARD.items():
        bar = "#" * val["score"] + "." * (10 - val["score"])
        print(f"  {key:>8} | {val['score']:>2}/10 [{bar}] {val['label']:<10} | {val['name']}")
    print("-" * 70)
    print(f"  {'TOTAL':>8} | {total}/{count * 10}  Average: {avg:.1f}/10")
    overall = (
        "EXCELLENT" if avg >= 8 else "GOOD" if avg >= 6 else
        "NEEDS IMPROVEMENT" if avg >= 4 else "CRITICAL ISSUES"
    )
    print(f"  Overall Assessment: {overall}")
    print("=" * 70)


atexit.register(_print_final_scorecard)


# ============================================================================
# 1. EMBEDDING SERVICE IMPORT & INITIALIZATION
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestEmbeddingServiceInit:
    """Test EmbeddingService class imports and basic construction."""

    def test_embedding_module_imports(self):
        """Verify the embedding module imports without errors."""
        try:
            from services.ai.ml.embedding import EmbeddingService, DocumentLoader
            _record_score("Embedding Module Import", 10, "EmbeddingService and DocumentLoader imported")
        except ImportError as e:
            try:
                from app.services.ai.ml.embedding import EmbeddingService, DocumentLoader
                _record_score("Embedding Module Import", 10,
                              "Imported via app.services path")
            except ImportError as e2:
                _record_score("Embedding Module Import", 3, f"Import failed: {e2}")
                pytest.fail(f"Cannot import EmbeddingService: {e2}")

    def test_embedding_service_instantiation(self):
        """EmbeddingService() should create an instance with expected attributes."""
        try:
            from services.ai.ml.embedding import EmbeddingService
        except ImportError:
            from app.services.ai.ml.embedding import EmbeddingService

        svc = EmbeddingService()
        assert hasattr(svc, "model_name"), "Missing model_name attribute"
        assert hasattr(svc, "dimension"), "Missing dimension attribute"
        assert hasattr(svc, "embedding_cache"), "Missing embedding_cache attribute"
        assert hasattr(svc, "faiss_index"), "Missing faiss_index attribute"
        assert hasattr(svc, "documents"), "Missing documents attribute"
        assert svc.is_initialized is False, "Should not be initialized yet"
        print(f"  model_name: {svc.model_name}")
        print(f"  dimension: {svc.dimension}")
        print(f"  device: {svc.device}")
        _record_score("EmbeddingService Instantiation", 10,
                       f"model={svc.model_name}, dim={svc.dimension}")

    def test_embedding_dimension_matches_settings(self):
        """Dimension should match what is configured in settings."""
        try:
            from services.ai.ml.embedding import EmbeddingService
        except ImportError:
            from app.services.ai.ml.embedding import EmbeddingService

        from core.settings import get_settings
        settings = get_settings()
        svc = EmbeddingService()
        assert svc.dimension == settings.embedding_dimension, (
            f"Dimension mismatch: service={svc.dimension}, settings={settings.embedding_dimension}"
        )
        _record_score("Dimension Matches Settings", 10,
                       f"dim={svc.dimension} == settings.embedding_dimension")

    def test_vectorstore_directory_exists(self):
        """The vectorstore directory should be created on instantiation."""
        try:
            from services.ai.ml.embedding import EmbeddingService
        except ImportError:
            from app.services.ai.ml.embedding import EmbeddingService

        svc = EmbeddingService()
        assert svc.vectorstore_dir.exists(), f"Vectorstore dir not created: {svc.vectorstore_dir}"
        _record_score("Vectorstore Dir Created", 10, str(svc.vectorstore_dir))


# ============================================================================
# 2. EMBEDDING GENERATION (with real model if available)
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestEmbeddingGeneration:
    """Test embedding generation. Uses real model if sentence_transformers is installed."""

    @pytest.fixture(scope="class")
    def embedding_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("BAAI/bge-large-en-v1.5")
            return model
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_single_text_embedding(self, embedding_model):
        """Embedding a single text should produce a 1024-dim vector."""
        emb = embedding_model.encode(["hello world"], convert_to_numpy=True)
        assert emb.shape == (1, 1024), f"Expected (1, 1024), got {emb.shape}"
        assert not np.all(emb == 0), "Embedding is all zeros"
        _record_score("Single Text Embedding", 10, f"shape={emb.shape}")

    def test_batch_embedding_generation(self, embedding_model):
        """Batch embedding should produce correct number of vectors."""
        texts = [
            "Kubernetes pod deployment",
            "Financial audit compliance",
            "Patient diagnosis treatment",
        ]
        emb = embedding_model.encode(texts, convert_to_numpy=True)
        assert emb.shape == (3, 1024), f"Expected (3, 1024), got {emb.shape}"
        # All embeddings should be different
        sim_01 = float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])))
        sim_02 = float(np.dot(emb[0], emb[2]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[2])))
        print(f"  sim(k8s, finance)={sim_01:.4f}, sim(k8s, health)={sim_02:.4f}")
        assert not np.allclose(emb[0], emb[1]), "First two embeddings are identical"
        _record_score("Batch Embedding Generation", 10, f"3 texts, distinct vectors")

    def test_embedding_normalization(self, embedding_model):
        """Normalized embeddings should have unit L2 norm."""
        emb = embedding_model.encode(
            ["Test normalization"], convert_to_numpy=True, normalize_embeddings=True
        )
        norm = np.linalg.norm(emb[0])
        print(f"  L2 norm: {norm:.6f}")
        assert abs(norm - 1.0) < 0.01, f"Norm should be ~1.0, got {norm:.6f}"
        _record_score("Embedding Normalization", 10, f"norm={norm:.6f}")


# ============================================================================
# 3. CACHE MECHANISM
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestEmbeddingCache:
    """Test the embedding cache hit/miss mechanism."""

    def test_cache_key_generation(self):
        """Cache key should be deterministic MD5 hash."""
        try:
            from services.ai.ml.embedding import EmbeddingService
        except ImportError:
            from app.services.ai.ml.embedding import EmbeddingService

        svc = EmbeddingService()
        key = svc._get_cache_key("test text")
        expected = hashlib.md5(f"{svc.model_name}:test text".encode()).hexdigest()
        assert key == expected, f"Cache key mismatch: {key} != {expected}"
        _record_score("Cache Key Generation", 10, f"key={key[:16]}...")

    def test_cache_key_different_texts(self):
        """Different texts should produce different cache keys."""
        try:
            from services.ai.ml.embedding import EmbeddingService
        except ImportError:
            from app.services.ai.ml.embedding import EmbeddingService

        svc = EmbeddingService()
        key1 = svc._get_cache_key("hello world")
        key2 = svc._get_cache_key("goodbye world")
        assert key1 != key2, "Different texts produced same cache key"
        _record_score("Cache Key Uniqueness", 10, "Different texts -> different keys")

    def test_cache_eviction_threshold(self):
        """Cache should have a max size and eviction logic."""
        try:
            from services.ai.ml.embedding import EmbeddingService
        except ImportError:
            from app.services.ai.ml.embedding import EmbeddingService

        svc = EmbeddingService()
        assert svc.max_cache_size == 10000, f"Expected max_cache_size=10000, got {svc.max_cache_size}"
        # Simulate filling the cache
        for i in range(100):
            svc.embedding_cache[f"key_{i}"] = np.zeros(1024)
        assert len(svc.embedding_cache) == 100
        print(f"  max_cache_size: {svc.max_cache_size}")
        _record_score("Cache Eviction Config", 10, f"max_cache_size={svc.max_cache_size}")


# ============================================================================
# 4. SIMILARITY CALCULATION
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestSimilarityCalculation:
    """Test cosine similarity between embeddings."""

    @pytest.fixture(scope="class")
    def embedding_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("BAAI/bge-large-en-v1.5")
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_same_text_high_similarity(self, embedding_model):
        """Identical text should have similarity ~1.0."""
        emb = embedding_model.encode(
            ["deploy Kubernetes cluster", "deploy Kubernetes cluster"],
            convert_to_numpy=True, normalize_embeddings=True
        )
        sim = float(np.dot(emb[0], emb[1]))
        print(f"  Same-text similarity: {sim:.6f}")
        assert sim > 0.99, f"Expected > 0.99, got {sim:.6f}"
        _record_score("Same-Text Similarity", 10, f"sim={sim:.6f}")

    def test_related_vs_unrelated(self, embedding_model):
        """Related texts should be more similar than unrelated texts."""
        texts = [
            "deploy Kubernetes pods to production cluster",
            "scale Kubernetes deployment with horizontal pod autoscaler",
            "bake a chocolate cake with vanilla frosting",
        ]
        emb = embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        sim_related = float(np.dot(emb[0], emb[1]))
        sim_unrelated = float(np.dot(emb[0], emb[2]))
        gap = sim_related - sim_unrelated
        print(f"  Related sim: {sim_related:.4f}")
        print(f"  Unrelated sim: {sim_unrelated:.4f}")
        print(f"  Gap: {gap:.4f}")
        assert gap > 0.1, f"Gap too small: {gap:.4f}"
        score = 10 if gap > 0.3 else 8 if gap > 0.2 else 6
        _record_score("Related vs Unrelated Similarity", score, f"gap={gap:.4f}")


# ============================================================================
# 5. FAISS INDEX
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestFAISSIndex:
    """Test FAISS index creation and basic operations."""

    def test_faiss_import(self):
        """FAISS should be importable."""
        try:
            import faiss
            _record_score("FAISS Import", 10, f"faiss version available")
        except ImportError:
            _record_score("FAISS Import", 3, "faiss not installed")
            pytest.skip("faiss not installed")

    def test_faiss_index_creation(self):
        """Create a FAISS IndexFlatL2 and add vectors."""
        import faiss
        dim = 1024
        index = faiss.IndexFlatL2(dim)
        assert index.ntotal == 0, "New index should be empty"

        # Add some random vectors
        vectors = np.random.randn(10, dim).astype("float32")
        index.add(vectors)
        assert index.ntotal == 10, f"Expected 10 vectors, got {index.ntotal}"
        _record_score("FAISS Index Creation", 10, f"10 vectors added, dim={dim}")

    def test_faiss_search(self):
        """FAISS search should return nearest neighbors."""
        import faiss
        dim = 128  # Smaller dim for speed
        index = faiss.IndexFlatL2(dim)
        # Add 100 random vectors
        data = np.random.randn(100, dim).astype("float32")
        index.add(data)

        # Query with the first vector — it should be its own nearest neighbor
        query = data[0:1]
        distances, indices = index.search(query, 5)
        assert indices[0][0] == 0, f"First result should be index 0, got {indices[0][0]}"
        assert distances[0][0] < 1e-6, f"Distance to self should be ~0, got {distances[0][0]}"
        print(f"  Top-5 indices: {indices[0]}")
        print(f"  Top-5 distances: {distances[0]}")
        _record_score("FAISS Search", 10, "Self-retrieval correct, distances valid")

    def test_existing_faiss_index_file(self):
        """Check if a precomputed FAISS index exists on disk."""
        index_path = _APP_DIR / "data" / "vectorstore" / "index.faiss"
        if index_path.exists():
            import faiss
            index = faiss.read_index(str(index_path))
            print(f"  Existing index: {index.ntotal} vectors, dim={index.d}")
            score = 10 if index.ntotal > 500 else 8 if index.ntotal > 100 else 6
            _record_score("Existing FAISS Index", score,
                           f"{index.ntotal} vectors, dim={index.d}")
        else:
            print(f"  No precomputed index at {index_path}")
            _record_score("Existing FAISS Index", 5, "No index.faiss on disk (run precompute)")


# ============================================================================
# 6. DOCUMENT LOADING
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestDocumentLoading:
    """Test DocumentLoader for txt and json formats."""

    def test_load_txt(self):
        """DocumentLoader should read .txt files."""
        try:
            from services.ai.ml.embedding import DocumentLoader
        except ImportError:
            from app.services.ai.ml.embedding import DocumentLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("This is a test document for embedding.\nSecond line here.")
            tmp_path = f.name

        try:
            text, fmt = DocumentLoader.load_document(tmp_path)
            assert fmt == "txt"
            assert "test document" in text
            assert len(text) > 10
            _record_score("Load TXT Document", 10, f"{len(text)} chars")
        finally:
            os.unlink(tmp_path)

    def test_load_json(self):
        """DocumentLoader should read .json files."""
        try:
            from services.ai.ml.embedding import DocumentLoader
        except ImportError:
            from app.services.ai.ml.embedding import DocumentLoader

        data = {"title": "Test Doc", "content": "Financial analysis report", "score": 42}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            text, fmt = DocumentLoader.load_document(tmp_path)
            assert fmt == "json"
            assert "Financial analysis report" in text
            _record_score("Load JSON Document", 10, f"{len(text)} chars")
        finally:
            os.unlink(tmp_path)

    def test_unsupported_format_raises(self):
        """DocumentLoader should raise ValueError for unsupported formats."""
        try:
            from services.ai.ml.embedding import DocumentLoader
        except ImportError:
            from app.services.ai.ml.embedding import DocumentLoader

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                DocumentLoader.load_document(tmp_path)
            _record_score("Unsupported Format Error", 10, "ValueError raised correctly")
        finally:
            os.unlink(tmp_path)


# ============================================================================
# 7. CROSS-ENCODER RERANKER AVAILABILITY
# ============================================================================

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestRerankerAvailability:
    """Check cross-encoder reranking availability."""

    def test_cross_encoder_importable(self):
        """Check if CrossEncoder is importable from sentence_transformers."""
        try:
            from sentence_transformers import CrossEncoder
            available = True
        except ImportError:
            available = False

        if available:
            _record_score("CrossEncoder Import", 10, "CrossEncoder available")
        else:
            _record_score("CrossEncoder Import", 6,
                           "CrossEncoder not available (reranking disabled)")

    def test_embedding_service_has_reranker_attribute(self):
        """EmbeddingService should have reranker-related attributes."""
        try:
            from services.ai.ml.embedding import EmbeddingService
        except ImportError:
            from app.services.ai.ml.embedding import EmbeddingService

        svc = EmbeddingService()
        assert hasattr(svc, "reranker"), "Missing reranker attribute"
        assert hasattr(svc, "reranker_model_name"), "Missing reranker_model_name attribute"
        print(f"  reranker_model_name: {svc.reranker_model_name}")
        print(f"  reranker (before init): {svc.reranker}")
        _record_score("Reranker Attributes", 10,
                       f"model={svc.reranker_model_name}")


# ============================================================================
# STANDALONE RUNNER
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VA LLM Specialist Model — Embedding Service Unit Tests")
    print("=" * 70)
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
