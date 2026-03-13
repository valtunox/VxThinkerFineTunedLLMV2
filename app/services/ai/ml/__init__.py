"""
VaLLM Specialist Model - ML / LLM Services.

Author: Joel Otepa Wembo
https://joelwembo.com

Package init: exposes direct OpenAI, embeddings, explainability, vector store,
ingestion, and fraud detection services for document analysis and business intelligence.
"""

try:
    from .direct_openai import direct_openai
except ImportError:
    direct_openai = None

try:
    from .embedding import embedding_service
except ImportError:
    embedding_service = None

try:
    from .vector_adapter import vector_store
except ImportError:
    vector_store = None

try:
    from .explainability import ExplainabilityService, explainability_service
except ImportError:
    ExplainabilityService = None
    explainability_service = None

try:
    from .ingestion import IngestionService, ingestion_service
except ImportError:
    IngestionService = None
    ingestion_service = None

try:
    from .fraud_detection import FraudDetectionService, fraud_detection_service
except ImportError:
    FraudDetectionService = None
    fraud_detection_service = None

__all__ = [
    "direct_openai",
    "embedding_service",
    "vector_store",
    "ExplainabilityService",
    "explainability_service",
    "IngestionService",
    "ingestion_service",
    "FraudDetectionService",
    "fraud_detection_service",
]
