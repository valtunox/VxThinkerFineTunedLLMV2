"""
VaLLM Query Schemas
====================
Request/response schemas for the VaLLM model API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# V1 Query Schemas (RAG + Reasoning)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Standard query request for VaLLM v1 endpoints."""
    query: str = Field(..., min_length=1, max_length=5000, description="The query text")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    threshold: float = Field(0.3, ge=0.0, le=1.0, description="Similarity threshold")
    include_reasoning: bool = Field(True, description="Include chain-of-thought reasoning")
    use_web_search: bool = Field(True, description="Use live web search to enrich the answer when available")
    web_search_max_results: int = Field(5, ge=1, le=10, description="Maximum live web results to use")
    workspace_id: Optional[str] = Field(None, description="Optional workspace UUID for scoped queries")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the query")


class QueryResult(BaseModel):
    """Single search result item."""
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Standard query response."""
    query: str
    answer: str = ""
    results: List[QueryResult] = []
    references: List[Dict[str, Any]] = []
    reasoning: Optional[str] = None
    confidence: float = 0.0
    model_version: str = "v1"
    processing_time_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# V1 Developer Assistant Schemas
# ---------------------------------------------------------------------------

class DeveloperQueryRequest(BaseModel):
    """Developer assistance query."""
    query: str = Field(..., min_length=1, max_length=5000)
    language: Optional[str] = Field(None, description="Programming language context")
    framework: Optional[str] = Field(None, description="Framework context (e.g., terraform, kubernetes)")
    use_web_search: bool = Field(True, description="Use live web search to enrich the answer when available")
    context: Optional[Dict[str, Any]] = None


class DeveloperQueryResponse(BaseModel):
    """Developer assistance response."""
    query: str
    answer: str
    code_snippets: List[str] = []
    references: List[Dict[str, Any]] = []
    confidence: float = 0.0
    processing_time_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# V1 Terminal Assistant Schemas
# ---------------------------------------------------------------------------

class TerminalQueryRequest(BaseModel):
    """Terminal/CLI assistance query."""
    query: str = Field(..., min_length=1, max_length=5000)
    shell: Optional[str] = Field("bash", description="Shell type (bash, zsh, powershell)")
    os_type: Optional[str] = Field("linux", description="Operating system")
    use_web_search: bool = Field(True, description="Use live web search to enrich the answer when available")
    context: Optional[Dict[str, Any]] = None


class TerminalQueryResponse(BaseModel):
    """Terminal/CLI assistance response."""
    query: str
    commands: List[str] = []
    explanation: str = ""
    warnings: List[str] = []
    confidence: float = 0.0
    processing_time_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# V2 NLP Schemas
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    """Entity extraction request."""
    text: str = Field(..., min_length=1, max_length=10000)
    entity_types: Optional[List[str]] = Field(None, description="Specific entity types to extract")


class ExtractResponse(BaseModel):
    """Entity extraction response."""
    entities: List[Dict[str, Any]] = []
    text_length: int = 0
    processing_time_ms: Optional[int] = None


# ---------------------------------------------------------------------------
# V3 Incident Pattern Schemas
# ---------------------------------------------------------------------------

class IncidentQueryRequest(BaseModel):
    """Cloud/DevOps incident pattern query."""
    query: str = Field(..., min_length=1, max_length=5000)
    metrics: Optional[Dict[str, Any]] = Field(None, description="System metrics for anomaly detection")
    severity: Optional[str] = Field(None, description="Incident severity level")


class IncidentQueryResponse(BaseModel):
    """Incident pattern analysis response."""
    query: str
    patterns: List[Dict[str, Any]] = []
    risk_assessment: Optional[Dict[str, Any]] = None
    recommended_actions: List[str] = []
    confidence: float = 0.0
    processing_time_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Health / Stats Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool = False
    vectorstore_loaded: bool = False
    database_connected: bool = False
    version: str = "1.0.0"
    uptime_seconds: Optional[float] = None


class StatsResponse(BaseModel):
    """System stats response."""
    total_vectors: int = 0
    total_documents: int = 0
    model_info: Optional[Dict[str, Any]] = None
    cache_stats: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
