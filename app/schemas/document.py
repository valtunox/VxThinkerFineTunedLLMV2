"""
VaLLM Specialist Model - Document, Chunk, Embedding & Financial Schemas.

Author: Joel Otepa Wembo
https://joelwembo.com

Pydantic schemas for document management, RAG chunking, verification,
billing analysis, financial transactions, and business recommendations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tenant schemas
# ---------------------------------------------------------------------------

class TenantCreate(BaseModel):
    name: str
    slug: str
    plan: str = "free"
    contact_email: Optional[str] = None
    contact_name: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class TenantResponse(BaseModel):
    id: str
    name: str
    slug: str
    is_active: bool = True
    plan: str = "free"
    max_documents: int = 100
    max_storage_mb: int = 500
    contact_email: Optional[str] = None
    contact_name: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Session schemas
# ---------------------------------------------------------------------------

class SessionCreate(BaseModel):
    tenant_id: str
    session_type: str = "user"
    external_user_id: Optional[str] = None
    agent_name: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    id: str
    tenant_id: str
    session_type: str
    external_user_id: Optional[str] = None
    agent_name: Optional[str] = None
    status: str = "active"
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    document_count: int = 0
    query_count: int = 0
    context: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Document schemas
# ---------------------------------------------------------------------------

class DocumentCreate(BaseModel):
    tenant_id: str
    session_id: Optional[str] = None
    document_type: str = Field(default="general", description="invoice, receipt, contract, identity, financial_statement, report, general")
    title: Optional[str] = None
    description: Optional[str] = None
    file_name: Optional[str] = None
    mime_type: Optional[str] = None
    language: Optional[str] = None
    tags: Optional[List[str]] = None
    custom_fields: Optional[Dict[str, Any]] = None


class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    document_type: Optional[str] = None
    verification_status: Optional[str] = None
    tags: Optional[List[str]] = None
    custom_fields: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    id: str
    tenant_id: str
    session_id: Optional[str] = None
    document_type: str
    title: Optional[str] = None
    description: Optional[str] = None
    file_name: Optional[str] = None
    file_size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    language: Optional[str] = None
    page_count: Optional[int] = None
    extraction_status: str = "pending"
    verification_status: Optional[str] = None
    confidence_score: Optional[float] = None
    tags: Optional[List[str]] = None
    custom_fields: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    success: bool = True
    count: int
    page: int = 1
    page_size: int = 20
    data: List[DocumentResponse]


# ---------------------------------------------------------------------------
# DocumentChunk schemas
# ---------------------------------------------------------------------------

class DocumentChunkCreate(BaseModel):
    document_id: str
    tenant_id: str
    chunk_index: int
    content_text: str
    token_count: Optional[int] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chunk_strategy: Optional[str] = "fixed_size"
    overlap_tokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentChunkResponse(BaseModel):
    id: str
    document_id: str
    tenant_id: str
    chunk_index: int
    content_text: str
    token_count: Optional[int] = None
    char_count: Optional[int] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chunk_strategy: Optional[str] = None
    content_hash: Optional[str] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class DocumentChunkListResponse(BaseModel):
    success: bool = True
    count: int
    document_id: str
    data: List[DocumentChunkResponse]


# ---------------------------------------------------------------------------
# DocumentEmbedding schemas
# ---------------------------------------------------------------------------

class DocumentEmbeddingCreate(BaseModel):
    chunk_id: str
    document_id: str
    tenant_id: str
    faiss_index_id: int
    faiss_index_name: str = "default"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    content_preview: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentEmbeddingResponse(BaseModel):
    id: str
    chunk_id: str
    document_id: str
    tenant_id: str
    faiss_index_id: int
    faiss_index_name: str
    embedding_model: str
    embedding_dim: int
    content_preview: Optional[str] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Verification schemas
# ---------------------------------------------------------------------------

class VerificationCreate(BaseModel):
    document_id: str
    verification_type: str = Field(description="authenticity, completeness, compliance, fraud_check")
    notes: Optional[str] = None
    verification_method: str = "ai"


class VerificationResponse(BaseModel):
    id: str
    document_id: str
    tenant_id: str
    verification_type: str
    status: str
    confidence_score: Optional[float] = None
    findings: Optional[Dict[str, Any]] = None
    risk_flags: Optional[List[str]] = None
    notes: Optional[str] = None
    verification_method: Optional[str] = None
    model_version: Optional[str] = None
    processing_time_ms: Optional[int] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Transaction schemas
# ---------------------------------------------------------------------------

class TransactionCreate(BaseModel):
    tenant_id: str
    transaction_type: str = Field(description="income, expense, transfer, invoice, payment")
    amount: float
    currency: str = "USD"
    reference_number: Optional[str] = None
    counterparty_name: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    description: Optional[str] = None
    line_items: Optional[List[Dict[str, Any]]] = None
    tax_amount: Optional[float] = None
    payment_method: Optional[str] = None
    due_date: Optional[datetime] = None
    document_id: Optional[str] = None


class TransactionUpdate(BaseModel):
    status: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    description: Optional[str] = None
    reconciliation_status: Optional[str] = None
    paid_at: Optional[datetime] = None


class TransactionResponse(BaseModel):
    id: str
    tenant_id: str
    document_id: Optional[str] = None
    transaction_type: str
    amount: float
    currency: str
    status: str
    reference_number: Optional[str] = None
    counterparty_name: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    description: Optional[str] = None
    line_items: Optional[List[Dict[str, Any]]] = None
    tax_amount: Optional[float] = None
    payment_method: Optional[str] = None
    due_date: Optional[datetime] = None
    paid_at: Optional[datetime] = None
    reconciliation_status: Optional[str] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class TransactionListResponse(BaseModel):
    success: bool = True
    count: int
    page: int = 1
    page_size: int = 20
    data: List[TransactionResponse]


# ---------------------------------------------------------------------------
# Business Recommendation schemas
# ---------------------------------------------------------------------------

class BusinessRecommendationResponse(BaseModel):
    id: str
    tenant_id: str
    recommendation_type: str
    title: str
    summary: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    impact_score: Optional[float] = None
    confidence: Optional[float] = None
    priority: str = "medium"
    status: str = "active"
    data_sources: Optional[List[str]] = None
    model_version: Optional[str] = None
    valid_until: Optional[datetime] = None
    implemented_at: Optional[datetime] = None
    feedback: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class BusinessRecommendationListResponse(BaseModel):
    success: bool = True
    count: int
    data: List[BusinessRecommendationResponse]


# ---------------------------------------------------------------------------
# Request/Response schemas for API endpoints
# ---------------------------------------------------------------------------

class DocumentVerifyRequest(BaseModel):
    document_id: str
    verification_types: List[str] = Field(default=["authenticity"], description="Types of verification to perform")
    notes: Optional[str] = None


class DocumentVerifyResponse(BaseModel):
    success: bool
    document_id: str
    verifications: List[VerificationResponse]
    overall_status: str
    overall_confidence: Optional[float] = None


class BillingAnalysisRequest(BaseModel):
    document_ids: List[str] = Field(description="Document IDs to analyze")
    analysis_type: str = "full"  # full, summary, line_items, tax


class BillingAnalysisResponse(BaseModel):
    success: bool
    transactions: List[TransactionResponse]
    summary: Dict[str, Any] = Field(default_factory=dict)
    total_amount: Optional[float] = None
    currency: str = "USD"


class FinancialSummaryResponse(BaseModel):
    success: bool
    tenant_id: Optional[str] = None
    period: Optional[str] = None
    total_income: float = 0.0
    total_expenses: float = 0.0
    net_amount: float = 0.0
    currency: str = "USD"
    transaction_count: int = 0
    categories: Dict[str, float] = Field(default_factory=dict)
    recommendations: List[BusinessRecommendationResponse] = Field(default_factory=list)
