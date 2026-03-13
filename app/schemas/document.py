"""
VaLLM Specialist Model - Document, Accounting, Excel,  & Financial Schemas.

Author: Joel Otepa Wembo
https://joelwembo.com

Pydantic schemas for document verification, billing analysis,
financial transactions, and business recommendations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Document schemas
# ---------------------------------------------------------------------------

class DocumentCreate(BaseModel):
    document_type: str = Field(default="general", description="invoice, receipt, contract, identity, financial_statement")
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
    organization_id: Optional[str] = None
    user_id: Optional[int] = None
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
    organization_id: Optional[str] = None
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
    organization_id: Optional[str] = None
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
    organization_id: Optional[str] = None
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
    organization_id: Optional[str] = None
    period: Optional[str] = None
    total_income: float = 0.0
    total_expenses: float = 0.0
    net_amount: float = 0.0
    currency: str = "USD"
    transaction_count: int = 0
    categories: Dict[str, float] = Field(default_factory=dict)
    recommendations: List[BusinessRecommendationResponse] = Field(default_factory=list)
