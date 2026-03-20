"""
ORM Models for VaLLM Specialist Model
=======================================
Lightweight schema for document analysis, verification, and financial AI.

Tables (8 total):
  - tenants                  : Multi-tenant isolation
  - sessions                 : User/agent session tracking
  - documents                : Document records (invoices, receipts, contracts, etc.)
  - document_chunks          : Chunked content for RAG embeddings
  - document_embeddings      : Vector references + metadata for FAISS search
  - verification_records     : Audit trail for document verification
  - transactions             : Financial transactions extracted from documents
  - business_recommendations : AI-generated insights

Author: Joel Otepa Wembo
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, BigInteger, Boolean, DateTime, Text,
    Numeric, Float, ForeignKey, func, Index,
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID

from app.orm.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


# ---------------------------------------------------------------------------
# 1. Tenant
# ---------------------------------------------------------------------------

class Tenant(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Multi-tenant isolation. Each tenant has its own documents, sessions,
    and financial data. Mirrors the tenants table from the Django admin.
    """

    __tablename__ = "tenants"

    name = Column(String(255), nullable=False)
    slug = Column(String(100), nullable=False, unique=True, index=True)
    is_active = Column(Boolean, nullable=False, default=True)

    # Subscription / billing tier
    plan = Column(String(50), nullable=False, server_default="free")  # free, starter, pro, enterprise
    max_documents = Column(Integer, nullable=False, server_default="100")
    max_storage_mb = Column(Integer, nullable=False, server_default="500")

    # Contact
    contact_email = Column(String(254), nullable=True)
    contact_name = Column(String(255), nullable=True)

    # Settings
    settings = Column(JSONB, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)

    # Relationships
    sessions = relationship("Session", back_populates="tenant", lazy="dynamic")
    documents = relationship("Document", back_populates="tenant", lazy="dynamic")
    transactions = relationship("Transaction", back_populates="tenant", lazy="dynamic")
    recommendations = relationship("BusinessRecommendation", back_populates="tenant", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<Tenant(id={self.id}, name={self.name!r}, slug={self.slug!r})>"


# ---------------------------------------------------------------------------
# 2. Session
# ---------------------------------------------------------------------------

class Session(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Tracks user or agent sessions. Each session belongs to a tenant
    and can be linked to documents processed during it.
    """

    __tablename__ = "sessions"

    tenant_id = Column(PG_UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)

    # Who / what initiated the session
    session_type = Column(String(30), nullable=False, server_default="user")  # user, agent, api, system
    external_user_id = Column(String(255), nullable=True)  # caller's user ID (opaque string)
    agent_name = Column(String(100), nullable=True)

    # State
    status = Column(String(20), nullable=False, server_default="active")  # active, completed, expired, failed
    started_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)

    # Context
    context = Column(JSONB, nullable=True)  # arbitrary session context
    metadata_ = Column("metadata", JSONB, nullable=True)

    # Stats
    document_count = Column(Integer, nullable=False, server_default="0")
    query_count = Column(Integer, nullable=False, server_default="0")

    # Relationships
    tenant = relationship("Tenant", back_populates="sessions")

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, tenant_id={self.tenant_id}, type={self.session_type!r})>"


# ---------------------------------------------------------------------------
# 3. Document
# ---------------------------------------------------------------------------

class Document(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Document record for verification, billing, and financial analysis."""

    __tablename__ = "documents"

    tenant_id = Column(PG_UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    session_id = Column(PG_UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=True, index=True)

    # Document identity
    document_type = Column(String(50), nullable=False, server_default="general")
    # Types: invoice, receipt, contract, identity, financial_statement, report, general
    title = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)

    # File info
    file_path = Column(String(1024), nullable=True)
    file_name = Column(String(500), nullable=True)
    file_size_bytes = Column(BigInteger, nullable=True)
    mime_type = Column(String(128), nullable=True)

    # Extracted content
    content_text = Column(Text, nullable=True)
    content_hash = Column(String(128), nullable=True)
    language = Column(String(10), nullable=True)
    page_count = Column(Integer, nullable=True)

    # Processing status
    extraction_status = Column(String(30), nullable=False, server_default="pending")
    # Status: pending, processing, completed, failed
    extraction_metadata = Column(JSONB, nullable=True)

    # Verification
    verification_status = Column(String(30), nullable=True)
    # Status: pending, verified, rejected, flagged
    confidence_score = Column(Numeric(5, 4), nullable=True)

    # Flexible fields
    tags = Column(JSONB, nullable=True)
    custom_fields = Column(JSONB, nullable=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan", lazy="dynamic")
    verification_records = relationship("VerificationRecord", back_populates="document", cascade="all, delete-orphan", lazy="dynamic")
    transactions = relationship("Transaction", back_populates="document", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, type={self.document_type!r}, title={self.title!r})>"


# ---------------------------------------------------------------------------
# 4. DocumentChunk (NEW — for RAG pipeline)
# ---------------------------------------------------------------------------

class DocumentChunk(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Chunked content from a document for RAG embeddings.
    Each document is split into chunks (paragraphs, sections, pages)
    that are individually embedded and indexed in FAISS.
    """

    __tablename__ = "document_chunks"

    document_id = Column(PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    tenant_id = Column(PG_UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)

    # Chunk identity
    chunk_index = Column(Integer, nullable=False)  # order within document (0-based)
    content_text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    char_count = Column(Integer, nullable=True)

    # Source location within document
    page_number = Column(Integer, nullable=True)
    section_title = Column(String(500), nullable=True)
    start_offset = Column(Integer, nullable=True)  # char offset in full doc text
    end_offset = Column(Integer, nullable=True)

    # Chunking strategy
    chunk_strategy = Column(String(50), nullable=True)  # fixed_size, paragraph, page, semantic
    overlap_tokens = Column(Integer, nullable=True)

    # Metadata
    content_hash = Column(String(128), nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)

    # Relationships
    document = relationship("Document", back_populates="chunks")
    embedding = relationship("DocumentEmbedding", back_populates="chunk", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_document_chunks_doc_idx", "document_id", "chunk_index", unique=True),
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk(id={self.id}, doc={self.document_id}, idx={self.chunk_index})>"


# ---------------------------------------------------------------------------
# 5. DocumentEmbedding (NEW — FAISS vector references)
# ---------------------------------------------------------------------------

class DocumentEmbedding(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Maps document chunks to FAISS vector index entries.
    Stores the FAISS index offset and embedding metadata so we can
    trace search results back to their source document and chunk.
    """

    __tablename__ = "document_embeddings"

    chunk_id = Column(PG_UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    document_id = Column(PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    tenant_id = Column(PG_UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)

    # FAISS reference
    faiss_index_id = Column(Integer, nullable=False)  # integer offset in FAISS index
    faiss_index_name = Column(String(100), nullable=False, server_default="default")  # supports multiple indices

    # Embedding info
    embedding_model = Column(String(100), nullable=False, server_default="BAAI/bge-large-en-v1.5")
    embedding_dim = Column(Integer, nullable=False, server_default="1024")
    embedding_hash = Column(String(128), nullable=True)  # hash of the vector for dedup

    # Search metadata
    content_preview = Column(String(500), nullable=True)  # first N chars for display
    metadata_ = Column("metadata", JSONB, nullable=True)

    # Relationships
    chunk = relationship("DocumentChunk", back_populates="embedding")

    __table_args__ = (
        Index("ix_doc_embeddings_faiss", "faiss_index_name", "faiss_index_id", unique=True),
    )

    def __repr__(self) -> str:
        return f"<DocumentEmbedding(id={self.id}, chunk={self.chunk_id}, faiss_idx={self.faiss_index_id})>"


# ---------------------------------------------------------------------------
# 6. VerificationRecord (audit trail)
# ---------------------------------------------------------------------------

class VerificationRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Audit trail for document verification results."""

    __tablename__ = "verification_records"

    document_id = Column(PG_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    tenant_id = Column(PG_UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)

    verification_type = Column(String(50), nullable=False)
    # Types: authenticity, completeness, compliance, fraud_check
    status = Column(String(30), nullable=False, server_default="pending")
    # Status: pending, passed, failed, flagged
    confidence_score = Column(Numeric(5, 4), nullable=True)
    findings = Column(JSONB, nullable=True)
    risk_flags = Column(JSONB, nullable=True)
    notes = Column(Text, nullable=True)

    verification_method = Column(String(50), nullable=True)  # ai, manual, hybrid
    model_version = Column(String(50), nullable=True)
    processing_time_ms = Column(Integer, nullable=True)

    # Relationships
    document = relationship("Document", back_populates="verification_records")

    def __repr__(self) -> str:
        return f"<VerificationRecord(id={self.id}, doc={self.document_id}, status={self.status!r})>"


# ---------------------------------------------------------------------------
# 7. Transaction (billing / accounting)
# ---------------------------------------------------------------------------

class Transaction(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Financial transaction extracted from documents or entered manually."""

    __tablename__ = "transactions"

    tenant_id = Column(PG_UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    document_id = Column(PG_UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True, index=True)

    transaction_type = Column(String(50), nullable=False)
    # Types: income, expense, transfer, invoice, payment
    amount = Column(Numeric(18, 4), nullable=False)
    currency = Column(String(3), nullable=False, server_default="USD")
    status = Column(String(30), nullable=False, server_default="pending")
    # Status: pending, completed, reconciled, disputed

    reference_number = Column(String(255), nullable=True)
    counterparty_name = Column(String(500), nullable=True)
    category = Column(String(100), nullable=True)
    subcategory = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    line_items = Column(JSONB, nullable=True)
    tax_amount = Column(Numeric(18, 4), nullable=True)
    payment_method = Column(String(50), nullable=True)
    due_date = Column(DateTime(timezone=True), nullable=True)
    paid_at = Column(DateTime(timezone=True), nullable=True)
    reconciliation_status = Column(String(30), nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="transactions")
    document = relationship("Document", back_populates="transactions")

    def __repr__(self) -> str:
        return f"<Transaction(id={self.id}, type={self.transaction_type!r}, amount={self.amount})>"


# ---------------------------------------------------------------------------
# 8. BusinessRecommendation (AI insights)
# ---------------------------------------------------------------------------

class BusinessRecommendation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """AI-generated business recommendation / insight."""

    __tablename__ = "business_recommendations"

    tenant_id = Column(PG_UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)

    recommendation_type = Column(String(50), nullable=False)
    # Types: cost_optimization, risk_alert, growth_opportunity, compliance
    title = Column(String(500), nullable=False)
    summary = Column(Text, nullable=True)
    details = Column(JSONB, nullable=True)
    impact_score = Column(Numeric(5, 4), nullable=True)
    confidence = Column(Numeric(5, 4), nullable=True)
    priority = Column(String(20), nullable=False, server_default="medium")
    # Priority: low, medium, high, critical
    status = Column(String(30), nullable=False, server_default="active")
    # Status: active, implemented, dismissed, expired

    data_sources = Column(JSONB, nullable=True)
    model_version = Column(String(50), nullable=True)
    valid_until = Column(DateTime(timezone=True), nullable=True)
    implemented_at = Column(DateTime(timezone=True), nullable=True)
    feedback = Column(JSONB, nullable=True)

    # Relationships
    tenant = relationship("Tenant", back_populates="recommendations")

    def __repr__(self) -> str:
        return f"<BusinessRecommendation(id={self.id}, type={self.recommendation_type!r})>"
