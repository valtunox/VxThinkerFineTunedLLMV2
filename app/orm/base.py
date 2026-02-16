"""
ORM Base Models & Mixins for VaLLM
====================================
Read-only models matching InfinityAI's vacloudopsdb1 schema.
VaLLM does NOT create tables -- it reads from InfinityAI's existing tables.
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, BigInteger, Boolean, DateTime, Text,
    Numeric, func,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all VaLLM read-only models."""
    pass


# ---------------------------------------------------------------------------
# Reusable Mixins (same as InfinityAI for schema compatibility)
# ---------------------------------------------------------------------------

class TimestampMixin:
    """Adds created_at and updated_at columns."""
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class UUIDPrimaryKeyMixin:
    """UUID primary key."""
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class BigSerialPrimaryKeyMixin:
    """BIGSERIAL primary key."""
    id = Column(BigInteger, primary_key=True, autoincrement=True)


class ExtendedFieldsMixin:
    """Standard extended fields pattern used across InfinityAI tables."""
    extended_editable_field_1 = Column(String(255), nullable=True)
    extended_editable_field_2 = Column(String(255), nullable=True)
    extended_editable_field_3 = Column(String(255), nullable=True)
    extended_editable_field_4 = Column(String(255), nullable=True)
    extended_editable_field_5 = Column(String(255), nullable=True)
    extended_boolean_field_1 = Column(Boolean, default=True, nullable=True)
    extended_boolean_field_2 = Column(Boolean, default=True, nullable=True)
    extended_boolean_field_3 = Column(Boolean, default=True, nullable=True)
    extended_json_field_1 = Column(JSONB, nullable=True)
    extended_json_field_2 = Column(JSONB, nullable=True)
    extended_json_field_3 = Column(JSONB, nullable=True)
    extended_json_field_4 = Column(JSONB, nullable=True)


class AIModelFieldsMixin:
    """AI model editable fields."""
    ai_model_editable_field_1 = Column(String(255), nullable=True)
    ai_model_editable_field_2 = Column(JSONB, nullable=True)
    ai_model_editable_field_3 = Column(JSONB, nullable=True)
    ai_model_editable_field_4 = Column(JSONB, nullable=True)
    ai_model_editable_field_5 = Column(String(255), nullable=True)
    ai_model_editable_field_6 = Column(String(255), nullable=True)


class QueryFieldsMixin:
    """Query/search tracking fields."""
    query_terms = Column(JSONB, nullable=True)
    inputs = Column(JSONB, nullable=True)
    search_keys = Column(JSONB, nullable=True)
    search_results = Column(JSONB, nullable=True)
