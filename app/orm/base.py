"""
ORM Base Models & Mixins for VaLLM Specialist Model
=====================================================
Lightweight base with only the mixins used by the specialist tables.
"""

import uuid
from sqlalchemy import Column, BigInteger, Boolean, DateTime, func
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all specialist model tables."""
    pass


# ---------------------------------------------------------------------------
# Reusable Mixins
# ---------------------------------------------------------------------------

class TimestampMixin:
    """Adds created_at and updated_at columns."""
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class UUIDPrimaryKeyMixin:
    """UUID primary key."""
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
