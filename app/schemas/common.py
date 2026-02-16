"""
Common Pydantic Schemas for VaLLM
==================================
Base schemas, enums, and shared response types.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EnvironmentEnum(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"


# ---------------------------------------------------------------------------
# Base Schemas
# ---------------------------------------------------------------------------

class BaseSchema(BaseModel):
    """Base for all read/response schemas."""
    model_config = {"from_attributes": True}


class TimestampSchema(BaseModel):
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Standard Responses
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    code: Optional[int] = None
    details: Optional[Any] = None
    timestamp: Optional[datetime] = None
    request_id: Optional[str] = None


class SuccessResponse(BaseModel):
    message: str
    data: Optional[Any] = None
    timestamp: Optional[datetime] = None
