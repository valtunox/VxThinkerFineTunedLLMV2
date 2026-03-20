"""
Authentication Schemas for VaLLM Specialist Model
===================================================
Simplified — tenant-based auth only, no user/API key models.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class TenantContext(BaseModel):
    """Tenant context extracted from request headers or token."""
    tenant_id: str
    tenant_slug: Optional[str] = None
    session_id: Optional[str] = None
    external_user_id: Optional[str] = None


class AuthResponse(BaseModel):
    """Generic auth check response."""
    authenticated: bool = False
    tenant_id: Optional[str] = None
    message: str = ""
