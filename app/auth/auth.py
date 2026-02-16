"""
VaLLM Authentication via X-API-Key
====================================
Validates incoming requests against InfinityAI's users_apikey table.
Only InfinityAI (or authorized services) should call VaLLM directly.

Usage in routes:
    from app.auth.auth import require_api_key, get_caller_identity

    @router.post("/query")
    async def query(
        request: QueryRequest,
        caller: CallerIdentity = Depends(get_caller_identity),
    ):
        ...
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Security, Request
from fastapi.security import APIKeyHeader

logger = logging.getLogger("vallm")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Optional: hardcoded internal service key for InfinityAI-to-VaLLM calls.
# Set VALLM_INTERNAL_SERVICE_KEY in env to enable bypass of DB lookup.
INTERNAL_SERVICE_KEY = os.getenv("VALLM_INTERNAL_SERVICE_KEY")

# Set VALLM_AUTH_DISABLED=true in dev to skip all auth checks.
AUTH_DISABLED = os.getenv("VALLM_AUTH_DISABLED", "false").lower() == "true"


def _hash_api_key(raw_key: str) -> str:
    """Hash a raw API key with SHA-256 to compare against token_hash in DB."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# FastAPI Dependencies
# ---------------------------------------------------------------------------

async def require_api_key(
    request: Request,
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> Optional[str]:
    """
    Validate the X-API-Key header.

    Returns the raw API key string on success.
    Raises 401 if missing or invalid.

    Supports three modes:
    1. AUTH_DISABLED=true  -> skip validation entirely (dev only)
    2. INTERNAL_SERVICE_KEY -> bypass DB if key matches (service-to-service)
    3. DB lookup           -> hash key and check users_apikey table
    """
    # Dev bypass
    if AUTH_DISABLED:
        return api_key or "dev-bypass"

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Internal service key bypass (InfinityAI -> VaLLM)
    if INTERNAL_SERVICE_KEY and api_key == INTERNAL_SERVICE_KEY:
        return api_key

    # Database validation
    try:
        from app.orm.session import SessionLocal
        from app.orm.models import APIKey

        token_hash = _hash_api_key(api_key)
        db = SessionLocal()
        try:
            key_record = (
                db.query(APIKey)
                .filter(
                    APIKey.token_hash == token_hash,
                    APIKey.is_active == True,
                )
                .first()
            )
            if not key_record:
                raise HTTPException(status_code=401, detail="Invalid API key")

            if key_record.is_expired:
                raise HTTPException(status_code=401, detail="API key has expired")

            # Check if "vallm" service is allowed (if scoping is configured)
            if key_record.disallowed_services:
                disallowed = key_record.disallowed_services
                if isinstance(disallowed, list) and "vallm" in disallowed:
                    raise HTTPException(status_code=403, detail="API key does not have access to VaLLM")
                elif isinstance(disallowed, dict) and disallowed.get("vallm"):
                    raise HTTPException(status_code=403, detail="API key does not have access to VaLLM")

            return api_key
        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"API key validation failed (DB error): {e}")
        # If DB is unavailable, reject the request
        raise HTTPException(status_code=503, detail="Authentication service unavailable")


async def get_caller_identity(
    request: Request,
    api_key: str = Depends(require_api_key),
):
    """
    Resolve the full caller identity from the validated API key.
    Returns a CallerIdentity schema with user, org, and key details.
    """
    from app.schemas.auth import CallerIdentity

    # Dev bypass returns a placeholder identity
    if AUTH_DISABLED:
        return CallerIdentity(
            user_id=0,
            username="dev-user",
            email="dev@localhost",
            full_name="Development User",
            api_key_name="dev-bypass",
            api_key_environment="DEVELOPMENT",
        )

    # Internal service key returns a service identity
    if INTERNAL_SERVICE_KEY and api_key == INTERNAL_SERVICE_KEY:
        return CallerIdentity(
            user_id=0,
            username="infinityai-service",
            email="service@infinityai.local",
            full_name="InfinityAI Service",
            api_key_name="internal-service-key",
            api_key_environment="PRODUCTION",
            is_superuser=True,
        )

    # Full DB lookup
    try:
        from app.orm.session import SessionLocal
        from app.orm.models import APIKey, User, Organization

        token_hash = _hash_api_key(api_key)
        db = SessionLocal()
        try:
            key_record = (
                db.query(APIKey)
                .filter(APIKey.token_hash == token_hash, APIKey.is_active == True)
                .first()
            )
            if not key_record:
                raise HTTPException(status_code=401, detail="Invalid API key")

            user = db.query(User).filter(User.id == key_record.user_id).first()
            if not user:
                raise HTTPException(status_code=401, detail="API key owner not found")

            org_name = None
            if user.organization_id:
                org = db.query(Organization).filter(Organization.id == user.organization_id).first()
                if org:
                    org_name = org.name

            return CallerIdentity(
                user_id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                company_name=user.company_name,
                organization_id=str(user.organization_id) if user.organization_id else None,
                organization_name=org_name,
                is_superuser=user.is_superuser,
                is_staff=user.is_staff,
                is_verified=user.is_verified,
                api_key_name=key_record.name,
                api_key_environment=key_record.environment,
                scopes=key_record.scopes,
            )
        finally:
            db.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Caller identity resolution failed: {e}")
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
