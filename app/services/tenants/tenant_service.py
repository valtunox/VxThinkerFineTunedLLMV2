"""
Tenant Management Service
=========================

FastAPI router for multi-tenant organization management.
Provides CRUD endpoints for the tenants table.
"""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import logging
import sys
import uuid
import re
from pathlib import Path
from datetime import datetime

# Add app directory to path for imports
app_dir = Path(__file__).parent.parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from core.db import get_db_connection
from auth.oauth import decode_access_token

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/tenants",
    tags=["Tenants"]
)

security = HTTPBearer(auto_error=False)

TENANT_COLUMNS = (
    "id, tenant_id, tenant_name, tenant_slug, tenant_type, "
    "contact_email, contact_phone, billing_email, "
    "address_line_1, address_line_2, city, state, postal_code, country, "
    "is_active, is_verified, status, "
    "max_users, max_workspaces, max_documents, max_verifications, storage_limit_mb, "
    "logo_url, primary_color, custom_domain, "
    "settings, metadata, created_at, updated_at, activated_at, tenant_group"
)


def _row_to_dict(row) -> dict:
    """Convert a tenant row tuple to a dict."""
    if not row:
        return None
    keys = [
        "id", "tenant_id", "tenant_name", "tenant_slug", "tenant_type",
        "contact_email", "contact_phone", "billing_email",
        "address_line_1", "address_line_2", "city", "state", "postal_code", "country",
        "is_active", "is_verified", "status",
        "max_users", "max_workspaces", "max_documents", "max_verifications", "storage_limit_mb",
        "logo_url", "primary_color", "custom_domain",
        "settings", "metadata", "created_at", "updated_at", "activated_at", "tenant_group",
    ]
    d = {}
    for i, key in enumerate(keys):
        val = row[i] if i < len(row) else None
        if isinstance(val, datetime):
            val = val.isoformat()
        d[key] = val
    return d


# ============================================================================
# Static routes FIRST (before {tenant_id} param route)
# ============================================================================

@router.get("/my-tenant")
async def get_my_tenant(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get the current authenticated user's tenant information."""
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT tenant_name, tenant_group FROM users WHERE username = %s", (username,))
        user_row = cursor.fetchone()

        if not user_row or not user_row[0]:
            cursor.close()
            conn.close()
            return {"success": True, "tenant": None, "message": "No tenant assigned to this user"}

        tenant_name = user_row[0]

        cursor.execute(f"SELECT {TENANT_COLUMNS} FROM tenants WHERE tenant_name = %s LIMIT 1", (tenant_name,))
        tenant_row = cursor.fetchone()
        cursor.close()
        conn.close()

        if tenant_row:
            return {"success": True, "tenant": _row_to_dict(tenant_row)}

        return {
            "success": True,
            "tenant": {"tenant_name": tenant_name, "tenant_group": user_row[1]},
            "message": "User has tenant_name but no matching record in tenants table"
        }

    except Exception as e:
        logger.error(f"Error fetching user tenant: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch tenant")


@router.get("/stats")
async def tenant_stats():
    """Get tenant statistics for dashboard display."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE is_active = true) as active,
                COUNT(*) FILTER (WHERE is_active = false) as inactive,
                COUNT(*) FILTER (WHERE status = 'suspended') as suspended,
                COUNT(*) FILTER (WHERE is_verified = true) as verified
            FROM tenants
        """)
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        return {
            "success": True,
            "stats": {
                "total_tenants": row[0] if row else 0,
                "active_tenants": row[1] if row else 0,
                "inactive_tenants": row[2] if row else 0,
                "suspended_tenants": row[3] if row else 0,
                "verified_tenants": row[4] if row else 0,
            }
        }
    except Exception as e:
        logger.error(f"Error fetching tenant stats: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch stats")


# ============================================================================
# CRUD routes
# ============================================================================

@router.get("/")
async def list_tenants(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    tenant_group: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
):
    """List all tenants with optional filtering and pagination."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        where_clauses = []
        params = []

        if tenant_group:
            where_clauses.append("tenant_group = %s")
            params.append(tenant_group)
        if status_filter:
            where_clauses.append("status = %s")
            params.append(status_filter)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Count
        cursor.execute(f"SELECT COUNT(*) FROM tenants {where_sql}", params)
        total = cursor.fetchone()[0]

        # Paginate
        offset = (page - 1) * page_size
        params_page = params + [page_size, offset]
        cursor.execute(
            f"SELECT {TENANT_COLUMNS} FROM tenants {where_sql} ORDER BY created_at DESC LIMIT %s OFFSET %s",
            params_page
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        return {
            "success": True,
            "count": total,
            "page": page,
            "page_size": page_size,
            "data": [_row_to_dict(r) for r in rows],
        }
    except Exception as e:
        logger.error(f"Error listing tenants: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list tenants")


@router.get("/{tenant_id}")
async def get_tenant(tenant_id: str):
    """Get a single tenant by its integer ID or UUID tenant_id."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if tenant_id.isdigit():
            cursor.execute(f"SELECT {TENANT_COLUMNS} FROM tenants WHERE id = %s", (int(tenant_id),))
        else:
            cursor.execute(f"SELECT {TENANT_COLUMNS} FROM tenants WHERE tenant_id = %s", (tenant_id,))

        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

        return {"success": True, "data": _row_to_dict(row)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching tenant: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch tenant")


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_tenant(
    tenant_name: str,
    contact_email: str,
    tenant_type: str = "standard",
    tenant_slug: Optional[str] = None,
    contact_phone: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    country: str = "Canada",
    max_users: int = 5,
    max_workspaces: int = 3,
):
    """Create a new tenant / organization."""
    tenant_id = uuid.uuid4().hex[:16]
    if not tenant_slug:
        tenant_slug = re.sub(r'[^a-z0-9-]', '-', tenant_name.lower().strip())
        tenant_slug = re.sub(r'-+', '-', tenant_slug).strip('-')

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(f"""
            INSERT INTO tenants (
                tenant_id, tenant_name, tenant_slug, tenant_type,
                contact_email, contact_phone, city, state, country,
                is_active, is_verified, status,
                max_users, max_workspaces, max_documents, max_verifications, storage_limit_mb,
                created_at, updated_at, tenant_group
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                true, false, 'active',
                %s, %s, 50000, 5000, 1000,
                NOW(), NOW(), %s
            )
            RETURNING {TENANT_COLUMNS}
        """, (
            tenant_id, tenant_name, tenant_slug, tenant_type,
            contact_email, contact_phone, city, state, country,
            max_users, max_workspaces, tenant_slug,
        ))

        row = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create tenant")

        logger.info(f"Created tenant: {tenant_name} ({tenant_id})")
        return {"success": True, "data": _row_to_dict(row)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating tenant: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create tenant: {e}")


@router.put("/{tenant_id}")
async def update_tenant(
    tenant_id: str,
    tenant_name: Optional[str] = None,
    contact_email: Optional[str] = None,
    contact_phone: Optional[str] = None,
    tenant_type: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    country: Optional[str] = None,
    is_active: Optional[bool] = None,
    status_val: Optional[str] = Query(None, alias="status"),
    max_users: Optional[int] = None,
    logo_url: Optional[str] = None,
    primary_color: Optional[str] = None,
):
    """Update a tenant's information."""
    update_fields = []
    update_values = []

    field_map = {
        "tenant_name": tenant_name, "contact_email": contact_email,
        "contact_phone": contact_phone, "tenant_type": tenant_type,
        "city": city, "state": state, "country": country,
        "is_active": is_active, "status": status_val,
        "max_users": max_users, "logo_url": logo_url, "primary_color": primary_color,
    }

    for col, val in field_map.items():
        if val is not None:
            update_fields.append(f"{col} = %s")
            update_values.append(val)

    if not update_fields:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update")

    update_fields.append("updated_at = NOW()")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if tenant_id.isdigit():
            where = "id = %s"
            update_values.append(int(tenant_id))
        else:
            where = "tenant_id = %s"
            update_values.append(tenant_id)

        cursor.execute(
            f"UPDATE tenants SET {', '.join(update_fields)} WHERE {where} RETURNING {TENANT_COLUMNS}",
            update_values
        )
        row = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

        return {"success": True, "data": _row_to_dict(row)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating tenant: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to update tenant: {e}")


@router.delete("/{tenant_id}")
async def deactivate_tenant(tenant_id: str):
    """Soft-deactivate a tenant (sets is_active=false, status=inactive)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if tenant_id.isdigit():
            where = "id = %s"
            param = int(tenant_id)
        else:
            where = "tenant_id = %s"
            param = tenant_id

        cursor.execute(
            f"UPDATE tenants SET is_active = false, status = 'inactive', updated_at = NOW() WHERE {where} RETURNING id, tenant_name",
            (param,)
        )
        row = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

        return {"success": True, "message": f"Tenant '{row[1]}' deactivated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating tenant: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to deactivate tenant")
