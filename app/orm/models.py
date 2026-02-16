"""
ORM Models for VaLLM -- Read-Only Access to InfinityAI Tables
==============================================================
These models map to EXISTING tables in vacloudopsdb1.
VaLLM never creates or migrates these tables -- InfinityAI owns them.

Tables used:
  - users_user          : Identify the calling user / developer
  - users_apikey        : Validate X-API-Key header
  - users_organization  : Tenant / organization context
  - infrastructure_workspace : Workspace context for scoped queries
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, BigInteger, Boolean, DateTime, Text,
    Numeric, ForeignKey, func,
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID

from app.orm.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    BigSerialPrimaryKeyMixin,
    ExtendedFieldsMixin,
    AIModelFieldsMixin,
    QueryFieldsMixin,
)


# ---------------------------------------------------------------------------
# 1. Organization (Tenant)
# ---------------------------------------------------------------------------

class Organization(Base, UUIDPrimaryKeyMixin, TimestampMixin, ExtendedFieldsMixin):
    """Tenant organization. Read-only from VaLLM's perspective."""

    __tablename__ = "users_organization"

    name = Column(String(255), nullable=False, server_default="test_org")

    # Tenant / subscription fields
    tenant_subscription_id = Column(String(255), nullable=True)
    tenant_primary_node_host = Column(String(255), nullable=True)
    tenant_primary_domain = Column(String(255), nullable=True)
    tenant_node_details = Column(Text, nullable=True)
    tenants_users_list = Column(JSONB, nullable=True)
    tenant_alternatives_host = Column(JSONB, nullable=True)

    # AI-specific organization fields
    ai_growth_prediction = Column(Numeric(5, 2), nullable=True)
    ai_security_posture_score = Column(Numeric(5, 2), nullable=True)
    ai_resource_optimization_potential = Column(JSONB, nullable=True)
    ai_onboarding_recommendations = Column(JSONB, nullable=True)

    # Relationships
    users = relationship("User", back_populates="organization")

    def __repr__(self) -> str:
        return f"<Organization(id={self.id}, name={self.name!r})>"


# ---------------------------------------------------------------------------
# 2. User (Developer)
# ---------------------------------------------------------------------------

class User(Base, BigSerialPrimaryKeyMixin, ExtendedFieldsMixin, AIModelFieldsMixin, QueryFieldsMixin):
    """
    Core user model mapping to InfinityAI's users_user table.
    In the VaLLM context, "developer" = a user with an active API key.
    There is no separate developer table -- this IS the developer table.
    """

    __tablename__ = "users_user"

    # ---- Authentication & identity -----------------------------------------
    password = Column(String(128), nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    is_superuser = Column(Boolean, nullable=False, default=False)
    username = Column(String(150), nullable=False, unique=True)
    first_name = Column(String(150), nullable=True)
    last_name = Column(String(150), nullable=True)
    is_staff = Column(Boolean, nullable=False, default=False)
    is_active = Column(Boolean, nullable=False, default=True)
    date_joined = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # ---- Deployment / region -----------------------------------------------
    deployed_where = Column(String(16), nullable=False, default="AMERICA-1")

    # ---- Contact -----------------------------------------------------------
    email = Column(String(254), nullable=False, unique=True)
    phone = Column(String(15), nullable=True)
    company_name = Column(String(255), nullable=True)

    # ---- MFA ---------------------------------------------------------------
    mfa_enabled = Column(Boolean, nullable=False, default=False)

    # ---- Verification ------------------------------------------------------
    is_verified = Column(Boolean, nullable=False, default=False)

    # ---- Restrictions ------------------------------------------------------
    is_restricted = Column(Boolean, nullable=False, default=False)

    # ---- Organization FK ---------------------------------------------------
    organization_id = Column(PG_UUID(as_uuid=True), ForeignKey("users_organization.id"), nullable=True, index=True)

    # ---- AI-specific user fields -------------------------------------------
    ai_risk_score = Column(Numeric(5, 2), nullable=True)
    ai_sentiment_analysis = Column(JSONB, nullable=True)
    ai_recommended_actions = Column(JSONB, nullable=True)
    ai_activity_anomaly_detected = Column(Boolean, nullable=False, default=False)
    user_feedback_on_ai = Column(JSONB, nullable=True)

    # ---- Relationships -----------------------------------------------------
    organization = relationship("Organization", back_populates="users")
    api_keys = relationship("APIKey", back_populates="user")

    @property
    def full_name(self) -> str:
        parts = [p for p in (self.first_name, self.last_name) if p]
        return " ".join(parts) if parts else self.username

    @property
    def is_locked(self) -> bool:
        return not self.is_active and self.is_restricted

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username!r}, email={self.email!r})>"


# ---------------------------------------------------------------------------
# 3. APIKey
# ---------------------------------------------------------------------------

class APIKey(Base, UUIDPrimaryKeyMixin):
    """
    Per-user API key for authenticating requests to VaLLM.
    InfinityAI creates these keys; VaLLM validates them via X-API-Key header.
    The token_hash is SHA-256 of the raw key that InfinityAI issued.
    """

    __tablename__ = "users_apikey"

    user_id = Column(BigInteger, ForeignKey("users_user.id"), nullable=False, index=True)
    environment = Column(String(20), nullable=False, default="DEVELOPMENT")
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    token_hash = Column(String(128), nullable=False)

    # Scope & access
    allowed_services = Column(JSONB, nullable=True)
    disallowed_services = Column(JSONB, nullable=True)
    scopes = Column(JSONB, nullable=True)
    is_read_only = Column(Boolean, nullable=False, default=False)
    rate_limit = Column(Integer, nullable=False, default=1000)
    last_ip = Column(String(45), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, nullable=False, default=0)

    # Security
    allowed_ips = Column(JSONB, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(self.expires_at.tzinfo) > self.expires_at

    @property
    def is_usable(self) -> bool:
        return self.is_active and not self.is_expired

    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name={self.name!r}, active={self.is_active})>"


# ---------------------------------------------------------------------------
# 4. Workspace (read-only context for scoped queries)
# ---------------------------------------------------------------------------

class Workspace(Base, UUIDPrimaryKeyMixin, TimestampMixin, AIModelFieldsMixin):
    """Infrastructure workspace. Read-only from VaLLM's perspective."""

    __tablename__ = "infrastructure_workspace"

    name = Column(String(255), nullable=False)
    user_id = Column(BigInteger, nullable=False)
    organization_id = Column(PG_UUID(as_uuid=True), nullable=False)
    insta_node_id = Column(BigInteger, nullable=True)
    is_active = Column(Boolean, default=True, nullable=True)
    is_default_workspace = Column(Boolean, default=True, nullable=True)
    session_id = Column(PG_UUID(as_uuid=True), nullable=False)
    configuration = Column(JSONB, nullable=False)

    # State / reports
    state_facts = Column(JSONB, nullable=True)
    reports = Column(JSONB, nullable=True)

    # Extended fields (VARCHAR 64 variant for this table)
    extended_editable_field_1 = Column(String(64), nullable=True)
    extended_editable_field_2 = Column(String(64), nullable=True)
    extended_editable_field_3 = Column(String(64), nullable=True)
    extended_editable_field_4 = Column(String(64), nullable=True)
    extended_editable_field_5 = Column(String(64), nullable=True)
    extended_boolean_field_1 = Column(Boolean, default=True, nullable=True)
    extended_boolean_field_2 = Column(Boolean, default=True, nullable=True)
    extended_boolean_field_3 = Column(Boolean, default=True, nullable=True)
    extended_json_field_1 = Column(JSONB, nullable=True)
    extended_json_field_2 = Column(JSONB, nullable=True)
    extended_json_field_3 = Column(JSONB, nullable=True)
    extended_json_field_4 = Column(JSONB, nullable=True)

    def __repr__(self) -> str:
        return f"<Workspace(id={self.id}, name={self.name!r}, user_id={self.user_id})>"
