"""
Database Migration Script for VaLLM Specialist Model
=====================================================
Creates all database tables defined in app.orm.models.

Run from the project root:
    python migration.py
    python migration.py --check-only
    python migration.py --force

Tables created:
    - tenants                  : Multi-tenant isolation
    - sessions                 : User/agent session tracking
    - documents                : Document records
    - document_chunks          : Chunked content for RAG
    - document_embeddings      : FAISS vector references
    - verification_records     : Verification audit trail
    - transactions             : Financial transactions
    - business_recommendations : AI-generated insights
"""

import sys
import os
import uuid
import hashlib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.orm.base import Base
from app.orm.models import (
    Tenant, Session, Document, DocumentChunk, DocumentEmbedding,
    VerificationRecord, Transaction, BusinessRecommendation,
)
from app.orm.session import engine, SessionLocal
from app.core.db import get_db_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REQUIRED_TABLES = [
    "tenants", "sessions", "documents", "document_chunks",
    "document_embeddings", "verification_records", "transactions",
    "business_recommendations",
]


def check_database_connection():
    """Verify database connection is working."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            logger.info(f"✅ Database connected: PostgreSQL {version}")
            return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


def check_tables_exist():
    """Check if tables already exist."""
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    missing_tables = [t for t in REQUIRED_TABLES if t not in existing_tables]

    if missing_tables:
        logger.info(f"📋 Missing tables: {', '.join(missing_tables)}")
        return False, existing_tables, missing_tables
    else:
        logger.info(f"✅ All {len(REQUIRED_TABLES)} required tables exist")
        return True, existing_tables, []


def create_tables():
    """Create all tables defined in the ORM models."""
    try:
        logger.info("🔨 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables created successfully!")
        return True
    except SQLAlchemyError as e:
        logger.error(f"❌ Error creating tables: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_default_tenant():
    """Create the default tenant from .env TENANT_ID or generate one."""
    try:
        with SessionLocal() as db:
            env_tenant_id = os.getenv("TENANT_ID")

            # Check if default already exists by slug
            existing = db.query(Tenant).filter_by(slug="va-specialist").first()
            if existing:
                logger.info(f"✅ Default tenant already exists: {existing.id} ({existing.name})")
                return str(existing.id)

            # Check by TENANT_ID from env
            if env_tenant_id:
                try:
                    env_uuid = uuid.UUID(env_tenant_id)
                    existing = db.query(Tenant).filter_by(id=env_uuid).first()
                    if existing:
                        logger.info(f"✅ Default tenant already exists (from TENANT_ID): {existing.id}")
                        return str(existing.id)
                except (ValueError, TypeError):
                    pass

            # Determine tenant UUID
            if env_tenant_id:
                try:
                    tenant_uuid = uuid.UUID(env_tenant_id)
                except (ValueError, TypeError):
                    tenant_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, "va-specialist.primary.tenant")
            else:
                tenant_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, "va-specialist.primary.tenant")

            default_tenant = Tenant(
                id=tenant_uuid,
                name="VaLLM Specialist AI",
                slug="va-specialist",
                is_active=True,
                plan="enterprise",
                max_documents=100000,
                max_storage_mb=50000,
                contact_email="admin@vallm-specialist.ai",
                contact_name="System Admin",
                settings={
                    "industries": ["finance", "healthcare", "cloud", "automation", "customer_service"],
                    "embedding_model": "BAAI/bge-large-en-v1.5",
                    "created_by": "migration",
                },
                metadata_={
                    "is_primary": True,
                    "note": "Primary default tenant created during migration",
                },
            )

            db.add(default_tenant)
            db.commit()
            db.refresh(default_tenant)

            logger.info(f"✅ Default tenant created!")
            logger.info(f"   ID:   {default_tenant.id}")
            logger.info(f"   Name: {default_tenant.name}")
            logger.info(f"   Slug: {default_tenant.slug}")
            logger.info(f"   Plan: {default_tenant.plan}")
            return str(default_tenant.id)

    except SQLAlchemyError as e:
        logger.error(f"❌ Error creating default tenant: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_tables():
    """Verify tables were created and show their structure."""
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        logger.info(f"\n📊 Database Tables ({len(tables)} total)")
        logger.info("=" * 70)

        for table_name in REQUIRED_TABLES:
            if table_name in tables:
                columns = inspector.get_columns(table_name)
                indexes = inspector.get_indexes(table_name)
                logger.info(f"\n  📋 {table_name} ({len(columns)} columns, {len(indexes)} indexes)")
                for col in columns:
                    col_type = str(col['type'])
                    nullable = "NULL" if col['nullable'] else "NOT NULL"
                    default = f" DEFAULT {col['default']}" if col.get('default') else ""
                    logger.info(f"     {col['name']:30s} {col_type:25s} {nullable}{default}")
                if indexes:
                    for idx in indexes:
                        unique = " UNIQUE" if idx.get('unique') else ""
                        logger.info(f"     🔑 {idx['name']}: ({', '.join(idx['column_names'])}){unique}")
            else:
                logger.warning(f"  ⚠️  {table_name} — NOT FOUND!")

        logger.info("\n" + "=" * 70)

        # Show tenant data
        try:
            with SessionLocal() as db:
                tenants = db.query(Tenant).all()
                if tenants:
                    logger.info(f"\n👤 Tenants ({len(tenants)}):")
                    for t in tenants:
                        logger.info(f"   {t.id} | {t.name} | slug={t.slug} | plan={t.plan} | active={t.is_active}")
                else:
                    logger.info("\n👤 No tenants in database yet")
        except Exception as e:
            logger.debug(f"Could not fetch tenants: {e}")

        return True
    except Exception as e:
        logger.error(f"❌ Error verifying tables: {e}")
        return False


def get_database_info():
    """Display database connection information."""
    try:
        config = get_db_config()
        logger.info("\n📡 Database Configuration:")
        logger.info("=" * 50)
        logger.info(f"   Host:     {config['host']}")
        logger.info(f"   Port:     {config['port']}")
        logger.info(f"   Database: {config['database']}")
        logger.info(f"   User:     {config['user']}")
        logger.info("=" * 50)
    except Exception as e:
        logger.warning(f"⚠️  Could not get database config: {e}")


def main():
    """Main migration function."""
    logger.info("=" * 70)
    logger.info("🚀 VaLLM Specialist Model — Database Migration")
    logger.info("=" * 70)

    get_database_info()

    if not check_database_connection():
        logger.error("❌ Cannot proceed without database connection")
        sys.exit(1)

    tables_exist, existing_tables, missing = check_tables_exist()

    if tables_exist:
        logger.info("\n✅ All tables already exist.")
        verify_tables()

        logger.info("\n👤 Checking default tenant...")
        tenant_id = create_default_tenant()
        if tenant_id:
            logger.info(f"   Default tenant ID: {tenant_id}")

        logger.info("\n💡 To recreate, run: python migration.py --force")
        return

    logger.info("\n🔨 Starting migration...")
    if not create_tables():
        logger.error("❌ Migration failed!")
        sys.exit(1)

    logger.info("\n👤 Creating default tenant...")
    tenant_id = create_default_tenant()
    if tenant_id:
        logger.info(f"   Default tenant ID: {tenant_id}")

    logger.info("\n🔍 Verifying migration...")
    if not verify_tables():
        logger.error("❌ Verification failed!")
        sys.exit(1)

    logger.info("\n" + "=" * 70)
    logger.info("✅ Migration completed successfully!")
    logger.info("=" * 70)
    logger.info("\n📝 Tables created:")
    for t in REQUIRED_TABLES:
        logger.info(f"   ✓ {t}")
    if tenant_id:
        logger.info(f"\n🔑 Default Tenant ID: {tenant_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VaLLM Specialist Model — Database Migration")
    parser.add_argument("--force", action="store_true", help="Drop and recreate all tables (DESTROYS DATA)")
    parser.add_argument("--check-only", action="store_true", help="Only check tables, don't create")
    args = parser.parse_args()

    if args.check_only:
        get_database_info()
        check_database_connection()
        tables_exist, _, _ = check_tables_exist()
        if tables_exist:
            verify_tables()
        sys.exit(0)

    if args.force:
        logger.warning("⚠️  FORCE MODE: Will drop and recreate ALL specialist tables!")
        response = input("Are you sure? This will DELETE ALL DATA! (yes/no): ")
        if response.lower() != "yes":
            logger.info("Migration cancelled.")
            sys.exit(0)
        try:
            logger.info("🗑️  Dropping existing tables...")
            Base.metadata.drop_all(bind=engine)
            logger.info("✅ Tables dropped")
        except Exception as e:
            logger.error(f"❌ Error dropping tables: {e}")
            sys.exit(1)

    main()
