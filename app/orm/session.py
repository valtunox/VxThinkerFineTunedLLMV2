"""
SQLAlchemy Session Factory for VaLLM
=====================================
Uses the same vacloudopsdb1 connection from app.core.db.
Read-only session -- VaLLM does not create or migrate tables.
"""

import os
import logging
from urllib.parse import quote_plus
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger("vallm")


def _build_database_url() -> str:
    """Build sync PostgreSQL URL from db.py config or env vars."""
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")
    try:
        from app.core.db import get_db_config
        cfg = get_db_config()
        user = quote_plus(cfg["user"])
        password = quote_plus(cfg["password"])
        return f"postgresql://{user}:{password}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    except Exception:
        pass
    return (
        f"postgresql://{quote_plus(os.getenv('POSTGRES_USER', 'postgres'))}:"
        f"{quote_plus(os.getenv('POSTGRES_PASSWORD', 'postgres'))}@"
        f"{os.getenv('POSTGRES_HOST', '127.0.0.1')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'vacloudopsdb1')}"
    )


DATABASE_URL = _build_database_url()

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=os.getenv("SQL_ECHO", "false").lower() == "true",
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database sessions outside FastAPI."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
