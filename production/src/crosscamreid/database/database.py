"""
database.py
===========
SQLAlchemy engine + session factory.
Call init_db() once at server startup to create tables and verify the connection.
"""

import logging
import os

import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .models import Base

logger = logging.getLogger("database")
logger.setLevel(logging.INFO)

load_dotenv()

_engine = None
_SessionLocal = None


def _build_url() -> str:
    user     = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    host     = os.environ["DB_HOST"]
    port     = os.environ.get("DB_PORT", "5432")
    dbname   = os.environ["DB_NAME"]
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"


def _db_settings() -> tuple[str, str, str, str, str]:
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    host = os.environ["DB_HOST"]
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ["DB_NAME"]
    return user, password, host, port, dbname


def ensure_database_exists() -> None:
    """
    Ensure target PostgreSQL database exists.
    Connects to maintenance DB 'postgres' and creates DB_NAME if missing.
    """
    user, password, host, port, dbname = _db_settings()
    conn = psycopg2.connect(
        dbname="postgres",
        user=user,
        password=password,
        host=host,
        port=port,
    )
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            exists = cur.fetchone() is not None
            if not exists:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
                logger.info("Created PostgreSQL database: %s", dbname)
                print(f"[DB] Created database: {dbname}")
    finally:
        conn.close()


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            _build_url(),
            pool_size=10,
            max_overflow=10,
            pool_pre_ping=True,   # drops stale connections before using them
        )
    return _engine


def get_session():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionLocal()


def init_db() -> None:
    """Create all tables and verify the connection. Prints result to console."""
    ensure_database_exists()
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("PostgreSQL connected — tables ready")
    print("[DB] PostgreSQL connected successfully")
