"""
TicketAI - Database Connection & Session Management
Provides robust Neon PostgreSQL connection handling, connection pooling,
parameterized execution, transaction support, and schema migrations.
"""

import os
import re
import json
import logging
from typing import Optional, Any, Dict, List
import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv

# Load environment
load_dotenv()

logger = logging.getLogger("ticketai.database")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def get_database_url() -> str:
    """Retrieve normalized PostgreSQL database URL from environment."""
    url = (
        os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
        or os.getenv("POSTGRES_PRISMA_URL")
        or ""
    ).strip()

    if not url:
        return ""

    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    if "sslmode" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"

    return url


class PGCursorWrapper:
    """
    Wraps a psycopg2 cursor for SQLite-compatible syntax:
    - Automatically converts '?' placeholders to '%s'
    - Auto-captures lastrowid on INSERT queries via RETURNING id
    - Supports fetchone, fetchall, and iteration with RealDictCursor rows
    """
    def __init__(self, raw_cursor):
        self._cursor = raw_cursor
        self.lastrowid = None

    def execute(self, sql: str, params: Any = None):
        if params is None:
            params = ()
        elif isinstance(params, list):
            params = tuple(params)
        elif not isinstance(params, tuple):
            params = (params,)

        pg_sql = sql.replace("?", "%s")
        trimmed = pg_sql.strip()
        is_insert = trimmed.upper().startswith("INSERT INTO")

        if is_insert:
            if "RETURNING" not in trimmed.upper():
                pg_sql_returning = trimmed.rstrip(";").rstrip() + " RETURNING id;"
            else:
                pg_sql_returning = pg_sql

            try:
                self._cursor.execute(pg_sql_returning, params)
                row = self._cursor.fetchone()
                if row:
                    if isinstance(row, dict):
                        self.lastrowid = row.get("id")
                    elif isinstance(row, (tuple, list)):
                        self.lastrowid = row[0]
                return self
            except Exception:
                # If RETURNING execution failed (e.g. table without id), retry standard
                try:
                    self._cursor.execute(pg_sql, params)
                except Exception:
                    pass
                self.lastrowid = None
                return self
        else:
            self._cursor.execute(pg_sql, params)
            self.lastrowid = None
            return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def fetchmany(self, size: Optional[int] = None):
        return self._cursor.fetchmany(size) if size else self._cursor.fetchmany()

    @property
    def description(self):
        return self._cursor.description

    @property
    def rowcount(self):
        return self._cursor.rowcount

    def close(self):
        try:
            self._cursor.close()
        except Exception:
            pass

    def __iter__(self):
        return iter(self._cursor)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PGConnectionWrapper:
    """Wraps psycopg2 connection with transaction and context management."""
    def __init__(self, raw_conn, pool=None):
        self._conn = raw_conn
        self._pool = pool

    def cursor(self):
        raw_cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        return PGCursorWrapper(raw_cur)

    def execute(self, sql: str, params: Any = None):
        cur = self.cursor()
        return cur.execute(sql, params)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        try:
            return self._conn.rollback()
        except Exception:
            pass

    def close(self):
        try:
            if self._pool is not None:
                self._pool.putconn(self._conn)
            else:
                self._conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        else:
            self.commit()
        self.close()


class DatabaseManager:
    """Thread-safe connection pooling & migration management for NeonDB PostgreSQL."""
    _instance: Optional["DatabaseManager"] = None
    _pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._init_pool()
        return cls._instance

    def _init_pool(self):
        db_url = get_database_url()
        if not db_url:
            logger.warning("DATABASE_URL not set. Running in uninitialized database state.")
            return

        try:
            # Min 1, Max 10 connections for lightweight pooling
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=db_url,
                connect_timeout=10
            )
            logger.info("NeonDB connection pool established successfully.")
        except Exception as e:
            logger.warning(f"Connection pool initialization warning (will retry on demand): {e}")
            self._pool = None

    def get_connection(self) -> PGConnectionWrapper:
        db_url = get_database_url()
        if not db_url:
            raise RuntimeError("DATABASE_URL is not set. Please configure NeonDB connection string in .env.")

        # Re-initialize pool if not active
        if self._pool is None:
            self._init_pool()

        if self._pool:
            try:
                raw_conn = self._pool.getconn()
                # Test connection liveness
                with raw_conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                return PGConnectionWrapper(raw_conn, pool=self._pool)
            except Exception as e:
                logger.warning(f"Re-initializing connection from pool due to: {e}")
                try:
                    self._pool.closeall()
                except Exception:
                    pass
                self._pool = None
                self._init_pool()
                if self._pool:
                    try:
                        raw_conn = self._pool.getconn()
                        return PGConnectionWrapper(raw_conn, pool=self._pool)
                    except Exception:
                        pass

        # Fallback to direct connection if pool exhausted or reconnecting
        raw_conn = psycopg2.connect(db_url, connect_timeout=10)
        return PGConnectionWrapper(raw_conn, pool=None)

    def run_migrations(self, embedding_dimension: int = 768):
        """
        Idempotent schema migration for NeonDB with pgvector:
        - users, tickets (existing tables preserved)
        - knowledge_documents, knowledge_chunks (pgvector)
        - ticket_resolutions, ticket_resolution_sources
        - ticket_feedback, ticket_events
        """
        db_url = get_database_url()
        if not db_url:
            logger.warning("Skipping migrations: DATABASE_URL not set.")
            return False

        logger.info(f"Running database migrations (embedding_dim={embedding_dimension})...")
        raw_conn = psycopg2.connect(db_url, connect_timeout=10)
        raw_conn.autocommit = True

        with raw_conn.cursor() as cur:
            # 1. Enable pgvector extension
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("Verified pgvector extension.")
            except Exception as e:
                logger.warning(f"pgvector extension check note: {e}")

            # 2. Existing Users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(150) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    tickets_created INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. Existing Tickets table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    ticket_text TEXT NOT NULL,
                    predicted_category VARCHAR(100) NOT NULL,
                    confidence REAL NOT NULL,
                    status VARCHAR(50) DEFAULT 'Open',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP NULL,
                    resolved_by INTEGER NULL,
                    resolution_notes TEXT NULL,
                    notification_email VARCHAR(255),
                    is_out_of_scope INTEGER DEFAULT 0,
                    needs_human_review INTEGER DEFAULT 0,
                    review_reason TEXT DEFAULT ''
                );
            """)

            # Idempotent ticket columns
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS tickets_created INTEGER DEFAULT 0;")
            cur.execute("ALTER TABLE tickets ADD COLUMN IF NOT EXISTS notification_email VARCHAR(255);")
            cur.execute("ALTER TABLE tickets ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMP NULL;")
            cur.execute("ALTER TABLE tickets ADD COLUMN IF NOT EXISTS resolved_by INTEGER NULL;")
            cur.execute("ALTER TABLE tickets ADD COLUMN IF NOT EXISTS resolution_notes TEXT NULL;")
            cur.execute("ALTER TABLE tickets ADD COLUMN IF NOT EXISTS is_out_of_scope INTEGER DEFAULT 0;")
            cur.execute("ALTER TABLE tickets ADD COLUMN IF NOT EXISTS needs_human_review INTEGER DEFAULT 0;")
            cur.execute("ALTER TABLE tickets ADD COLUMN IF NOT EXISTS review_reason TEXT DEFAULT '';")
            cur.execute("ALTER TABLE tickets ADD COLUMN IF NOT EXISTS priority VARCHAR(20) DEFAULT 'Medium';")
            cur.execute("ALTER TABLE tickets ADD COLUMN IF NOT EXISTS resolution_method VARCHAR(20) NULL;")
            cur.execute("ALTER TABLE tickets ADD COLUMN IF NOT EXISTS subject VARCHAR(255) DEFAULT '';")

            # Idempotent ticket_resolutions columns
            cur.execute("ALTER TABLE ticket_resolutions ADD COLUMN IF NOT EXISTS resolution_method VARCHAR(20) DEFAULT 'AI';")
            cur.execute("ALTER TABLE ticket_resolutions ADD COLUMN IF NOT EXISTS admin_id INTEGER NULL;")
            cur.execute("ALTER TABLE ticket_resolutions ADD COLUMN IF NOT EXISTS internal_notes TEXT DEFAULT '';")
            cur.execute("ALTER TABLE ticket_resolutions ADD COLUMN IF NOT EXISTS sent_at TIMESTAMP NULL;")

            # Idempotent ticket_events columns
            cur.execute("ALTER TABLE ticket_events ADD COLUMN IF NOT EXISTS admin_id INTEGER NULL;")
            cur.execute("ALTER TABLE ticket_events ADD COLUMN IF NOT EXISTS event_type VARCHAR(50) DEFAULT 'STATUS_CHANGE';")

            # 4. Knowledge Documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    category VARCHAR(100) NOT NULL,
                    source VARCHAR(255) NOT NULL,
                    version VARCHAR(50) DEFAULT '1.0',
                    is_active BOOLEAN DEFAULT TRUE,
                    file_hash VARCHAR(64) DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 5. Knowledge Chunks with pgvector embedding
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES knowledge_documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    chunk_text TEXT NOT NULL,
                    embedding vector({embedding_dimension}),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 6. Ticket Resolutions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ticket_resolutions (
                    id SERIAL PRIMARY KEY,
                    ticket_id INTEGER REFERENCES tickets(id) ON DELETE CASCADE,
                    problem_summary TEXT DEFAULT '',
                    identified_issue TEXT DEFAULT '',
                    proposed_solution TEXT NOT NULL,
                    final_solution TEXT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    resolution_status VARCHAR(50) DEFAULT 'needs_review',
                    requires_human_review BOOLEAN DEFAULT TRUE,
                    grounding_status VARCHAR(50) DEFAULT 'unverified',
                    escalation_reason TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP NULL
                );
            """)

            # 7. Ticket Resolution Sources table (Source Traceability)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ticket_resolution_sources (
                    id SERIAL PRIMARY KEY,
                    resolution_id INTEGER REFERENCES ticket_resolutions(id) ON DELETE CASCADE,
                    knowledge_chunk_id INTEGER REFERENCES knowledge_chunks(id) ON DELETE SET NULL,
                    document_id INTEGER REFERENCES knowledge_documents(id) ON DELETE SET NULL,
                    relevance_score REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 8. Ticket Feedback table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ticket_feedback (
                    id SERIAL PRIMARY KEY,
                    ticket_id INTEGER REFERENCES tickets(id) ON DELETE CASCADE,
                    resolution_id INTEGER REFERENCES ticket_resolutions(id) ON DELETE SET NULL,
                    feedback_rating INTEGER NOT NULL,
                    feedback_comment TEXT DEFAULT '',
                    was_resolved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 9. Ticket Events Audit Log
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ticket_events (
                    id SERIAL PRIMARY KEY,
                    ticket_id INTEGER REFERENCES tickets(id) ON DELETE CASCADE,
                    previous_status VARCHAR(50) DEFAULT '',
                    new_status VARCHAR(50) NOT NULL,
                    event_description TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 10. Performance Indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tickets_user_id ON tickets(user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_kb_doc_category ON knowledge_documents(category);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_kb_doc_active ON knowledge_documents(is_active);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc_id ON knowledge_chunks(document_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ticket_res_ticket_id ON ticket_resolutions(ticket_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ticket_res_method ON ticket_resolutions(resolution_method);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_ticket_events_ticket_id ON ticket_events(ticket_id);")

            # Try creating ivfflat or HNSW vector index if enough rows or supported
            try:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding_cosine
                    ON knowledge_chunks USING hnsw (embedding vector_cosine_ops);
                """)
                logger.info("Created HNSW vector cosine index on knowledge_chunks.")
            except Exception as e:
                logger.info(f"HNSW vector index note (will build dynamically): {e}")

        raw_conn.close()
        logger.info("NeonDB migrations completed successfully.")
        return True


def get_db():
    """Module-level function returning a database connection wrapper."""
    return DatabaseManager().get_connection()


def init_db(embedding_dimension: int = 768):
    """Run migrations on demand."""
    return DatabaseManager().run_migrations(embedding_dimension=embedding_dimension)
