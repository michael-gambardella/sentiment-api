"""PostgreSQL connection pool and prediction audit log.

Uses asyncpg for non-blocking I/O. All public functions swallow database
exceptions so that a Postgres outage never breaks prediction serving.
The pool is created at startup and stored on app.state.db.
"""
import hashlib

import asyncpg
import structlog

logger = structlog.get_logger(module=__name__)

_CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                BIGSERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    input_hash        TEXT             NOT NULL,
    input_text        TEXT             NOT NULL,
    label             TEXT             NOT NULL,
    confidence        DOUBLE PRECISION NOT NULL,
    model_version     TEXT             NOT NULL,
    latency_ms        DOUBLE PRECISION NOT NULL,
    served_from_cache BOOLEAN          NOT NULL DEFAULT FALSE,
    correlation_id    TEXT,
    client_ip         TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at    ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions (model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_input_hash    ON predictions (input_hash);
"""


async def init_db_pool(database_url: str) -> "asyncpg.Pool | None":
    try:
        pool = await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=10)
        async with pool.acquire() as conn:
            await conn.execute(_CREATE_SCHEMA)
        logger.info("Database pool ready")
        return pool
    except Exception as exc:
        logger.warning("Database unavailable, audit logging disabled", error=str(exc))
        return None


async def close_db_pool(pool: "asyncpg.Pool") -> None:
    try:
        await pool.close()
    except Exception as exc:
        logger.warning("Error closing database pool", error=str(exc))


async def log_prediction(
    pool: "asyncpg.Pool",
    *,
    text: str,
    label: str,
    confidence: float,
    model_version: str,
    latency_ms: float,
    served_from_cache: bool,
    correlation_id: str | None,
    client_ip: str | None,
) -> None:
    input_hash = hashlib.sha256(text.encode()).hexdigest()
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO predictions
                    (input_hash, input_text, label, confidence, model_version,
                     latency_ms, served_from_cache, correlation_id, client_ip)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                input_hash, text, label, float(confidence), model_version,
                float(latency_ms), served_from_cache, correlation_id, client_ip,
            )
    except Exception as exc:
        logger.warning("Failed to log prediction", error=str(exc))


async def get_recent_predictions(
    pool: "asyncpg.Pool",
    *,
    limit: int = 50,
    version: str | None = None,
) -> list[dict]:
    try:
        async with pool.acquire() as conn:
            if version:
                rows = await conn.fetch(
                    """
                    SELECT id, created_at, input_hash, label, confidence,
                           model_version, latency_ms, served_from_cache,
                           correlation_id, client_ip
                    FROM predictions
                    WHERE model_version = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    version, limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, created_at, input_hash, label, confidence,
                           model_version, latency_ms, served_from_cache,
                           correlation_id, client_ip
                    FROM predictions
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Failed to query predictions", error=str(exc))
        return []
