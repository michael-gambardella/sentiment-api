"""Redis-backed prediction cache.

Cache key: SHA-256 of "<version>:<text>" so identical inputs with the same
model version always hit the same slot. The "sentiment:v1:" prefix namespaces
keys away from Celery and allows future schema changes to invalidate all
entries by bumping the version prefix.

All public functions swallow Redis exceptions and return None / no-op so that
a Redis outage never breaks prediction serving.
"""
import hashlib
import json

import structlog
from redis.asyncio import Redis

logger = structlog.get_logger(module=__name__)

_PREFIX = "sentiment:v1:"


def _make_key(text: str, version: str) -> str:
    digest = hashlib.sha256(f"{version}:{text}".encode()).hexdigest()
    return f"{_PREFIX}{digest}"


async def get_cached(redis: Redis, text: str, version: str) -> dict | None:
    try:
        raw = await redis.get(_make_key(text, version))
        if raw is not None:
            return json.loads(raw)
    except Exception as exc:
        logger.warning("Cache read error", error=str(exc))
    return None


async def set_cached(redis: Redis, text: str, version: str, result: dict, ttl: int) -> None:
    try:
        await redis.set(_make_key(text, version), json.dumps(result), ex=ttl)
    except Exception as exc:
        logger.warning("Cache write error", error=str(exc))
