"""ASGI middleware for per-request correlation IDs and access logging.

X-Correlation-ID lifecycle:
  - Read from the incoming request header when present (gateway / client pass-through).
  - Generated as a UUID4 when absent.
  - Bound to structlog's contextvars so it appears in every log line for that request
    without any manual plumbing in route handlers or exception handlers.
  - Echoed back in the response header so callers can correlate client-side traces.
"""
import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(module=__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())

        # Reset per-request context so previous request's bindings don't bleed over.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            http_method=request.method,
            http_path=request.url.path,
        )

        logger.info("Request started")
        t0 = time.perf_counter()

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        structlog.contextvars.bind_contextvars(
            http_status=response.status_code,
            duration_ms=duration_ms,
        )
        logger.info("Request finished")

        response.headers["X-Correlation-ID"] = correlation_id
        return response
