"""Custom exception classes and FastAPI exception handlers.

All handlers return a consistent JSON body:
    {"error": "<ExceptionClassName>", "message": "<human-readable>", "detail": <optional>}

Registering handlers here keeps api/main.py free of error-handling logic.
"""
import logging
from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------

class ModelNotLoadedError(Exception):
    """Model artifacts are missing or failed to load at startup."""


class PredictionError(Exception):
    """An unexpected error occurred during inference."""


class InvalidInputError(Exception):
    """Input passes schema validation but violates a semantic constraint."""


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _body(error: str, message: str, detail: Any = None) -> dict:
    payload: dict = {"error": error, "message": message}
    if detail is not None:
        payload["detail"] = detail
    return payload


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
    logger.error("ModelNotLoadedError: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=_body("ModelNotLoadedError", str(exc)),
    )


async def prediction_error_handler(request: Request, exc: PredictionError) -> JSONResponse:
    logger.error("PredictionError on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_body("PredictionError", str(exc)),
    )


async def invalid_input_handler(request: Request, exc: InvalidInputError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_body("InvalidInputError", str(exc)),
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Replace FastAPI's verbose Pydantic error format with a concise, consistent shape."""
    errors = exc.errors()
    first = errors[0] if errors else {}
    field = " → ".join(str(loc) for loc in first.get("loc", []))
    message = first.get("msg", "Validation error")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_body(
            "ValidationError",
            f"{field}: {message}" if field else message,
            detail=errors,
        ),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_body("InternalServerError", "An unexpected error occurred."),
    )
