"""API key authentication dependency for protected endpoints."""
import structlog
from fastapi import Header

from api.errors import AuthenticationError
from config import settings

logger = structlog.get_logger(module=__name__)


async def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Validate the X-API-Key header against configured keys.

    No-ops when settings.api_keys is empty (auth disabled by default in development).
    Raises AuthenticationError when a key is required but missing or invalid.
    """
    if not settings.api_keys:
        return
    if x_api_key not in settings.api_keys:
        logger.warning("Rejected request: invalid or missing API key")
        raise AuthenticationError("Invalid or missing API key.")
