"""
Langfuse singleton client.

Reads credentials from the .env file (LANGFUSE_PUBLIC_KEY,
LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL / LANGFUSE_HOST) and exposes a
single `get_langfuse_client()` helper.  If any credential is missing the
module degrades gracefully: `get_langfuse_client()` returns None and the
callers skip tracing without raising an exception.
"""

import os
from typing import Optional

from dotenv import load_dotenv

from src.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

_langfuse_client = None
_initialized = False


def get_langfuse_client():
    """Return the singleton Langfuse client, or None if credentials are missing."""
    global _langfuse_client, _initialized

    if _initialized:
        return _langfuse_client

    _initialized = True

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    # Support both LANGFUSE_BASE_URL (set in .env) and LANGFUSE_HOST
    host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")

    if not public_key or not secret_key:
        logger.warning(
            "Langfuse credentials not found (LANGFUSE_PUBLIC_KEY / "
            "LANGFUSE_SECRET_KEY). Tracing is disabled."
        )
        return None

    try:
        from langfuse import Langfuse  # noqa: PLC0415

        kwargs = {
            "public_key": public_key,
            "secret_key": secret_key,
        }
        if host:
            kwargs["host"] = host

        _langfuse_client = Langfuse(**kwargs)
        logger.info("Langfuse client initialised successfully.")
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Failed to initialise Langfuse client: {exc}. Tracing is disabled.")
        _langfuse_client = None

    return _langfuse_client
