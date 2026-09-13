"""
llm/retry.py — Exponential-backoff retry utility.

Design decisions:
- Distinguishes "retriable" errors (transient: timeout, rate-limit, 5xx)
  from "non-retriable" errors (bad request, auth error, invalid input).
- Only retriable errors consume retry budget; non-retriable errors are
  re-raised immediately so the caller can surface them to the user.
- When ALL retry attempts are exhausted on a retriable error, raises
  RetriesExhaustedError so the LLM client knows to trigger fallback.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes that indicate a transient (retryable) server-side failure.
_RETRIABLE_HTTP_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Low-level exception types that are always transient.
_RETRIABLE_EXCEPTION_TYPES: tuple[type[Exception], ...] = (
    TimeoutError,
    asyncio.TimeoutError,
    ConnectionError,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
)

# Substrings in error messages that indicate a transient failure.
_RETRIABLE_MSG_FRAGMENTS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "rate limit",
    "rate_limit",
    "too many requests",
    "overloaded",
    "service unavailable",
    "connection reset",
    "temporarily unavailable",
    "try again",
    "internal server error",
)


class RetriesExhaustedError(Exception):
    """Raised when all retry attempts for a *retriable* error are consumed.

    The caller (LLM client) uses this signal to decide whether to fall back
    to an alternative model rather than surfacing the raw error to the user.
    """

    def __init__(self, last_error: Exception, attempts: int) -> None:
        self.last_error = last_error
        self.attempts = attempts
        super().__init__(
            f"All {attempts} retry attempt(s) failed. "
            f"Last error ({type(last_error).__name__}): {last_error}"
        )


def is_retriable_error(exc: Exception) -> bool:
    """Return True if *exc* represents a transient failure worth retrying."""
    # Check native exception types first.
    if isinstance(exc, _RETRIABLE_EXCEPTION_TYPES):
        return True

    # Check for HTTP status codes attached to the exception object.
    status: Optional[int] = (
        getattr(exc, "status_code", None)
        or getattr(exc, "status", None)
        or (
            getattr(getattr(exc, "response", None), "status_code", None)
        )
    )
    if status is not None and int(status) in _RETRIABLE_HTTP_CODES:
        return True

    # Check error message substrings.
    msg = str(exc).lower()
    if any(fragment in msg for fragment in _RETRIABLE_MSG_FRAGMENTS):
        return True

    return False


async def with_retry(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    on_retry: Optional[Callable[[int, Exception], Awaitable[None]]] = None,
    **kwargs: Any,
) -> T:
    """Execute *fn* with exponential back-off retry on transient errors.

    Args:
        fn:           Async callable to invoke.
        *args:        Positional arguments forwarded to *fn*.
        max_retries:  Maximum number of *additional* attempts after the first.
                      Total attempts = max_retries + 1.
        base_delay:   Base sleep time (seconds) for the first retry.
                      Subsequent retries use base * 2^attempt + jitter.
        max_delay:    Hard upper cap on the per-retry sleep duration.
        on_retry:     Optional async callback invoked before each retry with
                      (attempt_number: int, exc: Exception). Use for logging
                      retry events to Langfuse or structured loggers.
        **kwargs:     Keyword arguments forwarded to *fn*.

    Returns:
        The return value of a successful *fn* call.

    Raises:
        RetriesExhaustedError: If a retriable error persists after all retries.
        Exception:             Immediately, if the error is non-retriable.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)

        except Exception as exc:
            last_exc = exc

            if not is_retriable_error(exc):
                # Non-retriable — surface immediately, no fallback.
                logger.debug(
                    "Non-retriable error on attempt %d/%d: %s",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                )
                raise

            if attempt == max_retries:
                # Retriable, but budget exhausted → signal fallback.
                logger.warning(
                    "Retriable error persisted after %d attempt(s). "
                    "Raising RetriesExhaustedError. Last error: %s",
                    max_retries + 1,
                    exc,
                )
                raise RetriesExhaustedError(last_error=exc, attempts=max_retries + 1)

            # Calculate sleep with full-jitter exponential back-off.
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.2)  # ±20 % jitter
            sleep_time = delay + jitter

            logger.warning(
                "Attempt %d/%d failed (%s: %s). Retrying in %.2fs…",
                attempt + 1,
                max_retries + 1,
                type(exc).__name__,
                exc,
                sleep_time,
            )

            if on_retry is not None:
                try:
                    await on_retry(attempt + 1, exc)
                except Exception:
                    pass  # Never let the callback abort the retry loop.

            await asyncio.sleep(sleep_time)

    # Should never be reached, but satisfies the type checker.
    assert last_exc is not None
    raise RetriesExhaustedError(last_error=last_exc, attempts=max_retries + 1)
