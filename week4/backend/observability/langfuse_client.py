"""
observability/langfuse_client.py — Langfuse tracing wrapper.

Every chat request produces one Langfuse Trace that captures:
  - Session / user metadata
  - Each LLM call (model, tokens, latency, estimated cost)
  - Agent tool decisions
  - MCP tool executions (tool name, input, output, latency)
  - Errors and their types
  - Retry attempts
  - Fallback model usage

Disabled gracefully when LANGFUSE_ENABLED=false or keys are empty.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# Approximate cost per 1 K tokens (USD) — update as pricing changes.
_COST_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gemini-1.5-flash": {"prompt": 0.000075, "completion": 0.0003},
    "gemini/gemini-1.5-flash": {"prompt": 0.000075, "completion": 0.0003},
}


def _estimate_cost(model: str, usage: Dict[str, int]) -> float:
    rates = _COST_PER_1K.get(model, {"prompt": 0.001, "completion": 0.002})
    prompt_cost = usage.get("prompt_tokens", 0) / 1000 * rates["prompt"]
    completion_cost = usage.get("completion_tokens", 0) / 1000 * rates["completion"]
    return prompt_cost + completion_cost


class LangfuseTracer:
    """Thin wrapper around the Langfuse SDK for structured tracing."""

    def __init__(self) -> None:
        self._enabled = False
        self._client = None
        self._current_trace = None
        self._current_span = None

    def setup(self, public_key: str, secret_key: str, host: str, enabled: bool) -> None:
        """Initialize the Langfuse client. Called once at app startup."""
        if not enabled or not public_key or not secret_key:
            logger.info("Langfuse tracing disabled (keys not set or LANGFUSE_ENABLED=false).")
            return
        try:
            from langfuse import Langfuse  # type: ignore
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            self._enabled = True
            logger.info("Langfuse tracing enabled (host: %s).", host)
        except ImportError:
            logger.warning("langfuse package not installed. Tracing disabled.")
        except Exception as exc:
            logger.warning("Failed to initialize Langfuse: %s. Tracing disabled.", exc)

    # ------------------------------------------------------------------ #
    # Trace lifecycle
    # ------------------------------------------------------------------ #

    def start_trace(
        self,
        name: str,
        session_id: str,
        user_id: str,
        input_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Start a new top-level trace. Returns the trace ID."""
        if not self._enabled or not self._client:
            return None
        try:
            self._current_trace = self._client.trace(
                name=name,
                session_id=session_id,
                user_id=user_id,
                input=input_data,
                metadata=metadata or {},
            )
            return self._current_trace.id
        except Exception as exc:
            logger.debug("Langfuse start_trace error: %s", exc)
            return None

    def finish_trace(self, output: Any, trace_id: Optional[str] = None) -> None:
        """Mark the current trace as complete with its final output."""
        if not self._enabled or not self._current_trace:
            return
        try:
            self._current_trace.update(output=output)
            self._client.flush()
        except Exception as exc:
            logger.debug("Langfuse finish_trace error: %s", exc)

    # ------------------------------------------------------------------ #
    # LLM generation logging
    # ------------------------------------------------------------------ #

    def log_llm_call(
        self,
        name: str,
        model: str,
        input_messages: Any,
        output: str,
        usage: Dict[str, int],
        latency_ms: float,
        is_fallback: bool = False,
        trace_id: Optional[str] = None,
    ) -> None:
        if not self._enabled or not self._current_trace:
            return
        try:
            cost = _estimate_cost(model, usage)
            self._current_trace.generation(
                name=name,
                model=model,
                input=input_messages,
                output=output,
                usage={
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0),
                    "unit": "TOKENS",
                },
                metadata={
                    "latency_ms": round(latency_ms, 2),
                    "estimated_cost_usd": round(cost, 6),
                    "is_fallback": is_fallback,
                },
            )
        except Exception as exc:
            logger.debug("Langfuse log_llm_call error: %s", exc)

    # ------------------------------------------------------------------ #
    # Tool / MCP event logging
    # ------------------------------------------------------------------ #

    def log_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> None:
        if not self._enabled or not self._current_trace:
            return
        try:
            self._current_trace.span(
                name=f"mcp_tool:{tool_name}",
                input=tool_input,
                output=tool_output,
                metadata={
                    "latency_ms": round(latency_ms, 2),
                    "error": error,
                },
                level="ERROR" if error else "DEFAULT",
            )
        except Exception as exc:
            logger.debug("Langfuse log_tool_call error: %s", exc)

    def log_agent_decision(
        self,
        chosen_tools: list[str],
        reasoning: Optional[str] = None,
    ) -> None:
        if not self._enabled or not self._current_trace:
            return
        try:
            self._current_trace.span(
                name="agent_decision",
                input={"reasoning": reasoning},
                output={"chosen_tools": chosen_tools},
            )
        except Exception as exc:
            logger.debug("Langfuse log_agent_decision error: %s", exc)

    # ------------------------------------------------------------------ #
    # Reliability event logging
    # ------------------------------------------------------------------ #

    def log_retry(self, attempt: int, error: Exception, model: str) -> None:
        if not self._enabled or not self._current_trace:
            return
        try:
            self._current_trace.event(
                name="llm_retry",
                metadata={
                    "attempt": attempt,
                    "model": model,
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:500],
                },
                level="WARNING",
            )
        except Exception as exc:
            logger.debug("Langfuse log_retry error: %s", exc)

    def log_fallback(self, from_model: str, to_model: str, reason: str) -> None:
        if not self._enabled or not self._current_trace:
            return
        try:
            self._current_trace.event(
                name="llm_fallback",
                metadata={
                    "from_model": from_model,
                    "to_model": to_model,
                    "reason": reason,
                },
                level="WARNING",
            )
        except Exception as exc:
            logger.debug("Langfuse log_fallback error: %s", exc)

    def log_error(
        self,
        error_type: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._enabled or not self._current_trace:
            return
        try:
            self._current_trace.event(
                name="error",
                metadata={
                    "error_type": error_type,
                    "message": message[:1000],
                    **(context or {}),
                },
                level="ERROR",
            )
        except Exception as exc:
            logger.debug("Langfuse log_error error: %s", exc)

    def log_write_approval(
        self, tool_name: str, approved: bool, draft: Dict[str, Any]
    ) -> None:
        if not self._enabled or not self._current_trace:
            return
        try:
            self._current_trace.event(
                name="write_approval",
                metadata={
                    "tool_name": tool_name,
                    "approved": approved,
                    "draft_keys": list(draft.keys()),
                },
            )
        except Exception as exc:
            logger.debug("Langfuse log_write_approval error: %s", exc)


# Module-level singleton — imported by routers and agent.
tracer = LangfuseTracer()
