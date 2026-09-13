"""
llm/client.py — Unified LLM client with OpenAI primary + Gemini fallback.

Strategy:
  1. Try OpenAI with exponential-backoff retry (max_retries from config).
  2. If retries are exhausted on a *transient* error → fall back to Gemini.
  3. If the error is non-retriable (auth, bad request) → surface it immediately.
  4. Log which model responded and whether a fallback was triggered.

Uses LiteLLM under the hood so both OpenAI and Gemini share the same
request format (OpenAI-style messages / tool schemas).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import litellm
from litellm import acompletion

from config import settings
from llm.retry import RetriesExhaustedError, is_retriable_error, with_retry

logger = logging.getLogger(__name__)

# Keep LiteLLM from printing its own verbose logs.
litellm.set_verbose = False


class LLMResponse:
    """Wrapper returned by the agent-decision (non-streaming) call."""

    def __init__(
        self,
        content: str,
        tool_calls: list[dict],
        model: str,
        is_fallback: bool,
        usage: dict,
        latency_ms: float,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls       # list of {id, name, arguments}
        self.model = model                 # actual model that responded
        self.is_fallback = is_fallback
        self.usage = usage                 # {prompt_tokens, completion_tokens, total_tokens}
        self.latency_ms = latency_ms


class LLMClient:
    """Manages LLM interactions with retry, fallback, and streaming."""

    def __init__(self) -> None:
        self._primary_model = settings.openai_model
        self._fallback_model = settings.gemini_model_litellm
        self._openai_key = settings.openai_api_key
        self._gemini_key = settings.gemini_api_key

    # ------------------------------------------------------------------ #
    # Agent-decision call (non-streaming, returns tool_calls + content)
    # ------------------------------------------------------------------ #

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        on_retry=None,
    ) -> LLMResponse:
        """Call LLM non-streaming. Used for tool-decision phase of the agent loop."""
        start = time.perf_counter()
        is_fallback = False
        model_used = self._primary_model

        try:
            response = await with_retry(
                self._openai_complete,
                messages=messages,
                tools=tools,
                max_retries=settings.max_retries,
                base_delay=settings.retry_base_delay,
                max_delay=settings.retry_max_delay,
                on_retry=on_retry,
            )
        except RetriesExhaustedError as exc:
            logger.warning(
                "OpenAI retries exhausted (%s). Falling back to Gemini.", exc.last_error
            )
            is_fallback = True
            model_used = self._fallback_model
            response = await self._gemini_complete(messages=messages, tools=tools)
        # Non-retriable errors bubble up as-is.

        latency_ms = (time.perf_counter() - start) * 1000
        return self._parse_response(response, model_used, is_fallback, latency_ms)

    # ------------------------------------------------------------------ #
    # Streaming call (for final answer delivery)
    # ------------------------------------------------------------------ #

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        on_retry=None,
    ) -> AsyncGenerator[tuple[str, str, bool], None]:
        """Stream final LLM response tokens.

        Yields (token_text, model_name, is_fallback) tuples.
        """
        is_fallback = False
        model_used = self._primary_model

        try:
            stream_resp = await with_retry(
                self._openai_stream_start,
                messages=messages,
                max_retries=settings.max_retries,
                base_delay=settings.retry_base_delay,
                max_delay=settings.retry_max_delay,
                on_retry=on_retry,
            )
        except RetriesExhaustedError as exc:
            logger.warning(
                "OpenAI stream retries exhausted (%s). Falling back to Gemini.", exc.last_error
            )
            is_fallback = True
            model_used = self._fallback_model
            stream_resp = await self._gemini_stream_start(messages=messages)

        async for chunk in stream_resp:
            delta = self._extract_stream_token(chunk)
            if delta:
                yield delta, model_used, is_fallback

    # ------------------------------------------------------------------ #
    # Internal helpers — OpenAI
    # ------------------------------------------------------------------ #

    async def _openai_complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        kwargs: Dict[str, Any] = dict(
            model=self._primary_model,
            messages=messages,
            api_key=self._openai_key,
            api_base=settings.openai_base_url,
            stream=False,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return await acompletion(**kwargs)

    async def _openai_stream_start(self, messages: List[Dict[str, Any]]):
        return await acompletion(
            model=self._primary_model,
            messages=messages,
            api_key=self._openai_key,
            api_base=settings.openai_base_url,
            stream=True,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers — Gemini (via LiteLLM)
    # ------------------------------------------------------------------ #

    async def _gemini_complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        kwargs: Dict[str, Any] = dict(
            model=self._fallback_model,
            messages=messages,
            api_key=self._gemini_key,
            stream=False,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return await acompletion(**kwargs)

    async def _gemini_stream_start(self, messages: List[Dict[str, Any]]):
        return await acompletion(
            model=self._fallback_model,
            messages=messages,
            api_key=self._gemini_key,
            stream=True,
        )

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #

    def _parse_response(
        self,
        response,
        model: str,
        is_fallback: bool,
        latency_ms: float,
    ) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message

        content: str = message.content or ""
        tool_calls: list[dict] = []

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                import json as _json
                try:
                    arguments = _json.loads(tc.function.arguments)
                except Exception:
                    arguments = {}
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": arguments,
                    }
                )

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=model,
            is_fallback=is_fallback,
            usage=usage,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _extract_stream_token(chunk) -> str:
        """Extract text delta from a streaming chunk (LiteLLM format)."""
        try:
            delta = chunk.choices[0].delta
            return delta.content or ""
        except (AttributeError, IndexError):
            return ""
