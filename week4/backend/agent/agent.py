"""
agent/agent.py — Core AI agent loop.

The agent implements a ReAct-style loop:
  1. Call LLM (non-streaming) with conversation history + tool schemas.
  2. If the LLM requests tool calls:
     a. Execute all *read* tools immediately (no approval needed).
     b. If any *write* tools are requested → pause, emit approval_needed,
        save pending state so the route can resume after user confirms.
  3. When no tool calls remain → stream the final answer token-by-token.

The agent is a pure async generator that yields SSEEvent objects.
The FastAPI route serialises them to Server-Sent Events.

State management for pending approvals uses an in-memory dict keyed by
a UUID (action_id). In production, replace with Redis or a DB-backed store.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from agent.tools import (
    GITHUB_TOOL_SCHEMAS,
    format_draft_for_display,
    is_write_tool,
)
from llm.client import LLMClient
from mcp_client.client import MCPToolError, mcp_client
from models.schemas import (
    AgentState,
    ChatMessage,
    MessageRole,
    PendingToolCall,
    SSEEvent,
    SSEEventType,
)
from observability.langfuse_client import tracer

logger = logging.getLogger(__name__)

# In-memory store for pending write-action approvals.
# key: action_id (str UUID) → AgentState
_PENDING_APPROVALS: Dict[str, AgentState] = {}

# System prompt that shapes agent behaviour.
_SYSTEM_PROMPT = """You are a helpful AI assistant with access to GitHub tools.

When a user asks something that doesn't require GitHub (e.g., general questions, explanations, code help), answer directly without using any tools.

When GitHub access is needed, use the appropriate tool(s). Follow these rules strictly:
- Use ONLY the tools provided; do not invent tool names.
- For write operations (create_issue, add_issue_comment): first use read tools if needed to gather context, then call the write tool with well-formed, detailed content. The system will ask the user for approval before executing write actions — do NOT warn the user yourself; just call the tool.
- For search queries, compose proper GitHub search syntax (e.g., "is:open is:issue repo:owner/repo").
- Always reason from the user's intent, not just their literal words.

If a GitHub operation fails or returns an error, explain it clearly to the user in plain language.
"""


def _messages_to_dicts(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Convert Pydantic ChatMessage list to plain dicts for LiteLLM."""
    result = []
    for m in messages:
        d: Dict[str, Any] = {"role": m.role.value, "content": m.content}
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.name:
            d["name"] = m.name
        result.append(d)
    return result


def _user_friendly_error(exc: Exception) -> str:
    """Translate raw exceptions into clear user-facing messages."""
    msg = str(exc)

    if "rate limit" in msg.lower() or "429" in msg:
        return (
            "⚠️ The AI service is temporarily rate-limited. "
            "Please wait a moment and try again."
        )
    if "authentication" in msg.lower() or "401" in msg or "403" in msg:
        return "🔐 Authentication failed. Please check your API keys."
    if "timeout" in msg.lower():
        return "⏳ The request timed out. Please try again."
    if "not connected" in msg.lower() or "mcp" in msg.lower():
        return (
            "🔌 Could not reach the GitHub MCP server. "
            "Ensure GITHUB_PERSONAL_ACCESS_TOKEN is set."
        )
    if isinstance(exc, MCPToolError):
        return f"🐙 GitHub error: {msg}"

    return f"❌ Something went wrong: {msg}"


class Agent:
    """Stateless agent — each call runs an independent loop."""

    def __init__(self) -> None:
        self._llm = LLMClient()

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    async def run(
        self,
        messages: List[ChatMessage],
        session_id: str,
        user_id: str = "anonymous",
        trace_id: Optional[str] = None,
    ) -> AsyncGenerator[SSEEvent, None]:
        """
        Run the agent loop and yield SSEEvent objects.

        Callers (FastAPI routes) iterate over this generator and serialise
        each event to an SSE frame.
        """
        return self._agent_loop(
            messages=messages,
            session_id=session_id,
            user_id=user_id,
            trace_id=trace_id,
        )

    async def resume(
        self,
        action_id: str,
        approved: bool,
        trace_id: Optional[str] = None,
    ) -> AsyncGenerator[SSEEvent, None]:
        """
        Resume a paused agent after the user has approved/rejected a write action.
        """
        return self._resume_loop(
            action_id=action_id,
            approved=approved,
            trace_id=trace_id,
        )

    # ------------------------------------------------------------------ #
    # Internal loops
    # ------------------------------------------------------------------ #

    async def _agent_loop(
        self,
        messages: List[ChatMessage],
        session_id: str,
        user_id: str,
        trace_id: Optional[str],
    ) -> AsyncGenerator[SSEEvent, None]:
        """Core ReAct loop; yields SSEEvents until done or approval_needed."""

        # Build conversation with system prompt prepended.
        full_messages = [
            ChatMessage(role=MessageRole.system, content=_SYSTEM_PROMPT)
        ] + messages

        # Determine which tools to expose (only when MCP is connected).
        tools = GITHUB_TOOL_SCHEMAS if mcp_client.is_connected else []

        max_iterations = 10  # Safety cap to prevent infinite loops.

        async def _on_retry(attempt: int, exc: Exception):
            tracer.log_retry(attempt=attempt, error=exc, model="openai")
            yield SSEEvent(
                type=SSEEventType.error,
                content=f"⚠️ LLM request failed (attempt {attempt}), retrying…",
            )

        for iteration in range(max_iterations):
            # ---- Step 1: Non-streaming LLM call for tool decisions ---- #
            try:
                llm_resp = await self._llm.complete(
                    messages=_messages_to_dicts(full_messages),
                    tools=tools if tools else None,
                    on_retry=lambda attempt, exc: (
                        tracer.log_retry(attempt=attempt, error=exc, model="openai")
                    ),
                )
            except Exception as exc:
                tracer.log_error(
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
                yield SSEEvent(
                    type=SSEEventType.error,
                    content=_user_friendly_error(exc),
                )
                yield SSEEvent(type=SSEEventType.done)
                return

            # Log the LLM call to Langfuse.
            tracer.log_llm_call(
                name=f"agent_decision_iter{iteration}",
                model=llm_resp.model,
                input_messages=_messages_to_dicts(full_messages),
                output=llm_resp.content,
                usage=llm_resp.usage,
                latency_ms=llm_resp.latency_ms,
                is_fallback=llm_resp.is_fallback,
            )

            if llm_resp.is_fallback:
                tracer.log_fallback(
                    from_model="openai",
                    to_model=llm_resp.model,
                    reason="retries_exhausted",
                )
                yield SSEEvent(
                    type=SSEEventType.model_info,
                    model=llm_resp.model,
                    is_fallback=True,
                )

            # ---- Step 2: No tool calls → stream final answer --------- #
            if not llm_resp.tool_calls:
                if llm_resp.content:
                    # Append assistant turn to history.
                    full_messages.append(
                        ChatMessage(
                            role=MessageRole.assistant,
                            content=llm_resp.content,
                        )
                    )
                # Stream the final answer.
                async for token, model, is_fb in self._llm.stream(
                    messages=_messages_to_dicts(full_messages[:-1]),  # history without the dup
                ):
                    yield SSEEvent(
                        type=SSEEventType.token,
                        content=token,
                        model=model,
                        is_fallback=is_fb,
                    )
                yield SSEEvent(type=SSEEventType.done)
                return

            # ---- Step 3: Process tool calls ----------------------- #
            tracer.log_agent_decision(
                chosen_tools=[tc["name"] for tc in llm_resp.tool_calls]
            )

            # Add the assistant's tool-call turn to history.
            # We need to reconstruct it as the raw dict for LiteLLM.
            full_messages.append(
                ChatMessage(
                    role=MessageRole.assistant,
                    content=llm_resp.content or "",
                )
            )

            # Separate read vs write tool calls.
            read_calls = [tc for tc in llm_resp.tool_calls if not is_write_tool(tc["name"])]
            write_calls = [tc for tc in llm_resp.tool_calls if is_write_tool(tc["name"])]

            # Execute read tools immediately.
            pending_tool_calls = []
            completed_results = []

            for tc in read_calls:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                tool_id = tc["id"]

                yield SSEEvent(
                    type=SSEEventType.tool_start,
                    tool_name=tool_name,
                    tool_input=tool_args,
                )

                start_t = time.perf_counter()
                try:
                    result = await mcp_client.call_tool(tool_name, tool_args)
                    latency_ms = (time.perf_counter() - start_t) * 1000
                    tracer.log_tool_call(
                        tool_name=tool_name,
                        tool_input=tool_args,
                        tool_output=result,
                        latency_ms=latency_ms,
                    )
                    yield SSEEvent(
                        type=SSEEventType.tool_result,
                        tool_name=tool_name,
                        tool_output=result[:500] + "…" if len(result) > 500 else result,
                    )
                    # Add tool result to history.
                    full_messages.append(
                        ChatMessage(
                            role=MessageRole.tool,
                            content=result,
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )
                    completed_results.append(
                        {"tool_call_id": tool_id, "name": tool_name, "result": result}
                    )

                except Exception as exc:
                    latency_ms = (time.perf_counter() - start_t) * 1000
                    tracer.log_tool_call(
                        tool_name=tool_name,
                        tool_input=tool_args,
                        tool_output=None,
                        latency_ms=latency_ms,
                        error=str(exc),
                    )
                    error_msg = _user_friendly_error(exc)
                    full_messages.append(
                        ChatMessage(
                            role=MessageRole.tool,
                            content=f"ERROR: {error_msg}",
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )

            # Build pending tool call models for write actions.
            for tc in write_calls:
                pending_tool_calls.append(
                    PendingToolCall(
                        id=tc["id"],
                        name=tc["name"],
                        arguments=tc["arguments"],
                        is_write=True,
                    )
                )

            # If there are write actions → pause and ask for approval.
            if pending_tool_calls:
                action_id = str(uuid.uuid4())
                first_write = pending_tool_calls[0]

                state = AgentState(
                    session_id=session_id,
                    action_id=action_id,
                    user_id=user_id,
                    messages=full_messages,
                    pending_tool_calls=pending_tool_calls,
                    completed_tool_results=completed_results,
                    trace_id=trace_id,
                )
                _PENDING_APPROVALS[action_id] = state

                draft_display = format_draft_for_display(
                    first_write.name, first_write.arguments
                )
                tracer.log_write_approval(
                    tool_name=first_write.name,
                    approved=False,  # pending
                    draft=first_write.arguments,
                )
                yield SSEEvent(
                    type=SSEEventType.approval_needed,
                    action_id=action_id,
                    tool_name=first_write.name,
                    draft=first_write.arguments,
                    content=draft_display,
                )
                yield SSEEvent(type=SSEEventType.done)
                return

            # All tool calls were read-only → loop back.

        # Safety: exceeded max iterations without a final answer.
        yield SSEEvent(
            type=SSEEventType.error,
            content="❌ The agent exceeded its maximum number of reasoning steps. Please rephrase your request.",
        )
        yield SSEEvent(type=SSEEventType.done)

    async def _resume_loop(
        self,
        action_id: str,
        approved: bool,
        trace_id: Optional[str],
    ) -> AsyncGenerator[SSEEvent, None]:
        """Resume the agent after a write-action approval/rejection."""
        state = _PENDING_APPROVALS.pop(action_id, None)
        if state is None:
            yield SSEEvent(
                type=SSEEventType.error,
                content="❌ This approval request has expired or was already handled. Please try your request again.",
            )
            yield SSEEvent(type=SSEEventType.done)
            return

        tracer.log_write_approval(
            tool_name=state.pending_tool_calls[0].name,
            approved=approved,
            draft=state.pending_tool_calls[0].arguments,
        )

        if not approved:
            yield SSEEvent(
                type=SSEEventType.token,
                content="✋ Action cancelled. The write operation was not executed. Let me know if you'd like to do something else.",
            )
            yield SSEEvent(type=SSEEventType.done)
            return

        full_messages = list(state.messages)

        # Execute all pending write tools.
        for tc in state.pending_tool_calls:
            yield SSEEvent(
                type=SSEEventType.tool_start,
                tool_name=tc.name,
                tool_input=tc.arguments,
            )

            start_t = time.perf_counter()
            try:
                result = await mcp_client.call_tool(tc.name, tc.arguments)
                latency_ms = (time.perf_counter() - start_t) * 1000
                tracer.log_tool_call(
                    tool_name=tc.name,
                    tool_input=tc.arguments,
                    tool_output=result,
                    latency_ms=latency_ms,
                )
                yield SSEEvent(
                    type=SSEEventType.tool_result,
                    tool_name=tc.name,
                    tool_output=result[:500] + "…" if len(result) > 500 else result,
                )
                full_messages.append(
                    ChatMessage(
                        role=MessageRole.tool,
                        content=result,
                        tool_call_id=tc.id,
                        name=tc.name,
                    )
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - start_t) * 1000
                error_msg = _user_friendly_error(exc)
                tracer.log_tool_call(
                    tool_name=tc.name,
                    tool_input=tc.arguments,
                    tool_output=None,
                    latency_ms=latency_ms,
                    error=str(exc),
                )
                tracer.log_error(
                    error_type=type(exc).__name__,
                    message=str(exc),
                    context={"tool_name": tc.name},
                )
                full_messages.append(
                    ChatMessage(
                        role=MessageRole.tool,
                        content=f"ERROR: {error_msg}",
                        tool_call_id=tc.id,
                        name=tc.name,
                    )
                )

        # Generate final streaming answer after write execution.
        async for token, model, is_fb in self._llm.stream(
            messages=_messages_to_dicts(full_messages)
        ):
            yield SSEEvent(
                type=SSEEventType.token,
                content=token,
                model=model,
                is_fallback=is_fb,
            )

        yield SSEEvent(type=SSEEventType.done)


# Module-level singleton.
agent = Agent()
