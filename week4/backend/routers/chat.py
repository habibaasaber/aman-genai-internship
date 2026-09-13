"""
routers/chat.py — FastAPI chat endpoints.

POST /chat/stream
  Accepts a chat request, runs the agent loop, and streams SSE events back.

POST /chat/approve
  Accepts an approval decision for a pending write action, resumes the agent,
  and streams the final response as SSE.

SSE frame format:
  data: {"type": "token", "content": "Hello "}\n\n
  data: {"type": "tool_start", "tool_name": "list_repositories", "tool_input": {}}\n\n
  data: {"type": "approval_needed", "action_id": "...", "tool_name": "create_issue", "draft": {...}, "content": "..."}\n\n
  data: {"type": "done"}\n\n
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from agent.agent import agent
from models.schemas import ApprovalRequest, ChatRequest, SSEEvent
from observability.langfuse_client import tracer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _sse_frame(event: SSEEvent) -> str:
    """Serialise an SSEEvent into an SSE data frame string."""
    return f"data: {event.model_dump_json(exclude_none=True)}\n\n"


async def _stream_agent_events(
    gen: AsyncGenerator[SSEEvent, None],
) -> AsyncGenerator[str, None]:
    """Wrap an SSEEvent generator into raw SSE text frames."""
    try:
        async for event in gen:
            yield _sse_frame(event)
    except Exception as exc:
        logger.exception("Unexpected error in SSE stream: %s", exc)
        from models.schemas import SSEEventType
        yield _sse_frame(
            SSEEvent(
                type=SSEEventType.error,
                content="❌ An unexpected server error occurred. Please try again.",
            )
        )
        yield _sse_frame(SSEEvent(type=SSEEventType.done))


# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #

@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream an AI response for the given conversation.

    Returns: text/event-stream with SSE frames.
    """
    trace_id = tracer.start_trace(
        name="chat_stream",
        session_id=request.session_id,
        user_id=request.user_id or "anonymous",
        input_data={
            "message_count": len(request.messages),
            "last_user_message": next(
                (m.content for m in reversed(request.messages) if m.role.value == "user"),
                "",
            ),
        },
    )

    gen = await agent.run(
        messages=request.messages,
        session_id=request.session_id,
        user_id=request.user_id or "anonymous",
        trace_id=trace_id,
    )

    return StreamingResponse(
        _stream_agent_events(gen),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering.
            "Connection": "keep-alive",
        },
    )


@router.post("/approve")
async def chat_approve(request: ApprovalRequest):
    """
    Resume a paused agent after the user has approved or rejected a write action.

    Returns: text/event-stream with SSE frames (same format as /chat/stream).
    """
    if not request.action_id:
        raise HTTPException(status_code=400, detail="action_id is required.")

    gen = await agent.resume(
        action_id=request.action_id,
        approved=request.approved,
        trace_id=None,
    )

    return StreamingResponse(
        _stream_agent_events(gen),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
