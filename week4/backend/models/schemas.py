"""backend/models/schemas.py — Shared Pydantic models."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
# Chat messages
# ------------------------------------------------------------------ #

class MessageRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str = ""
    # Present when role == "tool" (maps to a prior tool_call_id)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # tool name for role=="tool" messages


# ------------------------------------------------------------------ #
# API request / response models
# ------------------------------------------------------------------ #

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: str
    user_id: Optional[str] = "anonymous"


class ApprovalRequest(BaseModel):
    session_id: str
    action_id: str
    approved: bool


# ------------------------------------------------------------------ #
# Agent internal state (stored server-side, keyed by action_id)
# ------------------------------------------------------------------ #

class PendingToolCall(BaseModel):
    """A single tool call that the LLM requested."""
    id: str                  # tool_call_id from the LLM response
    name: str                # tool function name
    arguments: Dict[str, Any]
    is_write: bool = False   # True → requires user approval


class AgentState(BaseModel):
    """Persisted state for a paused agent waiting for write-action approval."""
    session_id: str
    action_id: str           # UUID used as lookup key
    user_id: str = "anonymous"
    # The full message history up to the point the agent paused
    messages: List[ChatMessage]
    # All tool calls the LLM requested in the current turn
    pending_tool_calls: List[PendingToolCall]
    # Results already gathered for read-only tools (before pausing)
    completed_tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    # Langfuse trace ID so we can resume the same trace
    trace_id: Optional[str] = None


# ------------------------------------------------------------------ #
# SSE event envelope (streamed from FastAPI → Chainlit)
# ------------------------------------------------------------------ #

class SSEEventType(str, Enum):
    token = "token"                    # LLM text token
    tool_start = "tool_start"          # tool invocation started
    tool_result = "tool_result"        # tool returned a result
    approval_needed = "approval_needed"  # write action needs confirmation
    error = "error"                    # something went wrong
    model_info = "model_info"          # which model is responding
    done = "done"                      # stream complete


class SSEEvent(BaseModel):
    type: SSEEventType
    # Populated depending on event type:
    content: Optional[str] = None          # token text / error message
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None
    action_id: Optional[str] = None        # for approval_needed
    draft: Optional[Dict[str, Any]] = None # draft content for approval
    model: Optional[str] = None            # which model is being used
    is_fallback: bool = False              # True if using fallback model
