"""
frontend/app.py — Chainlit UI for the GitHub MCP Chatbot.

Features:
  - Password-based authentication (CHAINLIT_AUTH_SECRET + CHAINLIT_USERS).
  - Maintains per-session conversation history.
  - Streams LLM tokens from the FastAPI backend via Server-Sent Events.
  - Shows tool-use step indicators (which tool, what input/output).
  - Renders write-action approval dialogs using AskActionMessage.
  - Displays friendly error messages (never raw stack traces).
  - Shows a "Using fallback model (Gemini)" banner when triggered.

Environment variables (set in frontend/.env or inherited from Docker):
  BACKEND_URL      URL of the FastAPI backend  (default: http://localhost:8001)
  CHAINLIT_AUTH_SECRET  Secret key for auth token signing
  CHAINLIT_USERS   Comma-separated "user:pass" pairs

Run locally:
  cd week4/frontend
  chainlit run app.py --port 8000
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import chainlit as cl
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8001")
STREAM_ENDPOINT: str = f"{BACKEND_URL}/chat/stream"
APPROVE_ENDPOINT: str = f"{BACKEND_URL}/chat/approve"
HEALTH_ENDPOINT: str = f"{BACKEND_URL}/health"

# Users dict parsed from "admin:pass,demo:pass" format.
_RAW_USERS = os.getenv("CHAINLIT_USERS", "admin:admin123,demo:demo456")
VALID_USERS: dict[str, str] = {}
for _pair in _RAW_USERS.split(","):
    _pair = _pair.strip()
    if ":" in _pair:
        _u, _, _p = _pair.partition(":")
        VALID_USERS[_u.strip()] = _p.strip()

# HTTP client timeout settings.
_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0)


# ------------------------------------------------------------------ #
# Authentication
# ------------------------------------------------------------------ #

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Validate username/password and return a Chainlit User or None."""
    expected = VALID_USERS.get(username)
    if expected and expected == password:
        return cl.User(identifier=username, metadata={"role": "user"})
    return None


# ------------------------------------------------------------------ #
# Session initialisation
# ------------------------------------------------------------------ #

@cl.on_chat_start
async def on_start():
    """Called when a new chat session opens."""
    user: cl.User = cl.user_session.get("user")  # type: ignore
    username = user.identifier if user else "guest"

    # Initialise empty conversation history for this session.
    cl.user_session.set("history", [])
    cl.user_session.set("username", username)

    # Ping backend health.
    backend_ok = await _check_backend_health()

    welcome = (
        f"👋 Welcome, **{username}**!\n\n"
        "I'm your GitHub MCP assistant. I can:\n"
        "- 📋 **List** your repositories\n"
        "- 🔍 **Search** issues\n"
        "- 📄 **Get** issue details\n"
        "- ✏️ **Create** issues (with your approval)\n"
        "- 💬 **Comment** on issues (with your approval)\n"
        "- 💬 Answer general questions without any GitHub tools\n\n"
    )
    if not backend_ok:
        welcome += (
            "> ⚠️ **Warning:** Could not reach the backend API. "
            f"Check that FastAPI is running at `{BACKEND_URL}`.\n"
        )

    await cl.Message(content=welcome).send()


# ------------------------------------------------------------------ #
# Message handler
# ------------------------------------------------------------------ #

@cl.on_message
async def on_message(message: cl.Message):
    """Called on every user message. Calls FastAPI and streams the response."""
    history: list = cl.user_session.get("history", [])
    session_id: str = cl.user_session.get("id", "unknown")  # type: ignore
    username: str = cl.user_session.get("username", "anonymous")

    # Append user message to history.
    history.append({"role": "user", "content": message.content})

    # Run the streaming agent call.
    assistant_text = await _run_agent_stream(
        messages=history,
        session_id=session_id,
        user_id=username,
    )

    # Append assistant reply to history (only if we got a real response).
    if assistant_text:
        history.append({"role": "assistant", "content": assistant_text})

    cl.user_session.set("history", history)


# ------------------------------------------------------------------ #
# Core streaming helper
# ------------------------------------------------------------------ #

async def _run_agent_stream(
    messages: list,
    session_id: str,
    user_id: str,
    is_resume: bool = False,
    resume_action_id: Optional[str] = None,
    resume_approved: Optional[bool] = None,
) -> str:
    """
    Stream SSE events from the FastAPI backend and update the Chainlit UI.

    Returns the full assistant text produced during this call.
    """
    # Prepare the request payload.
    if is_resume:
        endpoint = APPROVE_ENDPOINT
        payload = {
            "session_id": session_id,
            "action_id": resume_action_id,
            "approved": resume_approved,
        }
    else:
        endpoint = STREAM_ENDPOINT
        payload = {
            "messages": messages,
            "session_id": session_id,
            "user_id": user_id,
        }

    # The main streaming message that accumulates LLM tokens.
    response_msg = cl.Message(content="")
    await response_msg.send()

    accumulated_text = ""
    active_step: Optional[cl.Step] = None

    try:
        async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
            async with client.stream("POST", endpoint, json=payload) as http_resp:
                if http_resp.status_code != 200:
                    body = await http_resp.aread()
                    await _show_error(
                        response_msg,
                        f"Backend returned HTTP {http_resp.status_code}: {body.decode()[:200]}",
                    )
                    return accumulated_text

                async for line in http_resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    raw_json = line[6:].strip()
                    if not raw_json:
                        continue

                    try:
                        event = json.loads(raw_json)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    # ---- Token: stream text to the message ---------- #
                    if event_type == "token":
                        token = event.get("content", "")
                        accumulated_text += token
                        await response_msg.stream_token(token)

                    # ---- Tool start --------------------------------- #
                    elif event_type == "tool_start":
                        tool_name = event.get("tool_name", "unknown_tool")
                        tool_input = event.get("tool_input", {})
                        active_step = cl.Step(
                            name=f"🔧 {_format_tool_name(tool_name)}",
                            type="tool",
                            show_input=True,
                        )
                        active_step.input = json.dumps(tool_input, indent=2)
                        await active_step.send()

                    # ---- Tool result -------------------------------- #
                    elif event_type == "tool_result":
                        if active_step is not None:
                            output = event.get("tool_output", "")
                            active_step.output = output
                            await active_step.update()
                            active_step = None

                    # ---- Approval needed ---------------------------- #
                    elif event_type == "approval_needed":
                        action_id = event.get("action_id")
                        tool_name = event.get("tool_name", "")
                        draft_display = event.get("content", "")

                        # Close the ongoing response message before showing dialog.
                        await response_msg.update()

                        result = await _ask_approval(
                            tool_name=tool_name,
                            draft_display=draft_display,
                        )
                        approved = result is not None and result.get("value") == "approve"

                        # Resume the agent with the approval decision.
                        resume_text = await _run_agent_stream(
                            messages=messages,
                            session_id=session_id,
                            user_id=user_id,
                            is_resume=True,
                            resume_action_id=action_id,
                            resume_approved=approved,
                        )
                        accumulated_text += resume_text
                        return accumulated_text

                    # ---- Model info (fallback triggered) ------------ #
                    elif event_type == "model_info":
                        is_fallback = event.get("is_fallback", False)
                        model = event.get("model", "")
                        if is_fallback:
                            await cl.Message(
                                content=(
                                    f"⚡ *Primary model unavailable — using fallback: **{model}**.*"
                                ),
                                author="System",
                            ).send()

                    # ---- Error -------------------------------------- #
                    elif event_type == "error":
                        error_msg = event.get("content", "Unknown error.")
                        await _show_error(response_msg, error_msg)

                    # ---- Done --------------------------------------- #
                    elif event_type == "done":
                        break

    except httpx.ConnectError:
        await _show_error(
            response_msg,
            f"🔌 Cannot connect to the backend at `{BACKEND_URL}`. "
            "Is the FastAPI server running?",
        )
    except httpx.TimeoutException:
        await _show_error(
            response_msg,
            "⏳ The request timed out. The backend may be overloaded — please try again.",
        )
    except Exception as exc:
        logger.exception("Unexpected error in _run_agent_stream: %s", exc)
        await _show_error(
            response_msg,
            "❌ An unexpected error occurred. Please try again.",
        )

    await response_msg.update()
    return accumulated_text


# ------------------------------------------------------------------ #
# Approval dialog
# ------------------------------------------------------------------ #

async def _ask_approval(tool_name: str, draft_display: str) -> Optional[dict]:
    """Show a Chainlit approval dialog and return the user's choice."""
    verb = "create the issue" if tool_name == "create_issue" else "post the comment"

    response = await cl.AskActionMessage(
        content=(
            f"{draft_display}\n\n"
            f"---\n"
            f"Would you like to **{verb}**?"
        ),
        actions=[
            cl.Action(
                name="approve",
                value="approve",
                label="✅ Approve — Execute",
                description="Confirm and send this to GitHub.",
            ),
            cl.Action(
                name="cancel",
                value="cancel",
                label="❌ Cancel",
                description="Do not execute this action.",
            ),
        ],
        timeout=120,  # 2-minute window for approval.
    ).send()
    return response


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

async def _show_error(msg: cl.Message, text: str) -> None:
    """Update a message with an error banner."""
    msg.content = f"> ⚠️ {text}"
    await msg.update()


async def _check_backend_health() -> bool:
    """Return True if the FastAPI backend is reachable and healthy."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(HEALTH_ENDPOINT)
            return resp.status_code == 200
    except Exception:
        return False


def _format_tool_name(name: str) -> str:
    """Convert snake_case tool names to Title Case for display."""
    return name.replace("_", " ").title()
