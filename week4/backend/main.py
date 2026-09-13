"""
main.py — FastAPI application entry point.

Responsibilities:
  - App factory and lifespan (startup / shutdown hooks).
  - CORS configuration (allows Chainlit UI to call the backend).
  - Health-check endpoint.
  - Mount routers.
  - Initialise singletons: Langfuse tracer, MCP client.

Run locally:
  cd week4/backend
  uvicorn main:app --reload --port 8001
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from mcp_client.client import mcp_client
from observability.langfuse_client import tracer
from routers.chat import router as chat_router

# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Lifespan (startup / shutdown)
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup ---------------------------------------------------- #
    logger.info("Starting GitHub MCP Chatbot backend…")

    # Initialise Langfuse tracing.
    tracer.setup(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        enabled=settings.langfuse_enabled,
    )

    # Connect to the GitHub MCP server.
    await mcp_client.connect(github_token=settings.github_personal_access_token)

    logger.info(
        "Backend ready. MCP connected: %s | Langfuse enabled: %s",
        mcp_client.is_connected,
        tracer._enabled,
    )

    yield  # Application runs here.

    # ---- Shutdown --------------------------------------------------- #
    logger.info("Shutting down backend…")
    await mcp_client.disconnect()
    logger.info("Backend shut down cleanly.")


# ------------------------------------------------------------------ #
# Application
# ------------------------------------------------------------------ #

app = FastAPI(
    title="GitHub MCP Chatbot API",
    description=(
        "Production-grade FastAPI backend powering the GitHub MCP Chatbot. "
        "Provides streaming chat, GitHub tool invocation via MCP, and "
        "write-action approval flows."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Chainlit (and localhost during development) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers.
app.include_router(chat_router)


# ------------------------------------------------------------------ #
# Health check
# ------------------------------------------------------------------ #

@app.get("/health", tags=["meta"])
async def health():
    """Returns the operational status of all backend components."""
    return {
        "status": "ok",
        "mcp_connected": mcp_client.is_connected,
        "langfuse_enabled": tracer._enabled,
        "available_tools": [t["name"] for t in mcp_client.available_tools],
        "primary_model": settings.openai_model,
        "fallback_model": settings.gemini_model,
    }


# ------------------------------------------------------------------ #
# Dev entrypoint
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
        log_level="info",
    )
