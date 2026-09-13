"""
mcp_client/client.py — Persistent MCP client for the GitHub MCP server.

Manages a long-lived subprocess connection to the official GitHub MCP server
(@modelcontextprotocol/server-github) via stdio JSON-RPC using the MCP
Python SDK.

Lifecycle:
  - connect()    called once at FastAPI startup
  - disconnect() called at shutdown
  - call_tool()  called per-request by the agent

Error handling:
  - Timeouts raise asyncio.TimeoutError → retriable
  - Process crashes raise RuntimeError → surfaced to user
  - Tool errors return structured error strings (not exceptions)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default tool call timeout (seconds).
TOOL_TIMEOUT_SECONDS = 30


class MCPToolError(Exception):
    """Raised when a tool call returns an error from the MCP server."""
    pass


class MCPClient:
    """Wrapper around the GitHub MCP server subprocess (stdio transport)."""

    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()
        self._session = None
        self._available_tools: List[Dict[str, Any]] = []
        self._connected = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def connect(self, github_token: str) -> None:
        """Start the GitHub MCP server and perform the MCP handshake."""
        if not github_token:
            logger.warning(
                "GITHUB_PERSONAL_ACCESS_TOKEN is not set. "
                "GitHub tools will be unavailable."
            )
            return

        try:
            from mcp.client.stdio import stdio_client, StdioServerParameters  # type: ignore
            from mcp import ClientSession  # type: ignore
        except ImportError:
            logger.error(
                "mcp package not installed. Run: pip install mcp"
            )
            return

        # Inherit PATH so npx can be found.
        env = {
            "GITHUB_PERSONAL_ACCESS_TOKEN": github_token,
        }
        for key in ("PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
            if key in os.environ:
                env[key] = os.environ[key]

        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env=env,
        )

        logger.info("Starting GitHub MCP server via npx…")
        try:
            read, write = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await asyncio.wait_for(self._session.initialize(), timeout=60)
            self._connected = True
            logger.info("GitHub MCP server connected successfully.")

            # Cache available tools.
            tools_result = await self._session.list_tools()
            self._available_tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema if hasattr(t, "inputSchema") else {},
                }
                for t in tools_result.tools
            ]
            logger.info(
                "Available MCP tools: %s",
                [t["name"] for t in self._available_tools],
            )

        except Exception as exc:
            logger.error("Failed to connect to GitHub MCP server: %s", exc)
            self._connected = False

    async def disconnect(self) -> None:
        """Shut down the MCP subprocess cleanly."""
        try:
            await self._exit_stack.aclose()
        except Exception as exc:
            logger.debug("MCP disconnect error (ignored): %s", exc)
        self._connected = False
        logger.info("GitHub MCP server disconnected.")

    # ------------------------------------------------------------------ #
    # Tool invocation
    # ------------------------------------------------------------------ #

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout: float = TOOL_TIMEOUT_SECONDS,
    ) -> str:
        """
        Call a GitHub MCP tool by name and return its output as a string.

        Raises:
            RuntimeError:         MCP server not connected.
            MCPToolError:         Tool returned an error response.
            asyncio.TimeoutError: Call exceeded *timeout* seconds.
        """
        if not self._connected or self._session is None:
            raise RuntimeError(
                "GitHub MCP server is not connected. "
                "Check that GITHUB_PERSONAL_ACCESS_TOKEN is set and npx is available."
            )

        logger.debug("MCP call_tool: %s(%s)", name, arguments)
        start = time.perf_counter()

        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("MCP tool %s timed out after %.1fs.", name, timeout)
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        logger.debug("MCP tool %s completed in %.1fms.", name, latency_ms)

        # Extract text content from the result.
        if result.isError:
            error_text = self._extract_text(result.content)
            raise MCPToolError(f"GitHub tool '{name}' failed: {error_text}")

        return self._extract_text(result.content)

    # ------------------------------------------------------------------ #
    # Tool discovery
    # ------------------------------------------------------------------ #

    @property
    def available_tools(self) -> List[Dict[str, Any]]:
        """Return tool metadata for the LLM function-calling schema."""
        return self._available_tools

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_text(content) -> str:
        """Pull plain text out of MCP content (list of TextContent / ImageContent)."""
        if not content:
            return ""
        parts = []
        for item in content:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)


# Module-level singleton.
mcp_client = MCPClient()
