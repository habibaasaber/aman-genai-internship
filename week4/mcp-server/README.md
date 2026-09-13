# GitHub MCP Server

This directory documents the GitHub MCP server integration.

## Overview

The GitHub MCP server is **not custom code** — it's the official npm package:

```
@modelcontextprotocol/server-github
```

It runs as a child process of the FastAPI backend, communicating over stdio using the Model Context Protocol JSON-RPC protocol.

## How it starts

When the FastAPI backend starts, `mcp_client/client.py` launches:

```bash
npx -y @modelcontextprotocol/server-github
```

The `-y` flag auto-installs the package on first run (cached subsequently).

## Authentication

The server reads the GitHub Personal Access Token from the environment variable:

```
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...
```

This is passed from the FastAPI backend's environment into the subprocess environment.

## Required GitHub Token Scopes

Generate a **Classic** Personal Access Token at https://github.com/settings/tokens with:

- `repo` — Full control of private repositories (needed for create_issue, add_comment)
- `read:org` — Read org membership (needed for list_repositories on orgs)
- `read:user` — Read user profile

For read-only demo use, `public_repo` is sufficient.

## Available Tools

The GitHub MCP server exposes many tools. The chatbot uses these 5:

| Tool | Type | Description |
|------|------|-------------|
| `list_repositories` | Read | List repos for a user/org |
| `search_issues` | Read | Search issues with GitHub query syntax |
| `get_issue` | Read | Get details of a specific issue |
| `create_issue` | **Write** | Create a new issue (requires approval) |
| `add_issue_comment` | **Write** | Add a comment to an issue (requires approval) |

## Tool Name Mapping

The GitHub MCP server's actual tool names may vary slightly between versions.
The backend's `mcp_client/client.py` calls `list_tools()` at startup and logs
all available tool names — check the backend logs if a tool call fails with "not found".

## config.json

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"
      }
    }
  }
}
```

This config is for reference (e.g., if you want to test the MCP server independently
using the MCP Inspector CLI: `npx @modelcontextprotocol/inspector`).
