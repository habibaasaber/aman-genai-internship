"""
agent/tools.py — Tool definitions for the GitHub MCP agent.

These tool schemas are passed to the LLM for function-calling / tool-use.
They describe the 5 required GitHub tools plus a small set of helpers
the LLM can use to reason about which GitHub actions are appropriate.

WRITE_TOOLS — set of tool names that require explicit user approval before
execution.  All other tools are READ_TOOLS and execute without confirmation.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ------------------------------------------------------------------ #
# Tool categories
# ------------------------------------------------------------------ #

READ_TOOLS: frozenset[str] = frozenset(
    {
        "list_repositories",
        "search_issues",
        "get_issue",
        "list_issue_comments",
    }
)

WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "create_issue",
        "add_issue_comment",
    }
)

ALL_TOOLS: frozenset[str] = READ_TOOLS | WRITE_TOOLS


def is_write_tool(name: str) -> bool:
    """Return True if *name* is a write (mutation) tool requiring approval."""
    return name in WRITE_TOOLS


# ------------------------------------------------------------------ #
# OpenAI / LiteLLM function-calling schema
# ------------------------------------------------------------------ #
# These schemas are passed as the `tools` parameter to the LLM.
# The schema format follows OpenAI's function-calling spec which LiteLLM
# forwards to Gemini automatically.

GITHUB_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    # ---- Read tools -------------------------------------------------- #
    {
        "type": "function",
        "function": {
            "name": "list_repositories",
            "description": (
                "List GitHub repositories for a user or organization. "
                "Use this when the user wants to see what repos exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "GitHub username or organization name. "
                        "If not specified, uses the token owner.",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["all", "owner", "public", "private", "member"],
                        "description": "Filter by repository type (default: all).",
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Number of results per page (max 100, default 30).",
                        "default": 30,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_issues",
            "description": (
                "Search for GitHub issues or pull requests using GitHub's search syntax. "
                "Use this to find issues by keyword, label, state, or assignee."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "GitHub search query string. Examples: "
                            "'repo:owner/repo is:open label:bug', "
                            "'is:open is:issue authentication in:title'. "
                            "Append 'is:issue' to exclude PRs."
                        ),
                    },
                    "per_page": {
                        "type": "integer",
                        "description": "Number of results (max 100, default 10).",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_issue",
            "description": (
                "Retrieve the full details of a specific GitHub issue by number, "
                "including its body, state, labels, assignees, and comments count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner (username or org).",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name (without the owner prefix).",
                    },
                    "issue_number": {
                        "type": "integer",
                        "description": "The number of the issue to retrieve.",
                    },
                },
                "required": ["owner", "repo", "issue_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_issue_comments",
            "description": "List the comments on a specific GitHub issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner."},
                    "repo": {"type": "string", "description": "Repository name."},
                    "issue_number": {
                        "type": "integer",
                        "description": "Issue number.",
                    },
                },
                "required": ["owner", "repo", "issue_number"],
            },
        },
    },
    # ---- Write tools (require approval) ------------------------------ #
    {
        "type": "function",
        "function": {
            "name": "create_issue",
            "description": (
                "Create a new GitHub issue in a repository. "
                "IMPORTANT: This is a write action — always draft the issue content "
                "based on the user's request and return it for user review before calling this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Repository owner.",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Issue title (concise, descriptive).",
                    },
                    "body": {
                        "type": "string",
                        "description": "Issue body in Markdown. Include: description, steps to reproduce (if a bug), expected vs actual behaviour, and any relevant context.",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels to apply (optional).",
                    },
                    "assignees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "GitHub usernames to assign (optional).",
                    },
                },
                "required": ["owner", "repo", "title", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_issue_comment",
            "description": (
                "Add a comment to an existing GitHub issue. "
                "IMPORTANT: This is a write action — retrieve the issue first with get_issue, "
                "then draft the comment body and return it for user approval before posting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Repository owner."},
                    "repo": {"type": "string", "description": "Repository name."},
                    "issue_number": {
                        "type": "integer",
                        "description": "Issue number to comment on.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Comment body in Markdown.",
                    },
                },
                "required": ["owner", "repo", "issue_number", "body"],
            },
        },
    },
]


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return the list of tool schemas to pass to the LLM."""
    return GITHUB_TOOL_SCHEMAS


def format_draft_for_display(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Return a human-readable draft string shown to the user for approval."""
    if tool_name == "create_issue":
        lines = [
            f"**📝 Draft Issue — `{arguments.get('owner', '?')}/{arguments.get('repo', '?')}`**",
            "",
            f"**Title:** {arguments.get('title', '(none)')}",
            "",
            "**Body:**",
            "```",
            arguments.get("body", "(empty)"),
            "```",
        ]
        if arguments.get("labels"):
            lines.append(f"\n**Labels:** {', '.join(arguments['labels'])}")
        if arguments.get("assignees"):
            lines.append(f"**Assignees:** {', '.join(arguments['assignees'])}")
        return "\n".join(lines)

    elif tool_name == "add_issue_comment":
        return "\n".join(
            [
                f"**💬 Draft Comment — `{arguments.get('owner', '?')}/{arguments.get('repo', '?')}` Issue #{arguments.get('issue_number', '?')}**",
                "",
                "**Comment:**",
                "```",
                arguments.get("body", "(empty)"),
                "```",
            ]
        )
    else:
        return f"**Tool:** `{tool_name}`\n**Arguments:** {arguments}"
