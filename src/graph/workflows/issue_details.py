"""Issue details workflow node."""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Dict

from src.client.mcp_client import MCPClient, MCPClientError
from src.graph.parsing import extract_issue_key
from src.graph.state import JiraGraphState
from src.llm.ollama import OllamaService
from src.logging_config import get_logger, logging_context


logger = get_logger(__name__)


ISSUE_DETAILS_TOOL_ERROR = (
    "I couldn't fetch Jira issue details right now. Please check the MCP server connection and try again."
)
ISSUE_DETAILS_PROMPT_ERROR = (
    "I fetched the issue details, but I couldn't format the response right now. Please try again."
)


def run_issue_details_workflow(
    state: JiraGraphState,
    mcp_client: MCPClient,
    llm_service: OllamaService,
    stream: bool = False,
) -> Dict[str, object]:
    issue_key = state.get("issue_key") or extract_issue_key(state.get("user_input", ""))
    with logging_context(workflow="issue_details", issue_key=issue_key or None):
        if not issue_key:
            logger.warning("Issue details request missing issue key")
            return {
                "requires_clarification": True,
                "error": "Missing issue key",
                "final_response": (
                    "Please provide a Jira issue key like ABC-123 so I can fetch the issue details."
                ),
            }

        tool_started_at = perf_counter()
        try:
            logger.info("Fetching Jira issue details")
            tool_output = asyncio.run(
                mcp_client.call_tool("get_issue_details", {"issue_id": issue_key})
            )
            logger.info(
                "Fetched Jira issue details",
                extra={
                    "duration_ms": round((perf_counter() - tool_started_at) * 1000, 2),
                    "text_length": len(tool_output.get("text", "")),
                },
            )
        except MCPClientError as exc:
            logger.warning("Issue details tool call failed", extra={"error_type": type(exc).__name__})
            return {
                "issue_key": issue_key,
                "error": ISSUE_DETAILS_TOOL_ERROR,
            }

        prompt_args = {"issue_json": tool_output.get("text", "")}

        prompt_started_at = perf_counter()
        try:
            logger.info("Fetching issue details prompt template")
            prompt_output = asyncio.run(
                mcp_client.run_prompt("format_issue_details", prompt_args)
            )
            prompt_text = prompt_output.get("text", "").strip()
            if not prompt_text:
                logger.warning("Issue details prompt template returned empty text")

            logger.info("Formatting issue details response via LLM")
            final_text = "" if stream else (llm_service.generate(prompt_text) if prompt_text else "")
            if not final_text:
                logger.warning("Issue details LLM response was empty")
            logger.info(
                "Formatted issue details response",
                extra={
                    "duration_ms": round((perf_counter() - prompt_started_at) * 1000, 2),
                    "response_length": len(final_text),
                },
            )
        except MCPClientError as exc:
            logger.warning(
                "Issue details prompt failed",
                extra={"error_type": type(exc).__name__},
            )
            return {
                "issue_key": issue_key,
                "tool_output": tool_output,
                "error": ISSUE_DETAILS_PROMPT_ERROR,
                "metadata": {
                    "workflow": "issue_details",
                    "tool": "get_issue_details",
                    "prompt": "format_issue_details",
                },
            }

        return {
            "issue_key": issue_key,
            "tool_output": tool_output,
            "prompt_output": prompt_text,
            "final_response": final_text,
            "metadata": {
                "workflow": "issue_details",
                "tool": "get_issue_details",
                "prompt": "format_issue_details",
            },
        }
