"""Team member status workflow node."""

from __future__ import annotations

import asyncio
import json
from time import perf_counter
from typing import Any, Dict, List

from src.client.mcp_client import MCPClient, MCPClientError
from src.graph.parsing import extract_board_name
from src.graph.state import JiraGraphState
from src.llm.ollama import OllamaService
from src.logging_config import get_logger, logging_context


logger = get_logger(__name__)


TEAM_STATUS_TOOL_ERROR = (
    "I couldn't fetch team member status updates right now. "
    "Please check the MCP server connection and try again."
)
TEAM_STATUS_LLM_ERROR = (
    "I collected the team status data, but I couldn't format the report right now. Please try again."
)


def _extract_assignee_name(issue: Dict[str, Any]) -> str:
    """Return the assignee display name from a Jira issue dict, handling multiple response shapes."""
    raw_assignee = issue.get("assignee")
    if isinstance(raw_assignee, str) and raw_assignee.strip():
        return raw_assignee.strip()
    if isinstance(raw_assignee, dict):
        name = (
            raw_assignee.get("displayName")
            or raw_assignee.get("name")
            or raw_assignee.get("emailAddress")
        )
        if isinstance(name, str) and name.strip():
            return name.strip()

    fields = issue.get("fields")
    if isinstance(fields, dict):
        nested = fields.get("assignee")
        if isinstance(nested, dict):
            name = nested.get("displayName") or nested.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()

    for key in ("assignee_name", "assigned_to", "owner"):
        val = issue.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return "Unassigned"


def _group_issues_by_assignee(issues: List[Any]) -> Dict[str, List[Any]]:
    """Group a list of Jira issue dicts by their assignee name."""
    grouped: Dict[str, List[Any]] = {}
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        assignee = _extract_assignee_name(issue)
        grouped.setdefault(assignee, []).append(issue)
    return grouped


def run_team_status_workflow(
    state: JiraGraphState,
    mcp_client: MCPClient,
    llm_service: OllamaService,
    stream: bool = False,
) -> Dict[str, object]:
    board_name = state.get("board_name") or extract_board_name(state.get("user_input", ""))
    with logging_context(workflow="team_status", board_name=board_name or None):
        if not board_name:
            logger.info("Board name not parsed from input; asking LLM to infer it")
            board_name = llm_service.infer_board_name(state.get("user_input", "") or "")

        with logging_context(board_name=board_name or None):
            if not board_name:
                logger.warning("Team status request missing board name")
                return {
                    "requires_clarification": True,
                    "error": "Missing board name",
                    "final_response": (
                        "Please include the scrum board name in your request "
                        "so I can collect team member status updates."
                    ),
                }

            tool_started_at = perf_counter()
            try:
                logger.info("Fetching active sprint issues for team status")
                tool_output = asyncio.run(
                    mcp_client.call_tool(
                        "get_active_sprint_issues", {"scrum_board_name": board_name}
                    )
                )
                logger.info(
                    "Fetched active sprint issues",
                    extra={
                        "duration_ms": round((perf_counter() - tool_started_at) * 1000, 2),
                        "text_length": len(tool_output.get("text", "")),
                    },
                )
            except MCPClientError:
                logger.warning(
                    "Team status tool call failed",
                    extra={"error_type": "MCPClientError"},
                )
                return {
                    "board_name": board_name,
                    "error": TEAM_STATUS_TOOL_ERROR,
                }

            issues_text = tool_output.get("text", "")
            try:
                raw_data = json.loads(issues_text)
            except json.JSONDecodeError:
                raw_data = None

            if isinstance(raw_data, list):
                issues_list: List[Any] = raw_data
            elif isinstance(raw_data, dict):
                issues_list = raw_data.get("issues", [])
                if not isinstance(issues_list, list):
                    issues_list = []
            else:
                issues_list = []

            by_assignee = _group_issues_by_assignee(issues_list)
            logger.info(
                "Grouped sprint issues by assignee",
                extra={"team_member_count": len(by_assignee)},
            )

            team_status_payload: Dict[str, Any] = {
                "board_name": board_name,
                "team_members": [
                    {"assignee": assignee, "issues": member_issues}
                    for assignee, member_issues in sorted(by_assignee.items())
                ],
            }

            prompt_args = {
                "team_status_json": json.dumps(team_status_payload),
            }

            prompt_started_at = perf_counter()
            prompt_text = ""
            try:
                logger.info("Fetching team status prompt template")
                prompt_output = asyncio.run(
                    mcp_client.run_prompt("format_team_status_report", prompt_args)
                )
                prompt_text = prompt_output.get("text", "").strip()
                if not prompt_text:
                    logger.warning(
                        "Team status prompt template returned empty text; "
                        "falling back to direct LLM prompt"
                    )
                    prompt_text = llm_service.build_team_status_prompt(team_status_payload)
            except MCPClientError:
                logger.warning(
                    "Team status prompt failed; falling back to direct LLM summarization"
                )
                prompt_text = llm_service.build_team_status_prompt(team_status_payload)

            formatted = "" if stream else (llm_service.generate(prompt_text) if prompt_text else "")
            if not formatted and not stream:
                logger.warning("Team status LLM response was empty")
                return {
                    "board_name": board_name,
                    "tool_output": tool_output,
                    "error": TEAM_STATUS_LLM_ERROR,
                    "metadata": {
                        "workflow": "team_status",
                        "tool": "get_active_sprint_issues",
                        "prompt": "format_team_status_report",
                        "team_member_count": len(by_assignee),
                    },
                }

            logger.info(
                "Generated team status report",
                extra={
                    "duration_ms": round((perf_counter() - prompt_started_at) * 1000, 2),
                    "response_length": len(formatted),
                    "team_member_count": len(by_assignee),
                },
            )

            return {
                "board_name": board_name,
                "tool_output": tool_output,
                "prompt_output": prompt_text,
                "final_response": formatted,
                "metadata": {
                    "workflow": "team_status",
                    "tool": "get_active_sprint_issues",
                    "prompt": "format_team_status_report",
                    "team_member_count": len(by_assignee),
                },
            }
