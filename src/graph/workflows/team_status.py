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


def format_assignee_presentation(
    assignee: str,
    issues: List[Any],
    index: int,
    total: int,
) -> str:
    """Format one assignee's issues as a structured collection prompt.

    Returns a Markdown block showing the assignee's current sprint issues
    and asking them to provide a status update.

    Args:
        assignee: Assignee display name.
        issues: List of Jira issue dicts for this assignee.
        index: Zero-based position of this assignee in the full list.
        total: Total number of assignees being collected.

    Returns:
        Markdown-formatted string ready for display in the chat UI.
    """
    lines = [
        f"**{assignee}** ({index + 1} of {total})",
        "",
        f"Here are **{assignee}'s** current issues:",
        "",
    ]
    for issue in issues:
        key = issue.get("id", "?")
        summary = issue.get("summary", "(no summary)")
        status = issue.get("status", "Unknown")
        lines.append(f"- **{key}**: {summary}  `{status}`")
    lines.extend([
        "",
        f"{assignee}, please provide your status update:",
    ])
    return "\n".join(lines)


def generate_scrum_master_report(
    board_name: str,
    assignees: List[str],
    issues_by_assignee: Dict[str, List[Any]],
    collected_updates: Dict[str, str],
    llm_service: OllamaService,
) -> str:
    """Synthesize all collected status updates into a scrum master report.

    Builds a structured prompt that combines each assignee's issue list with
    their self-reported update, then delegates to the LLM for the final
    analytical narrative.

    Args:
        board_name: Name of the Scrum board.
        assignees: Ordered list of assignee names.
        issues_by_assignee: Sprint issues grouped by assignee name.
        collected_updates: Self-reported status text keyed by assignee name.
        llm_service: Configured LLM service.

    Returns:
        Markdown-formatted scrum master report.
    """
    started_at = perf_counter()
    payload = {
        "board_name": board_name,
        "team_members": [
            {
                "assignee": assignee,
                "issues": issues_by_assignee.get(assignee, []),
                "status_update": collected_updates.get(assignee, "(no update provided)"),
            }
            for assignee in assignees
        ],
    }
    prompt = llm_service.build_scrum_master_report_prompt(payload)
    report = llm_service.generate(prompt)
    logger.info(
        "Generated scrum master report",
        extra={
            "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            "assignee_count": len(assignees),
            "response_length": len(report),
        },
    )
    return report


def run_team_status_workflow(
    state: JiraGraphState,
    mcp_client: MCPClient,
    llm_service: OllamaService,
    stream: bool = False,
) -> Dict[str, object]:
    """Phase 1 of the team status collection cycle.

    Fetches all active sprint issues for the board, groups them by assignee,
    and returns the first assignee's issues as a structured presentation.
    The returned state carries ``team_status_phase="collecting"`` together
    with the full ``assignees`` list, ``assignee_issues`` map, and an empty
    ``collected_updates`` dict so the UI can resume the loop across
    subsequent user turns.

    Args:
        state: Current workflow state; must contain ``"board_name"`` or a
            parseable ``"user_input"``.
        mcp_client: Configured MCP client used to fetch sprint issues.
        llm_service: Configured LLM service used to infer the board name
            when it cannot be extracted via regex.
        stream: Unused in this phase; preserved for interface compatibility.

    Returns:
        Partial state dict containing:

        * ``"final_response"`` — intro header plus first assignee's issue
          presentation.
        * ``"team_status_phase"`` — ``"collecting"``.
        * ``"assignees"`` — sorted list of all assignee names.
        * ``"assignee_issues"`` — issues grouped by assignee.
        * ``"current_assignee_index"`` — ``0``.
        * ``"collected_updates"`` — empty dict.
        * ``"metadata"`` — observability metadata.
    """
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

            if not by_assignee:
                return {
                    "board_name": board_name,
                    "tool_output": tool_output,
                    "final_response": (
                        f"No active sprint issues found for board **{board_name}**."
                    ),
                    "metadata": {
                        "workflow": "team_status",
                        "phase": "complete",
                        "tool": "get_active_sprint_issues",
                        "team_member_count": 0,
                    },
                }

            assignees = sorted(by_assignee.keys())
            intro = (
                f"## Team Status Collection — {board_name}\n\n"
                f"Found **{len(assignees)} team member(s)** with active issues. "
                "Please provide status updates for each assignee.\n\n"
                "---\n\n"
            )
            first_presentation = format_assignee_presentation(
                assignees[0], by_assignee[assignees[0]], 0, len(assignees)
            )

            logger.info(
                "Initiated team status collection",
                extra={
                    "team_member_count": len(assignees),
                    "first_assignee": assignees[0],
                },
            )

            return {
                "board_name": board_name,
                "tool_output": tool_output,
                "final_response": intro + first_presentation,
                "team_status_phase": "collecting",
                "assignees": assignees,
                "assignee_issues": by_assignee,
                "current_assignee_index": 0,
                "collected_updates": {},
                "metadata": {
                    "workflow": "team_status",
                    "phase": "collecting",
                    "tool": "get_active_sprint_issues",
                    "team_member_count": len(assignees),
                    "current_assignee": assignees[0],
                },
            }
