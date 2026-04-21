"""Sprint insights workflow node."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta
from time import perf_counter
from typing import Dict

from src.client.mcp_client import MCPClient, MCPClientError
from src.graph.parsing import extract_board_name
from src.graph.state import JiraGraphState
from src.llm.ollama import OllamaService
from src.logging_config import get_logger, logging_context


logger = get_logger(__name__)


SPRINT_INSIGHTS_TOOL_ERROR = (
    "I couldn't fetch active sprint issues right now. Please check the MCP server connection and try again."
)
SPRINT_INSIGHTS_PROMPT_ERROR = (
    "I analyzed the sprint data, but I couldn't format the sprint insights right now. Please try again."
)


def _as_non_empty_string(value: object, default: str) -> str:
    """Return *value* as a stripped string, or *default* if blank or ``None``.

    Args:
        value: Any object; converted to ``str`` before stripping.
        default: Fallback string returned when *value* is ``None`` or
            produces an empty string after stripping.

    Returns:
        Stripped string value, or *default*.
    """
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_iso_date(value: object, fallback: date) -> str:
    """Convert *value* to an ISO-8601 date string (``YYYY-MM-DD``).

    Handles the following input types:

    * :class:`datetime.datetime` — converted via ``.date().isoformat()``.
    * :class:`datetime.date` — converted via ``.isoformat()``.
    * ``str`` — stripped and parsed as ISO-8601; a trailing ``"Z"`` is
      replaced with ``"+00:00"`` for compatibility with
      :func:`datetime.fromisoformat`.

    Returns *fallback* formatted as ISO-8601 for any unrecognised type or
    parse failure.

    Args:
        value: Date value in one of the supported formats described above.
        fallback: :class:`datetime.date` to use when *value* cannot be
            parsed.

    Returns:
        An ISO-8601 date string in ``YYYY-MM-DD`` format.
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        text = value.strip()
        if text:
            normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
            try:
                return datetime.fromisoformat(normalized).date().isoformat()
            except ValueError:
                try:
                    return date.fromisoformat(text).isoformat()
                except ValueError:
                    pass
    return fallback.isoformat()


def _normalize_sprint_meta(raw_meta: object, board_name: str) -> Dict[str, object]:
    """Normalise a raw sprint metadata object into a consistent schema.

    Handles field aliasing (e.g. ``sprint_id`` → ``id``,
    ``startDate`` → ``start_date``) and fills missing fields with safe
    defaults: ``"unknown-sprint"`` for missing IDs, ``"Active Sprint"`` for
    missing names, today's date as the start date, and today + 14 days as
    the end date.

    Args:
        raw_meta: Raw sprint metadata dict from the MCP tool response, or
            any non-dict value (treated as an empty dict).
        board_name: Board name used as a fallback when ``board_name`` is
            absent from *raw_meta*.

    Returns:
        A normalised dict with the following guaranteed keys:
        ``id``, ``name``, ``start_date``, ``end_date``, ``goal``,
        ``board_id``, ``board_name``.
    """
    meta = raw_meta if isinstance(raw_meta, dict) else {}
    today = date.today()
    default_end = today + timedelta(days=14)

    return {
        "id": _as_non_empty_string(meta.get("id", meta.get("sprint_id")), "unknown-sprint"),
        "name": _as_non_empty_string(meta.get("name") or meta.get("sprint_name"), "Active Sprint"),
        "start_date": _as_iso_date(meta.get("start_date") or meta.get("startDate"), today),
        "end_date": _as_iso_date(meta.get("end_date") or meta.get("endDate"), default_end),
        "goal": meta.get("goal") or "",
        "board_id": _as_non_empty_string(meta.get("board_id", meta.get("boardId")), "unknown-board"),
        "board_name": _as_non_empty_string(meta.get("board_name"), board_name),
    }


def run_sprint_insights_workflow(
    state: JiraGraphState,
    mcp_client: MCPClient,
    llm_service: OllamaService,
    stream: bool = False,
) -> Dict[str, object]:
    """Execute the sprint-insights workflow and return a partial state update.

    Orchestrates the full sprint-insights pipeline:

    1. Resolves the target board name from the graph state, falling back to
       LLM inference when regex extraction yields nothing.
    2. Calls the ``get_active_sprint_issues`` MCP tool.
    3. Normalises the tool response into a consistent sprint payload.
    4. Calls the ``format_sprint_progress`` MCP prompt with the normalised
       payload.
    5. Passes the formatted prompt text to the LLM for final generation
       (unless *stream* is ``True``, in which case generation is deferred
       to the streaming layer).

    Args:
        state: Current LangGraph workflow state; must contain
            ``"board_name"`` or a parseable ``"user_input"``.
        mcp_client: Configured MCP client used to call tools and prompts.
        llm_service: Configured Ollama service used for board name inference
            and final response generation.
        stream: When ``True``, skips eager LLM generation so that the
            streaming layer can generate the response incrementally.

    Returns:
        A partial state dict suitable for merging into
        :class:`~src.graph.state.JiraGraphState`.  On success, contains:
        ``"board_name"``, ``"tool_output"``, ``"prompt_output"``,
        ``"final_response"``, and ``"metadata"``.  On failure, contains
        ``"error"`` and, where applicable, ``"requires_clarification"``.
    """
    board_name = state.get("board_name") or extract_board_name(state.get("user_input", ""))
    with logging_context(workflow="sprint_insights", board_name=board_name or None):
        if not board_name:
            logger.info("Board name not parsed from input; asking LLM to infer it")
            board_name = llm_service.infer_board_name(state.get("user_input", "") or "")

        with logging_context(board_name=board_name or None):
            if not board_name:
                logger.warning("Sprint insights request missing board name")
                return {
                    "requires_clarification": True,
                    "error": "Missing board name",
                    "final_response": (
                        "Please include the scrum board name in your request so I can fetch active sprint insights."
                    ),
                }

            tool_started_at = perf_counter()
            try:
                logger.info("Fetching active sprint issues")
                tool_output = asyncio.run(
                    mcp_client.call_tool("get_active_sprint_issues", {"scrum_board_name": board_name})
                )
                logger.info(
                    "Fetched active sprint issues",
                    extra={
                        "duration_ms": round((perf_counter() - tool_started_at) * 1000, 2),
                        "text_length": len(tool_output.get("text", "")),
                    },
                )
            except MCPClientError as exc:
                logger.warning(
                    "Sprint insights tool call failed",
                    extra={"error_type": type(exc).__name__},
                )
                return {
                    "board_name": board_name,
                    "error": SPRINT_INSIGHTS_TOOL_ERROR,
                }

            issues_text = tool_output.get("text", "")
            try:
                parsed_issues = json.loads(issues_text)
            except json.JSONDecodeError:
                parsed_issues = None

            if isinstance(parsed_issues, dict):
                prompt_payload = dict(parsed_issues)
                source_meta = prompt_payload.get("sprint_meta")
                if not isinstance(source_meta, dict):
                    source_meta = prompt_payload.get("metadata")
                prompt_payload["sprint_meta"] = _normalize_sprint_meta(source_meta, board_name)
                if not isinstance(prompt_payload.get("metadata"), dict):
                    prompt_payload["metadata"] = {"board_name": board_name}
                else:
                    prompt_payload["metadata"]["board_name"] = (
                        prompt_payload["metadata"].get("board_name") or board_name
                    )
            elif isinstance(parsed_issues, list):
                prompt_payload = {
                    "metadata": {"board_name": board_name},
                    "sprint_meta": _normalize_sprint_meta(None, board_name),
                    "issues": parsed_issues,
                }
            else:
                prompt_payload = {
                    "metadata": {"board_name": board_name},
                    "sprint_meta": _normalize_sprint_meta(None, board_name),
                    "issues": [{"raw": issues_text}],
                }

            prompt_args = {
                # MCP prompt `format_sprint_progress` expects a single JSON string argument.
                "sprint_data_json": json.dumps(prompt_payload),
            }

            prompt_started_at = perf_counter()
            try:
                logger.info("Fetching sprint insights prompt template")
                prompt_output = asyncio.run(
                    mcp_client.run_prompt("format_sprint_progress", prompt_args)
                )
                prompt_text = prompt_output.get("text", "").strip()
                if not prompt_text:
                    logger.warning("Sprint insights prompt template returned empty text")

                logger.info("Generating sprint insights report via LLM")
                formatted = "" if stream else (llm_service.generate(prompt_text) if prompt_text else "")
                if not formatted:
                    logger.warning("Sprint insights LLM response was empty")
                logger.info(
                    "Generated sprint insights report",
                    extra={
                        "duration_ms": round((perf_counter() - prompt_started_at) * 1000, 2),
                        "response_length": len(formatted),
                    },
                )
            except MCPClientError as exc:
                logger.warning(
                    "Sprint insights prompt failed",
                    extra={"error_type": type(exc).__name__},
                )
                return {
                    "board_name": board_name,
                    "tool_output": tool_output,
                    "error": SPRINT_INSIGHTS_PROMPT_ERROR,
                    "metadata": {
                        "workflow": "sprint_insights",
                        "tool": "get_active_sprint_issues",
                        "prompt": "format_sprint_progress",
                    },
                }

            return {
                "board_name": board_name,
                "tool_output": tool_output,
                "prompt_output": prompt_text,
                "final_response": formatted,
                "metadata": {
                    "workflow": "sprint_insights",
                    "tool": "get_active_sprint_issues",
                    "prompt": "format_sprint_progress",
                },
            }
