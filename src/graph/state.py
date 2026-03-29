"""Shared LangGraph state definitions."""

from typing import Any, Dict, Optional, TypedDict


class JiraGraphState(TypedDict, total=False):
    user_input: str
    intent: str
    issue_key: str
    board_name: str
    tool_output: Dict[str, Any]
    prompt_output: str
    analysis: str
    final_response: str
    error: str
    requires_clarification: bool
    metadata: Dict[str, Any]


class WorkflowResult(TypedDict):
    response: str
    metadata: Dict[str, Any]
    error: Optional[str]
