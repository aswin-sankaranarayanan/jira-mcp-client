"""Shared LangGraph state definitions."""

from typing import Any, Dict, Optional, TypedDict


class JiraGraphState(TypedDict, total=False):
    """Mutable state carried through every node of the LangGraph workflow.

    All fields are optional (``total=False``) so that individual workflow
    nodes can populate only the keys they produce without having to declare
    defaults for the rest.

    Attributes:
        user_input: The raw natural-language query supplied by the user.
        stream: When ``True``, workflow nodes skip eager LLM generation and
            leave ``prompt_output`` populated for the streaming layer.
        intent: Resolved workflow intent — one of ``"issue_details"``,
            ``"sprint_insights"``, or ``"clarify"``.
        issue_key: Jira issue key extracted from the user input
            (e.g. ``"PROJ-123"``).
        board_name: Scrum board name extracted or inferred from the user
            input (e.g. ``"Platform Team"``).
        tool_output: Raw response envelope returned by
            :meth:`~src.client.mcp_client.MCPClient.call_tool`.
        prompt_output: Formatted prompt template text returned by
            :meth:`~src.client.mcp_client.MCPClient.run_prompt`, intended
            for downstream LLM generation.
        analysis: Intermediate analytical text produced by the LLM.
        final_response: The fully generated response text presented to the
            user.
        error: Human-readable error message when a node encounters a failure.
        requires_clarification: ``True`` when the workflow cannot proceed
            without additional user input.
        metadata: Arbitrary key/value pairs attached by workflow nodes for
            observability (e.g. ``{"workflow": "issue_details", "tool":
            "get_issue_details"}``).
    """
    user_input: str
    stream: bool
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
    """Return type for synchronous, non-streaming workflow execution.

    Attributes:
        response: The final generated text to be displayed to the user.
            Empty string if the workflow ended with an error.
        metadata: Structured metadata produced by the last active workflow
            node, useful for observability and debugging.
        error: Human-readable error message, or ``None`` when the workflow
            completed successfully.
    """
    response: str
    metadata: Dict[str, Any]
    error: Optional[str]


class WorkflowEvent(TypedDict, total=False):
    """Structured event emitted by the streaming workflow iterator.

    Events are yielded by
    :meth:`~src.graph.router.JiraWorkflowEngine.stream_events` to allow the
    UI layer to display incremental progress and token-by-token text
    generation.

    Attributes:
        type: Event type discriminator.  One of:

            * ``"progress"`` — a workflow stage transition; display as a
              status indicator.
            * ``"token"`` — a single LLM output chunk; append to the
              response buffer.
            * ``"final"`` — the complete response text; replace the current
              response buffer.
            * ``"error"`` — a recoverable error; display as an error banner.
            * ``"done"`` — signals the end of the event stream; no further
              events will follow.

        request_id: Opaque hex identifier that groups all events belonging
            to a single user request.
        stage: The active workflow stage when this event was emitted
            (e.g. ``"detect_intent"``, ``"issue_details"``, ``"compose"``).
        message: Human-readable text payload.  Interpretation depends on
            *type*.
        data: Optional structured payload for machine-readable event data.
    """
    type: str
    request_id: str
    stage: str
    message: str
    data: Dict[str, Any]
