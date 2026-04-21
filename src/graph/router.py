"""LangGraph router and orchestrator for Jira workflows."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, Iterable, cast
from uuid import uuid4

from langgraph.graph import END, StateGraph

from src.client.mcp_client import MCPClient
from src.graph.parsing import extract_board_name, extract_issue_key
from src.graph.state import JiraGraphState, WorkflowEvent, WorkflowResult
from src.graph.workflows.issue_details import run_issue_details_workflow
from src.graph.workflows.sprint_insights import run_sprint_insights_workflow
from src.graph.workflows.team_status import run_team_status_workflow
from src.llm.ollama import OllamaService
from src.logging_config import get_logger, logging_context


logger = get_logger(__name__)


class JiraWorkflowEngine:
    def __init__(self, mcp_client: MCPClient, llm_service: OllamaService) -> None:
        self._mcp_client = mcp_client
        self._llm_service = llm_service
        self._graph = self._build_graph()
        logger.info("Initialized Jira workflow engine")

    def _build_graph(self):
        builder = StateGraph(JiraGraphState)
        builder.add_node("detect_intent", self._detect_intent)
        builder.add_node("issue_details", self._issue_details)
        builder.add_node("sprint_insights", self._sprint_insights)
        builder.add_node("team_status", self._team_status)
        builder.add_node("clarify", self._clarify)

        builder.set_entry_point("detect_intent")
        builder.add_conditional_edges(
            "detect_intent",
            self._intent_route,
            {
                "issue_details": "issue_details",
                "sprint_insights": "sprint_insights",
                "team_status": "team_status",
                "clarify": "clarify",
            },
        )

        builder.add_edge("issue_details", END)
        builder.add_edge("sprint_insights", END)
        builder.add_edge("team_status", END)
        builder.add_edge("clarify", END)
        logger.debug("Compiled workflow graph")
        return builder.compile()

    def run(self, user_input: str) -> WorkflowResult:
        state: JiraGraphState = {
            "user_input": user_input,
            "issue_key": extract_issue_key(user_input) or "",
            "board_name": extract_board_name(user_input) or "",
        }
        logger.info(
            "Running workflow request",
            extra={
                "input_length": len(user_input),
                "issue_key": state["issue_key"] or None,
                "board_name": state["board_name"] or None,
            },
        )

        with logging_context(
            issue_key=state["issue_key"] or None,
            board_name=state["board_name"] or None,
        ):
            try:
                result = self._graph.invoke(state)
            except Exception as exc:
                logger.exception("Workflow execution failed")
                return {
                    "response": "",
                    "metadata": {"workflow": "error"},
                    "error": str(exc),
                }

        metadata = dict(result.get("metadata", {}))
        logger.info(
            "Workflow request completed",
            extra={
                "workflow": metadata.get("workflow", "unknown"),
                "has_error": bool(result.get("error")),
            },
        )
        return {
            "response": str(result.get("final_response") or ""),
            "metadata": metadata,
            "error": result.get("error"),
        }

    def stream_response(self, user_input: str) -> Iterable[str]:
        for event in self.stream_events(user_input):
            event_type = str(event.get("type") or "")
            if event_type in {"token", "final", "error"}:
                text = str(event.get("message") or "")
                if text:
                    yield text

    def stream_events(self, user_input: str) -> Iterable[WorkflowEvent]:
        request_id = uuid4().hex[:12]
        state: JiraGraphState = {
            "user_input": user_input,
            "stream": True,
            "issue_key": extract_issue_key(user_input) or "",
            "board_name": extract_board_name(user_input) or "",
        }
        yield self._event(
            request_id,
            "progress",
            stage="detect_intent",
            message="Understanding your request...",
        )

        with logging_context(
            issue_key=state["issue_key"] or None,
            board_name=state["board_name"] or None,
        ):
            try:
                result: Dict[str, object] = {}
                last_stage = "detect_intent"
                for snapshot in self._iter_graph_snapshots(state):
                    if not isinstance(snapshot, dict):
                        continue
                    result = dict(snapshot)
                    stage = self._resolve_stage(result)
                    if stage != last_stage:
                        yield self._event(
                            request_id,
                            "progress",
                            stage=stage,
                            message=self._progress_message_for_stage(stage),
                        )
                        last_stage = stage
            except Exception:
                logger.exception("Workflow stream failed")
                yield self._event(
                    request_id,
                    "error",
                    stage="workflow",
                    message="I couldn't stream the workflow response right now. Please try again.",
                )
                yield self._event(request_id, "done", stage="workflow", message="done")
                return

        error_message = str(result.get("error") or "").strip()
        if error_message:
            logger.warning("Workflow stream using error response")
            yield self._event(
                request_id,
                "error",
                stage=self._resolve_stage(result),
                message=error_message,
            )
            yield self._event(request_id, "done", stage=self._resolve_stage(result), message="done")
            return

        prompt_text = str(result.get("prompt_output") or "").strip()
        if prompt_text:
            yield self._event(
                request_id,
                "progress",
                stage="compose",
                message="Drafting the final response...",
            )
            chunk_count = 0
            try:
                for chunk in self._llm_service.stream_generate(prompt_text):
                    chunk_count += 1
                    yield self._event(request_id, "token", stage="compose", message=chunk)
            except Exception:
                logger.exception("Workflow stream failed; falling back to unary generation")
                fallback_text = self._llm_service.generate(prompt_text)
                if fallback_text:
                    yield self._event(
                        request_id,
                        "final",
                        stage="compose",
                        message=fallback_text,
                    )
                yield self._event(request_id, "done", stage="compose", message="done")
                return

            if chunk_count == 0:
                logger.warning("LLM stream returned no chunks; using unary fallback")
                fallback_text = self._llm_service.generate(prompt_text)
                if fallback_text:
                    yield self._event(
                        request_id,
                        "final",
                        stage="compose",
                        message=fallback_text,
                    )
            yield self._event(request_id, "done", stage="compose", message="done")
            return

        content = str(result.get("final_response") or "").strip()
        if not content:
            logger.warning("Workflow response was empty")
            content = "I could not generate a response for that request."
        yield self._event(
            request_id,
            "final",
            stage=self._resolve_stage(result),
            message=content,
        )
        yield self._event(request_id, "done", stage=self._resolve_stage(result), message="done")

    def _iter_graph_snapshots(self, state: JiraGraphState) -> Iterable[Dict[str, object]]:
        stream_fn = getattr(self._graph, "stream", None)
        if callable(stream_fn):
            stream_values = cast(Iterable[Any], stream_fn(state, stream_mode="values"))
            for snapshot in stream_values:
                if isinstance(snapshot, dict):
                    yield dict(snapshot)
            return

        yield asyncio.run(self._stream_graph_state(state))

    async def _stream_graph_state(self, state: JiraGraphState) -> Dict[str, object]:
        latest_state: Dict[str, object] = {}
        async for snapshot in self._graph.astream(state, stream_mode="values"):
            if isinstance(snapshot, dict):
                latest_state = dict(snapshot)
        return latest_state

    def _event(
        self,
        request_id: str,
        event_type: str,
        *,
        stage: str,
        message: str,
        data: Dict[str, object] | None = None,
    ) -> WorkflowEvent:
        event: WorkflowEvent = {
            "type": event_type,
            "request_id": request_id,
            "stage": stage,
            "message": message,
        }
        if data:
            event["data"] = data
        return event

    def _resolve_stage(self, state: Dict[str, object]) -> str:
        metadata = state.get("metadata")
        if isinstance(metadata, dict):
            workflow = metadata.get("workflow")
            if workflow in {"issue_details", "sprint_insights", "team_status", "clarify"}:
                return str(workflow)
        intent = state.get("intent")
        if intent in {"issue_details", "sprint_insights", "team_status", "clarify"}:
            return str(intent)
        return "detect_intent"

    def _progress_message_for_stage(self, stage: str) -> str:
        if stage == "issue_details":
            return "Fetching issue details..."
        if stage == "sprint_insights":
            return "Analyzing active sprint data..."
        if stage == "team_status":
            return "Collecting team member status updates..."
        if stage == "clarify":
            return "Preparing clarification..."
        return "Understanding your request..."

    def stream_text(self, text: str) -> Iterable[str]:
        # Stream precomputed workflow text in small chunks to keep UI updates responsive.
        for chunk in re.findall(r"\S+\s*", text):
            yield chunk

    def _detect_intent(self, state: JiraGraphState) -> Dict[str, object]:
        user_input = state.get("user_input", "")
        intent = self._llm_service.detect_intent(user_input)
        logger.info("Resolved workflow route", extra={"workflow": intent})
        return {"intent": intent}

    def _intent_route(self, state: JiraGraphState) -> str:
        return state.get("intent", "clarify")

    def _issue_details(self, state: JiraGraphState) -> Dict[str, object]:
        return run_issue_details_workflow(
            state,
            self._mcp_client,
            self._llm_service,
            stream=bool(state.get("stream")),
        )

    def _sprint_insights(self, state: JiraGraphState) -> Dict[str, object]:
        return run_sprint_insights_workflow(
            state,
            self._mcp_client,
            self._llm_service,
            stream=bool(state.get("stream")),
        )

    def _team_status(self, state: JiraGraphState) -> Dict[str, object]:
        return run_team_status_workflow(
            state,
            self._mcp_client,
            self._llm_service,
            stream=bool(state.get("stream")),
        )

    def _clarify(self, state: JiraGraphState) -> Dict[str, object]:
        logger.info("Returning clarification response", extra={"workflow": "clarify"})
        return {
            "requires_clarification": True,
            "final_response": (
                "I can help with three Jira workflows:\n"
                "1) Sprint insights for a specific board\n"
                "2) Issue details for a specific issue key\n"
                "3) Team member status report for a specific board\n\n"
                "Try prompts like:\n"
                "- Show sprint progress for board Platform Team\n"
                "- Get details for issue PROJ-123\n"
                "- Show team status for board Platform Team"
            ),
            "metadata": {"workflow": "clarify"},
        }
