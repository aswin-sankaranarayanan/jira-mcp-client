"""LangGraph router and orchestrator for Jira workflows."""

from __future__ import annotations

import re
from typing import Dict, Iterable

from langgraph.graph import END, StateGraph

from src.client.mcp_client import MCPClient
from src.graph.parsing import extract_board_name, extract_issue_key
from src.graph.state import JiraGraphState, WorkflowResult
from src.graph.workflows.issue_details import run_issue_details_workflow
from src.graph.workflows.sprint_insights import run_sprint_insights_workflow
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
        builder.add_node("clarify", self._clarify)

        builder.set_entry_point("detect_intent")
        builder.add_conditional_edges(
            "detect_intent",
            self._intent_route,
            {
                "issue_details": "issue_details",
                "sprint_insights": "sprint_insights",
                "clarify": "clarify",
            },
        )

        builder.add_edge("issue_details", END)
        builder.add_edge("sprint_insights", END)
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
        result = self.run(user_input)
        content = result.get("response", "")
        if not content.strip():
            logger.warning("Workflow response was empty")
            content = "I could not generate a response for that request."

        yield from self.stream_text(content)

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
        return run_issue_details_workflow(state, self._mcp_client, self._llm_service)

    def _sprint_insights(self, state: JiraGraphState) -> Dict[str, object]:
        return run_sprint_insights_workflow(state, self._mcp_client, self._llm_service)

    def _clarify(self, state: JiraGraphState) -> Dict[str, object]:
        logger.info("Returning clarification response", extra={"workflow": "clarify"})
        return {
            "requires_clarification": True,
            "final_response": (
                "I can help with two Jira workflows:\n"
                "1) Sprint insights for a specific board\n"
                "2) Issue details for a specific issue key\n\n"
                "Try prompts like:\n"
                "- Show sprint progress for board Platform Team\n"
                "- Get details for issue PROJ-123"
            ),
            "metadata": {"workflow": "clarify"},
        }
