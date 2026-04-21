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
from src.graph.workflows.team_status import run_team_status_workflow, format_assignee_presentation
from src.llm.ollama import OllamaService
from src.logging_config import get_logger, logging_context


logger = get_logger(__name__)


class JiraWorkflowEngine:
    """LangGraph-based orchestrator for Jira MCP workflows.

    Builds and executes a directed workflow graph that classifies the user's
    intent, routes to the appropriate workflow node (issue details, sprint
    insights, or clarification), fetches data via the MCP server, and
    generates a response using the Ollama LLM service.

    The engine supports two execution modes:

    * **Synchronous** (:meth:`run`) — blocks until the full response is ready
      and returns a :class:`~src.graph.state.WorkflowResult`.
    * **Streaming** (:meth:`stream_events`) — yields
      :class:`~src.graph.state.WorkflowEvent` objects as the workflow
      progresses, including per-token LLM output for a real-time UI.

    Args:
        mcp_client: Configured :class:`~src.client.mcp_client.MCPClient`
            used to call MCP tools and prompts.
        llm_service: Configured :class:`~src.llm.ollama.OllamaService` used
            for intent detection and response generation.
    """

    def __init__(self, mcp_client: MCPClient, llm_service: OllamaService) -> None:
        """Initialise the engine and compile the LangGraph workflow graph.

        Args:
            mcp_client: Configured MCP client for tool and prompt calls.
            llm_service: Configured Ollama service for LLM inference.
        """
        self._mcp_client = mcp_client
        self._llm_service = llm_service
        self._graph = self._build_graph()
        logger.info("Initialized Jira workflow engine")

    def _build_graph(self):
        """Construct and compile the LangGraph ``StateGraph`` for Jira workflows.

        Registers four nodes — ``detect_intent``, ``issue_details``,
        ``sprint_insights``, and ``clarify`` — and wires conditional edges
        from ``detect_intent`` to each terminal node based on the resolved
        intent.

        Returns:
            A compiled LangGraph ``CompiledGraph`` ready for invocation or
            streaming.
        """
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
        """Execute the workflow synchronously and return the complete result.

        Builds an initial :class:`~src.graph.state.JiraGraphState` from
        *user_input*, pre-populating ``issue_key`` and ``board_name`` via
        regex extraction, then invokes the compiled graph and assembles a
        :class:`~src.graph.state.WorkflowResult`.

        Args:
            user_input: Raw natural-language query from the user.

        Returns:
            A :class:`~src.graph.state.WorkflowResult` containing:

            * ``"response"`` — generated text (empty string on error).
            * ``"metadata"`` — workflow metadata dict.
            * ``"error"`` — error message string, or ``None`` on success.
        """
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
        """Yield plain text chunks as the workflow generates its response.

        A thin convenience wrapper around :meth:`stream_events` that filters
        for ``"token"``, ``"final"``, and ``"error"`` events and yields their
        ``"message"`` strings directly.

        Args:
            user_input: Raw natural-language query from the user.

        Yields:
            Text chunks suitable for concatenation into the final response.
        """
        for event in self.stream_events(user_input):
            event_type = str(event.get("type") or "")
            if event_type in {"token", "final", "error"}:
                text = str(event.get("message") or "")
                if text:
                    yield text

    def stream_events(self, user_input: str) -> Iterable[WorkflowEvent]:
        """Execute the workflow and yield structured progress and content events.

        Drives the LangGraph graph in streaming mode, emitting ``"progress"``
        events at each stage transition.  After the graph completes:

        * If ``prompt_output`` is populated, streams LLM generation
          token-by-token as ``"token"`` events, falling back to a single
          ``"final"`` event if streaming fails.
        * If ``final_response`` is populated, emits a single ``"final"``
          event.
        * Emits an ``"error"`` event if the workflow raised an exception or
          produced an error message.

        All event streams are terminated with a ``"done"`` event.

        Args:
            user_input: Raw natural-language query from the user.

        Yields:
            :class:`~src.graph.state.WorkflowEvent` dicts in order:
            ``"progress"`` → ``"token"``/``"final"``/``"error"`` → ``"done"``.
        """
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

        # When the team-status workflow initiates a collection cycle, emit a
        # structured "session" event so the UI can store the session state and
        # route subsequent user inputs through advance_team_status_collection.
        if result.get("team_status_phase") == "collecting":
            session_data: Dict[str, object] = {
                "board_name": str(result.get("board_name") or ""),
                "assignees": list(result.get("assignees") or []),
                "issues_by_assignee": dict(result.get("assignee_issues") or {}),
                "current_index": int(result.get("current_assignee_index") or 0),
                "collected_updates": dict(result.get("collected_updates") or {}),
            }
            yield self._event(
                request_id,
                "session",
                stage="team_status",
                message="",
                data={"action": "team_status_start", "session": session_data},
            )

        yield self._event(request_id, "done", stage=self._resolve_stage(result), message="done")

    def _iter_graph_snapshots(self, state: JiraGraphState) -> Iterable[Dict[str, object]]:
        """Iterate over LangGraph state snapshots in streaming mode.

        Prefers the synchronous ``graph.stream`` API (available in most
        LangGraph builds).  Falls back to ``asyncio.run`` over
        ``graph.astream`` for environments where the sync API is absent.

        Args:
            state: Initial workflow state.

        Yields:
            Successive state snapshot dicts produced as each graph node
            completes.
        """
        stream_fn = getattr(self._graph, "stream", None)
        if callable(stream_fn):
            stream_values = cast(Iterable[Any], stream_fn(state, stream_mode="values"))
            for snapshot in stream_values:
                if isinstance(snapshot, dict):
                    yield dict(snapshot)
            return

        yield asyncio.run(self._stream_graph_state(state))

    async def _stream_graph_state(self, state: JiraGraphState) -> Dict[str, object]:
        """Async fallback that collects the final state via ``graph.astream``.

        Consumes all snapshots from the async graph stream and returns only
        the last one, which represents the fully-populated final state after
        all nodes have executed.

        Args:
            state: Initial workflow state.

        Returns:
            The final state dict after all graph nodes have completed.
        """
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
        """Construct a :class:`~src.graph.state.WorkflowEvent` dict.

        Args:
            request_id: Opaque hex identifier grouping events for a single
                request.
            event_type: Discriminator string — one of ``"progress"``,
                ``"token"``, ``"final"``, ``"error"``, or ``"done"``.
            stage: Active workflow stage name at the time of the event.
            message: Human-readable text payload for the event.
            data: Optional structured payload merged into the event dict.

        Returns:
            A fully-populated :class:`~src.graph.state.WorkflowEvent` dict.
        """
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
        """Determine the active workflow stage from a state snapshot.

        Prefers the ``metadata.workflow`` value (set by completed nodes),
        then falls back to the top-level ``intent`` field, and finally to
        ``"detect_intent"`` when neither is available.

        Args:
            state: A LangGraph state snapshot dict.

        Returns:
            One of ``"issue_details"``, ``"sprint_insights"``,
            ``"clarify"``, or ``"detect_intent"``.
        """
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
        """Return a human-readable progress message for a given workflow stage.

        Args:
            stage: Workflow stage name (e.g. ``"issue_details"``).

        Returns:
            A short, user-facing status string describing the current stage.
        """
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
        """Yield *text* in small word-boundary chunks for responsive UI updates.

        Splits on whitespace boundaries so that complete words (plus their
        trailing whitespace) are emitted as individual chunks, producing a
        natural typewriter effect in the chat UI.

        Args:
            text: Pre-generated response text to stream.

        Yields:
            Successive word-and-whitespace chunks of *text*.
        """
        # Stream precomputed workflow text in small chunks to keep UI updates responsive.
        for chunk in re.findall(r"\S+\s*", text):
            yield chunk

    def _detect_intent(self, state: JiraGraphState) -> Dict[str, object]:
        """LangGraph node: classify the user's intent via the LLM.

        Calls :meth:`~src.llm.ollama.OllamaService.detect_intent` with the
        raw user input and returns a partial state update containing the
        resolved intent label.

        Args:
            state: Current workflow state; must contain ``"user_input"``.

        Returns:
            A partial state dict with key ``"intent"`` set to one of
            ``"issue_details"``, ``"sprint_insights"``, or ``"clarify"``.
        """
        user_input = state.get("user_input", "")
        intent = self._llm_service.detect_intent(user_input)
        logger.info("Resolved workflow route", extra={"workflow": intent})
        return {"intent": intent}

    def _intent_route(self, state: JiraGraphState) -> str:
        """LangGraph conditional edge: return the next node name based on intent.

        Args:
            state: Current workflow state after ``detect_intent`` has run.

        Returns:
            One of ``"issue_details"``, ``"sprint_insights"``, or
            ``"clarify"``.  Defaults to ``"clarify"`` when intent is absent.
        """
        return state.get("intent", "clarify")

    def _issue_details(self, state: JiraGraphState) -> Dict[str, object]:
        """LangGraph node: execute the issue-details workflow.

        Delegates to
        :func:`~src.graph.workflows.issue_details.run_issue_details_workflow`.

        Args:
            state: Current workflow state; must contain ``"issue_key"`` or a
                parseable ``"user_input"``.

        Returns:
            Partial state dict produced by the issue-details workflow.
        """
        return run_issue_details_workflow(
            state,
            self._mcp_client,
            self._llm_service,
            stream=bool(state.get("stream")),
        )

    def _sprint_insights(self, state: JiraGraphState) -> Dict[str, object]:
        """LangGraph node: execute the sprint-insights workflow.

        Delegates to
        :func:`~src.graph.workflows.sprint_insights.run_sprint_insights_workflow`.

        Args:
            state: Current workflow state; must contain ``"board_name"`` or a
                parseable ``"user_input"``.

        Returns:
            Partial state dict produced by the sprint-insights workflow.
        """
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

    def advance_team_status_collection(
        self,
        session: Dict[str, Any],
        status_update: str,
    ) -> Iterable[WorkflowEvent]:
        """Process one status update in the interactive team status collection cycle.

        Records *status_update* for the current assignee, then either presents
        the next assignee's issues or generates and streams the final scrum
        master report when all assignees have been covered.

        Args:
            session: Active collection session dict maintained by the UI layer.
                Must contain ``"board_name"``, ``"assignees"``,
                ``"issues_by_assignee"``, ``"current_index"``, and
                ``"collected_updates"`` keys.
            status_update: The status text provided by the current assignee.

        Yields:
            :class:`~src.graph.state.WorkflowEvent` dicts.  When another
            assignee remains: a ``"final"`` event with the next presentation
            followed by a ``"session"`` event carrying the updated session.
            When all assignees are done: ``"progress"``, token/``"final"``
            events for the LLM report, a ``"session"`` event with
            ``action="team_status_complete"`` and ``session=None``, then
            ``"done"``.
        """
        request_id = uuid4().hex[:12]
        assignees: list = session["assignees"]
        current_index: int = session["current_index"]
        current_assignee: str = assignees[current_index]
        collected_updates: Dict[str, Any] = session["collected_updates"]

        # Record the update for the current assignee.
        collected_updates[current_assignee] = status_update
        logger.info(
            "Recorded team status update",
            extra={"assignee": current_assignee, "index": current_index},
        )

        next_index = current_index + 1

        if next_index < len(assignees):
            # More assignees remain — present the next one.
            next_assignee: str = assignees[next_index]
            issues_by_assignee: Dict[str, Any] = session["issues_by_assignee"]
            presentation = format_assignee_presentation(
                next_assignee,
                issues_by_assignee[next_assignee],
                next_index,
                len(assignees),
            )
            updated_session = {**session, "current_index": next_index}
            yield self._event(request_id, "final", stage="team_status", message=presentation)
            yield self._event(
                request_id,
                "session",
                stage="team_status",
                message="",
                data={"action": "team_status_advance", "session": updated_session},
            )
            yield self._event(request_id, "done", stage="team_status", message="done")
            return

        # All assignees covered — generate the final scrum master report.
        yield self._event(
            request_id,
            "progress",
            stage="team_status",
            message="All updates collected. Generating scrum master report...",
        )

        board_name: str = session["board_name"]
        issues_by_assignee = session["issues_by_assignee"]
        payload = {
            "board_name": board_name,
            "team_members": [
                {
                    "assignee": a,
                    "issues": issues_by_assignee.get(a, []),
                    "status_update": collected_updates.get(a, "(no update provided)"),
                }
                for a in assignees
            ],
        }
        prompt = self._llm_service.build_scrum_master_report_prompt(payload)

        chunk_count = 0
        try:
            for chunk in self._llm_service.stream_generate(prompt):
                chunk_count += 1
                yield self._event(request_id, "token", stage="team_status", message=chunk)
        except Exception:
            logger.exception("Scrum master report streaming failed; falling back to unary")
            fallback = self._llm_service.generate(prompt)
            if fallback:
                yield self._event(request_id, "final", stage="team_status", message=fallback)
            yield self._event(
                request_id,
                "session",
                stage="team_status",
                message="",
                data={"action": "team_status_complete", "session": None},
            )
            yield self._event(request_id, "done", stage="team_status", message="done")
            return

        if chunk_count == 0:
            logger.warning("LLM stream returned no chunks for scrum report; using unary fallback")
            fallback = self._llm_service.generate(prompt)
            if fallback:
                yield self._event(request_id, "final", stage="team_status", message=fallback)

        logger.info(
            "Completed scrum master report stream",
            extra={"assignee_count": len(assignees), "chunk_count": chunk_count},
        )
        yield self._event(
            request_id,
            "session",
            stage="team_status",
            message="",
            data={"action": "team_status_complete", "session": None},
        )
        yield self._event(request_id, "done", stage="team_status", message="done")

    def _clarify(self, state: JiraGraphState) -> Dict[str, object]:
        """LangGraph node: return a clarification prompt to the user.

        Emitted when the intent cannot be determined with confidence.  Produces
        a static response listing the two supported workflows and example
        queries.

        Args:
            state: Current workflow state (not used; accepted for LangGraph
                node signature compatibility).

        Returns:
            Partial state dict with ``"requires_clarification"``,
            ``"final_response"``, and ``"metadata"`` keys populated.
        """
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
