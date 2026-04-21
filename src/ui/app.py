"""Streamlit chat interface for Jira MCP client."""

from __future__ import annotations

from typing import Dict, List
from uuid import uuid4

import streamlit as st

from src.client.mcp_client import MCPClient
from src.config.settings import APP_SUBTITLE, APP_TITLE, LOG_FORMAT, LOG_LEVEL, MCP_SERVER_URL, OLLAMA_MODEL
from src.graph.router import JiraWorkflowEngine
from src.llm.ollama import OllamaService
from src.logging_config import configure_logging, get_logger, logging_context
from src.ui.styles import APP_CSS


logger = get_logger(__name__)


def _init_state() -> None:
    """Bootstrap Streamlit session state for the current browser session.

    Initialises three session-state keys if they are not already present:

    * ``"session_id"`` — a 12-character hex identifier that persists for the
      lifetime of the browser tab and is attached to all log records via
      :func:`~src.logging_config.logging_context`.
    * ``"messages"`` — an ordered list of ``{"role": str, "content": str}``
      dicts representing the chat history, pre-seeded with the assistant's
      welcome message.
    * ``"engine"`` — the :class:`~src.graph.router.JiraWorkflowEngine`
      instance shared across all interactions within the session.

    This function is idempotent: calling it multiple times within the same
    Streamlit session is safe and has no side effects beyond the first call.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex[:12]

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "I can help with Jira sprint insights and issue details. "
                    "Ask about an issue key like PROJ-101 or request sprint insights for a board."
                ),
            }
        ]
        logger.info("Initialized session messages")

    if "engine" not in st.session_state:
        client = MCPClient(MCP_SERVER_URL)
        llm = OllamaService(OLLAMA_MODEL)
        st.session_state.engine = JiraWorkflowEngine(client, llm)
        logger.info("Created workflow engine for session")

    # Holds the active team-status collection session; None when not collecting.
    if "team_status_session" not in st.session_state:
        st.session_state.team_status_session = None


def _render_messages(messages: List[Dict[str, str]]) -> None:
    """Render a list of chat messages into the active Streamlit chat container.

    Iterates over *messages* and uses :func:`streamlit.chat_message` to
    render each entry with the appropriate avatar emoji (🧑 for user messages,
    🧭 for assistant messages).  Message content is rendered as Markdown.

    Args:
        messages: Ordered list of message dicts, each containing ``"role"``
            (``"user"`` or ``"assistant"``) and ``"content"`` keys.
    """
    for message in messages:
        avatar = "🧑" if message["role"] == "user" else "🧭"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])


def run_app() -> None:
    """Configure and launch the Streamlit Jira Copilot chat interface.

    Entry point called by :func:`src.main.main`.  Performs the following
    steps on every Streamlit re-run:

    1. Configures structured logging for the Streamlit process.
    2. Sets the page title, icon, and layout via
       :func:`streamlit.set_page_config`.
    3. Injects global CSS from :data:`~src.ui.styles.APP_CSS`.
    4. Initialises per-session state via :func:`_init_state`.
    5. Renders the sticky header panel and the scrollable chat history.
    6. Waits for user input via :func:`streamlit.chat_input`.
    7. On new input: appends the user message, invokes
       :meth:`~src.graph.router.JiraWorkflowEngine.stream_events`, and
       streams the assistant response into the chat panel token-by-token.
    8. Handles all event types (``"progress"``, ``"token"``, ``"final"``,
       ``"error"``, ``"done"``) and falls back gracefully on exceptions.

    Note:
        ``st.set_page_config`` must be the first Streamlit call in the
        script; this function satisfies that constraint by calling it before
        any other ``st.*`` API.
    """
    configure_logging(LOG_LEVEL, LOG_FORMAT)
    st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)

    _init_state()

    with st.container(key="header_panel"):
        st.title(APP_TITLE)
        st.caption(APP_SUBTITLE)

    chat_panel = st.container(key="chat_panel", height=560, border=True)
    with chat_panel:
        _render_messages(st.session_state.messages)

    prompt = st.chat_input("Ask Jira questions...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_panel:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

    with logging_context(session_id=st.session_state.session_id, request_id=uuid4().hex[:12]):
        logger.info("Received chat prompt", extra={"prompt_length": len(prompt)})

        with chat_panel:
            with st.chat_message("assistant", avatar="🧭"):
                assistant_text = ""
                error_message = None
                try:
                    progress_placeholder = st.empty()
                    response_placeholder = st.empty()
                    progress_placeholder.info("Thinking...")

                    # Route to the team-status collection handler when a
                    # collection cycle is already in progress; otherwise use
                    # the normal workflow engine.
                    engine = st.session_state.engine
                    if st.session_state.team_status_session is not None:
                        event_source = engine.advance_team_status_collection(
                            st.session_state.team_status_session, prompt
                        )
                    else:
                        event_source = engine.stream_events(prompt)

                    for event in event_source:
                        event_type = str(event.get("type") or "")
                        message = str(event.get("message") or "")
                        if event_type == "progress":
                            progress_placeholder.info(message or "Working...")
                            continue
                        if event_type == "token":
                            progress_placeholder.empty()
                            assistant_text += message
                            response_placeholder.markdown(assistant_text)
                            continue
                        if event_type == "final":
                            progress_placeholder.empty()
                            assistant_text = message
                            response_placeholder.markdown(assistant_text)
                            continue
                        if event_type == "session":
                            # Update the team-status collection session in
                            # Streamlit state without affecting the UI display.
                            data = event.get("data") or {}
                            action = str(data.get("action") or "")
                            if action in {"team_status_start", "team_status_advance"}:
                                st.session_state.team_status_session = data.get("session")
                            elif action == "team_status_complete":
                                st.session_state.team_status_session = None
                            continue
                        if event_type == "error":
                            progress_placeholder.empty()
                            error_message = message or "I couldn't render the response right now."
                            st.error(error_message)
                            break
                        if event_type == "done":
                            progress_placeholder.empty()
                except Exception:
                    logger.exception("Assistant streaming failed")
                    error_message = "I couldn't render the response in the chat UI right now. Please try again."
                    st.error(error_message)

        if not assistant_text and error_message:
            assistant_text = error_message

        logger.info("Completed chat response", extra={"response_length": len(assistant_text)})

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
