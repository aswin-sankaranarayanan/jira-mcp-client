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


def _render_messages(messages: List[Dict[str, str]]) -> None:
    for message in messages:
        avatar = "🧑" if message["role"] == "user" else "🧭"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])


def run_app() -> None:
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
                with st.spinner("Gathering Jira insights..."):
                    result = st.session_state.engine.run(prompt)

                error_message = result.get("error")
                assistant_text = str(result.get("response", "")).strip()

                if error_message:
                    st.error(str(error_message))
                    if assistant_text:
                        st.markdown(assistant_text)
                else:
                    try:
                        streamed = st.write_stream(st.session_state.engine.stream_text(assistant_text))
                        assistant_text = str(streamed).strip() if streamed is not None else assistant_text
                    except Exception:
                        logger.exception("Assistant streaming failed")
                        stream_error = "I couldn't render the response in the chat UI right now. Please try again."
                        st.error(stream_error)
                        if assistant_text:
                            st.markdown(assistant_text)

        if not assistant_text and error_message:
            assistant_text = str(error_message)

        logger.info("Completed chat response", extra={"response_length": len(assistant_text)})

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
