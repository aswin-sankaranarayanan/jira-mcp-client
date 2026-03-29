"""Static runtime configuration for the Jira MCP chat app."""

from __future__ import annotations

import os


def _get_env(name: str, default: str) -> str:
	value = os.getenv(name, default).strip()
	return value or default


def _get_bool_env(name: str, default: bool) -> bool:
	value = os.getenv(name)
	if value is None:
		return default

	normalized = value.strip().lower()
	if normalized in {"1", "true", "yes", "on"}:
		return True
	if normalized in {"0", "false", "no", "off"}:
		return False
	return default


MCP_SERVER_URL = _get_env("MCP_SERVER_URL", "http://localhost:8000/sse")
OLLAMA_MODEL = _get_env("OLLAMA_MODEL", "llama3.2:latest")
APP_TITLE = _get_env("APP_TITLE", "Jira Copilot")
APP_SUBTITLE = _get_env("APP_SUBTITLE", "LangGraph + MCP powered Jira assistant")
STREAMING_ENABLED = _get_bool_env("STREAMING_ENABLED", True)
LOG_LEVEL = _get_env("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = _get_env("LOG_FORMAT", "json").lower()
