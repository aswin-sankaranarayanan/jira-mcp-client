"""Static runtime configuration for the Jira MCP chat app."""

from __future__ import annotations

import os


def _get_env(name: str, default: str) -> str:
	"""Read a string environment variable with a non-empty default.

	Strips leading and trailing whitespace from the raw environment value.
	If the result is an empty string, *default* is returned so that
	``SOME_VAR=""`` behaves identically to the variable being unset.

	Args:
		name: Name of the environment variable to read.
		default: Value to return when the variable is absent or blank.

	Returns:
		The stripped environment value, or *default* if absent or blank.
	"""
	value = os.getenv(name, default).strip()
	return value or default


def _get_bool_env(name: str, default: bool) -> bool:
	"""Read a boolean environment variable.

	Accepts the following truthy strings (case-insensitive):
	``"1"``, ``"true"``, ``"yes"``, ``"on"``.

	Accepts the following falsy strings (case-insensitive):
	``"0"``, ``"false"``, ``"no"``, ``"off"``.

	Unrecognised non-empty values return *default* rather than raising.

	Args:
		name: Name of the environment variable to read.
		default: Value to return when the variable is absent or its value
			cannot be interpreted as a boolean.

	Returns:
		Parsed boolean, or *default* when the variable is unset or its
		value is not in the recognised set.
	"""
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
