"""Centralized structured logging for the Jira MCP client."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Iterator, Mapping, TextIO


APP_LOGGER_NAME = "jira_mcp_client"
_LOG_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("jira_log_context", default={})
_LOG_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__.keys()) | {"message", "asctime"}
_LOGGING_LOCK = Lock()
_CONFIGURED = False


class ContextFilter(logging.Filter):
    """Attach per-request context fields to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = _LOG_CONTEXT.get()
        for key, value in context.items():
            setattr(record, key, value)

        for key in ("session_id", "request_id", "workflow", "issue_key", "board_name"):
            if not hasattr(record, key):
                setattr(record, key, "-")

        return True


class JsonFormatter(logging.Formatter):
    """Render logs as JSON for production ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key in ("session_id", "request_id", "workflow", "issue_key", "board_name"):
            value = getattr(record, key, None)
            if value not in (None, "", "-"):
                payload[key] = value

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _LOG_RECORD_FIELDS and value not in (None, "", "-")
        }
        payload.update(extras)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True, default=str)


class TextFormatter(logging.Formatter):
    """Render readable console logs for local development."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        context_parts = []

        for key in ("session_id", "request_id", "workflow", "issue_key", "board_name"):
            value = getattr(record, key, None)
            if value not in (None, "", "-"):
                context_parts.append(f"{key}={value}")

        if context_parts:
            return f"{message} [{' '.join(context_parts)}]"

        return message


def get_logger(name: str) -> logging.Logger:
    normalized = name.removeprefix("src.")
    return logging.getLogger(f"{APP_LOGGER_NAME}.{normalized}")


def configure_logging(
    level: str = "INFO",
    log_format: str = "json",
    *,
    force: bool = False,
    stream: TextIO | None = None,
) -> None:
    """Configure application logging once per process."""

    global _CONFIGURED

    with _LOGGING_LOCK:
        if _CONFIGURED and not force:
            return

        handler = logging.StreamHandler(stream or sys.stdout)
        handler.addFilter(ContextFilter())

        if log_format.lower() == "text":
            handler.setFormatter(
                TextFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
            )
        else:
            handler.setFormatter(JsonFormatter())

        logger = logging.getLogger(APP_LOGGER_NAME)
        logger.handlers.clear()
        logger.setLevel(_resolve_log_level(level))
        logger.propagate = False
        logger.addHandler(handler)

        _CONFIGURED = True


@contextmanager
def logging_context(**context: Mapping[str, Any] | Any) -> Iterator[None]:
    """Temporarily enrich logs with request-scoped context."""

    current = dict(_LOG_CONTEXT.get())
    updates = {key: value for key, value in context.items() if value not in (None, "", "-")}
    token = _LOG_CONTEXT.set({**current, **updates})
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def _resolve_log_level(level: str) -> int:
    resolved = getattr(logging, level.upper(), logging.INFO)
    return resolved if isinstance(resolved, int) else logging.INFO