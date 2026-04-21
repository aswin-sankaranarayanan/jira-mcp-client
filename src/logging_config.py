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
    """Attach per-request context fields to every log record.

    Reads context variables from the thread/coroutine-local ``_LOG_CONTEXT``
    ``ContextVar`` and copies them as attributes onto each ``LogRecord``
    before formatting.  Additionally, ensures that the canonical Jira context
    fields (``session_id``, ``request_id``, ``workflow``, ``issue_key``,
    ``board_name``) are always present — defaulting to ``"-"`` when absent.

    This filter is intended to be registered on a single ``StreamHandler``
    and should not be reused across handlers.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Enrich *record* with the current request-scoped context fields.

        Args:
            record: The log record to be emitted.

        Returns:
            ``True`` — the record is always passed through; this filter is
            purely side-effecting.
        """
        context = _LOG_CONTEXT.get()
        for key, value in context.items():
            setattr(record, key, value)

        for key in ("session_id", "request_id", "workflow", "issue_key", "board_name"):
            if not hasattr(record, key):
                setattr(record, key, "-")

        return True


class JsonFormatter(logging.Formatter):
    """Render log records as structured JSON for production log ingestion.

    Produces a single JSON object per line, suitable for consumption by log
    aggregation systems such as Datadog, Splunk, or AWS CloudWatch.  Each
    object always includes ``timestamp`` (ISO-8601 UTC with millisecond
    precision), ``level``, ``logger``, and ``message`` fields.  Optional
    context fields and any extra kwargs passed to the logger call are merged
    in automatically.

    Exception tracebacks are serialized under the ``"exception"`` key when
    ``exc_info`` is present on the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serialize *record* to a single-line JSON string.

        Args:
            record: The log record to be formatted.

        Returns:
            A UTF-8-safe JSON string (``ensure_ascii=True``).  Non-JSON-
            serializable extra values are coerced to ``str`` via the
            ``default`` hook.
        """
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
    """Render human-readable log lines for local development.

    Delegates base formatting to :class:`logging.Formatter` and appends a
    bracketed context annotation (e.g. ``[session_id=abc workflow=issue_details]``)
    whenever non-null Jira context fields are present on the record.

    This formatter is used when ``LOG_FORMAT`` is set to ``"text"`` and is
    *not* recommended for production because it cannot be reliably parsed by
    log aggregation tools.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format *record* as a human-readable line with optional context annotation.

        Args:
            record: The log record to be formatted.

        Returns:
            A plain-text log line.  Context fields are appended as a
            space-separated ``[key=value ...]`` block when at least one
            non-null context field is present.
        """
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
    """Return a child logger scoped under the application root logger.

    Strips the ``"src."`` prefix from *name* before constructing the full
    logger name so that module-level calls like ``get_logger(__name__)``
    produce consistent, shorter names regardless of package layout.

    Args:
        name: Typically the calling module's ``__name__``.  The ``"src."``
            prefix, if present, is removed before creating the child logger.

    Returns:
        A :class:`logging.Logger` named
        ``"jira_mcp_client.<normalized_name>"``.
    """
    normalized = name.removeprefix("src.")
    return logging.getLogger(f"{APP_LOGGER_NAME}.{normalized}")


def configure_logging(
    level: str = "INFO",
    log_format: str = "json",
    *,
    force: bool = False,
    stream: TextIO | None = None,
) -> None:
    """Configure application logging once per process.

    Sets up a single :class:`logging.StreamHandler` on the root application
    logger (``"jira_mcp_client"``).  The handler is decorated with a
    :class:`ContextFilter` and either a :class:`JsonFormatter` (default) or a
    :class:`TextFormatter` depending on *log_format*.

    This function is **idempotent**: repeated calls are silently ignored
    unless ``force=True`` is passed.  A threading :class:`~threading.Lock`
    ensures that concurrent initialisation attempts during startup are safe.

    Args:
        level: Logging level string (e.g. ``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``).  Unknown values fall back to ``INFO``.
        log_format: ``"json"`` (default) emits structured JSON; any other
            value (commonly ``"text"``) emits human-readable lines.
        force: When ``True``, reconfigures logging even if it was already
            initialised.  Useful in tests that require a fresh handler.
        stream: Output stream for the handler.  Defaults to ``sys.stdout``.
    """

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
    """Temporarily enrich log records with request-scoped context fields.

    Uses a :class:`contextvars.ContextVar` to attach key/value pairs to every
    log record emitted within the managed block, including those produced in
    nested coroutines or threads that inherit the context.  The previous
    context is restored when the block exits — even on exception.

    ``None``, ``""`` and ``"-"`` values are silently filtered out so that
    placeholder sentinels are never written to the log payload.

    Args:
        **context: Arbitrary key/value pairs to attach to log records.
            Keys that map to falsy sentinel values (``None``, ``""``,
            ``"-"``) are ignored.

    Yields:
        ``None`` — the managed block runs with the enriched context.

    Example::

        with logging_context(session_id="abc123", workflow="issue_details"):
            logger.info("Processing request")  # record carries session_id & workflow
    """

    current = dict(_LOG_CONTEXT.get())
    updates = {key: value for key, value in context.items() if value not in (None, "", "-")}
    token = _LOG_CONTEXT.set({**current, **updates})
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def _resolve_log_level(level: str) -> int:
    """Convert a level name string to the corresponding :mod:`logging` integer.

    Args:
        level: A case-insensitive log level name such as ``"DEBUG"``,
            ``"INFO"``, ``"WARNING"``, ``"ERROR"``, or ``"CRITICAL"``.

    Returns:
        The integer constant from :mod:`logging` (e.g. ``logging.DEBUG ==
        10``).  Returns ``logging.INFO`` (20) for any unrecognised value.
    """
    resolved = getattr(logging, level.upper(), logging.INFO)
    return resolved if isinstance(resolved, int) else logging.INFO