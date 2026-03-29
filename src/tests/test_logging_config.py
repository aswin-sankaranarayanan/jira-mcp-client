"""Tests for application logging configuration."""

import io
import json
import unittest

from src.logging_config import configure_logging, get_logger, logging_context


class LoggingConfigTests(unittest.TestCase):
    def test_configure_logging_emits_json_with_context(self) -> None:
        stream = io.StringIO()
        configure_logging("INFO", "json", force=True, stream=stream)

        logger = get_logger(__name__)
        with logging_context(request_id="req-123", workflow="issue_details", issue_key="PROJ-1"):
            logger.info("Structured log message", extra={"duration_ms": 12.5})

        payload = json.loads(stream.getvalue().strip())
        self.assertEqual(payload["level"], "INFO")
        self.assertEqual(payload["message"], "Structured log message")
        self.assertEqual(payload["request_id"], "req-123")
        self.assertEqual(payload["workflow"], "issue_details")
        self.assertEqual(payload["issue_key"], "PROJ-1")
        self.assertEqual(payload["duration_ms"], 12.5)


if __name__ == "__main__":
    unittest.main()