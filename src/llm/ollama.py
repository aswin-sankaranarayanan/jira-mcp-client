"""Ollama-powered helper methods for intent and analysis."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Iterable, Optional

from langchain_ollama import ChatOllama

from src.logging_config import get_logger


logger = get_logger(__name__)


class OllamaService:
    def __init__(self, model: str) -> None:
        self._llm = ChatOllama(model=model, temperature=0)
        logger.info("Configured Ollama service", extra={"model": model})

    def detect_intent(self, user_input: str) -> str:
        started_at = perf_counter()
        prompt = (
            "Classify this Jira request into one label: issue_details, sprint_insights, or clarify. "
            "Return JSON only with schema {\\\"intent\\\": \\\"<label>\\\"}.\n"
            f"Request: {user_input}"
        )
        raw = self._llm.invoke(prompt).content
        intent = self._safe_json_extract(raw).get("intent", "clarify")
        resolved_intent = (
            intent if intent in {"issue_details", "sprint_insights", "clarify"} else "clarify"
        )
        logger.info(
            "Detected workflow intent",
            extra={
                "intent": resolved_intent,
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                "input_length": len(user_input),
            },
        )
        return resolved_intent

    def infer_board_name(self, user_input: str) -> Optional[str]:
        started_at = perf_counter()
        prompt = (
            "Extract scrum board name from this Jira request. "
            "Return JSON only with schema {\\\"board_name\\\": \\\"<name or empty>\\\"}.\n"
            f"Request: {user_input}"
        )
        raw = self._llm.invoke(prompt).content
        board_name = self._safe_json_extract(raw).get("board_name", "").strip()
        resolved = board_name or None
        logger.info(
            "Inferred board name",
            extra={
                "board_name_detected": bool(resolved),
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        return resolved

    def summarize_issues(self, issues_text: str) -> str:
        started_at = perf_counter()
        prompt = (
            "Analyze the active sprint Jira issues and provide concise observations for a scrum master. "
            "Focus on delivery risk, blockers, and next actions.\n\n"
            f"Issues:\n{issues_text}"
        )
        summary = str(self._llm.invoke(prompt).content)
        logger.info(
            "Generated sprint analysis",
            extra={
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                "issues_length": len(issues_text),
                "summary_length": len(summary),
            },
        )
        return summary

    def generate(self, prompt: str) -> str:
        """Send a fully-formed prompt to the LLM and return its text response."""
        started_at = perf_counter()
        response = str(self._llm.invoke(prompt).content)
        logger.info(
            "Generated LLM response",
            extra={
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                "prompt_length": len(prompt),
                "response_length": len(response),
            },
        )
        return response

    def stream_generate(self, prompt: str) -> Iterable[str]:
        """Send a fully-formed prompt to the LLM and stream response chunks."""
        started_at = perf_counter()
        chunk_count = 0
        try:
            for chunk in self._llm.stream(prompt):
                content = getattr(chunk, "content", "")
                if content:
                    chunk_count += 1
                    yield str(content)
        except Exception:
            logger.exception("LLM streaming failed")
            raise
        finally:
            logger.info(
                "Completed LLM response stream",
                extra={
                    "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                    "prompt_length": len(prompt),
                    "chunk_count": chunk_count,
                },
            )

    def stream_markdown(self, text: str) -> Iterable[str]:
        logger.info("Streaming markdown response", extra={"input_length": len(text)})
        prompt = (
            "You are a Jira assistant. Improve readability while preserving meaning and markdown structure. "
            "Do not invent facts.\n\n"
            f"Input:\n{text}"
        )
        chunk_count = 0
        try:
            for chunk in self._llm.stream(prompt):
                content = getattr(chunk, "content", "")
                if content:
                    chunk_count += 1
                    yield str(content)
        except Exception:
            logger.exception("Markdown streaming failed")
            raise
        finally:
            logger.info("Completed markdown stream", extra={"chunk_count": chunk_count})

    def _safe_json_extract(self, raw: object) -> dict:
        text = str(raw).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Basic fallback for models that wrap JSON with text
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    logger.warning("Failed to decode wrapped JSON response")
                    return {}
            logger.warning("Failed to decode JSON response")
            return {}
