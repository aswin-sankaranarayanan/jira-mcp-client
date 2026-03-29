"""Centralized MCP SSE client.

All interactions with MCP tools and prompts must flow through this class.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any, Dict, List

from mcp import ClientSession
from mcp.client.sse import sse_client

from src.logging_config import get_logger


logger = get_logger(__name__)


class MCPClientError(RuntimeError):
    """Raised when an MCP operation fails."""


class MCPClient:
    """Thin abstraction over MCP SSE transport and session lifecycle."""

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url
        logger.info("Configured MCP client", extra={"server_url": server_url})

    @asynccontextmanager
    async def _session(self):
        logger.debug("Opening MCP session", extra={"server_url": self.server_url})
        try:
            async with sse_client(self.server_url) as streams:
                read_stream, write_stream = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    logger.debug("MCP session initialized", extra={"server_url": self.server_url})
                    yield session
        except Exception as exc:
            logger.exception("Failed to open MCP session", extra={"server_url": self.server_url})
            raise MCPClientError(f"Failed to connect to MCP server at {self.server_url}: {exc}") from exc

    async def list_tools(self) -> List[str]:
        async with self._session() as session:
            result = await session.list_tools()
            tools = [tool.name for tool in getattr(result, "tools", [])]
            logger.info("Fetched MCP tools", extra={"tool_count": len(tools)})
            return tools

    async def list_prompts(self) -> List[str]:
        async with self._session() as session:
            result = await session.list_prompts()
            prompts = [prompt.name for prompt in getattr(result, "prompts", [])]
            logger.info("Fetched MCP prompts", extra={"prompt_count": len(prompts)})
            return prompts

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        started_at = perf_counter()
        logger.info("Calling MCP tool", extra={"tool_name": tool_name, "argument_keys": sorted(arguments)})
        async with self._session() as session:
            try:
                result = await session.call_tool(tool_name, arguments=arguments)
                structured = self._extract_structured_content(result)
                text = self._extract_text(result, structured_fallback=structured)
                duration_ms = round((perf_counter() - started_at) * 1000, 2)
                logger.info(
                    "MCP tool call completed",
                    extra={
                        "tool_name": tool_name,
                        "duration_ms": duration_ms,
                        "text_length": len(text),
                        "has_structured": structured is not None,
                    },
                )
                return {
                    "name": tool_name,
                    "arguments": arguments,
                    "text": text,
                    "structured": structured,
                    "raw": str(result),
                }
            except Exception as exc:
                logger.exception(
                    "MCP tool call failed",
                    extra={
                        "tool_name": tool_name,
                        "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                    },
                )
                raise MCPClientError(f"Tool call failed for '{tool_name}': {exc}") from exc

    async def run_prompt(self, prompt_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        started_at = perf_counter()
        logger.info(
            "Running MCP prompt",
            extra={"prompt_name": prompt_name, "argument_keys": sorted(arguments)},
        )
        async with self._session() as session:
            try:
                result = await session.get_prompt(prompt_name, arguments=arguments)
                text = self._strip_leading_json_blob(self._extract_text(result))
                duration_ms = round((perf_counter() - started_at) * 1000, 2)
                logger.info(
                    "MCP prompt completed",
                    extra={
                        "prompt_name": prompt_name,
                        "duration_ms": duration_ms,
                        "text_length": len(text),
                    },
                )
                return {
                    "name": prompt_name,
                    "arguments": arguments,
                    "text": text,
                    "raw": str(result),
                }
            except Exception as exc:
                logger.exception(
                    "MCP prompt failed",
                    extra={
                        "prompt_name": prompt_name,
                        "duration_ms": round((perf_counter() - started_at) * 1000, 2),
                    },
                )
                raise MCPClientError(f"Prompt execution failed for '{prompt_name}': {exc}") from exc

    def _strip_leading_json_blob(self, text: str) -> str:
        """Remove a leading JSON object or array from text, returning only the trailing human-readable portion."""
        stripped = text.lstrip()
        if not stripped.startswith(("{" , "[")):
            return text
        try:
            _, end = json.JSONDecoder().raw_decode(stripped)
            return stripped[end:].lstrip("\n").lstrip()
        except json.JSONDecodeError:
            return text

    def _extract_text(self, payload: Any, structured_fallback: Any = None) -> str:
        chunks: List[str] = []

        # Tool outputs often expose .content
        if hasattr(payload, "content"):
            for item in getattr(payload, "content", []):
                text = getattr(item, "text", None)
                if text:
                    chunks.append(str(text))

        # Prompt outputs often expose .messages with nested .content
        if hasattr(payload, "messages"):
            assistant_chunks: List[str] = []
            fallback_chunks: List[str] = []

            for message in getattr(payload, "messages", []):
                text = self._extract_message_text(message)
                if not text:
                    continue

                role = str(getattr(message, "role", "") or "").lower()
                if role in {"assistant", "model"}:
                    assistant_chunks.append(text)
                else:
                    fallback_chunks.append(text)

            chunks.extend(assistant_chunks or fallback_chunks)

        return "\n".join(chunk for chunk in chunks if chunk).strip()

    def _extract_message_text(self, message: Any) -> str:
        content = getattr(message, "content", None)

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
            return "\n".join(parts).strip()

        text = getattr(content, "text", None)
        return str(text).strip() if text else ""

    def _extract_structured_content(self, payload: Any) -> Any:
        # Newer MCP tool responses can provide structuredContent directly.
        structured = getattr(payload, "structuredContent", None)
        if structured is not None:
            return structured

        # Fallback: inspect content blocks for non-text structured data.
        for item in getattr(payload, "content", []):
            item_data = getattr(item, "data", None)
            if item_data is not None:
                return item_data

            item_json = getattr(item, "json", None)
            if item_json is not None:
                return item_json

        return None
