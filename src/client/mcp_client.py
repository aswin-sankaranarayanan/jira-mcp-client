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
    """Raised when an MCP operation fails.

    Wraps lower-level transport, protocol, or server errors so that callers
    can catch a single exception type without coupling to the underlying MCP
    SDK exceptions.
    """


class MCPClient:
    """Thin abstraction over MCP SSE transport and session lifecycle.

    Provides a simple async API for listing tools and prompts, invoking MCP
    tools, and running MCP prompts against a remote MCP server over
    Server-Sent Events (SSE).

    Each public method opens a *fresh* SSE session for the duration of the
    call and closes it afterwards, which keeps connection management simple
    at the cost of per-call overhead.

    Args:
        server_url: Base URL of the MCP SSE endpoint
            (e.g. ``"http://localhost:8000/sse"``).

    Raises:
        MCPClientError: Re-raised for any transport-level failure during
            session initialisation or method invocation.
    """

    def __init__(self, server_url: str) -> None:
        """Initialise the client with the MCP server endpoint.

        Args:
            server_url: Full URL of the SSE endpoint exposed by the MCP
                server (e.g. ``"http://localhost:8000/sse"``).
        """
        self.server_url = server_url
        logger.info("Configured MCP client", extra={"server_url": server_url})

    @asynccontextmanager
    async def _session(self):
        """Open an authenticated MCP session as an async context manager.

        Establishes the SSE transport, creates a :class:`mcp.ClientSession`,
        calls ``initialize()`` to complete the MCP handshake, and yields the
        ready session.  The connection is always closed when the context exits.

        Yields:
            An initialised :class:`mcp.ClientSession` ready for tool and
            prompt calls.

        Raises:
            MCPClientError: If the SSE connection or session initialisation
                fails.
        """
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
        """Return the names of all tools registered on the MCP server.

        Opens a temporary session, calls ``list_tools`` on the MCP protocol,
        and extracts tool names from the response.

        Returns:
            A list of tool name strings, possibly empty if the server
            registers no tools.

        Raises:
            MCPClientError: If the session cannot be established or the
                server returns an error.
        """
        async with self._session() as session:
            result = await session.list_tools()
            tools = [tool.name for tool in getattr(result, "tools", [])]
            logger.info("Fetched MCP tools", extra={"tool_count": len(tools)})
            return tools

    async def list_prompts(self) -> List[str]:
        """Return the names of all prompts registered on the MCP server.

        Opens a temporary session, calls ``list_prompts`` on the MCP
        protocol, and extracts prompt names from the response.

        Returns:
            A list of prompt name strings, possibly empty if the server
            registers no prompts.

        Raises:
            MCPClientError: If the session cannot be established or the
                server returns an error.
        """
        async with self._session() as session:
            result = await session.list_prompts()
            prompts = [prompt.name for prompt in getattr(result, "prompts", [])]
            logger.info("Fetched MCP prompts", extra={"prompt_count": len(prompts)})
            return prompts

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a named MCP tool and return a structured result envelope.

        Opens a fresh session, calls the tool, extracts both plain-text and
        structured content from the response, and measures call latency.

        Args:
            tool_name: The registered name of the tool to invoke.
            arguments: Key/value arguments forwarded to the tool as-is.

        Returns:
            A dictionary with the following keys:

            * ``"name"`` — echoed *tool_name*.
            * ``"arguments"`` — echoed *arguments*.
            * ``"text"`` — concatenated plain-text content from the response.
            * ``"structured"`` — parsed structured content, or ``None``.
            * ``"raw"`` — string representation of the raw MCP response.

        Raises:
            MCPClientError: If the session fails to open or the tool call
                returns an error.
        """
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
        """Execute a named MCP prompt and return the resulting text.

        Opens a fresh session, calls ``get_prompt``, strips any leading JSON
        blob from the assistant reply (which some MCP servers prepend), and
        measures call latency.

        Args:
            prompt_name: The registered name of the prompt to execute.
            arguments: Key/value arguments forwarded to the prompt as-is.

        Returns:
            A dictionary with the following keys:

            * ``"name"`` — echoed *prompt_name*.
            * ``"arguments"`` — echoed *arguments*.
            * ``"text"`` — the human-readable portion of the assistant reply,
              with any leading JSON blob stripped.
            * ``"raw"`` — string representation of the raw MCP response.

        Raises:
            MCPClientError: If the session fails to open or the prompt
                execution returns an error.
        """
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
        """Extract all plain-text content from an MCP response payload.

        Handles both tool responses (which expose ``.content`` with text
        blocks) and prompt responses (which expose ``.messages`` with nested
        content).  For prompt responses, assistant/model messages are
        preferred; messages from other roles act as a fallback.

        Args:
            payload: A raw MCP SDK response object — either a tool result or
                a prompt result.
            structured_fallback: Unused; reserved for future structured-to-
                text conversion.

        Returns:
            A single string formed by joining all extracted text chunks with
            newlines.  Returns an empty string if no text content is found.
        """
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
        """Extract the plain-text body from a single MCP message object.

        Handles three content shapes:

        * A bare ``str`` — returned directly after stripping.
        * A ``list`` of content blocks — each block's ``.text`` attribute is
          joined with newlines.
        * An object with a ``.text`` attribute — the attribute value is
          returned as a stripped string.

        Args:
            message: An MCP message object with a ``content`` attribute.

        Returns:
            Stripped plain text, or an empty string if no text can be
            extracted.
        """
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
        """Extract structured (non-text) content from an MCP tool response.

        First checks for the ``structuredContent`` attribute introduced in
        newer MCP tool response schemas.  Falls back to inspecting content
        blocks for ``data`` or ``json`` attributes.

        Args:
            payload: A raw MCP tool result object.

        Returns:
            The structured payload (a dict, list, or primitive), or ``None``
            if no structured content is present.
        """
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
