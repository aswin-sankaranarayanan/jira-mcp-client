"""Regression tests for sprint insights workflow."""

import json
import unittest

from src.client.mcp_client import MCPClientError
from src.graph.workflows.sprint_insights import run_sprint_insights_workflow


class FakeMCPClient:
    def __init__(self) -> None:
        self.last_tool_args = None
        self.last_prompt_args = None
        self.fail_prompt = False
        self.prompt_text = "Formatted sprint insights"

    async def call_tool(self, tool_name, arguments):
        self.last_tool_args = arguments
        return {
            "name": tool_name,
            "arguments": arguments,
            "text": "Issue list text",
        }

    async def run_prompt(self, prompt_name, arguments):
        self.last_prompt_args = arguments
        if self.fail_prompt:
            raise MCPClientError("prompt failed")
        return {
            "name": prompt_name,
            "arguments": arguments,
            "text": self.prompt_text,
        }


class FakeOllamaService:
    def infer_board_name(self, user_input: str):
        return "Platform Team"


class SprintInsightsWorkflowTests(unittest.TestCase):
    def test_sprint_insights_uses_prompt_output_without_analysis_fallback(self) -> None:
        client = FakeMCPClient()
        llm = FakeOllamaService()

        result = run_sprint_insights_workflow(
            {"user_input": "Show sprint progress for board Platform Team"},
            client,
            llm,
        )

        self.assertEqual(result.get("final_response"), "Formatted sprint insights")
        self.assertEqual(client.last_tool_args, {"board_name": "Platform Team"})
        self.assertIn("sprint_data_json", client.last_prompt_args)
        sprint_payload = json.loads(client.last_prompt_args["sprint_data_json"])
        self.assertEqual(sprint_payload.get("metadata"), {"board_name": "Platform Team"})
        self.assertEqual(sprint_payload.get("issues"), [{"raw": "Issue list text"}])

        sprint_meta = sprint_payload.get("sprint_meta")
        self.assertIsInstance(sprint_meta, dict)
        self.assertEqual(sprint_meta.get("id"), "unknown-sprint")
        self.assertEqual(sprint_meta.get("name"), "Active Sprint")
        self.assertEqual(sprint_meta.get("goal"), "")
        self.assertEqual(sprint_meta.get("board_id"), "unknown-board")
        self.assertEqual(sprint_meta.get("board_name"), "Platform Team")

        self.assertRegex(sprint_meta.get("start_date", ""), r"^\d{4}-\d{2}-\d{2}$")
        self.assertRegex(sprint_meta.get("end_date", ""), r"^\d{4}-\d{2}-\d{2}$")

    def test_sprint_insights_returns_friendly_error_when_prompt_fails(self) -> None:
        client = FakeMCPClient()
        client.fail_prompt = True
        llm = FakeOllamaService()

        result = run_sprint_insights_workflow(
            {"user_input": "Show sprint progress for board Platform Team"},
            client,
            llm,
        )

        self.assertEqual(
            result.get("error"),
            "I analyzed the sprint data, but I couldn't format the sprint insights right now. Please try again.",
        )
        self.assertIsNone(result.get("final_response"))

    def test_sprint_insights_returns_friendly_error_when_prompt_text_is_empty(self) -> None:
        client = FakeMCPClient()
        client.prompt_text = ""
        llm = FakeOllamaService()

        result = run_sprint_insights_workflow(
            {"user_input": "Show sprint progress for board Platform Team"},
            client,
            llm,
        )

        self.assertIsNone(result.get("error"))
        self.assertEqual(result.get("final_response"), "")


if __name__ == "__main__":
    unittest.main()