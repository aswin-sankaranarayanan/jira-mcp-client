"""Regression tests for issue details workflow."""

import unittest

from src.client.mcp_client import MCPClientError
from src.graph.workflows.issue_details import run_issue_details_workflow


class FakeMCPClient:
    def __init__(self) -> None:
        self.last_tool_args = None
        self.last_prompt_args = None
        self.fail_prompt = False
        self.prompt_text = "Formatted issue details"

    async def call_tool(self, tool_name, arguments):
        self.last_tool_args = arguments
        return {
            "name": tool_name,
            "arguments": arguments,
            "text": "Issue summary text",
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
    def generate(self, prompt: str) -> str:
        return prompt


class IssueDetailsWorkflowTests(unittest.TestCase):
    def test_issue_details_workflow_uses_declared_mcp_arguments(self) -> None:
        client = FakeMCPClient()
        llm = FakeOllamaService()
        state = {"user_input": "Get the details of the issue PROJ-15"}

        result = run_issue_details_workflow(state, client, llm)

        self.assertEqual(result.get("issue_key"), "PROJ-15")
        self.assertEqual(result.get("final_response"), "Formatted issue details")
        self.assertEqual(client.last_tool_args, {"issue_id": "PROJ-15"})
        self.assertEqual(client.last_prompt_args, {"issue_json": "Issue summary text"})

    def test_issue_details_workflow_returns_friendly_error_when_prompt_fails(self) -> None:
        client = FakeMCPClient()
        llm = FakeOllamaService()
        client.fail_prompt = True

        result = run_issue_details_workflow({"user_input": "Get details for PROJ-15"}, client, llm)

        self.assertEqual(
            result.get("error"),
            "I fetched the issue details, but I couldn't format the response right now. Please try again.",
        )
        self.assertIsNone(result.get("final_response"))

    def test_issue_details_workflow_returns_friendly_error_when_prompt_text_is_empty(self) -> None:
        client = FakeMCPClient()
        llm = FakeOllamaService()
        client.prompt_text = "   "

        result = run_issue_details_workflow({"user_input": "Get details for PROJ-15"}, client, llm)

        self.assertIsNone(result.get("error"))
        self.assertEqual(result.get("final_response"), "")

    def test_issue_details_workflow_returns_prompt_for_streaming_mode(self) -> None:
        client = FakeMCPClient()
        llm = FakeOllamaService()

        result = run_issue_details_workflow(
            {"user_input": "Get details for PROJ-15"},
            client,
            llm,
            stream=True,
        )

        self.assertIsNone(result.get("error"))
        self.assertEqual(result.get("final_response"), "")
        self.assertEqual(result.get("prompt_output"), "Formatted issue details")


if __name__ == "__main__":
    unittest.main()
