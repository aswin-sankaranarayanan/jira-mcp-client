"""Unit tests for team member status workflow."""

import json
import unittest

from src.client.mcp_client import MCPClientError
from src.graph.workflows.team_status import (
    _extract_assignee_name,
    _group_issues_by_assignee,
    run_team_status_workflow,
)


class FakeMCPClient:
    def __init__(self) -> None:
        self.last_tool_args = None
        self.last_prompt_args = None
        self.fail_prompt = False
        self.prompt_text = "Formatted team status report"
        self.issues_json: str = json.dumps(
            [
                {"key": "PROJ-1", "summary": "Fix login bug", "status": "In Progress", "assignee": "Alice"},
                {"key": "PROJ-2", "summary": "Add tests", "status": "Open", "assignee": "Bob"},
                {"key": "PROJ-3", "summary": "Deploy service", "status": "Blocked", "assignee": "Alice"},
            ]
        )

    async def call_tool(self, tool_name, arguments):
        self.last_tool_args = arguments
        return {
            "name": tool_name,
            "arguments": arguments,
            "text": self.issues_json,
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

    def generate(self, prompt: str) -> str:
        return f"LLM: {prompt}"

    def build_team_status_prompt(self, payload: dict) -> str:
        return f"Direct prompt for {payload.get('board_name', '')}"


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class ExtractAssigneeNameTests(unittest.TestCase):
    def test_plain_string_field(self) -> None:
        self.assertEqual(_extract_assignee_name({"assignee": "Alice"}), "Alice")

    def test_assignee_object_display_name(self) -> None:
        self.assertEqual(
            _extract_assignee_name({"assignee": {"displayName": "Bob Jones"}}),
            "Bob Jones",
        )

    def test_assignee_object_name_fallback(self) -> None:
        self.assertEqual(
            _extract_assignee_name({"assignee": {"name": "cjones"}}),
            "cjones",
        )

    def test_nested_fields_structure(self) -> None:
        self.assertEqual(
            _extract_assignee_name(
                {"fields": {"assignee": {"displayName": "Carol"}}}
            ),
            "Carol",
        )

    def test_assignee_name_key(self) -> None:
        self.assertEqual(
            _extract_assignee_name({"assignee_name": "Dave"}),
            "Dave",
        )

    def test_missing_assignee_returns_unassigned(self) -> None:
        self.assertEqual(_extract_assignee_name({}), "Unassigned")

    def test_blank_string_assignee_returns_unassigned(self) -> None:
        self.assertEqual(_extract_assignee_name({"assignee": "   "}), "Unassigned")


class GroupIssuesByAssigneeTests(unittest.TestCase):
    def test_groups_correctly(self) -> None:
        issues = [
            {"key": "A-1", "assignee": "Alice"},
            {"key": "A-2", "assignee": "Bob"},
            {"key": "A-3", "assignee": "Alice"},
        ]
        result = _group_issues_by_assignee(issues)
        self.assertEqual(set(result.keys()), {"Alice", "Bob"})
        self.assertEqual(len(result["Alice"]), 2)
        self.assertEqual(len(result["Bob"]), 1)

    def test_non_dict_items_are_skipped(self) -> None:
        issues = [{"assignee": "Alice"}, "not-a-dict", None]
        result = _group_issues_by_assignee(issues)
        self.assertEqual(list(result.keys()), ["Alice"])

    def test_empty_list(self) -> None:
        self.assertEqual(_group_issues_by_assignee([]), {})


# ---------------------------------------------------------------------------
# Workflow integration tests
# ---------------------------------------------------------------------------


class TeamStatusWorkflowTests(unittest.TestCase):
    def test_workflow_calls_correct_tool_and_groups_by_assignee(self) -> None:
        client = FakeMCPClient()
        llm = FakeOllamaService()

        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            client,
            llm,
        )

        self.assertEqual(client.last_tool_args, {"scrum_board_name": "Platform Team"})
        self.assertIn("team_status_json", client.last_prompt_args)

        payload = json.loads(client.last_prompt_args["team_status_json"])
        self.assertEqual(payload["board_name"], "Platform Team")

        members = {m["assignee"]: m["issues"] for m in payload["team_members"]}
        self.assertIn("Alice", members)
        self.assertIn("Bob", members)
        self.assertEqual(len(members["Alice"]), 2)
        self.assertEqual(len(members["Bob"]), 1)

    def test_workflow_returns_final_response_from_prompt_output(self) -> None:
        client = FakeMCPClient()
        llm = FakeOllamaService()

        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            client,
            llm,
        )

        self.assertEqual(result.get("final_response"), "LLM: Formatted team status report")
        self.assertIsNone(result.get("error"))

    def test_workflow_falls_back_to_direct_llm_when_prompt_fails(self) -> None:
        client = FakeMCPClient()
        client.fail_prompt = True
        llm = FakeOllamaService()

        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            client,
            llm,
        )

        self.assertIsNone(result.get("error"))
        self.assertIn("Direct prompt for Platform Team", result.get("final_response", ""))

    def test_workflow_returns_streaming_mode_without_final_response(self) -> None:
        client = FakeMCPClient()
        llm = FakeOllamaService()

        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            client,
            llm,
            stream=True,
        )

        self.assertIsNone(result.get("error"))
        self.assertEqual(result.get("final_response"), "")
        self.assertEqual(result.get("prompt_output"), "Formatted team status report")

    def test_workflow_includes_team_member_count_in_metadata(self) -> None:
        client = FakeMCPClient()
        llm = FakeOllamaService()

        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            client,
            llm,
        )

        metadata = result.get("metadata", {})
        self.assertEqual(metadata.get("workflow"), "team_status")
        self.assertEqual(metadata.get("team_member_count"), 2)

    def test_workflow_requires_clarification_when_board_name_missing(self) -> None:
        class NoInferLLM(FakeOllamaService):
            def infer_board_name(self, user_input: str):
                return None

        result = run_team_status_workflow(
            {"user_input": "show team status"},
            FakeMCPClient(),
            NoInferLLM(),
        )

        self.assertTrue(result.get("requires_clarification"))
        self.assertIsNotNone(result.get("error"))

    def test_workflow_handles_issues_wrapped_in_dict(self) -> None:
        client = FakeMCPClient()
        client.issues_json = json.dumps(
            {
                "issues": [
                    {"key": "X-1", "assignee": "Eve", "status": "Open"},
                    {"key": "X-2", "assignee": "Eve", "status": "Done"},
                ]
            }
        )
        llm = FakeOllamaService()

        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            client,
            llm,
        )

        payload = json.loads(client.last_prompt_args["team_status_json"])
        members = {m["assignee"]: m["issues"] for m in payload["team_members"]}
        self.assertIn("Eve", members)
        self.assertEqual(len(members["Eve"]), 2)

    def test_workflow_handles_unassigned_issues(self) -> None:
        client = FakeMCPClient()
        client.issues_json = json.dumps(
            [
                {"key": "P-1", "status": "Open"},
                {"key": "P-2", "assignee": "Frank", "status": "In Progress"},
            ]
        )
        llm = FakeOllamaService()

        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            client,
            llm,
        )

        payload = json.loads(client.last_prompt_args["team_status_json"])
        assignees = [m["assignee"] for m in payload["team_members"]]
        self.assertIn("Unassigned", assignees)
        self.assertIn("Frank", assignees)


if __name__ == "__main__":
    unittest.main()
