"""Unit tests for team member status workflow."""

import json
import unittest

from src.client.mcp_client import MCPClientError
from src.graph.workflows.team_status import (
    _extract_assignee_name,
    _group_issues_by_assignee,
    format_assignee_presentation,
    generate_scrum_master_report,
    run_team_status_workflow,
)


class FakeMCPClient:
    def __init__(self) -> None:
        self.last_tool_args = None
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


class FakeOllamaService:
    def infer_board_name(self, user_input: str):
        return "Platform Team"

    def generate(self, prompt: str) -> str:
        return f"LLM: {prompt[:40]}"

    def build_scrum_master_report_prompt(self, payload: dict) -> str:
        return f"Scrum report prompt for {payload.get('board_name', '')}"

    # Kept for backward-compatibility with any remaining callers.
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
# Unit tests for format_assignee_presentation
# ---------------------------------------------------------------------------


class FormatAssigneePresentationTests(unittest.TestCase):
    def _issues(self):
        return [
            {"key": "PROJ-1", "summary": "Fix login bug", "status": "In Progress"},
            {"key": "PROJ-3", "summary": "Deploy service", "status": "Blocked"},
        ]

    def test_contains_assignee_name(self) -> None:
        text = format_assignee_presentation("Alice", self._issues(), 0, 3)
        self.assertIn("Alice", text)

    def test_shows_position_in_sequence(self) -> None:
        text = format_assignee_presentation("Alice", self._issues(), 0, 3)
        self.assertIn("1 of 3", text)

    def test_lists_each_issue(self) -> None:
        text = format_assignee_presentation("Alice", self._issues(), 0, 3)
        self.assertIn("PROJ-1", text)
        self.assertIn("PROJ-3", text)
        self.assertIn("In Progress", text)
        self.assertIn("Blocked", text)

    def test_ends_with_update_prompt(self) -> None:
        text = format_assignee_presentation("Alice", self._issues(), 0, 3)
        self.assertIn("please provide your status update", text)


# ---------------------------------------------------------------------------
# Unit tests for generate_scrum_master_report
# ---------------------------------------------------------------------------


class GenerateScrumMasterReportTests(unittest.TestCase):
    def test_calls_llm_with_all_assignees(self) -> None:
        llm = FakeOllamaService()
        issues_by = {
            "Alice": [{"key": "PROJ-1", "summary": "Fix login", "status": "Done"}],
            "Bob": [{"key": "PROJ-2", "summary": "Add tests", "status": "Open"}],
        }
        updates = {"Alice": "Finished the fix.", "Bob": "Tests in progress."}
        report = generate_scrum_master_report("Platform Team", ["Alice", "Bob"], issues_by, updates, llm)
        self.assertIsInstance(report, str)
        self.assertTrue(len(report) > 0)

    def test_uses_no_update_placeholder_for_missing_assignee(self) -> None:
        prompts_seen = []

        class CaptureLLM(FakeOllamaService):
            def build_scrum_master_report_prompt(self, payload):
                prompts_seen.append(payload)
                return "prompt"

        generate_scrum_master_report(
            "Board",
            ["Alice"],
            {"Alice": []},
            {},  # no update for Alice
            CaptureLLM(),
        )
        member = prompts_seen[0]["team_members"][0]
        self.assertEqual(member["status_update"], "(no update provided)")


# ---------------------------------------------------------------------------
# Workflow integration tests
# ---------------------------------------------------------------------------


class TeamStatusWorkflowTests(unittest.TestCase):
    def test_workflow_calls_correct_tool(self) -> None:
        client = FakeMCPClient()
        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            client,
            FakeOllamaService(),
        )
        self.assertEqual(client.last_tool_args, {"scrum_board_name": "Platform Team"})

    def test_workflow_initiates_collecting_phase(self) -> None:
        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            FakeMCPClient(),
            FakeOllamaService(),
        )
        self.assertEqual(result.get("team_status_phase"), "collecting")

    def test_workflow_returns_sorted_assignees(self) -> None:
        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            FakeMCPClient(),
            FakeOllamaService(),
        )
        self.assertEqual(result.get("assignees"), ["Alice", "Bob"])

    def test_workflow_groups_issues_by_assignee(self) -> None:
        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            FakeMCPClient(),
            FakeOllamaService(),
        )
        issues_by = result.get("assignee_issues", {})
        self.assertIn("Alice", issues_by)
        self.assertIn("Bob", issues_by)
        self.assertEqual(len(issues_by["Alice"]), 2)
        self.assertEqual(len(issues_by["Bob"]), 1)

    def test_workflow_starts_at_index_zero_with_empty_updates(self) -> None:
        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            FakeMCPClient(),
            FakeOllamaService(),
        )
        self.assertEqual(result.get("current_assignee_index"), 0)
        self.assertEqual(result.get("collected_updates"), {})

    def test_workflow_final_response_contains_first_assignee_presentation(self) -> None:
        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            FakeMCPClient(),
            FakeOllamaService(),
        )
        response = result.get("final_response", "")
        # First assignee alphabetically is Alice.
        self.assertIn("Alice", response)
        self.assertIn("PROJ-1", response)

    def test_workflow_final_response_includes_intro_header(self) -> None:
        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            FakeMCPClient(),
            FakeOllamaService(),
        )
        self.assertIn("Team Status Collection", result.get("final_response", ""))

    def test_workflow_metadata_contains_phase_collecting(self) -> None:
        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            FakeMCPClient(),
            FakeOllamaService(),
        )
        metadata = result.get("metadata", {})
        self.assertEqual(metadata.get("workflow"), "team_status")
        self.assertEqual(metadata.get("phase"), "collecting")
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
        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            client,
            FakeOllamaService(),
        )
        issues_by = result.get("assignee_issues", {})
        self.assertIn("Eve", issues_by)
        self.assertEqual(len(issues_by["Eve"]), 2)

    def test_workflow_handles_unassigned_issues(self) -> None:
        client = FakeMCPClient()
        client.issues_json = json.dumps(
            [
                {"key": "P-1", "status": "Open"},
                {"key": "P-2", "assignee": "Frank", "status": "In Progress"},
            ]
        )
        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            client,
            FakeOllamaService(),
        )
        issues_by = result.get("assignee_issues", {})
        self.assertIn("Unassigned", issues_by)
        self.assertIn("Frank", issues_by)

    def test_workflow_no_issues_returns_complete_phase(self) -> None:
        client = FakeMCPClient()
        client.issues_json = json.dumps([])
        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            client,
            FakeOllamaService(),
        )
        metadata = result.get("metadata", {})
        self.assertEqual(metadata.get("phase"), "complete")
        self.assertIsNone(result.get("error"))

    def test_workflow_tool_error_returns_error_key(self) -> None:
        class FailingClient(FakeMCPClient):
            async def call_tool(self, tool_name, arguments):
                raise MCPClientError("connection refused")

        result = run_team_status_workflow(
            {"user_input": "Show team status for board Platform Team"},
            FailingClient(),
            FakeOllamaService(),
        )
        self.assertIsNotNone(result.get("error"))
        self.assertIsNone(result.get("team_status_phase"))


if __name__ == "__main__":
    unittest.main()
