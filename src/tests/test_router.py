"""Regression tests for router error handling."""

import unittest

from src.graph.router import JiraWorkflowEngine


class FakeGraph:
    def invoke(self, state):
        raise RuntimeError("boom")


class FakeOllamaService:
    def detect_intent(self, user_input: str) -> str:
        return "clarify"

    def generate(self, prompt: str) -> str:
        return "fallback response"

    def stream_generate(self, prompt: str):
        yield "streamed "
        yield "response"

    def stream_markdown(self, text: str):
        yield text


class FakeClient:
    async def call_tool(self, tool_name, arguments):
        return {
            "name": tool_name,
            "arguments": arguments,
            "text": "payload",
        }

    async def run_prompt(self, prompt_name, arguments):
        return {
            "name": prompt_name,
            "arguments": arguments,
            "text": "Prompt body",
        }


class RouterTests(unittest.TestCase):
    def test_run_returns_friendly_error_when_workflow_execution_fails(self) -> None:
        engine = JiraWorkflowEngine(object(), FakeOllamaService())
        engine._graph = FakeGraph()

        result = engine.run("Show issue PROJ-1")

        self.assertIsNotNone(result.get("error"))
        self.assertEqual(result.get("response"), "")

    def test_stream_response_yields_clarification_for_unknown_intent(self) -> None:
        engine = JiraWorkflowEngine(FakeClient(), FakeOllamaService())
        chunks = list(engine.stream_response("hello"))

        self.assertTrue(chunks)
        self.assertIn("I can help with three Jira workflows", "".join(chunks))

    def test_stream_response_streams_from_prompt_when_workflow_is_issue_details(self) -> None:
        class IssueIntentLLM(FakeOllamaService):
            def detect_intent(self, user_input: str) -> str:
                return "issue_details"

        engine = JiraWorkflowEngine(FakeClient(), IssueIntentLLM())
        chunks = list(engine.stream_response("Get issue PROJ-1"))

        self.assertEqual("".join(chunks), "streamed response")

    def test_stream_events_emits_progress_and_tokens(self) -> None:
        class IssueIntentLLM(FakeOllamaService):
            def detect_intent(self, user_input: str) -> str:
                return "issue_details"

        engine = JiraWorkflowEngine(FakeClient(), IssueIntentLLM())
        events = list(engine.stream_events("Get issue PROJ-1"))

        event_types = [event.get("type") for event in events]
        self.assertIn("progress", event_types)
        self.assertIn("token", event_types)
        self.assertEqual(event_types[-1], "done")


if __name__ == "__main__":
    unittest.main()