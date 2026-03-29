"""Regression tests for router error handling."""

import unittest

from src.graph.router import JiraWorkflowEngine


class FakeGraph:
    def invoke(self, state):
        raise RuntimeError("boom")


class FakeOllamaService:
    def detect_intent(self, user_input: str) -> str:
        return "clarify"

    def stream_markdown(self, text: str):
        yield text


class RouterTests(unittest.TestCase):
    def test_run_returns_friendly_error_when_workflow_execution_fails(self) -> None:
        engine = JiraWorkflowEngine(object(), FakeOllamaService())
        engine._graph = FakeGraph()

        result = engine.run("Show issue PROJ-1")

        self.assertIsNotNone(result.get("error"))
        self.assertEqual(result.get("response"), "")


if __name__ == "__main__":
    unittest.main()