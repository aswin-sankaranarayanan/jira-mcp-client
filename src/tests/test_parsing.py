"""Basic tests for entity parsing helpers."""

import unittest

from src.graph.parsing import extract_board_name, extract_issue_key


class ParsingTests(unittest.TestCase):
    def test_extract_issue_key(self) -> None:
        self.assertEqual(extract_issue_key("show PROJ-42 details"), "PROJ-42")

    def test_extract_board_name(self) -> None:
        self.assertEqual(
            extract_board_name("show sprint progress for board Platform Team"),
            "Platform Team",
        )


if __name__ == "__main__":
    unittest.main()
