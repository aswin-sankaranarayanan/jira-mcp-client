"""Entity extraction helpers for workflow routing."""

from __future__ import annotations

import re
from typing import Optional

ISSUE_KEY_REGEX = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")


def extract_issue_key(text: str) -> Optional[str]:
    match = ISSUE_KEY_REGEX.search(text)
    return match.group(1) if match else None


def extract_board_name(text: str) -> Optional[str]:
    quoted = re.search(r"[\"']([A-Za-z0-9 _-]{2,80})[\"']", text)
    if quoted:
        return _normalize_board_name(quoted.group(1))

    patterns = [
        r"\bboard\s*(?:name)?\s*(?:is|=|:)?\s*([A-Za-z0-9 _-]{2,80})",
        r"\bfor\s+board\s+([A-Za-z0-9 _-]{2,80})",
        r"\bon\s+board\s+([A-Za-z0-9 _-]{2,80})",
        r"\bactive\s+sprint\s+for\s+([A-Za-z0-9 _-]{2,80})\s+board",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _normalize_board_name(match.group(1))

    return None


def _normalize_board_name(value: str) -> str:
    cleaned = value.strip(" \t\n\r.,;:!?()[]{}")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned
