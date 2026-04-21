"""Entity extraction helpers for workflow routing."""

from __future__ import annotations

import re
from typing import Optional

ISSUE_KEY_REGEX = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")


def extract_issue_key(text: str) -> Optional[str]:
    """Extract the first Jira issue key from *text* using a regex pattern.

    A valid Jira issue key matches the pattern ``[A-Z][A-Z0-9]+-\\d+``
    (e.g. ``"PROJ-123"`` or ``"AB-1"``).  Only the first match is returned.

    Args:
        text: Arbitrary string that may contain a Jira issue key.

    Returns:
        The first matched issue key in uppercase, or ``None`` if no key is
        found.
    """
    match = ISSUE_KEY_REGEX.search(text)
    return match.group(1) if match else None


def extract_board_name(text: str) -> Optional[str]:
    """Extract a Scrum board name from a natural-language string.

    Uses a two-pass strategy:

    1. **Quoted extraction** — looks for a value enclosed in single or double
       quotes (2–80 characters).
    2. **Pattern matching** — tries a set of common English phrasings such as
       ``"for board <name>"``, ``"on board <name>"``, and ``"active sprint
       for <name> board"``.

    The extracted name is normalised with :func:`_normalize_board_name`
    before being returned.

    Args:
        text: A natural-language query that may reference a Scrum board.

    Returns:
        The normalised board name, or ``None`` if no board name can be
        extracted.
    """
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
    """Normalise a raw board name string for consistent downstream use.

    Strips common punctuation and whitespace from both ends and collapses
    internal runs of whitespace to a single space.

    Args:
        value: Raw board name string, possibly containing leading/trailing
            whitespace or punctuation.

    Returns:
        A cleaned board name string, or an empty string if *value* was blank
        after stripping.
    """
    cleaned = value.strip(" \t\n\r.,;:!?()[]{}")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned
