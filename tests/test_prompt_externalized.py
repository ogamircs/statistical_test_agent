"""System prompt externalization sanity checks."""

from __future__ import annotations

import re

from src.prompts import PROMPT_VERSION, load_system_prompt


def test_system_prompt_loads_non_empty() -> None:
    text = load_system_prompt()
    assert isinstance(text, str)
    assert len(text) > 200
    assert "A/B Testing Analyst" in text


def test_system_prompt_contains_all_tool_capabilities() -> None:
    text = load_system_prompt()
    for token in (
        "load_and_auto_analyze",
        "load_csv",
        "configure_and_analyze",
        "answer_data_question",
        "plan_sample_size",
    ):
        assert token in text, f"prompt missing capability marker: {token}"


def test_prompt_version_format() -> None:
    assert re.match(r"^\d{4}-\d{2}-\d{2}\.\d+$", PROMPT_VERSION), PROMPT_VERSION
