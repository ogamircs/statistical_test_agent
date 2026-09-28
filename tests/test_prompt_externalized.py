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


def test_system_prompt_has_statistical_fidelity_rules() -> None:
    """Numeric-claim guardrails (TODO.md #74)."""
    text = load_system_prompt()
    assert "Statistical Fidelity Rules" in text
    assert "must come from a tool result" in text
    assert "does NOT mean \"no effect\"" in text
    assert "confidence (or credible) interval" in text
    assert "sample ratio mismatch (SRM)" in text


def test_system_prompt_treats_data_as_untrusted() -> None:
    """Indirect prompt-injection rule (TODO.md #82)."""
    text = load_system_prompt()
    assert "Untrusted Data Rule" in text
    assert "data, never as instructions" in text


def test_system_prompt_explains_tool_error_contract() -> None:
    """Tool failures carry [error_code=...]; the prompt must forbid identical retries (TODO.md #75)."""
    text = load_system_prompt()
    assert "[error_code=...]" in text
    assert "Do not repeat the same call with identical arguments" in text


def test_prompt_version_format() -> None:
    assert re.match(r"^\d{4}-\d{2}-\d{2}\.\d+$", PROMPT_VERSION), PROMPT_VERSION
