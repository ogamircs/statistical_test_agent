"""Tests for observability hooks (token usage callback + JSON formatter)."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from src.observability import TokenUsageCallback, _JsonFormatter, configure_json_logging


def _llm_result_with_usage(prompt: int, completion: int) -> SimpleNamespace:
    return SimpleNamespace(
        llm_output={
            "token_usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }
        },
        generations=[],
    )


def _llm_result_with_message_metadata(prompt: int, completion: int) -> SimpleNamespace:
    message = SimpleNamespace(
        response_metadata={
            "token_usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }
        }
    )
    generation = SimpleNamespace(message=message)
    return SimpleNamespace(llm_output=None, generations=[[generation]])


def test_token_usage_aggregates_across_calls() -> None:
    cb = TokenUsageCallback()
    cb.on_llm_end(_llm_result_with_usage(10, 5))
    cb.on_llm_end(_llm_result_with_usage(20, 7))
    snap = cb.snapshot()
    assert snap == {
        "calls": 2,
        "prompt_tokens": 30,
        "completion_tokens": 12,
        "total_tokens": 42,
    }


def test_token_usage_extracts_from_message_metadata() -> None:
    cb = TokenUsageCallback()
    cb.on_llm_end(_llm_result_with_message_metadata(8, 3))
    assert cb.snapshot()["total_tokens"] == 11


def test_token_usage_reset_zeros_state() -> None:
    cb = TokenUsageCallback()
    cb.on_llm_end(_llm_result_with_usage(10, 5))
    cb.reset()
    assert cb.snapshot() == {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_token_usage_handles_response_without_usage() -> None:
    cb = TokenUsageCallback()
    cb.on_llm_end(SimpleNamespace(llm_output=None, generations=[]))
    assert cb.snapshot()["total_tokens"] == 0


def test_configure_json_logging_returns_false_when_disabled() -> None:
    assert configure_json_logging({"STATAGENT_LOG_FORMAT": "text"}) is False
    assert configure_json_logging({}) is False


def test_configure_json_logging_swaps_root_formatter() -> None:
    root = logging.getLogger()
    original = list(root.handlers)
    try:
        assert configure_json_logging({"STATAGENT_LOG_FORMAT": "json"}) is True
        for handler in root.handlers:
            assert isinstance(handler.formatter, _JsonFormatter)

        record = logging.LogRecord(
            name="testlogger",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        formatted = root.handlers[0].formatter.format(record)
        payload = json.loads(formatted)
        assert payload["level"] == "INFO"
        assert payload["logger"] == "testlogger"
        assert payload["message"] == "hello world"
    finally:
        for handler in root.handlers:
            handler.setFormatter(logging.Formatter())
        root.handlers = original
