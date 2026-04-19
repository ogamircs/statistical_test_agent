"""Lightweight observability hooks: token usage + optional JSON logging.

This module is intentionally dependency-free (stdlib + langchain core).
``TokenUsageCallback`` is a LangChain callback that tallies prompt /
completion tokens across one ``agent.run`` and emits a single info log
when the run finishes. ``configure_json_logging`` swaps the root
formatter for a one-line JSON formatter when ``STATAGENT_LOG_FORMAT=json``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from langchain_core.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)


class TokenUsageCallback(BaseCallbackHandler):
    """Sum prompt + completion + total tokens across a single agent run."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.calls = 0

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.calls = 0

    def on_llm_end(self, response: Any, **_: Any) -> None:
        usage = self._extract_usage(response)
        if not usage:
            return
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(
            usage.get("total_tokens")
            or (int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0))
        )

    @staticmethod
    def _extract_usage(response: Any) -> Optional[Dict[str, Any]]:
        # LangChain's LLMResult exposes usage via .llm_output or per-message
        # response_metadata depending on the provider. Try both.
        usage = None
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            usage = llm_output.get("token_usage") or llm_output.get("usage")
        if usage:
            return usage
        for generation_list in getattr(response, "generations", []) or []:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                metadata = getattr(message, "response_metadata", None) if message else None
                if isinstance(metadata, dict):
                    candidate = metadata.get("token_usage") or metadata.get("usage")
                    if candidate:
                        return candidate
        return None

    def snapshot(self) -> Dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class _JsonFormatter(logging.Formatter):
    """Emit one JSON line per record. No batching, no rotation."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


def configure_json_logging(env: Optional[Dict[str, str]] = None) -> bool:
    """Swap the root formatter for JSON when ``STATAGENT_LOG_FORMAT=json``.

    Returns ``True`` if JSON logging was enabled, ``False`` otherwise.
    Safe to call multiple times; it only mutates handlers it can see.
    """
    source = env if env is not None else os.environ
    if source.get("STATAGENT_LOG_FORMAT", "").lower() != "json":
        return False

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        root.addHandler(handler)
    for handler in root.handlers:
        handler.setFormatter(_JsonFormatter())
    return True
