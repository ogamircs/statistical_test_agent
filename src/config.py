"""Centralized runtime configuration for the conversational agent.

All knobs that previously lived as scattered constants — LLM model name,
temperature, backend file-size threshold, default SQL row limit — collapse
into one ``Config`` dataclass loadable from environment variables. The
defaults preserve current behavior so existing callers can adopt
``Config()`` lazily without behavior change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_LLM_MODEL = "gpt-5.2"
_DEFAULT_LLM_TEMPERATURE = 0.0
_DEFAULT_FILE_SIZE_THRESHOLD_MB = 2.0
_DEFAULT_SQL_ROW_LIMIT = 20
_DEFAULT_QUERY_TIMEOUT_SECONDS = 5.0
_DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 120.0
_DEFAULT_LLM_MAX_RETRIES = 2
# LangGraph's own default; one ReAct step (model call or tool call) per unit.
_DEFAULT_AGENT_RECURSION_LIMIT = 25
# Prior chat messages (human + AI) re-sent to the model each turn. The full
# history stays persisted; only the model-bound window is bounded.
_DEFAULT_MAX_HISTORY_MESSAGES = 40


@dataclass(frozen=True)
class Config:
    """Resolved runtime configuration."""

    llm_model: str = _DEFAULT_LLM_MODEL
    llm_temperature: float = _DEFAULT_LLM_TEMPERATURE
    file_size_threshold_mb: float = _DEFAULT_FILE_SIZE_THRESHOLD_MB
    sql_default_row_limit: int = _DEFAULT_SQL_ROW_LIMIT
    query_timeout_seconds: float = _DEFAULT_QUERY_TIMEOUT_SECONDS
    llm_request_timeout_seconds: float = _DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS
    llm_max_retries: int = _DEFAULT_LLM_MAX_RETRIES
    agent_recursion_limit: int = _DEFAULT_AGENT_RECURSION_LIMIT
    max_history_messages: int = _DEFAULT_MAX_HISTORY_MESSAGES

    @classmethod
    def from_env(cls, environ: dict | None = None) -> "Config":
        """Build a Config from environment variables (or a provided mapping)."""
        env = environ if environ is not None else os.environ

        return cls(
            llm_model=env.get("STATAGENT_LLM_MODEL", _DEFAULT_LLM_MODEL),
            llm_temperature=_coerce_float(
                env.get("STATAGENT_LLM_TEMPERATURE"), _DEFAULT_LLM_TEMPERATURE
            ),
            file_size_threshold_mb=_coerce_float(
                env.get("STATAGENT_FILE_SIZE_THRESHOLD_MB"),
                _DEFAULT_FILE_SIZE_THRESHOLD_MB,
            ),
            sql_default_row_limit=_coerce_int(
                env.get("STATAGENT_SQL_ROW_LIMIT"), _DEFAULT_SQL_ROW_LIMIT
            ),
            query_timeout_seconds=_coerce_float(
                env.get("STATAGENT_QUERY_TIMEOUT_SECONDS"),
                _DEFAULT_QUERY_TIMEOUT_SECONDS,
            ),
            llm_request_timeout_seconds=_coerce_float(
                env.get("STATAGENT_LLM_TIMEOUT_SECONDS"),
                _DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
            ),
            llm_max_retries=_coerce_int(
                env.get("STATAGENT_LLM_MAX_RETRIES"), _DEFAULT_LLM_MAX_RETRIES
            ),
            agent_recursion_limit=_coerce_int(
                env.get("STATAGENT_AGENT_RECURSION_LIMIT"),
                _DEFAULT_AGENT_RECURSION_LIMIT,
            ),
            max_history_messages=_coerce_int(
                env.get("STATAGENT_MAX_HISTORY_MESSAGES"),
                _DEFAULT_MAX_HISTORY_MESSAGES,
            ),
        )

    def validate(self) -> None:
        """Raise ValueError when a numeric knob is out of range."""
        if not self.llm_model:
            raise ValueError("Config.llm_model must be a non-empty string")
        if self.llm_temperature < 0 or self.llm_temperature > 2:
            raise ValueError("Config.llm_temperature must be in [0, 2]")
        if self.file_size_threshold_mb <= 0:
            raise ValueError("Config.file_size_threshold_mb must be > 0")
        if self.sql_default_row_limit <= 0:
            raise ValueError("Config.sql_default_row_limit must be > 0")
        if self.query_timeout_seconds <= 0:
            raise ValueError("Config.query_timeout_seconds must be > 0")
        if self.llm_request_timeout_seconds <= 0:
            raise ValueError("Config.llm_request_timeout_seconds must be > 0")
        if self.llm_max_retries < 0:
            raise ValueError("Config.llm_max_retries must be >= 0")
        if self.agent_recursion_limit < 2:
            raise ValueError("Config.agent_recursion_limit must be >= 2")
        if self.max_history_messages < 1:
            raise ValueError("Config.max_history_messages must be >= 1")


def _coerce_float(value: object, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        if isinstance(value, (int, float)):
            return int(value)
        return int(str(value))
    except (TypeError, ValueError):
        return default
