"""Tests for Config.from_env + validation."""

from __future__ import annotations

import pytest

from src.config import Config


def test_defaults_match_legacy_constants() -> None:
    cfg = Config()
    assert cfg.llm_model == "gpt-5.2"
    assert cfg.llm_temperature == 0.0
    assert cfg.file_size_threshold_mb == 2.0
    assert cfg.sql_default_row_limit == 20
    assert cfg.query_timeout_seconds == 5.0


def test_from_env_overrides_each_field() -> None:
    env = {
        "STATAGENT_LLM_MODEL": "claude-sonnet-4-6",
        "STATAGENT_LLM_TEMPERATURE": "0.3",
        "STATAGENT_FILE_SIZE_THRESHOLD_MB": "8.5",
        "STATAGENT_SQL_ROW_LIMIT": "50",
        "STATAGENT_QUERY_TIMEOUT_SECONDS": "12",
    }
    cfg = Config.from_env(env)
    assert cfg.llm_model == "claude-sonnet-4-6"
    assert cfg.llm_temperature == 0.3
    assert cfg.file_size_threshold_mb == 8.5
    assert cfg.sql_default_row_limit == 50
    assert cfg.query_timeout_seconds == 12.0


def test_from_env_falls_back_on_garbage_values() -> None:
    cfg = Config.from_env({"STATAGENT_LLM_TEMPERATURE": "not-a-number"})
    assert cfg.llm_temperature == 0.0


def test_from_env_ignores_missing_keys() -> None:
    cfg = Config.from_env({})
    assert cfg.llm_model == "gpt-5.2"


def test_validate_rejects_bad_temperature() -> None:
    with pytest.raises(ValueError):
        Config(llm_temperature=-0.1).validate()
    with pytest.raises(ValueError):
        Config(llm_temperature=2.5).validate()


def test_validate_rejects_nonpositive_threshold() -> None:
    with pytest.raises(ValueError):
        Config(file_size_threshold_mb=0).validate()
    with pytest.raises(ValueError):
        Config(query_timeout_seconds=0).validate()
    with pytest.raises(ValueError):
        Config(sql_default_row_limit=0).validate()


def test_validate_rejects_empty_model() -> None:
    with pytest.raises(ValueError):
        Config(llm_model="").validate()


def test_agent_picks_up_config_threshold(monkeypatch) -> None:
    """Agent should pick up file_size_threshold_mb from injected Config."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    from src.agent import ABTestingAgent

    cfg = Config(file_size_threshold_mb=7.5)
    agent = ABTestingAgent(config=cfg)
    assert agent.runtime.file_size_threshold_mb == 7.5
