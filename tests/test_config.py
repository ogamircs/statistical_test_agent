"""Tests for Config.from_env + validation."""

from __future__ import annotations

import pytest

from src.config import Config


def test_defaults_match_legacy_constants() -> None:
    cfg = Config()
    assert cfg.llm_model == "gpt-5.2"
    assert cfg.llm_temperature == 0.0
    assert cfg.sql_default_row_limit == 20
    assert cfg.query_timeout_seconds == 5.0


def test_from_env_overrides_each_field() -> None:
    env = {
        "STATAGENT_LLM_MODEL": "claude-sonnet-4-6",
        "STATAGENT_LLM_TEMPERATURE": "0.3",
        "STATAGENT_SQL_ROW_LIMIT": "50",
        "STATAGENT_QUERY_TIMEOUT_SECONDS": "12",
    }
    cfg = Config.from_env(env)
    assert cfg.llm_model == "claude-sonnet-4-6"
    assert cfg.llm_temperature == 0.3
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


def test_validate_rejects_nonpositive_limits() -> None:
    with pytest.raises(ValueError):
        Config(query_timeout_seconds=0).validate()
    with pytest.raises(ValueError):
        Config(sql_default_row_limit=0).validate()


def test_validate_rejects_empty_model() -> None:
    with pytest.raises(ValueError):
        Config(llm_model="").validate()


def test_agent_picks_up_injected_config(monkeypatch, tmp_path) -> None:
    """Agent should use the injected Config rather than re-reading the env."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    from src.agent import ABTestingAgent

    cfg = Config(agent_recursion_limit=7, query_store_dir=str(tmp_path))
    agent = ABTestingAgent(config=cfg)
    assert agent.config.agent_recursion_limit == 7
    # The injected Config (not the env) decides where the session store lives.
    assert agent.session.query_store_path.parent == tmp_path


def test_query_store_dir_comes_from_env_and_is_validated() -> None:
    assert Config.from_env({"STATAGENT_QUERY_STORE_DIR": "/data/stores"}).query_store_dir == "/data/stores"
    assert Config.from_env({}).query_store_dir == "output/query_store"
    with pytest.raises(ValueError):
        Config(query_store_dir="  ").validate()


@pytest.mark.parametrize(
    "raw, expected",
    [("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("false", False), ("", False), ("maybe", False)],
)
def test_require_auth_parses_truthy_env_values(raw: str, expected: bool) -> None:
    assert Config.from_env({"STATAGENT_REQUIRE_AUTH": raw}).require_auth is expected


def test_require_auth_defaults_off() -> None:
    assert Config.from_env({}).require_auth is False
