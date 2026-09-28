"""Tests for optional web UI password auth (credentials come from Config)."""

from __future__ import annotations

from src.auth import is_auth_enabled, verify_credentials
from src.config import Config

_CONFIGURED = Config(auth_username="admin", auth_password="s3cr3t")


def test_auth_disabled_without_credentials() -> None:
    assert is_auth_enabled(Config()) is False
    assert verify_credentials(Config(), "anyone", "anything") is None


def test_auth_disabled_with_only_username() -> None:
    config = Config(auth_username="admin")
    assert is_auth_enabled(config) is False
    assert verify_credentials(config, "admin", "x") is None


def test_verify_credentials_accepts_match() -> None:
    assert is_auth_enabled(_CONFIGURED) is True
    assert verify_credentials(_CONFIGURED, "admin", "s3cr3t") == "admin"


def test_verify_credentials_rejects_wrong_password() -> None:
    assert verify_credentials(_CONFIGURED, "admin", "nope") is None


def test_verify_credentials_rejects_wrong_username() -> None:
    assert verify_credentials(_CONFIGURED, "attacker", "s3cr3t") is None


def test_verify_credentials_rejects_non_strings() -> None:
    assert verify_credentials(_CONFIGURED, None, "s3cr3t") is None  # type: ignore[arg-type]
    assert verify_credentials(_CONFIGURED, "admin", None) is None  # type: ignore[arg-type]


def test_auth_ignores_process_env_when_config_is_injected(monkeypatch) -> None:
    # PR #10 review: an injected Config must fully control auth.
    monkeypatch.setenv("STATAGENT_AUTH_USERNAME", "env-user")
    monkeypatch.setenv("STATAGENT_AUTH_PASSWORD", "env-pass")
    assert is_auth_enabled(Config()) is False
    assert verify_credentials(Config(), "env-user", "env-pass") is None


def test_credentials_are_hidden_from_repr() -> None:
    assert "s3cr3t" not in repr(_CONFIGURED)
