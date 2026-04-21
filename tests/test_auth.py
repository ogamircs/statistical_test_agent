"""Tests for optional Chainlit password auth."""

from __future__ import annotations

import pytest

from src.auth import is_auth_enabled, verify_credentials


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("STATAGENT_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("STATAGENT_AUTH_PASSWORD", raising=False)
    yield


def test_auth_disabled_when_env_unset() -> None:
    assert is_auth_enabled() is False
    assert verify_credentials("anyone", "anything") is None


def test_auth_disabled_when_only_username_set(monkeypatch) -> None:
    monkeypatch.setenv("STATAGENT_AUTH_USERNAME", "admin")
    assert is_auth_enabled() is False
    assert verify_credentials("admin", "x") is None


def test_verify_credentials_accepts_match(monkeypatch) -> None:
    monkeypatch.setenv("STATAGENT_AUTH_USERNAME", "admin")
    monkeypatch.setenv("STATAGENT_AUTH_PASSWORD", "s3cr3t")

    assert is_auth_enabled() is True
    assert verify_credentials("admin", "s3cr3t") == "admin"


def test_verify_credentials_rejects_wrong_password(monkeypatch) -> None:
    monkeypatch.setenv("STATAGENT_AUTH_USERNAME", "admin")
    monkeypatch.setenv("STATAGENT_AUTH_PASSWORD", "s3cr3t")

    assert verify_credentials("admin", "nope") is None


def test_verify_credentials_rejects_wrong_username(monkeypatch) -> None:
    monkeypatch.setenv("STATAGENT_AUTH_USERNAME", "admin")
    monkeypatch.setenv("STATAGENT_AUTH_PASSWORD", "s3cr3t")

    assert verify_credentials("attacker", "s3cr3t") is None


def test_verify_credentials_rejects_non_strings(monkeypatch) -> None:
    monkeypatch.setenv("STATAGENT_AUTH_USERNAME", "admin")
    monkeypatch.setenv("STATAGENT_AUTH_PASSWORD", "s3cr3t")

    assert verify_credentials(None, "s3cr3t") is None  # type: ignore[arg-type]
    assert verify_credentials("admin", None) is None  # type: ignore[arg-type]
