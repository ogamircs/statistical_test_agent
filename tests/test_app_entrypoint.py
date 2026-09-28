"""app.py must configure root logging so src.* logs survive under uvicorn."""

from __future__ import annotations

import importlib
import logging


def test_app_import_installs_a_root_log_handler(monkeypatch) -> None:
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])

    import app

    importlib.reload(app)

    assert root.handlers, "src.* INFO logs would be dropped under uvicorn"
    assert logging.getLogger("src.agent").isEnabledFor(logging.INFO)


def test_app_refuses_to_start_on_malformed_require_auth(monkeypatch) -> None:
    # PR #10 review: STATAGENT_REQUIRE_AUTH=tru must not silently disable the guard.
    import pytest

    import app

    monkeypatch.setenv("STATAGENT_REQUIRE_AUTH", "tru")
    with pytest.raises(ValueError, match="STATAGENT_REQUIRE_AUTH"):
        importlib.reload(app)
    monkeypatch.delenv("STATAGENT_REQUIRE_AUTH")
    importlib.reload(app)


def test_bad_tuning_knob_keeps_auth_settings(monkeypatch) -> None:
    import app

    monkeypatch.setenv("STATAGENT_LLM_TEMPERATURE", "9")  # out of range -> fallback
    monkeypatch.setenv("STATAGENT_REQUIRE_AUTH", "1")
    monkeypatch.setenv("STATAGENT_AUTH_USERNAME", "ana")
    monkeypatch.setenv("STATAGENT_AUTH_PASSWORD", "pw")
    try:
        importlib.reload(app)
        assert app._STARTUP_CONFIG.llm_temperature == 0.0
        assert app._STARTUP_CONFIG.auth_enabled is True
        assert app._STARTUP_CONFIG.require_auth is True
    finally:
        for name in ("STATAGENT_LLM_TEMPERATURE", "STATAGENT_REQUIRE_AUTH",
                     "STATAGENT_AUTH_USERNAME", "STATAGENT_AUTH_PASSWORD"):
            monkeypatch.delenv(name)
        importlib.reload(app)
