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
