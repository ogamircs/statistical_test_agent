"""Suite-wide fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_query_store_dir(tmp_path_factory, monkeypatch):
    """Keep per-session SQLite stores out of output/query_store.

    Agents built without an explicit path write there by default, and the
    React UI lists that directory as conversation history.
    """
    monkeypatch.setenv("STATAGENT_QUERY_STORE_DIR", str(tmp_path_factory.mktemp("query_store")))
