"""Tests for the query-store garbage collector."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.query_store_gc import (
    cleanup_query_stores,
    reset_gc_state,
    run_startup_gc,
)


def _backdate(path: Path, days: int) -> None:
    cutoff = time.time() - days * 86400
    os.utime(path, (cutoff, cutoff))


@pytest.fixture(autouse=True)
def _clean_gc_state():
    reset_gc_state()
    yield
    reset_gc_state()


def test_cleanup_deletes_old_files(tmp_path: Path) -> None:
    old = tmp_path / "session-old.sqlite"
    midway = tmp_path / "session-mid.sqlite"
    fresh = tmp_path / "session-new.sqlite"
    for f in (old, midway, fresh):
        f.write_bytes(b"\0" * 16)

    _backdate(old, 10)
    _backdate(midway, 5)
    _backdate(fresh, 1)

    result = cleanup_query_stores(tmp_path, max_age_days=7)

    assert str(old) in result["deleted"]
    assert not old.exists()
    assert midway.exists()
    assert fresh.exists()
    assert result["oversized"] == []


def test_cleanup_reports_oversized(tmp_path: Path) -> None:
    big = tmp_path / "session-big.sqlite"
    big.write_bytes(b"\0" * (60 * 1024 * 1024))

    result = cleanup_query_stores(tmp_path, max_age_days=7, size_warn_mb=50.0)

    assert str(big) in result["oversized"]
    assert big.exists()
    assert result["deleted"] == []


def test_cleanup_handles_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    result = cleanup_query_stores(missing)
    assert result == {"deleted": [], "oversized": []}


def test_cleanup_ignores_non_sqlite(tmp_path: Path) -> None:
    sqlite_old = tmp_path / "session-old.sqlite"
    txt = tmp_path / "session-old.txt"
    journal = tmp_path / "session-old.sqlite-journal"
    wal = tmp_path / "session-old.sqlite-wal"
    for f in (sqlite_old, txt, journal, wal):
        f.write_bytes(b"x")
        _backdate(f, 30)

    result = cleanup_query_stores(tmp_path, max_age_days=7)

    assert str(sqlite_old) in result["deleted"]
    assert not sqlite_old.exists()
    assert txt.exists()
    assert journal.exists()
    assert wal.exists()


def test_run_startup_gc_invoked_once_per_directory(tmp_path: Path) -> None:
    with patch("src.query_store_gc.cleanup_query_stores") as mocked:
        mocked.return_value = {"deleted": [], "oversized": []}

        run_startup_gc(tmp_path)
        run_startup_gc(tmp_path)
        run_startup_gc(tmp_path)

    assert mocked.call_count == 1
