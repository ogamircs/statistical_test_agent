"""Tests for the data-loading runtime helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent_runtime import AgentRuntime


class _FakeAnalyzer:
    def __init__(self):
        self.df = None
        self.load_calls = []

    def load_data(self, filepath, **kwargs):
        self.load_calls.append((filepath, kwargs))
        self.df = object()
        return {"columns": ["a", "b"], "shape": (10, 2)}


def test_runtime_load_data_returns_analyzer_info_and_size(tmp_path: Path) -> None:
    analyzer = _FakeAnalyzer()
    runtime = AgentRuntime(analyzer=analyzer)
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    loaded, info, size_mb = runtime.load_data(str(csv_path))

    assert loaded is analyzer
    assert runtime.get_active_analyzer() is analyzer
    assert info["shape"] == (10, 2)
    assert size_mb > 0
    assert analyzer.load_calls[0][0] == str(csv_path.resolve())


def test_get_file_size_mb_returns_size_for_existing_file(tmp_path: Path) -> None:
    runtime = AgentRuntime(analyzer=_FakeAnalyzer())
    target = tmp_path / "sample.csv"
    payload = b"x" * (1024 * 1024 + 256)
    target.write_bytes(payload)

    size_mb = runtime.get_file_size_mb(str(target))

    assert size_mb == pytest.approx(len(payload) / (1024 * 1024))


def test_get_file_size_mb_warns_on_missing_file_and_returns_zero(tmp_path, caplog) -> None:
    runtime = AgentRuntime(analyzer=_FakeAnalyzer())
    missing = tmp_path / "does_not_exist.csv"

    with caplog.at_level("WARNING", logger="src.agent_runtime"):
        size_mb = runtime.get_file_size_mb(str(missing))

    assert size_mb == 0.0
    assert any(
        "get_file_size_mb failed" in rec.message and str(missing) in rec.message
        for rec in caplog.records
    )


def test_get_file_size_mb_propagates_unexpected_exception(monkeypatch) -> None:
    import os as _os

    runtime = AgentRuntime(analyzer=_FakeAnalyzer())

    def _boom(_path):
        raise RuntimeError("boom")

    monkeypatch.setattr(_os.path, "getsize", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        runtime.get_file_size_mb("/anything")


def test_runtime_normalizes_shape_from_row_count_and_columns() -> None:
    runtime = AgentRuntime(analyzer=_FakeAnalyzer())

    assert runtime.normalize_shape({"row_count": 12, "columns": ["a", "b", "c"]}) == (12, 3)


def test_runtime_raises_when_shape_metadata_missing() -> None:
    runtime = AgentRuntime(analyzer=_FakeAnalyzer())

    with pytest.raises(KeyError):
        runtime.normalize_shape({"columns": ["a"]})
