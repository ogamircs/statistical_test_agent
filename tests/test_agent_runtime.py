"""Tests for the extracted backend runtime helper."""

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


class _FailingAnalyzer(_FakeAnalyzer):
    def load_data(self, filepath, **kwargs):
        self.load_calls.append((filepath, kwargs))
        raise RuntimeError("spark load failed")


def test_runtime_prefers_spark_for_large_files(monkeypatch, tmp_path: Path) -> None:
    pandas_analyzer = _FakeAnalyzer()
    spark_analyzer = _FakeAnalyzer()
    runtime = AgentRuntime(
        analyzer=pandas_analyzer,
        spark_factory=lambda: spark_analyzer,
        spark_available=lambda: True,
        file_size_threshold_mb=2.0,
    )
    csv_path = tmp_path / "large.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "get_file_size_mb", lambda _filepath: 10.0)

    analyzer, info, backend, file_size_mb, spark_selected, fallback_note = runtime.load_data_with_backend(str(csv_path))

    assert analyzer is spark_analyzer
    assert info["shape"] == (10, 2)
    assert backend == "spark"
    assert file_size_mb == 10.0
    assert spark_selected is True
    assert fallback_note is None
    assert runtime.using_spark is True
    assert len(spark_analyzer.load_calls) == 1
    assert len(pandas_analyzer.load_calls) == 0


def test_runtime_prefers_pandas_for_small_files_even_when_spark_available(
    monkeypatch, tmp_path: Path
) -> None:
    pandas_analyzer = _FakeAnalyzer()
    runtime = AgentRuntime(
        analyzer=pandas_analyzer,
        spark_factory=lambda: pytest.fail("Spark factory should not be called for small files"),
        spark_available=lambda: True,
        file_size_threshold_mb=2.0,
    )
    csv_path = tmp_path / "small.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "get_file_size_mb", lambda _filepath: 0.5)

    analyzer, info, backend, file_size_mb, spark_selected, fallback_note = runtime.load_data_with_backend(str(csv_path))

    assert analyzer is pandas_analyzer
    assert info["shape"] == (10, 2)
    assert backend == "pandas"
    assert file_size_mb == 0.5
    assert spark_selected is False
    assert fallback_note is None
    assert runtime.using_spark is False
    assert len(pandas_analyzer.load_calls) == 1


def test_runtime_never_selects_spark_when_unavailable(monkeypatch, tmp_path: Path) -> None:
    pandas_analyzer = _FakeAnalyzer()
    runtime = AgentRuntime(
        analyzer=pandas_analyzer,
        spark_factory=lambda: pytest.fail("Spark factory should not be called when unavailable"),
        spark_available=lambda: False,
        file_size_threshold_mb=2.0,
    )
    csv_path = tmp_path / "large.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "get_file_size_mb", lambda _filepath: 10.0)

    assert runtime.should_use_spark(str(csv_path)) is False

    analyzer, info, backend, file_size_mb, spark_selected, fallback_note = runtime.load_data_with_backend(str(csv_path))

    assert analyzer is pandas_analyzer
    assert info["shape"] == (10, 2)
    assert backend == "pandas"
    assert file_size_mb == 10.0
    assert spark_selected is False
    assert fallback_note is None
    assert runtime.using_spark is False
    assert len(pandas_analyzer.load_calls) == 1


def test_runtime_reuses_cached_spark_analyzer_instance() -> None:
    spark_analyzer = _FakeAnalyzer()
    factory_calls = []

    def _factory():
        factory_calls.append(1)
        return spark_analyzer

    runtime = AgentRuntime(
        analyzer=_FakeAnalyzer(),
        spark_factory=_factory,
        spark_available=lambda: True,
    )

    first = runtime.init_spark_analyzer()
    second = runtime.init_spark_analyzer()

    assert first is spark_analyzer
    assert second is spark_analyzer
    assert len(factory_calls) == 1


def test_runtime_falls_back_to_pandas_when_spark_load_fails(monkeypatch, tmp_path: Path) -> None:
    pandas_analyzer = _FakeAnalyzer()
    spark_analyzer = _FailingAnalyzer()
    runtime = AgentRuntime(
        analyzer=pandas_analyzer,
        spark_factory=lambda: spark_analyzer,
        spark_available=lambda: True,
        file_size_threshold_mb=2.0,
    )
    csv_path = tmp_path / "large.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(runtime, "get_file_size_mb", lambda _filepath: 10.0)

    analyzer, info, backend, file_size_mb, spark_selected, fallback_note = runtime.load_data_with_backend(str(csv_path))

    assert analyzer is pandas_analyzer
    assert info["shape"] == (10, 2)
    assert backend == "pandas"
    assert file_size_mb == 10.0
    assert spark_selected is True
    assert "spark load failed" in fallback_note
    assert runtime.using_spark is False
    assert len(spark_analyzer.load_calls) == 1
    assert len(pandas_analyzer.load_calls) == 1


def test_get_file_size_mb_returns_size_for_existing_file(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        analyzer=_FakeAnalyzer(),
        spark_factory=None,
        spark_available=lambda: False,
    )
    target = tmp_path / "sample.csv"
    payload = b"x" * (1024 * 1024 + 256)
    target.write_bytes(payload)

    size_mb = runtime.get_file_size_mb(str(target))

    assert size_mb == pytest.approx(len(payload) / (1024 * 1024))


def test_get_file_size_mb_warns_on_missing_file_and_returns_zero(tmp_path, caplog) -> None:
    runtime = AgentRuntime(
        analyzer=_FakeAnalyzer(),
        spark_factory=None,
        spark_available=lambda: False,
    )
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

    runtime = AgentRuntime(
        analyzer=_FakeAnalyzer(),
        spark_factory=None,
        spark_available=lambda: False,
    )

    def _boom(_path):
        raise RuntimeError("boom")

    monkeypatch.setattr(_os.path, "getsize", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        runtime.get_file_size_mb("/anything")


def test_runtime_normalizes_shape_from_row_count_and_columns() -> None:
    runtime = AgentRuntime(
        analyzer=_FakeAnalyzer(),
        spark_factory=lambda: _FakeAnalyzer(),
        spark_available=lambda: True,
    )

    assert runtime.normalize_shape({"row_count": 12, "columns": ["a", "b", "c"]}) == (12, 3)


def test_runtime_raises_when_shape_metadata_missing() -> None:
    runtime = AgentRuntime(
        analyzer=_FakeAnalyzer(),
        spark_factory=lambda: _FakeAnalyzer(),
        spark_available=lambda: True,
    )

    with pytest.raises(KeyError):
        runtime.normalize_shape({"columns": ["a"]})
