"""Tests for CSV data-path confinement (TODO.md #48).

The LLM controls the ``filepath`` argument of the loading tools, so every
path must be confined to explicitly allowed roots before it reaches
``pd.read_csv`` (which would otherwise read arbitrary local files or fetch
URLs).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent_runtime import AgentRuntime
from src.data_paths import (
    DataPathNotAllowedError,
    default_data_roots,
    resolve_data_path,
)


class _FakeAnalyzer:
    def __init__(self):
        self.df = None
        self.load_calls = []

    def load_data(self, filepath, **kwargs):
        self.load_calls.append((filepath, kwargs))
        self.df = object()
        return {"columns": ["a", "b"], "shape": (10, 2)}


@pytest.mark.parametrize(
    "url",
    [
        "http://internal-host/data.csv",
        "https://example.com/data.csv",
        "ftp://example.com/data.csv",
        "s3://bucket/key.csv",
        "file:///etc/passwd",
    ],
)
def test_rejects_url_schemes(url: str) -> None:
    with pytest.raises(DataPathNotAllowedError):
        resolve_data_path(url)


def test_rejects_absolute_path_outside_allowed_roots() -> None:
    with pytest.raises(DataPathNotAllowedError):
        resolve_data_path("/etc/passwd")


def test_rejects_project_dotenv(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=secret", encoding="utf-8")

    with pytest.raises(DataPathNotAllowedError):
        resolve_data_path(str(env_file), allowed_roots=[tmp_path / "data"])


def test_rejects_traversal_escaping_allowed_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(DataPathNotAllowedError):
        resolve_data_path("data/../secret.txt", allowed_roots=[tmp_path / "data"])


def test_rejects_hidden_file_inside_allowed_root(tmp_path: Path) -> None:
    hidden = tmp_path / ".hidden.csv"
    hidden.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(DataPathNotAllowedError):
        resolve_data_path(str(hidden), allowed_roots=[tmp_path])


def test_accepts_file_in_data_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "sample.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    resolved = resolve_data_path("data/sample.csv")

    assert resolved == csv_path.resolve()


def test_accepts_chainlit_upload_in_files_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    upload_dir = tmp_path / ".files" / "session-abc"
    upload_dir.mkdir(parents=True)
    csv_path = upload_dir / "upload.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    resolved = resolve_data_path(str(csv_path))

    assert resolved == csv_path.resolve()


def test_accepts_file_in_system_tempdir(tmp_path: Path) -> None:
    csv_path = tmp_path / "upload.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    resolved = resolve_data_path(str(csv_path))

    assert resolved == csv_path.resolve()


def test_env_var_extends_default_roots(monkeypatch, tmp_path: Path) -> None:
    extra_root = tmp_path / "warehouse"
    monkeypatch.setenv("STATAGENT_DATA_ROOTS", str(extra_root))

    roots = default_data_roots()

    assert extra_root.resolve() in roots


def test_error_carries_stable_code_and_user_message() -> None:
    with pytest.raises(DataPathNotAllowedError) as excinfo:
        resolve_data_path("/etc/passwd")

    assert excinfo.value.code == "DATA_PATH_NOT_ALLOWED"
    assert isinstance(excinfo.value.user_message, str)
    assert excinfo.value.user_message


def test_runtime_load_rejects_disallowed_path() -> None:
    pandas_analyzer = _FakeAnalyzer()
    runtime = AgentRuntime(
        analyzer=pandas_analyzer,
        spark_factory=None,
        spark_available=lambda: False,
    )

    with pytest.raises(DataPathNotAllowedError):
        runtime.load_data_with_backend("/etc/passwd")

    assert pandas_analyzer.load_calls == []


def test_runtime_load_accepts_allowed_path(tmp_path: Path) -> None:
    pandas_analyzer = _FakeAnalyzer()
    runtime = AgentRuntime(
        analyzer=pandas_analyzer,
        spark_factory=None,
        spark_available=lambda: False,
    )
    csv_path = tmp_path / "ok.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    analyzer, info, backend, *_ = runtime.load_data_with_backend(str(csv_path))

    assert analyzer is pandas_analyzer
    assert backend == "pandas"
    assert pandas_analyzer.load_calls[0][0] == str(csv_path.resolve())
