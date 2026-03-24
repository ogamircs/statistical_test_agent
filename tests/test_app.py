from pathlib import Path

from app import _ensure_chainlit_files_root


def test_ensure_chainlit_files_root_creates_files_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert not Path(".files").exists()

    _ensure_chainlit_files_root()

    assert Path(".files").is_dir()
