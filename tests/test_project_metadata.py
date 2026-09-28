"""Regression tests for project metadata, CI configuration, and repo docs."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_pyproject_declares_canonical_metadata_and_extras() -> None:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml should be the canonical metadata file"

    pyproject = pyproject_path.read_text(encoding="utf-8")
    assert "[project]" in pyproject
    assert "dependencies = [" in pyproject
    assert "[project.optional-dependencies]" in pyproject
    assert "dev = [" in pyproject
    assert "pyspark" not in pyproject, "the project is pandas-only; Spark was removed"


def test_requirements_file_delegates_to_project_metadata() -> None:
    requirements = _read("requirements.txt")
    assert ".[dev]" in requirements, "requirements.txt should install the project from pyproject metadata"


def test_ci_installs_from_lockfile() -> None:
    workflow = _read(".github/workflows/ci.yml")
    assert "uv sync --frozen --extra dev" in workflow, (
        "CI must install from the committed uv.lock so it tests the same "
        "dependency versions the Docker image ships (TODO.md #64)"
    )
    assert "spark" not in workflow.lower()


def test_ci_enforces_mypy_and_repo_wide_ruff() -> None:
    workflow = _read(".github/workflows/ci.yml")
    assert "continue-on-error" not in workflow, (
        "mypy must be blocking (TODO.md #61)"
    )
    assert "mypy src app.py" in workflow
    assert "ruff check ." in workflow, "lint the whole repo including scripts/ (TODO.md #71)"


def test_gitignore_excludes_generated_output_artifacts() -> None:
    gitignore = _read(".gitignore")
    assert "output/" in gitignore
    assert "*.md" in gitignore
    assert "!README.md" in gitignore
    assert "!docs/architecture.md" in gitignore
    assert "!docs/development.md" in gitignore
    assert "!docs/testing.md" in gitignore


def test_readme_documents_modern_install_flow() -> None:
    readme = _read("README.md")
    assert "uv sync --extra dev" in readme
    assert "pandas" in readme
    assert "spark" not in readme.lower()


def test_curated_docs_cover_architecture_development_and_testing() -> None:
    architecture = _read("docs/architecture.md")
    development = _read("docs/development.md")
    testing = _read("docs/testing.md")

    assert "app.py" in architecture
    assert "uv sync --extra dev" in development
    assert "chainlit" in development.lower() or "python app.py" in development
    assert "pytest -q" in testing
    for doc in (architecture, development, testing):
        assert "spark" not in doc.lower()


def test_chainlit_is_fully_removed() -> None:
    pyproject = _read("pyproject.toml")
    assert "chainlit" not in pyproject.lower()
    assert "fastapi" in pyproject and "uvicorn" in pyproject
    assert not (REPO_ROOT / ".chainlit").exists()
    assert not (REPO_ROOT / "public").exists()
    assert "chainlit" not in _read("Dockerfile").lower()
    assert "chainlit" not in _read("app.py").lower()


def test_frontend_package_defines_ci_scripts_and_lockfile() -> None:
    package = json.loads(_read("frontend/package.json"))
    for script in ("dev", "build", "typecheck", "test"):
        assert script in package["scripts"], f"frontend is missing the {script!r} script"
    assert (REPO_ROOT / "frontend" / "package-lock.json").exists(), "commit the npm lockfile"
    for dependency in ("react", "react-markdown", "remark-gfm", "plotly.js-dist-min"):
        assert dependency in package["dependencies"]


def test_frontend_loads_no_external_assets() -> None:
    """No CDN/hot-linked assets: the UI must work air-gapped (TODO.md #72)."""
    sources = [REPO_ROOT / "frontend" / "index.html"]
    sources += [
        path
        for path in (REPO_ROOT / "frontend" / "src").rglob("*")
        if path.suffix in {".ts", ".tsx", ".css"} and not path.name.endswith((".test.ts", ".test.tsx"))
    ]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert "http://" not in text and "https://" not in text, f"external URL in {path.name}"


def test_ci_builds_and_tests_the_frontend() -> None:
    workflow = _read(".github/workflows/ci.yml")
    assert "frontend:" in workflow
    for step in ("npm ci", "npm run typecheck", "npm test", "npm run build"):
        assert step in workflow, f"CI frontend job must run {step!r}"


def test_dockerfile_builds_ui_and_ships_runtime_only() -> None:
    dockerfile = _read("Dockerfile")
    assert "FROM node:" in dockerfile and "npm run build" in dockerfile
    assert "--extra dev" not in dockerfile, "runtime image must not ship dev tooling (TODO.md #70)"
    assert "HEALTHCHECK" in dockerfile and "/api/health" in dockerfile, "TODO.md #116"
    assert 'CMD ["uvicorn", "app:app"' in dockerfile


def test_agents_md_is_tracked_and_documents_ci_gates() -> None:
    gitignore = _read(".gitignore")
    assert "!AGENTS.md" in gitignore, "AGENTS.md must be allowlisted past the *.md ignore"
    agents = _read("AGENTS.md")
    for gate in ("ruff check .", "mypy src app.py", "--cov-fail-under=78"):
        assert gate in agents, f"AGENTS.md must document the CI gate: {gate}"
