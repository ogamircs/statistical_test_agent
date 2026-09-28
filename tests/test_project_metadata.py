"""Regression tests for project metadata, CI configuration, and repo docs."""

from __future__ import annotations

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


def test_chainlit_config_references_custom_ui_assets() -> None:
    config = _read(".chainlit/config.toml")
    assert 'custom_css = "/public/custom.css"' in config
    assert 'custom_js = "/public/custom.js"' in config
    assert (REPO_ROOT / "public" / "custom.css").exists()
    assert (REPO_ROOT / "public" / "custom.js").exists()


def test_custom_ui_assets_define_centered_conversation_layout_hooks() -> None:
    custom_js = _read("public/custom.js")
    custom_css = _read("public/custom.css")
    assert "layout-centered-conversation" in custom_js
    assert "centered-conversation-root" in custom_js
    assert "centered-conversation-scroll" in custom_js
    assert ".centered-conversation-root" in custom_css
    assert ".centered-conversation-scroll" in custom_css


def test_custom_ui_assets_define_conversation_sidebar_hooks() -> None:
    custom_js = _read("public/custom.js")
    custom_css = _read("public/custom.css")
    assert "ab-testing-agent.conversation-list" in custom_js
    assert "ab-testing-agent.active-conversation" in custom_js
    assert "ab-testing-agent.clear-history-suppression" in custom_js
    assert "firstUserMessageTitle" in custom_js
    assert "loadClearHistorySuppression" in custom_js
    assert "conversation-history-title-text" in custom_js
    assert ".conversation-history-title-text" in custom_css
    assert ".conversation-history-item.is-active" in custom_css


def test_custom_ui_assets_define_processing_loader_hooks() -> None:
    custom_js = _read("public/custom.js")
    custom_css = _read("public/custom.css")
    assert "Ajax-loader.gif" in custom_js
    assert "enhanceProcessingIndicators" in custom_js
    assert "processing-indicator" in custom_js
    assert ".processing-indicator" in custom_css
    assert ".processing-indicator-gif" in custom_css
    assert "@keyframes processing-indicator-spin" in custom_css


def test_agents_md_is_tracked_and_documents_ci_gates() -> None:
    gitignore = _read(".gitignore")
    assert "!AGENTS.md" in gitignore, "AGENTS.md must be allowlisted past the *.md ignore"
    agents = _read("AGENTS.md")
    for gate in ("ruff check .", "mypy src app.py", "--cov-fail-under=78"):
        assert gate in agents, f"AGENTS.md must document the CI gate: {gate}"
