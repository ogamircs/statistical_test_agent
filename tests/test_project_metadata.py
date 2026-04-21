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
    assert "spark = [" in pyproject
    assert "pyspark" in pyproject


def test_requirements_file_delegates_to_project_metadata() -> None:
    requirements = _read("requirements.txt")
    assert ".[dev]" in requirements, "requirements.txt should install the project from pyproject metadata"


def test_ci_installs_from_project_metadata_and_has_spark_job() -> None:
    workflow = _read(".github/workflows/ci.yml")
    assert '".[dev]"' in workflow or "'.[dev]'" in workflow
    assert '".[dev,spark]"' in workflow or "'.[dev,spark]'" in workflow
    assert "spark" in workflow.lower()
    assert "tests/test_parity_pandas_spark.py" in workflow
    assert "tests/test_pyspark_analyzer.py" in workflow


def test_gitignore_excludes_generated_output_artifacts() -> None:
    gitignore = _read(".gitignore")
    assert "output/" in gitignore
    assert "*.md" in gitignore
    assert "!README.md" in gitignore
    assert "!docs/architecture.md" in gitignore
    assert "!docs/development.md" in gitignore
    assert "!docs/testing.md" in gitignore


def test_readme_documents_backend_capabilities_and_modern_install_flow() -> None:
    readme = _read("README.md")
    assert "## Backend Capability Matrix" in readme
    assert "uv sync --extra dev" in readme
    assert "PySpark" in readme
    assert "pandas" in readme


def test_curated_docs_cover_architecture_development_and_testing() -> None:
    architecture = _read("docs/architecture.md")
    development = _read("docs/development.md")
    testing = _read("docs/testing.md")

    assert "PySpark" in architecture
    assert "app.py" in architecture
    assert "uv sync --extra dev" in development
    assert "chainlit" in development.lower() or "python app.py" in development
    assert "pytest -q" in testing
    assert "tests/test_pyspark_analyzer.py" in testing
    assert "tests/test_parity_pandas_spark.py" in testing


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
