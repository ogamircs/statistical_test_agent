# P3 Packaging, Docs, and CI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Modernize project metadata, tighten generated-artifact hygiene, document backend capability differences clearly, and add explicit Spark CI coverage.

**Architecture:** Move dependency definitions into a canonical `pyproject.toml` with extras for `dev` and `spark`, keep `requirements.txt` as a compatibility shim, and make CI install from project metadata instead of raw requirements. Treat docs and hygiene as executable contract surfaces by adding tests that assert the presence of the new metadata, backend capability documentation, and Spark CI job.

**Tech Stack:** Python packaging (`pyproject.toml`, setuptools, uv-compatible metadata), GitHub Actions, pytest, Markdown docs

---

### Task 1: Add metadata and CI regression tests

**Files:**
- Create: `tests/test_project_metadata.py`

**Step 1: Write the failing test**

Add tests that assert:
- `pyproject.toml` exists and declares project metadata plus `dev` and `spark` extras
- `requirements.txt` delegates to project metadata instead of duplicating the dependency graph
- `.github/workflows/ci.yml` installs from project metadata and contains a Spark-specific job/check path
- `.gitignore` ignores `output/`
- `README.md` and `LARGE_FILE_SUPPORT.md` contain backend capability documentation markers

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_project_metadata.py -q`
Expected: FAIL because `pyproject.toml` does not exist and the docs/CI markers are not present yet.

**Step 3: Write minimal implementation**

Create the test file with simple text assertions against repo files.

**Step 4: Run test to verify it passes after implementation work**

Run: `pytest tests/test_project_metadata.py -q`
Expected: PASS once metadata, CI, and docs are updated.

### Task 2: Introduce canonical project metadata

**Files:**
- Create: `pyproject.toml`
- Modify: `requirements.txt`

**Step 1: Write the failing test**

Covered by Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_project_metadata.py -q`

**Step 3: Write minimal implementation**

Add `pyproject.toml` that:
- defines the package metadata
- includes runtime dependencies
- separates `dev` extras from `spark` extras
- configures setuptools to package the existing `src` package tree

Update `requirements.txt` to be a compatibility entry point that installs the project with the `dev` extra.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_project_metadata.py -q`

### Task 3: Upgrade CI to use project metadata and explicit Spark coverage

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1: Write the failing test**

Covered by Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_project_metadata.py -q`

**Step 3: Write minimal implementation**

Update CI to:
- install from `.[dev]` as the canonical metadata source
- keep compile/smoke/pytest coverage for the default path
- add a dedicated Spark-enabled job that installs `.[dev,spark]`
- run Spark parity/analyzer tests in that job

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_project_metadata.py -q`

### Task 4: Tighten repo hygiene and refresh backend docs

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `LARGE_FILE_SUPPORT.md`
- Modify: `TEST_RESULTS.md`

**Step 1: Write the failing test**

Covered by Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_project_metadata.py -q`

**Step 3: Write minimal implementation**

Update:
- `.gitignore` to exclude generated `output/` artifacts
- `README.md` with modern install instructions and a backend capability section
- `LARGE_FILE_SUPPORT.md` with explicit pandas/Spark parity and limitations
- `TEST_RESULTS.md` so it no longer reports stale test counts and reflects current verification expectations

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_project_metadata.py -q`

### Task 5: Verify end-to-end

**Files:**
- Verify only

**Step 1: Run focused tests**

Run: `pytest tests/test_project_metadata.py -q`
Expected: PASS

**Step 2: Run full suite**

Run: `pytest -q`
Expected: PASS with current skip behavior unless Spark is installed locally

**Step 3: Run compile check**

Run: `python -m compileall -q src tests app.py`
Expected: clean exit
