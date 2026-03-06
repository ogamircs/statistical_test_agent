# P2 Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the analysis, reporting, visualization, and logging layers to reduce complexity without changing user-visible statistical behavior.

**Architecture:** Split the current monoliths into smaller private stages and helper modules while preserving the existing public API. Introduce typed summary/report models that still support legacy dictionary-style access during migration, then move renderers/tests onto the typed interface. Keep the critical path on the pandas analyzer stable and use regression tests to lock behavior before extracting logic.

**Tech Stack:** Python 3.9, pandas, statsmodels, SciPy, Plotly, pytest, dataclasses

---

### Task 1: Add Typed Summary Report Models

**Files:**
- Modify: `src/statistics/models.py`
- Modify: `src/statistics/summary_builder.py`
- Modify: `src/agent_reporting.py`
- Test: `tests/test_analyzer_comprehensive.py`

**Step 1: Write the failing test**

Add tests asserting that `generate_summary()` returns a typed object with attribute access for top-level fields and detail rows.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_analyzer_comprehensive.py -k 'typed_summary' -q`
Expected: FAIL because summary is still a raw dict.

**Step 3: Write minimal implementation**

Add dataclasses for summary reports and detail rows, and update `ABTestSummaryBuilder.generate_summary()` to return the typed model with a compatibility method for legacy dict-style consumers.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_analyzer_comprehensive.py -k 'typed_summary' -q`
Expected: PASS

### Task 2: Decompose `ABTestAnalyzer.run_ab_test`

**Files:**
- Modify: `src/statistics/analyzer.py`
- Test: `tests/test_analyzer_comprehensive.py`

**Step 1: Write the failing test**

Add tests that exercise one extracted stage directly or through a smaller helper boundary, such as filtering/pre-period alignment or diagnostics assembly.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_analyzer_comprehensive.py -k 'analysis_stage' -q`
Expected: FAIL because helper stages do not exist yet.

**Step 3: Write minimal implementation**

Extract private stage helpers for scope preparation, pre-period handling, covariate preparation, model/diagnostic computation, and result assembly. Keep `run_ab_test()` as the orchestration shell.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_analyzer_comprehensive.py -k 'analysis_stage or run_ab_test' -q`
Expected: PASS

### Task 3: Split `StatsmodelsABTestEngine`

**Files:**
- Create: `src/statistics/model_families.py`
- Create: `src/statistics/diagnostics.py`
- Modify: `src/statistics/statsmodels_engine.py`
- Modify: `src/statistics/__init__.py`
- Test: `tests/test_analyzer_comprehensive.py`

**Step 1: Write the failing test**

Add tests that pin one model-family path and one diagnostics path so extracted helpers must still produce the same metadata and guardrails.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_analyzer_comprehensive.py -k 'model_family or diagnostics_module' -q`
Expected: FAIL because the extracted modules are not wired in yet.

**Step 3: Write minimal implementation**

Move metric/model-family inference and fitting helpers into `model_families.py`, move SRM/assumption/outlier diagnostics into `diagnostics.py`, and keep `StatsmodelsABTestEngine` as a thin composition wrapper.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_analyzer_comprehensive.py -k 'model_family or diagnostics_module' -q`
Expected: PASS

### Task 4: Refactor Plot Builders

**Files:**
- Create: `src/statistics/chart_builders.py`
- Modify: `src/statistics/visualizer.py`
- Test: `tests/test_visualizations.py`

**Step 1: Write the failing test**

Add stronger chart assertions for titles, axis labels, and representative trace names on the statistical summary and dashboard charts.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_visualizations.py -q`
Expected: FAIL because the current tests do not enforce those details.

**Step 3: Write minimal implementation**

Extract reusable layout/trace helpers into `chart_builders.py` and rewire the visualizer to use them while preserving current chart semantics.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_visualizations.py -q`
Expected: PASS

### Task 5: Replace Runtime Prints with Structured Logging

**Files:**
- Modify: `app.py`
- Modify: `src/statistics/pyspark_analyzer.py`
- Test: `tests/test_agent.py`

**Step 1: Write the failing test**

Add tests that capture logging for settings updates or runtime events and assert that no runtime path depends on `print`.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_agent.py -k 'logging' -q`
Expected: FAIL because runtime print calls still exist.

**Step 3: Write minimal implementation**

Replace runtime `print` usage with contextual logger calls. Leave CLI/demo-only script output alone if it is outside normal app/runtime flow.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_agent.py -k 'logging' -q`
Expected: PASS

### Task 6: Full Verification

**Files:**
- Verify: `src/`
- Verify: `tests/`

**Step 1: Run focused suites**

Run:
- `.venv/bin/python -m pytest tests/test_analyzer_comprehensive.py -q`
- `.venv/bin/python -m pytest tests/test_visualizations.py -q`
- `.venv/bin/python -m pytest tests/test_agent.py -q`

Expected: PASS

**Step 2: Run full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS

**Step 3: Compile check**

Run: `.venv/bin/python -m compileall -q src tests app.py`
Expected: exit 0
