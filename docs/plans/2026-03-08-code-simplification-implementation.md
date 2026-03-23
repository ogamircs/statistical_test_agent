# Code Simplification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify the agent/runtime/tooling layer by extracting session state, modularizing the tool registry, and centralizing chart selection while keeping behavior stable.

**Architecture:** Introduce a dedicated session service for persistence, data-question answering, chart/result state, and conversation history. Convert the tool layer into a thin facade over smaller tool-group modules, and drive chart selection through one shared registry.

**Tech Stack:** Python, LangChain, LangGraph, Plotly, pandas, SQLite, pytest

---

### Task 1: Capture the target behavior with tests

**Files:**
- Modify: `tests/test_agent.py`
- Create: `tests/test_agent_session.py`

**Step 1: Write the failing test**

Add tests that verify:

- the agent session object can store and clear chart/result/summary state
- the session service persists raw data and analysis outputs through its store abstraction
- the public tool registry still exposes the expected tool names after modularization

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_agent.py tests/test_agent_session.py -q`

Expected: FAIL because the session abstraction does not exist yet.

**Step 3: Write minimal implementation**

Create the session abstraction and wire the agent to it.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_agent.py tests/test_agent_session.py -q`

Expected: PASS

### Task 2: Extract agent session handling

**Files:**
- Create: `src/agent_session.py`
- Modify: `src/agent.py`

**Step 1: Write the failing test**

Add tests that use the new session object for:

- message-history tracking
- chart/result state
- raw-data persistence
- analysis-output persistence
- SQL data-question answering

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_agent_session.py -q`

Expected: FAIL because the module and service are missing.

**Step 3: Write minimal implementation**

Implement `AgentSessionState` and `AgentAnalysisSession`, then make `ABTestingAgent` delegate to them while keeping the current public surface stable.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_agent_session.py -q`

Expected: PASS

### Task 3: Split the tool layer into smaller modules

**Files:**
- Modify: `src/agent_tools.py`
- Create: `src/tooling/common.py`
- Create: `src/tooling/loading.py`
- Create: `src/tooling/analysis.py`
- Create: `src/tooling/visualization.py`
- Create: `src/tooling/__init__.py`

**Step 1: Write the failing test**

Add a test that asserts the public `create_agent_tools()` facade still returns the same core tool names and that chart generation still uses the stored backend-aware state.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_agent.py::test_agent_has_expected_tools -q`

Expected: FAIL once the facade is reduced before the modular imports are wired.

**Step 3: Write minimal implementation**

Move tool handlers into grouped modules and keep `src/agent_tools.py` as a composition layer only.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_agent.py -q`

Expected: PASS

### Task 4: Centralize chart selection

**Files:**
- Create: `src/statistics/chart_catalog.py`
- Modify: `src/statistics/visualizer.py`
- Modify: `src/tooling/visualization.py`
- Modify: `tests/test_visualizations.py`

**Step 1: Write the failing test**

Add tests for a shared chart catalog that verifies:

- core chart aliases resolve consistently
- the “all” selection produces the expected core set

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_visualizations.py -q`

Expected: FAIL because the shared chart-selection layer does not exist yet.

**Step 3: Write minimal implementation**

Introduce a chart catalog used by both the visualizer entrypoint and the chart-generation tool.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_visualizations.py -q`

Expected: PASS

### Task 5: Update backlog and run full verification

**Files:**
- Modify: `TODO.md`

**Step 1: Record current simplification status**

Add a dedicated simplification section that marks the implemented wave complete and leaves the heavier follow-up items as explicit next steps.

**Step 2: Run verification**

Run:

- `pytest -q`
- `python3 -m compileall -q src tests app.py`

Expected: all tests pass and compileall exits cleanly.
