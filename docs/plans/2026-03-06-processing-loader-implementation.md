# Processing Loader Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace UI `Processing...` placeholders with a small loading GIF and a CSS fallback spinner.

**Architecture:** Leave the Python placeholder messages intact and enhance them in the existing custom Chainlit browser layer. JavaScript will detect processing articles and replace their contents with a loader component, while CSS will size the GIF and provide a fallback spinner.

**Tech Stack:** Chainlit custom assets, vanilla JavaScript, CSS, pytest, Playwright browser verification

---

### Task 1: Document the approved loader behavior

**Files:**
- Create: `docs/plans/2026-03-06-processing-loader-design.md`
- Create: `docs/plans/2026-03-06-processing-loader-implementation.md`

**Step 1: Save the design**

Document the DOM-enhancement approach and the chosen remote GIF source.

**Step 2: Save the implementation handoff**

Write the JS/CSS execution steps for the loader enhancement.

### Task 2: Add processing-placeholder enhancement logic

**Files:**
- Modify: `public/custom.js`

**Step 1: Write the failing test**

Extend the asset regression test to require the loading GIF URL and processing-indicator hooks.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_project_metadata.py -q`

Expected: FAIL until the new hooks exist.

**Step 3: Write minimal implementation**

Add logic that:
- detects article nodes whose visible text is `Processing...`
- swaps the text for a small loader element
- listens for image errors and enables a fallback spinner state
- avoids repeatedly re-enhancing the same node

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_project_metadata.py -q`

Expected: PASS

### Task 3: Style the loader and fallback state

**Files:**
- Modify: `public/custom.css`

**Step 1: Write the failing test**

Reuse the asset regression test for the new loader classes.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_project_metadata.py -q`

Expected: FAIL until the styles exist.

**Step 3: Write minimal implementation**

Add styles for:
- the inline processing indicator wrapper
- the GIF sizing
- a fallback spinner
- a hidden label for accessibility

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_project_metadata.py -q`

Expected: PASS

### Task 4: Verify in the running app

**Files:**
- Test: `tests/test_project_metadata.py`
- Inspect: `public/custom.js`
- Inspect: `public/custom.css`

**Step 1: Run focused verification**

Run: `pytest tests/test_project_metadata.py -q`

Expected: PASS

**Step 2: Run full verification**

Run: `pytest -q`

Expected: PASS

**Step 3: Browser smoke test**

Trigger a flow that shows the current processing placeholder and confirm it renders as a small animated loading indicator instead of the literal text.
