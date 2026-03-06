# Centered Composer Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep the chat composer centered after the conversation begins while preserving the current custom Chainlit sidebar and starter-chip behavior.

**Architecture:** Extend the existing custom UI layer instead of replacing Chainlit layout primitives. JavaScript will compute a centered-conversation state from the current DOM, and CSS will style the active conversation shell when that state is enabled.

**Tech Stack:** Chainlit custom assets, vanilla JavaScript, CSS, pytest, Playwright browser smoke testing

---

### Task 1: Document the approved UI change

**Files:**
- Create: `docs/plans/2026-03-06-centered-composer-layout-design.md`
- Create: `docs/plans/2026-03-06-centered-composer-layout-implementation.md`

**Step 1: Save the approved design**

Write the centered-composer goal, constraints, and chosen approach into the design doc.

**Step 2: Save the implementation handoff**

Write the JS/CSS-focused execution steps into the implementation plan.

**Step 3: Commit**

Skip unless the user asks for commits.

### Task 2: Add a persistent centered-conversation state

**Files:**
- Modify: `public/custom.js`

**Step 1: Write the failing test**

Use a lightweight asset regression test that looks for the new centered-conversation hooks.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_project_metadata.py -q`

Expected: FAIL because the new layout hooks are not present yet.

**Step 3: Write minimal implementation**

Add helpers that:
- detect whether messages exist
- locate the active conversation shell, thread scroller, and composer section
- compute whether the current conversation should remain centered
- apply/remove stable classes on the DOM

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_project_metadata.py -q`

Expected: PASS

### Task 3: Style the centered active conversation

**Files:**
- Modify: `public/custom.css`

**Step 1: Write the failing test**

Reuse the asset regression test so CSS hooks are required.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_project_metadata.py -q`

Expected: FAIL until the CSS hooks exist.

**Step 3: Write minimal implementation**

Add styles for:
- the centered active conversation root
- the centered message scroller
- the centered composer shell
- suggestion visibility that depends on actual message presence instead of only the empty-state class

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_project_metadata.py -q`

Expected: PASS

### Task 4: Verify the behavior in the running app

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

Reload the Chainlit app, start or reuse a conversation, and confirm:
- the composer stays centered after the first response
- the left history panel still renders
- the layout naturally falls back to normal scrolling when the thread becomes tall

**Step 4: Commit**

Skip unless the user asks for commits.
