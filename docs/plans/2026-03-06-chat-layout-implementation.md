# Chat Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Customize the Chainlit UI so the empty-state composer sits in the middle, starter suggestions appear under it, and a persistent left-side conversation history panel is visible on desktop.

**Architecture:** Keep the existing Chainlit frontend and layer custom behavior through `custom_css` and `custom_js` served from `public/`. Use a small DOM enhancer to insert the sidebar, manage starter chips, and toggle empty-state classes from the rendered page structure.

**Tech Stack:** Chainlit config, custom CSS, custom JavaScript, Playwright verification, pytest metadata checks

---

### Task 1: Wire Chainlit to Custom Assets

**Files:**
- Modify: `.chainlit/config.toml`
- Test: `tests/test_project_metadata.py`

**Step 1: Write the failing test**

Assert that the Chainlit config references `/public/custom.css` and `/public/custom.js`, and that both asset files exist.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_project_metadata.py -k custom_ui_assets -q`

**Step 3: Write minimal implementation**

Add `custom_css` and `custom_js` to the UI config and create the matching files in `public/`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_project_metadata.py -k custom_ui_assets -q`

### Task 2: Add Empty-State Layout and Sidebar Styling

**Files:**
- Create: `public/custom.css`

**Step 1: Implement CSS**

Add desktop grid layout, persistent left sidebar styling, centered welcome-screen overrides, and starter-chip styling.

**Step 2: Verify in browser**

Run the app and confirm the empty-state composer is centered and the sidebar is visible on desktop.

### Task 3: Add DOM Enhancer for Suggestions and Conversation History

**Files:**
- Create: `public/custom.js`

**Step 1: Implement JavaScript**

Insert a left history panel into the existing Chainlit shell, populate it from current step messages, persist recent items in `sessionStorage`, and inject starter suggestion chips that submit prompts.

**Step 2: Verify in browser**

Open the app, send a message, and confirm the sidebar updates and the composer transitions from centered empty state to normal chat layout.

### Task 4: Final Verification

**Files:**
- Verify: `tests/test_project_metadata.py`
- Verify: app in browser

**Step 1: Run focused metadata test**

Run: `pytest tests/test_project_metadata.py -k custom_ui_assets -q`

**Step 2: Run full test suite**

Run: `pytest -q`

**Step 3: Manual browser verification**

Refresh the live app, confirm the centered empty-state composer, suggestion chips, and left conversation history panel behave correctly on desktop.
