# Conversation History Sidebar Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the step-by-step left sidebar with a ChatGPT-style conversation list that shows a single title per chat.

**Architecture:** Move the sidebar persistence model from message excerpts to conversation metadata. JavaScript will create and update a browser-local conversation list based on the first user message in the active chat, and CSS will restyle the sidebar rows as compact conversation titles with an active state.

**Tech Stack:** Chainlit custom assets, vanilla JavaScript, CSS, pytest, Playwright browser verification

---

### Task 1: Document the approved sidebar behavior

**Files:**
- Create: `docs/plans/2026-03-06-conversation-history-design.md`
- Create: `docs/plans/2026-03-06-conversation-history-implementation.md`

**Step 1: Save the design**

Describe the browser-local conversation list approach and the first-message title rule.

**Step 2: Save the implementation handoff**

Write the JavaScript/CSS execution steps for the sidebar rewrite.

### Task 2: Replace message history storage with conversation storage

**Files:**
- Modify: `public/custom.js`

**Step 1: Write the failing test**

Extend the lightweight asset regression test so it requires the new conversation-list hooks and storage keys.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_project_metadata.py -q`

Expected: FAIL until the new hooks exist.

**Step 3: Write minimal implementation**

Add logic that:
- reads the first user message from the current chat
- creates an active conversation id only when a conversation begins
- stores one entry per conversation in local/session storage
- highlights the active conversation entry
- stops rendering per-message rows

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_project_metadata.py -q`

Expected: PASS

### Task 3: Restyle the sidebar rows as conversation titles

**Files:**
- Modify: `public/custom.css`

**Step 1: Write the failing test**

Reuse the asset regression test for the new class hooks.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_project_metadata.py -q`

Expected: FAIL until the new styles exist.

**Step 3: Write minimal implementation**

Add compact title-row styling, active-state styling, and updated empty-state copy.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_project_metadata.py -q`

Expected: PASS

### Task 4: Verify the new sidebar behavior

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

Confirm:
- one row appears for the current conversation
- replies do not add extra rows
- a fresh conversation adds a new row with the next first user message
- the current conversation is highlighted
