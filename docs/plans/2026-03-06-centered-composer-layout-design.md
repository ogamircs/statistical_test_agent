# Centered Composer Layout Design

**Date:** 2026-03-06

## Goal

Keep the chat composer visually centered after the conversation starts instead of dropping it to the bottom immediately.

## Problem

The current custom Chainlit UI only applies the centered layout in the empty state. As soon as the first message is sent, the app falls back to the default chat structure where the message list consumes the available height and the composer docks to the bottom of the viewport.

## Chosen Approach

Add a persistent centered-conversation layout state in the existing custom JavaScript and CSS.

- The JavaScript will detect whether the active conversation is short enough to center cleanly.
- The CSS will center the active chat shell when that state is enabled.
- The layout will naturally fall back to the standard top-aligned flow once the thread becomes too tall to center comfortably.

## Why This Approach

- It preserves the current left history panel and starter suggestions without rebuilding the page structure.
- It avoids fighting Chainlit internals with a heavier custom shell.
- It keeps the change local to `public/custom.js` and `public/custom.css`.

## Expected Behavior

- Empty state remains centered.
- After the first message, short and medium conversations stay centered.
- Long conversations stop forcing centering and scroll normally.
- Starter suggestions remain hidden after the first send.

## Verification

- Add a lightweight regression check for the new layout hooks.
- Run the Python test suite.
- Reload the app and confirm in the browser that the composer remains centered after the conversation starts.
