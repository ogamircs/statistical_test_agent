# Conversation History Sidebar Design

**Date:** 2026-03-06

## Goal

Make the left sidebar behave more like ChatGPT by showing one row per conversation instead of one row per message.

## Problem

The current custom sidebar saves and renders every prompt and reply. That makes the panel noisy and unlike the expected chat-history interaction, where each conversation is represented by a single title.

## Chosen Approach

Store a browser-local list of conversations and derive each title from the first user message in that chat.

- A conversation row is created only after the first user message appears.
- Additional messages in the same chat do not create more rows.
- When the UI returns to a fresh empty state, the next first user message starts a new conversation entry.

## Why This Approach

- It produces the behavior the user asked for without requiring backend thread storage.
- It keeps the change inside the current `public/custom.js` and `public/custom.css` layer.
- It avoids coupling the sidebar to undocumented Chainlit internals.

## Expected Behavior

- The sidebar shows one compact row per conversation.
- Each row title is the first user message.
- Rows are appended only when a new conversation starts.
- The current conversation is visually highlighted.
- The empty state copy refers to conversations rather than prompts and replies.

## Limitation

Because this is browser-local UI state, older rows are a visual chat list rather than fully restorable backend threads. The active conversation can still be highlighted accurately.
