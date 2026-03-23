# Code Simplification Design

**Date:** 2026-03-08

## Goal

Reduce maintenance complexity in the agent/runtime/tooling path without changing the user-facing analysis workflow.

## Problem

The current codebase works, but the complexity is concentrated in a few modules:

- `src/agent.py` mixes orchestration, backend selection, persistence, query answering, and chat state.
- `src/agent_tools.py` contains the full tool registry plus all tool handlers in one file.
- `src/statistics/visualizer.py` and `src/agent_tools.py` both carry chart-selection logic.
- `src/statistics/models.py` still carries a broad legacy-compatibility surface during the typed-model migration.

This makes changes slower, increases regression risk, and keeps file-level cognitive load high.

## Design

### 1. Move session and persistence logic out of `agent.py`

Create a dedicated session service that owns:

- chat history
- last results / summary / charts
- raw-data persistence
- analysis-output persistence
- SQL question answering

`ABTestingAgent` should become a thin orchestrator that wires the LLM, active analyzer selection, tool registry, and session service together.

### 2. Split the tool layer by responsibility

Keep `src/agent_tools.py` as a facade only. Move handlers into smaller modules grouped by concern:

- loading/configuration
- analysis/querying
- visualization
- shared tool-context helpers

This keeps the public import stable while reducing the size of the hot file.

### 3. Centralize chart selection

Define one chart-selection registry that maps:

- chart keys
- user aliases
- chart builder callables

Use that registry in the tool layer and the visualizer entrypoint so chart wiring is not duplicated.

### 4. Record the next simplification wave explicitly

Not everything should be forced into one refactor. Keep a backlog for the heavier follow-up work:

- reduce the remaining legacy mapping compatibility in `models.py`
- narrow the Spark contract to the paths that justify it
- trim the visualization surface to a smaller stable core
- add file-size guardrails once the large-module refactor lands

## Non-Goals

- Removing Spark support
- Changing the user-visible statistical workflow
- Rewriting the stats engine again
- Removing existing charts from the UI in this wave

## Validation

The simplification refactor is acceptable if:

- existing agent tests still pass
- visualization tests still pass
- no tool names or core outputs regress
- `agent.py` and `agent_tools.py` become materially smaller and narrower in responsibility
