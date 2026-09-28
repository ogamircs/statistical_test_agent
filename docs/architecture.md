# Architecture

## Overview

The A/B Testing Analysis Agent is a FastAPI server (`app.py`, `src/api/`) with a React + TypeScript UI (`frontend/`). Uploaded experiment data flows through an in-memory pandas + statsmodels analysis stack driven by a LangGraph ReAct agent.

```text
Browser (React, frontend/)
   │  REST + Server-Sent Events (/api/*)
   ▼
FastAPI (src/api/app.py) ── SessionRegistry: one ABTestingAgent per chat session
   │
   ▼
ABTestingAgent (src/agent.py) ── LangGraph ReAct loop over tools in src/tooling/
   │
   ▼
pandas analyzer (src/statistics/) ── typed results → markdown reports + Plotly figures
```

## Request Flow

1. The browser creates a session (`POST /api/sessions`) and optionally uploads a CSV (`POST /api/sessions/{id}/upload`). Uploads are size/extension-checked, stored under `.uploads/<session>/<file_id>.csv` (an allowed data root in `src/data_paths.py`), and answered with a data preview.
2. `POST /api/sessions/{id}/chat` starts `ABTestingAgent.run` in a worker thread and streams progress back as SSE (contract below). A per-session lock rejects concurrent runs with `409 SESSION_BUSY`.
3. `src/agent.py` runs the LangGraph agent. Tools in `src/tooling/` load data through `src/agent_runtime.py` (path confinement + pandas) and run the analyzer.
4. `src/agent_reporting.py` renders the markdown report. `src/statistics/visualizer.py` + `chart_catalog.py` build Plotly figures, which the API serializes to Plotly JSON.
5. Each session persists to `output/query_store/session-<id>.sqlite`: chat history, the raw uploaded data (`raw_data`), and a hidden `_session_state` table with the column mapping, group labels and latest charts. `GET /api/sessions` lists past conversations. After a restart, opening a session returns its chat and charts immediately, and the first analysis or chart request lazily reloads the data, reapplies the mapping and labels, and re-runs the analysis.

## HTTP API

| Method & path | Purpose |
| --- | --- |
| `GET /api/health` | Liveness (also the Docker `HEALTHCHECK`). |
| `GET /api/config` | `auth_required`, `max_upload_mb`, chart-type options, model name. |
| `POST /api/login` | Exchange `STATAGENT_AUTH_*` credentials for a bearer token (only when auth is on). |
| `GET /api/sessions` | Past sessions with a title (first user message) and last-updated time. |
| `POST /api/sessions` / `DELETE /api/sessions/{id}` | Create / delete a session (store + uploads). |
| `GET /api/sessions/{id}/messages` | Persisted chat plus the latest charts of this server process. |
| `DELETE /api/sessions/{id}/messages` | Clear the conversation (memory and persisted history). |
| `POST /api/sessions/{id}/upload` | Multipart CSV upload → `{file_id, filename, size_bytes, preview}`. |
| `POST /api/sessions/{id}/chat` | `{message, file_id?}` → SSE stream. |
| `GET /api/sessions/{id}/charts?type=` | Build charts from the last analysis without an LLM call (`dashboard`, `all`, `bayesian`, or any `chart_catalog` key/alias). `409 NO_ANALYSIS` before an analysis exists. |

Errors are JSON: `{"error": {"code": "...", "message": "..."}}`. When `STATAGENT_AUTH_USERNAME`/`PASSWORD` are set, every `/api` route except health, config and login requires `Authorization: Bearer <token>`.

## SSE Contract (`POST /api/sessions/{id}/chat`)

Events arrive in this order; each `data` line is one JSON object:

| Event | Data |
| --- | --- |
| `status` | `{"state": "started"}` |
| `tool_start` | `{"id", "name", "label"}`, zero or more, from a LangChain callback on the agent run |
| `tool_end` | `{"id", "name", "ok"}` (`ok` is false when the tool reported an `[error_code=…]`) |
| `token` | `{"text": string}`, zero or more, interleaved with tool events: the model's visible text as it is generated (never tool-call arguments or tool output). Text before a tool call is a preamble; the UI clears its live buffer on each `tool_start`. |
| `message` | `{"content": markdown, "error_code": string or null}`, the complete final answer; clients replace any streamed text with it |
| `charts` | `{"charts": [{"name", "title", "figure"}]}` (`figure` is Plotly JSON; empty when the turn made no charts) |
| `done` | `{}` |

If the run crashes, an `error` event (`{"code", "message"}`) replaces `message`/`charts`, followed by `done`. The run is started before streaming begins, so a client disconnect never leaves the session locked, and the reply still lands in the persisted history.

## Analysis Layout

- `src/statistics/analyzer.py`: the pandas-backed analysis facade.
- `src/statistics/models.py`: canonical typed result and summary models consumed by reports and charts.
- `src/statistics/chart_catalog.py` and `src/statistics/visualizer.py`: chart selection and figure orchestration. Charts render from canonical result objects, not the dataframe.

## UI Layer (`frontend/`)

- `src/App.tsx` owns session/chat state; `src/lib/api.ts` is the typed API client, including the `fetch` + `ReadableStream` SSE reader (`src/lib/sse.ts`).
- Components: history sidebar, chat with GFM markdown tables and live progress steps, composer with drag-and-drop CSV upload and data preview, and a chart workspace (tabs/grid, chart-type picker, fullscreen, PNG/SVG export).
- Plotly (`plotly.js-dist-min`) is lazy-loaded on the first chart and re-themed client-side for dark mode with a colorblind-safe palette.
- The browser layer stays presentation-only; analysis logic lives in Python.
