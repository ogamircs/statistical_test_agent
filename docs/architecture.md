# Architecture

## Overview

The A/B Testing Analysis Agent is a Chainlit app that routes uploaded experiment data through a pandas-first analysis stack and switches to PySpark for large CSV files when the Spark runtime is available.

## Request Flow

1. `app.py` receives chat input or file uploads from Chainlit.
2. `src/agent.py` coordinates the conversational workflow and delegates backend loading to `src/agent_runtime.py`.
3. `src/agent_runtime.py` chooses pandas or PySpark based on file size and runtime availability.
4. The active analyzer runs statistical analysis and returns canonical typed results.
5. `src/agent_reporting.py` formats summaries for chat responses.
6. `src/statistics/visualizer.py` builds Plotly figures for dashboard and chart requests.

## Backend Layout

- `src/statistics/analyzer.py`: default pandas-backed analysis path for interactive work.
- `src/statistics/pyspark_analyzer.py`: distributed backend for larger files and Spark parity coverage.
- `src/statistics/models.py`: canonical result and summary models shared across both backends.
- `src/statistics/chart_catalog.py` and `src/statistics/visualizer.py`: chart selection and figure orchestration.

## Spark Behavior

- Files at or below the runtime threshold stay on pandas.
- Files above the threshold request PySpark through the runtime helper.
- Spark session creation pins both the driver and worker interpreter to the current Python executable so local runs do not silently fall back to a different system Python.
- If Spark initialization or file loading fails, the app falls back to pandas and reports that fallback clearly.
- Charts render from canonical result objects, so the visualization layer stays backend-agnostic once analysis has completed.

## UI Layer

- `public/custom.js` adds conversation-history behavior, clear-history suppression, and processing-indicator enhancements.
- `public/custom.css` owns the custom chat layout, composer positioning, and chart container sizing.
- The browser layer is intentionally thin and keeps analysis logic in Python modules.
