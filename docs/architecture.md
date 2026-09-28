# Architecture

## Overview

The A/B Testing Analysis Agent is a Chainlit app that routes uploaded experiment data through an in-memory pandas + statsmodels analysis stack.

## Request Flow

1. `app.py` receives chat input or file uploads from Chainlit.
2. `src/agent.py` coordinates the conversational workflow and delegates CSV loading to `src/agent_runtime.py`.
3. `src/agent_runtime.py` confines the path to the allowed data roots and loads it into the pandas analyzer.
4. The analyzer runs statistical analysis and returns canonical typed results.
5. `src/agent_reporting.py` formats summaries for chat responses.
6. `src/statistics/visualizer.py` builds Plotly figures for dashboard and chart requests.

## Analysis Layout

- `src/statistics/analyzer.py`: the pandas-backed analysis facade.
- `src/statistics/models.py`: canonical typed result and summary models consumed by reports and charts.
- `src/statistics/chart_catalog.py` and `src/statistics/visualizer.py`: chart selection and figure orchestration. Charts render from canonical result objects, not the dataframe.

## UI Layer

- `public/custom.js` adds conversation-history behavior, clear-history suppression, and processing-indicator enhancements.
- `public/custom.css` owns the custom chat layout, composer positioning, and chart container sizing.
- The browser layer is intentionally thin and keeps analysis logic in Python modules.
