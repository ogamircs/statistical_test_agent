# Testing

## Core Regression Suite

Run the full Python test suite (the same gates CI runs):

```bash
uv run ruff check .
uv run mypy src app.py
uv run pytest -q -ra --cov=src --cov-fail-under=78
```

API and SSE behavior is covered by `tests/test_api.py` with a stub agent (no network). For metadata, CI and packaging regressions:

```bash
uv run pytest tests/test_project_metadata.py -q
```

## Frontend

```bash
cd frontend
npm run typecheck
npm test
npm run build
```

## Manual Smoke Testing

Recommended browser smoke checks:

1. Start the app (see `docs/development.md`).
2. Upload `data/sample_ab_data.csv`, send "best guess analysis", and check that the progress steps, the markdown tables and the data preview render.
3. Open the chart workspace, switch chart types (dashboard, p-values, Bayesian), try fullscreen and PNG export, and toggle dark mode.
4. Reload the page and resume the conversation from the history sidebar.
