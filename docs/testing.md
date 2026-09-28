# Testing

## Core Regression Suite

Run the full project test suite:

```bash
./.venv/bin/pytest -q
```

For metadata and custom UI regressions:

```bash
./.venv/bin/pytest tests/test_project_metadata.py -q
```

## Manual Smoke Testing

Recommended browser smoke checks:

1. Start the app locally with Chainlit.
2. Upload `data/sample_ab_data.csv` and verify the standard analysis flow.
3. Request `dashboard` or `full dashboard` and verify the figures render cleanly in the chat UI.
