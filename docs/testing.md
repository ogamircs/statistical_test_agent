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

## Spark-Focused Coverage

The highest-signal Spark suites are:

```bash
./.venv/bin/pytest -q tests/test_pyspark_analyzer.py tests/test_parity_pandas_spark.py -ra
```

Related backend-selection coverage also lives in:

- `tests/test_agent.py`
- `tests/test_agent_runtime.py`
- `tests/test_analyzer_protocol.py`

## Local Spark Prerequisites

To exercise the real Spark path locally, this machine needs:

- the `spark` dependency extra installed
- a working Java runtime
- a large enough CSV to cross the backend-selection threshold

The Spark session bootstrap pins the worker and driver interpreter to the active Python executable, which helps avoid macOS system-Python mismatches during local runs.

If those prerequisites are missing, the dedicated Spark suites skip or the app falls back to pandas.

## Manual Smoke Testing

Recommended browser smoke checks:

1. Start the app locally with Chainlit.
2. Upload `data/sample_ab_data.csv` and verify the standard analysis flow.
3. Upload `data/sample_ab_data_large.csv` and confirm the response reports the PySpark backend when Spark is available.
4. Request `dashboard` or `full dashboard` and verify the figures render cleanly in the chat UI.
