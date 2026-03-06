# Test Results

## Current baseline

Local verification is tracked against the current codebase rather than the original bootstrapping phase.

- Main regression suite: `pytest -q`
- Syntax check: `python -m compileall -q src tests app.py`
- Spark-specific suites: `pytest -q tests/test_pyspark_analyzer.py tests/test_parity_pandas_spark.py -ra`

## Latest local run

As of March 6, 2026:

- `pytest -q` passed with `85 passed, 2 skipped in 3.57s`
- The warning-producing tests were cleaned up, so project test output is now warning-free
- Any remaining `urllib3` / LibreSSL startup warning is environment-specific and not emitted by the project test code

## CI expectations

GitHub Actions now validates two paths:

- Core job using `".[dev]"` for the default pandas-first workflow
- Spark job using `".[dev,spark]"` plus Java for distributed-backend coverage

## Coverage intent

The highest-signal suites currently are:

- `tests/test_analyzer_comprehensive.py` for pandas/statistics behavior
- `tests/test_agent.py` for agent and tool contract coverage
- `tests/test_visualizations.py` for chart semantics
- `tests/test_pyspark_analyzer.py` for Spark-specific behavior
- `tests/test_parity_pandas_spark.py` for golden-dataset parity
