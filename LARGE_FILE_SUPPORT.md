# Large File Support

## Overview

The agent uses a pandas-first workflow and switches to PySpark for large CSV files when Spark is installed and the runtime starts successfully. If Spark initialization or loading fails, the agent falls back to pandas automatically.

- `pandas`: default backend for normal interactive analysis
- `PySpark`: optional backend for large-file, distributed processing

The current threshold is `2 MB`, configured in `src/agent.py`.

## Installation

Default setup:

```bash
uv sync --extra dev
```

Enable Spark support:

```bash
uv sync --extra dev --extra spark
```

## Backend Capability Matrix

| Capability | pandas | PySpark | Notes |
| --- | --- | --- | --- |
| Automatic backend selection from file size | Yes | Yes | Spark is selected only when installed and the file crosses the threshold. |
| Core A/B analysis and segmented summaries | Yes | Yes | Golden-dataset parity tests cover this path. |
| Summary and distribution helpers | Yes | Yes | Spark implements native `get_data_summary()` and `get_segment_distribution()`. |
| Plotly charts | Yes | Yes | Charts consume canonical result payloads rather than raw backend-specific frames. |
| pandas-style `query_data()` exploration | Yes | Unsupported | The Spark backend explicitly rejects pandas-query syntax. |
| pandas-only helper tools (`get_column_values`, `calculate_statistics`) | Yes | Unsupported | Those tools require direct pandas series/dataframe access. |
| Large-file execution reliability | Limited by local memory | Best-effort | Spark depends on Java, local Spark availability, and environment compatibility. |

## What "Best-Effort" Means

Spark support is intentionally best-effort rather than guaranteed:

- The agent will try Spark first for large files.
- If Spark cannot start or cannot load the file, the agent falls back to pandas.
- CI now includes a Spark-specific job so regressions are caught earlier, but local runtime issues can still make Spark unavailable.

## Running Spark Tests Locally

```bash
pytest -q tests/test_pyspark_analyzer.py tests/test_parity_pandas_spark.py -ra
```

If Spark is not installed, those suites skip. If Spark is installed but Java or the local runtime is misconfigured, the tests may also skip or fail at session startup.

## Platform Notes

- `pandas`: expected to work on macOS, Linux, and Windows
- `PySpark`: works best on macOS and Linux
- Windows Spark usage should be treated as best-effort because local Java/Spark compatibility varies more widely

## Practical Guidance

- Stay on pandas for smaller files and interactive data inspection.
- Install Spark when you need larger-file execution or parity validation against the distributed path.
- Do not assume every pandas convenience helper is available in Spark mode; the main parity target is the statistical analysis path, not the full dataframe exploration surface.
