# A/B Testing Analysis Agent

Conversational A/B test analysis with a pandas-first statistical stack and an optional PySpark backend for larger files. The default backend model is `gpt-5.2`.

## Features

- Conversational workflow via LangChain, LangGraph, and Chainlit
- Automatic CSV loading with pandas-by-default and Spark auto-selection for large files
- Automatic column and treatment/control label inference
- Frequentist, Bayesian, and experiment-design helpers built around `statsmodels`, `scipy`, and `numpy`
- Segment-level analysis, summary generation, and Plotly charts
- Canonical result schema shared across pandas and Spark analysis paths
- Smoke-tested core path plus dedicated CI coverage for Spark-specific tests

## Installation

1. Clone the repository and enter it:

   ```bash
   git clone <repository-url>
   cd statistical_test_agent
   ```

2. Create the virtual environment:

   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. Install the default development environment:

   ```bash
   uv sync --extra dev
   ```

4. Install Spark support when you need the large-file backend:

   ```bash
   uv sync --extra dev --extra spark
   ```

5. Add your API key to `.env`:

   ```dotenv
   OPENAI_API_KEY=your-api-key-here
   ```

`requirements.txt` remains as a compatibility shim for tooling that still expects it, but `pyproject.toml` is now the canonical source of dependency metadata.

## Usage

Start the Chainlit app:

```bash
python app.py
```

Generate sample data:

```bash
python scripts/generate_sample_data.py
```

Run the local test suite:

```bash
pytest -q
```

## Backend Capability Matrix

| Capability | pandas | PySpark | Notes |
| --- | --- | --- | --- |
| CSV loading and auto backend selection | Yes | Yes | Spark is selected for large files when available, with automatic pandas fallback on Spark init/load failure. |
| Auto column detection and label inference | Yes | Yes | Both backends use the shared label-inference rules introduced in P1. |
| Core A/B analysis (`run_ab_test`, segmented analysis, summaries) | Yes | Yes | This is the parity-tested path and the main reason the Spark backend exists. |
| Data summary and segment distribution | Yes | Yes | Spark implements native summary/distribution helpers. |
| Interactive charts | Yes | Yes | Charts render from canonical result objects, not directly from the dataframe backend. |
| `query_data` with pandas query syntax | Yes | Unsupported | Spark explicitly rejects pandas-query semantics. |
| `get_column_values` / `calculate_statistics` tool helpers | Yes | Unsupported | Those helpers currently require a pandas dataframe and are not exposed for Spark dataframes. |
| Large-file production workflow | Best for small/medium files | Best-effort for large files | Spark depends on a working Java/Spark runtime and can still fall back to pandas. |

## Architecture

```text
app.py
src/
  agent.py                  LangGraph/LLM orchestration and backend switching
  agent_tools.py            Tool contract exposed to the conversational agent
  agent_reporting.py        User-facing reports and structured error rendering
  statistics/
    analyzer.py             High-level analysis facade
    data_manager.py         pandas data loading and schema inference
    pyspark_analyzer.py     Spark-specific large-file backend
    statsmodels_engine.py   Facade over modular inference helpers
    diagnostics.py          Assumption checks and guardrails
    power_analysis.py       Power and sample-size helpers
    bayesian.py             Bayesian routines
    summary_builder.py      Typed summary generation
    visualizer.py           Plotly chart orchestration
    chart_builders.py       Reusable chart composition helpers
tests/
  test_analyzer_comprehensive.py
  test_agent.py
  test_parity_pandas_spark.py
  test_pyspark_analyzer.py
  test_visualizations.py
```

## Data Expectations

Input CSVs should include:

| Column Type | Required | Description |
| --- | --- | --- |
| Group | Yes | Treatment vs control indicator |
| Effect value | Yes | Outcome metric used for inference |
| Customer ID | No | Entity identifier |
| Segment | No | Segment or cohort column |
| Duration | No | Exposure or experiment duration |

Example:

```csv
customer_id,experiment_group,customer_segment,effect_value,experiment_duration_days
CUST_001,treatment,Premium,58.50,14
CUST_002,control,Standard,28.30,21
CUST_003,treatment,Basic,12.80,7
```

## CI

GitHub Actions runs two paths:

- Core job: installs `".[dev]"`, performs `compileall`, runs a smoke analysis, and executes `pytest -q -ra`
- Spark job: installs `".[dev,spark]"`, configures Java, and runs `tests/test_pyspark_analyzer.py` plus `tests/test_parity_pandas_spark.py`

## Related Docs

- [LARGE_FILE_SUPPORT.md](LARGE_FILE_SUPPORT.md)
- [TEST_RESULTS.md](TEST_RESULTS.md)

## Support

If you enjoy this, buy me some tokens: [buymeacoffee.com/amircs](https://buymeacoffee.com/amircs)
