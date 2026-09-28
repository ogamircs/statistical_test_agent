# A/B Testing Analysis Agent

Conversational A/B test analysis built on a pandas + statsmodels statistical stack. The default LLM is `gpt-5.2`.

## Features

- Conversational workflow via LangChain, LangGraph, and Chainlit
- Automatic CSV loading and schema inference with pandas
- Automatic column and treatment/control label inference
- Frequentist, Bayesian, and experiment-design helpers built around `statsmodels`, `scipy`, and `numpy`
- Segment-level analysis, summary generation, and Plotly charts
- Canonical typed result schema shared by reports and charts
- Smoke-tested core path, blocking ruff/mypy, and a coverage-gated test suite in CI

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

4. Add your API key to `.env`:

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

## Architecture

```text
app.py
src/
  agent.py                  LangGraph/LLM orchestration
  agent_tools.py            Tool contract exposed to the conversational agent
  agent_reporting.py        User-facing reports and structured error rendering
  statistics/
    analyzer.py             High-level analysis facade
    data_manager.py         pandas data loading and schema inference
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

GitHub Actions installs from `uv.lock` (`uv sync --frozen --extra dev`), then runs `compileall`, `ruff check .`, blocking `mypy src app.py`, a smoke analysis on the sample CSV, and `pytest` with a 78% coverage floor.

## Related Docs

- [docs/architecture.md](docs/architecture.md)
- [docs/development.md](docs/development.md)
- [docs/testing.md](docs/testing.md)

## Support

If you enjoy this, buy me some tokens: [buymeacoffee.com/amircs](https://buymeacoffee.com/amircs)
