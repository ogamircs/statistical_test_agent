# A/B Testing Analysis Agent

An intelligent conversational agent for analyzing A/B test experiments, powered by LangChain and OpenAI models (default backend model: `gpt-5.2`). Upload experiment data and get comprehensive statistical analysis through natural language interaction.

## Features

- **Conversational Interface**: Chat naturally about your A/B test data
- **Automatic Backend Switching**: Uses pandas by default and automatically switches to PySpark for larger files with safe pandas fallback
- **Automatic Column Detection**: Smart detection of experiment columns based on naming patterns
- **Canonical Result Schema**: Consistent segment result payload across pandas and Spark analysis paths
- **Comprehensive Statistical Analysis (statsmodels-first)**:
  - OLS/GLM/robust model-based treatment effect estimation
  - Effect size calculations (Cohen's d)
  - 95% confidence intervals and Bayesian credible intervals
  - Statistical power analysis and sample size adequacy assessment
  - Multiple-testing correction (Benjamini-Hochberg / FDR)
  - Guardrails for unstable inference and proportion-test edge cases
- **Experiment Diagnostics**:
  - SRM (sample ratio mismatch) checks
  - Assumption diagnostics (normality/variance checks)
  - Outlier sensitivity diagnostics
- **Sequential Decision Support**: Optional interim-look recommendations (continue/stop efficacy/stop futility)
- **Metric/Model Support**:
  - Continuous metrics (OLS HC3)
  - Binary metrics (Binomial GLM)
  - Count metrics (Poisson/Negative Binomial)
  - Heavy-tail robust strategy
  - Optional covariate-adjusted effects
- **Segment-level Analysis**: Analyze results across customer segments
- **Interactive Visualizations**: Plotly-powered charts and dashboards
- **Actionable Recommendations**: Get clear guidance based on your results
- **Reliability & Observability**: Structured user-facing error codes, logging, and safer query behavior
- **CI-Ready Test Suite**: GitHub Actions workflow runs compile checks, smoke analysis, and pytest

## Project Structure

```
ab_testing_agent/
├── app.py                      # Chainlit UI entry point
├── requirements.txt            # Python dependencies
├── chainlit.md                 # Chainlit welcome screen
├── .env                        # Environment variables (API keys)
├── .gitignore                  # Git ignore rules
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── agent.py                # ABTestingAgent orchestration
│   ├── agent_tools.py          # Agent tool handlers
│   ├── agent_reporting.py      # Tool/report rendering and structured errors
│   └── statistics/             # Statistical analysis module
│       ├── __init__.py
│       ├── models.py           # Data structures (ABTestResult)
│       ├── analyzer.py         # Analysis facade/orchestrator
│       ├── data_manager.py     # Data loading and auto-mapping
│       ├── statsmodels_engine.py # Statsmodels-based inference engine
│       ├── summary_builder.py  # Summary and recommendations
│       └── visualizer.py       # Plotly visualization charts
│
├── data/                       # Data files directory
│   ├── sample_ab_data.csv      # Sample experiment data
│   └── sample_ab_data_alt.csv  # Alternative format sample
│
├── scripts/                    # Utility scripts
│   └── generate_sample_data.py # Generate sample test data
│
├── tests/                      # Test files
│   ├── __init__.py
│   ├── test_ab_module.py       # Statistical module tests
│   ├── test_agent.py           # Agent tests
│   ├── test_parity_pandas_spark.py # Golden parity tests (pandas vs Spark)
│   └── test_visualizations.py  # Visualization tests
│
├── .github/workflows/          # CI workflows
│   └── ci.yml                  # Compile/smoke/pytest checks
│
└── .chainlit/                  # Chainlit configuration
    └── config.toml
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Chainlit UI (app.py)                     │
│                    Web-based chat interface                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ABTestingAgent (src/agent.py)                 │
│              LangChain Agent with LangGraph ReAct               │
│  - backend selection + Spark fallback                           │
│  - modular tool layer (agent_tools.py)                          │
│  - modular report/error layer (agent_reporting.py)              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                Statistics Module (src/statistics/)              │
├─────────────────────────────────────────────────────────────────┤
│  ABTestAnalyzer (analyzer.py)                                   │
│  - Thin facade that orchestrates modular components             │
│                                                                 │
│  ABTestDataManager (data_manager.py)                            │
│  - Data loading, column detection, auto-configuration           │
│                                                                 │
│  StatsmodelsABTestEngine (statsmodels_engine.py)                │
│  - OLS-based treatment effect estimation (HC3 robust SE)        │
│  - GLM support (binary/count), heavy-tail robust strategy        │
│  - Proportion tests, AA tests, power analysis, Bayesian MC      │
│  - FDR/BH multiple-testing correction + inference guardrails     │
│  - SRM/assumption/outlier diagnostics                            │
│  - Sequential decision support (interim looks)                  │
│                                                                 │
│  ABTestSummaryBuilder (summary_builder.py)                      │
│  - Aggregation and recommendation generation                    │
├─────────────────────────────────────────────────────────────────┤
│  ABTestVisualizer (visualizer.py)                               │
│  - Treatment vs Control charts                                  │
│  - Effect sizes with confidence intervals                       │
│  - P-value visualization                                        │
│  - Power analysis charts                                        │
│  - Cohen's d interpretation bands                               │
│  - Summary dashboards                                           │
├─────────────────────────────────────────────────────────────────┤
│  ABTestResult (models.py)                                       │
│  - Dataclass for test results                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ab_testing_agent
   ```

2. **Create a virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Start the Chainlit UI

```bash
python app.py
```

This will start the web interface at `http://localhost:8000`.

### Generate Sample Data

To generate sample A/B test data for testing:

```bash
python scripts/generate_sample_data.py
```

This creates two sample files in the `data/` directory:
- `sample_ab_data.csv` - Main sample with 4 customer segments
- `sample_ab_data_alt.csv` - Alternative format with different column names

### Example Workflow

1. **Upload your CSV file** using the attachment button in the chat
2. **Confirm column mappings** - The agent will detect and suggest columns
3. **Specify group labels** - Tell the agent which values represent treatment/control
4. **Run analysis**:
   - "Run a full A/B test analysis"
   - "What's the effect size for the Premium segment?"
   - "Show me the segment distribution"

## Data Requirements

Your CSV file should contain:

| Column Type | Description | Required |
|-------------|-------------|----------|
| Group | Treatment/control indicator | Yes |
| Effect Value | Numeric metric to analyze | Yes |
| Customer ID | Unique identifier | No |
| Segment | Customer segments | No |
| Duration | Experiment duration | No |

### Example CSV Structure

```csv
customer_id,experiment_group,customer_segment,effect_value,experiment_duration_days
CUST_001,treatment,Premium,58.50,14
CUST_002,control,Standard,28.30,21
CUST_003,treatment,Basic,12.80,7
```

## Statistical Measures

The agent provides the following statistical measures:

| Measure | Description |
|---------|-------------|
| Sample Sizes | Treatment and control group counts |
| Means | Group averages for the metric |
| Effect Size | Absolute difference (Treatment - Control) |
| Cohen's d | Standardized effect size |
| T-statistic | Test statistic value |
| P-value | Statistical significance |
| 95% CI | Confidence interval for effect size |
| Power | Probability of detecting true effect |
| Required n | Minimum sample size for adequate power |

### Effect Size Interpretation (Cohen's d)

| Value | Interpretation |
|-------|----------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |

## Modules

### src/agent.py
Main LangChain agent orchestration layer:
- backend switching (pandas vs Spark)
- conversational `run`/`arun` lifecycle
- tool registration via modular tool handlers

### src/agent_tools.py
Tool implementation layer for the conversational agent:
- data loading/configuration/analysis tools
- chart generation and query/data helper tools
- centralized tool-level error handling

### src/agent_reporting.py
Rendering and reliability layer:
- markdown/text formatting for tool outputs
- structured user-facing errors and stable error codes

### src/statistics/analyzer.py
Facade/orchestration layer that coordinates data, inference, and reporting components.

### src/statistics/data_manager.py
Data lifecycle and schema-inference layer:
- CSV loading and dataframe access
- Column auto-detection
- Best-guess auto-configuration
- Data summary and distribution helpers

### src/statistics/statsmodels_engine.py
Statsmodels-first inferential layer:
- OLS treatment-effect estimation with robust covariance
- GLM binomial/count model support
- Heavy-tail robust inference path
- AA test, two-proportion z-test, power/sample-size calculations
- Difference-in-differences estimation
- Bayesian Monte Carlo effect estimation
- Sequential decision support for interim looks
- SRM/assumption/outlier diagnostics and guardrails

### src/statistics/summary_builder.py
Transforms segment-level results into:
- Executive summary metrics
- Detailed tabular payloads for UI/reporting
- Actionable recommendations

### src/statistics/visualizer.py
Plotly-based visualization module:
- Treatment vs Control bar charts
- Effect sizes with confidence intervals
- P-value charts with significance threshold
- Power analysis visualization
- Cohen's d interpretation bands
- Comprehensive multi-chart dashboards
- Waterfall charts for effect contribution

### src/statistics/models.py
Data structures for analysis results:
- `ABTestResult`: Canonical dataclass used across pandas and Spark paths
- Includes adjusted p-values, diagnostics, model metadata, and sequential fields

## Configuration

### Environment Variables

Create a `.env` file with:
```
OPENAI_API_KEY=your-api-key-here
```

### Chainlit Settings

Edit `.chainlit/config.toml` to customize:
- Project name and description
- UI theme and colors
- File upload settings (CSV, max size)
- Feature toggles

### Agent Settings

Configure in `src/agent.py`:
- `model_name`: LLM model (default: `"gpt-5.2"`)
- `temperature`: Response randomness (default: 0)

### Statistical Settings

Configure in `src/statistics/analyzer.py`:
- `significance_level`: Alpha for tests (default: 0.05)
- `power_threshold`: Minimum power (default: 0.8)

## Testing and CI

Run locally:

```bash
uv run python -m compileall -q src tests app.py
uv run pytest -q -ra
```

CI:
- `.github/workflows/ci.yml` runs syntax checks, a smoke analyzer flow, and the pytest suite.
- Spark-dependent tests skip cleanly when PySpark/runtime is unavailable.

## Troubleshooting

### Common Issues

1. **"No data loaded"**: Upload a CSV file first before running analysis
2. **"Column not found"**: Check column names match exactly (case-sensitive)
3. **"Insufficient data"**: Segment may have too few samples for analysis

### API Key Issues

Ensure your OpenAI API key is set correctly in the `.env` file.

## License

MIT License
