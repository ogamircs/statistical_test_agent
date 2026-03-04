# A/B Testing Analysis Agent

An intelligent conversational agent for analyzing A/B test experiments, powered by LangChain and GPT-4. Upload your experiment data and get comprehensive statistical analysis through natural language interaction.

## Features

- **Conversational Interface**: Chat naturally about your A/B test data
- **Automatic Column Detection**: Smart detection of experiment columns based on naming patterns
- **Comprehensive Statistical Analysis**:
  - T-tests for significance testing
  - Effect size calculations (Cohen's d)
  - 95% Confidence intervals
  - Statistical power analysis
  - Sample size adequacy assessment
- **Segment-level Analysis**: Analyze results across customer segments
- **Interactive Visualizations**: Plotly-powered charts and dashboards
- **Actionable Recommendations**: Get clear guidance based on your results

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
│   ├── agent.py                # LangChain conversational agent
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
│   └── test_visualizations.py  # Visualization tests
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
│                                                                 │
│  Tools:                                                         │
│  - load_csv           - run_ab_test        - query_data         │
│  - set_column_mapping - run_full_analysis  - get_data_summary   │
│  - set_group_labels   - get_column_values  - calculate_stats    │
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
│  - Proportion tests, AA tests, power analysis, Bayesian MC      │
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
The main LangChain agent that provides a conversational interface. Uses LangGraph's ReAct pattern with custom tools for data analysis.

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
- AA test, two-proportion z-test, power/sample-size calculations
- Difference-in-differences estimation
- Bayesian Monte Carlo effect estimation

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
- `ABTestResult`: Dataclass containing all test metrics for a segment

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
- `model_name`: LLM model (default: "gpt-4o")
- `temperature`: Response randomness (default: 0)

### Statistical Settings

Configure in `src/statistics/analyzer.py`:
- `significance_level`: Alpha for tests (default: 0.05)
- `power_threshold`: Minimum power (default: 0.8)

## Troubleshooting

### Common Issues

1. **"No data loaded"**: Upload a CSV file first before running analysis
2. **"Column not found"**: Check column names match exactly (case-sensitive)
3. **"Insufficient data"**: Segment may have too few samples for analysis

### API Key Issues

Ensure your OpenAI API key is set correctly in the `.env` file.

## License

MIT License
