# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a conversational A/B testing analysis agent powered by LangChain, LangGraph, and OpenAI's GPT-4. Users interact through a Chainlit web UI to upload CSV experiment data and receive comprehensive statistical analysis through natural language conversation.

The agent performs:
- **Frequentist analysis**: T-tests, proportion tests (z-tests), effect sizes, power analysis
- **Bayesian analysis**: Posterior probabilities, credible intervals, expected loss
- **Causal inference**: Difference-in-Differences (DiD) analysis when pre/post data available
- **Balance checking**: AA tests with bootstrap balancing for imbalanced groups
- **Segment-level analysis**: Automatic segmentation and cross-segment comparison

## Development Commands

### Running the Application

```bash
# Start the Chainlit web UI (default: http://localhost:8000)
python app.py
# or
chainlit run app.py
```

### Dependency Management

This project uses `uv` for faster dependency management:

```bash
# Create virtual environment
uv venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Generating Test Data

```bash
# Generate small sample A/B test CSV files with pre/post effects (~350KB)
python scripts/generate_sample_data.py

# Generate large sample dataset for testing PySpark backend (~30MB, 500K rows)
python scripts/generate_large_sample_data.py
```

This creates `data/sample_ab_data.csv` and `data/sample_ab_data_alt.csv` with realistic experiment data including multiple segments, pre/post effects, and intentionally imbalanced groups for testing AA tests.

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_ab_module.py
pytest tests/test_agent.py
pytest tests/test_visualizations.py
```

## Architecture

### Component Flow

```
User (Browser)
    ↓
Chainlit UI (app.py)
    ↓
ABTestingAgent (src/agent.py) ← LangGraph ReAct Agent with Tools
    ↓                              [Automatic Backend Selection]
    ├─→ ABTestAnalyzer (pandas) ← For files ≤ 2MB (in-memory)
    └─→ PySparkABTestAnalyzer ← For files > 2MB (distributed)
    ↓
ABTestVisualizer (src/statistics/visualizer.py) ← Plotly Charts
```

### Large File Support (PySpark)

The agent **automatically detects file size** and switches backends:

- **Files ≤ 2MB**: Uses `ABTestAnalyzer` (pandas) for fast in-memory processing
- **Files > 2MB**: Uses `PySparkABTestAnalyzer` (PySpark) for distributed processing

This happens automatically in the `load_csv` and `load_and_auto_analyze` tools. The statistical results are identical - only the processing method differs.

**To enable PySpark support**, uncomment the PySpark line in `requirements.txt` and install:
```bash
uv pip install pyspark>=3.5.0
```

See [LARGE_FILE_SUPPORT.md](LARGE_FILE_SUPPORT.md) for detailed documentation.

### Key Design Patterns

**LangGraph ReAct Agent**: The agent (`src/agent.py`) uses LangGraph's ReAct (Reasoning + Acting) pattern with a tool-based architecture. Each statistical operation is exposed as a tool that the LLM can invoke based on user intent.

**Stateful Analyzer**: `ABTestAnalyzer` maintains state across tool calls (loaded DataFrame, column mappings, treatment/control labels) so the agent can perform multi-step workflows conversationally.

**Two-Mode Operation**:
1. **Best-Guess Mode**: Single-step auto-configuration and analysis (`load_and_auto_analyze` tool)
2. **Manual Mode**: Step-by-step configuration with user confirmation (`load_csv`, `set_column_mapping`, `set_group_labels`, then analysis tools)

**Chart Storage Pattern**: The agent stores generated charts in `_last_charts` dict. The Chainlit app retrieves and displays them after each agent response, then clears them.

### Statistical Analysis Pipeline

The analyzer (`src/statistics/analyzer.py`) performs analysis in this order:

1. **AA Test** (if `pre_effect` column exists): Check treatment/control balance before experiment
   - If imbalanced (p < 0.05), apply bootstrap balancing to find balanced control subset

2. **Frequentist Tests**:
   - **T-test**: Compare `post_effect` means between treatment and control
   - **Proportion Test**: Z-test for conversion/activation rate differences (non-zero effects)
   - Calculate combined effect: t-test effect + proportion-based effect

3. **Bayesian Test**: Monte Carlo simulation with t-distribution
   - Uses DiD effect if pre/post data available
   - Outputs P(Treatment > Control), credible intervals, expected loss

4. **DiD Analysis** (if both `pre_effect` and `post_effect` exist):
   - Calculate true causal effect: (post_treatment - pre_treatment) - (post_control - pre_control)
   - This removes baseline differences and secular trends

### Data Model Hierarchy

All statistical results are stored in dataclasses (`src/statistics/models.py`):

- **`ABTestResult`**: Complete results for a single segment (40+ fields covering frequentist, Bayesian, DiD, AA test, proportion test)
- **`AATestResult`**: Balance check results including bootstrap information

These dataclasses are returned by analyzer methods and consumed by the visualizer and agent tools.

### Column Auto-Detection

The analyzer includes smart column detection (`detect_columns()`) based on naming patterns:
- **Group**: "group", "treatment", "control", "variant", "experiment_group"
- **Pre-effect**: "pre_effect", "pre_value", "baseline", "before"
- **Post-effect**: "post_effect", "post_value", "effect_value", "revenue", "amount"
- **Segment**: "segment", "customer_segment", "cohort", "category"
- **Customer ID**: "customer_id", "user_id", "id"

The auto-configuration flow (`auto_configure()`) attempts to detect all columns and labels, falling back gracefully with warnings.

## Code Structure Conventions

### Agent Tools Pattern

Each tool in `src/agent.py` follows this structure:
1. Pydantic input schema (for structured tools)
2. Implementation function that calls analyzer/visualizer
3. Tool wrapper with clear description
4. Chart storage if visualizations are generated

**Example**:
```python
class GenerateChartsInput(BaseModel):
    chart_types: List[str] = Field(description="List of chart types")

def generate_charts(chart_types: List[str]) -> str:
    # Implementation
    for chart_type in chart_types:
        fig = self.visualizer.plot_xyz(...)
        self._last_charts[chart_type] = fig
    return "Charts generated"

generate_charts_tool = StructuredTool.from_function(
    func=generate_charts,
    name="generate_charts",
    description="Generate charts...",
    args_schema=GenerateChartsInput
)
```

### Statistical Test Implementation

When adding new statistical tests to `analyzer.py`:
1. Add fields to `ABTestResult` dataclass in `models.py`
2. Implement test calculation method in analyzer (e.g., `run_bayesian_test()`)
3. Call from `run_ab_test()` or `run_segmented_analysis()`
4. Update `generate_summary()` to include new metrics
5. Add visualization method to `visualizer.py`
6. Expose via agent tool if user-facing

### Visualization Conventions

All visualizations in `visualizer.py`:
- Use the class color palette (`self.colors`)
- Apply consistent layout via `_apply_layout()`
- Return `go.Figure` objects
- Include hover data and clear axis labels
- Use subplots for multi-metric dashboards

## Environment Configuration

Required environment variable in `.env`:
```
OPENAI_API_KEY=your-api-key-here
```

Chainlit configuration in `.chainlit/config.toml`:
- Project name and UI theme
- File upload settings (CSV only, max size)
- Feature toggles

## Important Implementation Notes

### Statistical Calculations

**Effect Decomposition**: The agent calculates multiple effect measures:
- **T-test effect**: Difference in means for customers who engaged
- **Proportion effect**: Value from incremental conversions (customers who converted ONLY due to treatment)
- **DiD effect**: True causal effect removing baseline differences
- **Combined effect**: T-test + proportion (total business impact)

**Power Analysis**: Uses `statsmodels.stats.power.TTestIndPower` to assess sample adequacy. Flags segments with power < 0.8 or samples below required N.

**Bootstrap Balancing**: When AA test fails (pre-treatment imbalance), randomly samples control group subsets to find one that balances with treatment (max 1000 iterations, p > 0.1 threshold).

### Chainlit Integration

**File Upload Flow** (`app.py`):
1. User uploads CSV via attachment button
2. File saved to temp path via `element.path`
3. User's message text (if any) combined with file path
4. Passed to agent: "User request: {text}\n\nCSV file path: {path}"
5. Agent detects intent and chooses best-guess or manual mode

**Chart Display**:
- Agent stores charts in `agent._last_charts` dict
- After each message, `display_charts()` retrieves and displays them as `cl.Plotly` elements
- Charts cleared after display to avoid duplication

### Agent System Prompt

The agent's system prompt (in `_create_agent()`) instructs it to:
- Prefer combined tools (`configure_and_analyze`, `load_and_auto_analyze`) over multi-step flows
- Default to analyzing ALL segments unless user specifies one
- Use best-guess mode when user says "auto", "best guess", or provides no specific instructions
- Always offer to show charts after analysis

## Common Modification Patterns

**Adding a new statistical test**:
1. Add fields to `ABTestResult` in [models.py](src/statistics/models.py)
2. Implement test method in [analyzer.py](src/statistics/analyzer.py)
3. Call from `run_ab_test()` around line 400-500
4. Update summary generation around line 800
5. Add chart method in [visualizer.py](src/statistics/visualizer.py)
6. Update dashboard composition if needed
7. Add tool in [agent.py](src/agent.py) if user-facing

**Adding a new visualization**:
1. Add method to `ABTestVisualizer` class
2. Use `_apply_layout()` for consistency
3. Add to `generate_charts` tool's available chart types
4. Document in agent system prompt

**Modifying auto-detection**:
1. Update patterns in `detect_columns()` method
2. Test with sample data using `python scripts/generate_sample_data.py`
3. Verify auto-configuration via best-guess mode

## Debugging Tips

**Agent not detecting columns**: Check `detect_columns()` return value; add new patterns if needed.

**Statistical test failing**: Verify data has sufficient samples per segment (n ≥ 30 recommended).

**Charts not displaying**: Ensure chart is stored in `self._last_charts` and name matches expected pattern.

**AA test always failing**: Check if pre_effect data has high variance; may need to adjust p-value threshold or bootstrap iterations.
