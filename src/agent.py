"""
LangChain A/B Testing Agent

An intelligent agent that can:
- Load and analyze CSV data
- Perform comprehensive A/B testing
- Answer questions about the data
- Provide statistical insights and recommendations
- Generate interactive visualizations
"""

import os
from typing import Optional, List, Any, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool, StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
import plotly.graph_objects as go

from .statistics import ABTestAnalyzer, ABTestVisualizer

load_dotenv()


class ABTestingAgent:
    """
    LangChain Agent for A/B Testing Analysis

    Provides conversational interface for:
    - Loading and exploring CSV data
    - Configuring column mappings
    - Running A/B tests
    - Answering data-related questions
    - Generating interactive visualizations
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.analyzer = ABTestAnalyzer()
        self.visualizer = ABTestVisualizer()
        self.chat_history: List[Any] = []
        self.agent = self._create_agent()
        self._pending_confirmation = None
        self._last_charts: Dict[str, go.Figure] = {}
        self._last_results = None
        self._last_summary = None

    def get_charts(self) -> Dict[str, go.Figure]:
        """Get the last generated charts"""
        return self._last_charts

    def clear_charts(self):
        """Clear the stored charts"""
        self._last_charts = {}

    def _create_tools(self) -> List[Tool]:
        """Create the tools for the agent"""

        # Tool: Load CSV Data
        def load_csv(filepath: str) -> str:
            """Load a CSV file for analysis"""
            try:
                info = self.analyzer.load_data(filepath)
                columns = info["columns"]
                shape = info["shape"]

                suggestions = self.analyzer.detect_columns()

                result = f"Successfully loaded data from '{filepath}'\n"
                result += f"Shape: {shape[0]} rows, {shape[1]} columns\n\n"
                result += f"Columns found: {', '.join(columns)}\n\n"
                result += "Column suggestions based on naming patterns:\n"

                for col_type, suggested in suggestions.items():
                    if suggested:
                        result += f"  - {col_type}: {', '.join(suggested)}\n"
                    else:
                        result += f"  - {col_type}: No automatic match found\n"

                result += "\nPlease confirm or specify the correct column mappings."
                return result

            except Exception as e:
                return f"Error loading file: {str(e)}"

        load_csv_tool = Tool(
            name="load_csv",
            func=load_csv,
            description="Load a CSV file for A/B test analysis. Input should be the file path. Use this when you need to inspect the data before analysis. For best-guess mode, use load_and_auto_analyze instead."
        )

        # Tool: Load and Auto-Analyze (Best Guess - Single Step)
        def load_and_auto_analyze(filepath: str) -> str:
            """Load CSV and automatically run full analysis using best guesses"""
            try:
                # Load the data
                info = self.analyzer.load_data(filepath)

                # Auto-configure
                config = self.analyzer.auto_configure()

                if not config["success"]:
                    return f"Loaded file but auto-configuration failed: {config.get('error', 'Unknown error')}"

                # Run analysis
                results = self.analyzer.run_segmented_analysis()
                summary = self.analyzer.generate_summary(results)

                # Store for charts
                self._last_results = results
                self._last_summary = summary

                # Build output with markdown formatting
                output = "## Best Guess Mode - Analysis Complete\n\n"

                output += f"**File:** {filepath.split('/')[-1].split(chr(92))[-1]}\n"
                output += f"**Shape:** {info['shape'][0]:,} rows × {info['shape'][1]} columns\n\n"

                output += "### Auto-Detected Configuration\n"
                output += "| Setting | Value |\n|---------|-------|\n"
                for key, value in config["mapping"].items():
                    output += f"| {key} | `{value}` |\n"
                output += f"| Treatment Label | `{config['labels'].get('treatment', 'N/A')}` |\n"
                output += f"| Control Label | `{config['labels'].get('control', 'N/A')}` |\n\n"

                if config["warnings"]:
                    output += "**Warnings:**\n"
                    for warning in config["warnings"]:
                        output += f"- {warning}\n"
                    output += "\n"

                output += "---\n\n"
                output += "## A/B Test Results\n\n"

                output += "### Overview\n"
                output += f"- **Segments Analyzed:** {summary['total_segments_analyzed']}\n"
                output += f"- **T-test Significant:** {summary['t_test_significant_segments']} ({summary['t_test_significance_rate']:.1%})\n"
                output += f"- **Proportion Test Significant:** {summary['prop_test_significant_segments']} ({summary['prop_test_significance_rate']:.1%})\n\n"

                output += "### Sample Information\n"
                output += f"- **Total Treatment:** {summary['total_treatment_customers']:,}\n"
                output += f"- **Total Control:** {summary['total_control_customers']:,}\n\n"

                output += "### Effect Summary\n"
                output += f"- **T-test Effect:** {summary['t_test_effect_calculation']}\n"
                output += f"- **Proportion Effect:** {summary['prop_test_effect_calculation']}\n"
                output += f"- **Combined Total Effect:** {summary['combined_effect_calculation']}\n\n"

                output += "### Statistical Results Summary\n\n"
                output += "| Segment | Treat N | Ctrl N | T-test p-val | T-test Effect | Prop p-val | Prop Effect | Total Effect |\n"
                output += "|---------|---------|--------|--------------|---------------|------------|-------------|-------------|\n"

                for r in summary['detailed_results']:
                    t_effect = r['effect']
                    t_pval = r['p_value']
                    prop_pval = r['prop_p_value']
                    prop_effect_per_cust = r.get('prop_effect_per_customer', 0)

                    # Calculate total effect: t-test effect × treatment_n + prop effect × control_n
                    t_total = t_effect * r['treatment_n'] if r['significant'] else 0
                    prop_total = prop_effect_per_cust * r['control_n'] if r['prop_significant'] else 0
                    total_effect = t_total + prop_total

                    # Add significance markers
                    t_sig_marker = "*" if r['significant'] else ""
                    p_sig_marker = "*" if r['prop_significant'] else ""

                    output += f"| {r['segment']} | {r['treatment_n']:,} | {r['control_n']:,} | {t_pval:.4f}{t_sig_marker} | {t_effect:.4f} | {prop_pval:.4f}{p_sig_marker} | {prop_effect_per_cust:.4f} | {total_effect:.2f} |\n"

                output += "\n*\\* indicates statistical significance (p < 0.05)*\n"
                output += "*Total Effect = (T-test Effect × Treat N) + (Prop Effect × Ctrl N) for significant results*\n"

                output += "\n### Recommendations\n\n"
                for i, rec in enumerate(summary['recommendations'], 1):
                    output += f"{i}. {rec}\n"

                output += "\n---\n*Would you like to see the visualizations?*"

                return output

            except Exception as e:
                return f"Error in load and auto-analyze: {str(e)}"

        load_auto_analyze_tool = Tool(
            name="load_and_auto_analyze",
            func=load_and_auto_analyze,
            description="""Load a CSV file AND automatically run full A/B test analysis using best guesses - ALL IN ONE STEP.
Use this when the user says 'best guess', 'auto', 'automatic', 'figure it out', or wants quick analysis without manual configuration.
Input: file path. This is the FASTEST way to get results."""
        )

        # Tool: Set Column Mapping
        class ColumnMappingInput(BaseModel):
            customer_id: Optional[str] = Field(None, description="Column name for customer ID")
            group: str = Field(..., description="Column name for treatment/control group indicator")
            effect_value: str = Field(..., description="Column name for the metric/effect value to analyze")
            segment: Optional[str] = Field(None, description="Column name for customer segments")
            duration: Optional[str] = Field(None, description="Column name for experiment duration")

        def set_column_mapping(
            customer_id: Optional[str] = None,
            group: str = "",
            effect_value: str = "",
            segment: Optional[str] = None,
            duration: Optional[str] = None
        ) -> str:
            """Set the column mapping for analysis"""
            mapping = {}
            if customer_id:
                mapping["customer_id"] = customer_id
            if group:
                mapping["group"] = group
            if effect_value:
                mapping["effect_value"] = effect_value
            if segment:
                mapping["segment"] = segment
            if duration:
                mapping["duration"] = duration

            try:
                self.analyzer.set_column_mapping(mapping)

                group_info = self.analyzer.get_group_values()

                result = f"Column mapping set successfully:\n"
                for key, value in mapping.items():
                    result += f"  - {key}: {value}\n"

                result += f"\nUnique values in '{group}' column: {group_info['unique_values']}\n"
                result += "Please specify which value represents 'treatment' and which represents 'control'."

                return result

            except Exception as e:
                return f"Error setting column mapping: {str(e)}"

        set_mapping_tool = StructuredTool.from_function(
            func=set_column_mapping,
            name="set_column_mapping",
            description="Set the column mapping for A/B test analysis. Specify which columns contain customer ID, group indicator, effect value, segments, and duration."
        )

        # Tool: Set Group Labels
        def set_group_labels(treatment_label: str, control_label: str) -> str:
            """Set the labels used for treatment and control groups"""
            try:
                self.analyzer.set_group_labels(treatment_label, control_label)
                return f"Group labels set: Treatment='{treatment_label}', Control='{control_label}'"
            except Exception as e:
                return f"Error setting group labels: {str(e)}"

        set_labels_tool = Tool(
            name="set_group_labels",
            func=lambda x: set_group_labels(*[s.strip() for s in x.split(",")]),
            description="Set the treatment and control group labels. Input format: 'treatment_label, control_label'"
        )

        # Tool: Run A/B Test
        def run_ab_test(segment: Optional[str] = None) -> str:
            """Run A/B test for a specific segment or overall"""
            try:
                if segment and segment.lower() not in ["none", "overall", "all", ""]:
                    result = self.analyzer.run_ab_test(segment_filter=segment)
                else:
                    result = self.analyzer.run_ab_test()

                output = f"\n{'='*60}\n"
                output += f"A/B TEST RESULTS - {result.segment}\n"
                output += f"{'='*60}\n\n"

                output += f"Sample Sizes:\n"
                output += f"  Treatment: {result.treatment_size}\n"
                output += f"  Control: {result.control_size}\n\n"

                output += f"Means:\n"
                output += f"  Treatment: {result.treatment_mean:.4f}\n"
                output += f"  Control: {result.control_mean:.4f}\n\n"

                output += f"Effect Size:\n"
                output += f"  Absolute Difference: {result.effect_size:.4f}\n"
                output += f"  Cohen's d: {result.cohens_d:.4f}\n"
                output += f"  95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]\n\n"

                output += f"Statistical Test:\n"
                output += f"  t-statistic: {result.t_statistic:.4f}\n"
                output += f"  p-value: {result.p_value:.6f}\n"
                output += f"  Significant (p < 0.05): {'YES' if result.is_significant else 'NO'}\n\n"

                output += f"Power Analysis:\n"
                output += f"  Statistical Power: {result.power:.2%}\n"
                output += f"  Required Sample Size (per group): {result.required_sample_size}\n"
                output += f"  Sample Adequate: {'YES' if result.is_sample_adequate else 'NO'}\n"

                return output

            except Exception as e:
                return f"Error running A/B test: {str(e)}"

        run_test_tool = Tool(
            name="run_ab_test",
            func=run_ab_test,
            description="Run A/B test for a specific segment or overall. Input: segment name (or 'overall' for all data)"
        )

        # Tool: Run Full Segmented Analysis
        def run_full_analysis(_: str = "") -> str:
            """Run A/B tests for all segments and generate summary"""
            try:
                results = self.analyzer.run_segmented_analysis()
                summary = self.analyzer.generate_summary(results)

                # Store results for chart generation
                self._last_results = results
                self._last_summary = summary

                output = f"\n{'='*60}\n"
                output += f"FULL A/B TEST ANALYSIS SUMMARY\n"
                output += f"{'='*60}\n\n"

                output += f"OVERVIEW:\n"
                output += f"  Segments Analyzed: {summary['total_segments_analyzed']}\n"
                output += f"  Significant Results: {summary['significant_segments']}\n"
                output += f"  Non-Significant: {summary['non_significant_segments']}\n"
                output += f"  Significance Rate: {summary['significance_rate']:.1%}\n\n"

                output += f"SAMPLE INFORMATION:\n"
                output += f"  Total Treatment Customers: {summary['total_treatment_customers']}\n"
                output += f"  Total Control Customers: {summary['total_control_customers']}\n"
                if summary['treatment_control_ratio']:
                    output += f"  Treatment/Control Ratio: {summary['treatment_control_ratio']:.2f}\n\n"

                output += f"EFFECT SIZE SUMMARY:\n"
                output += f"  Average Significant Effect: {summary['average_significant_effect']:.4f}\n"
                output += f"  Treatment in Significant Segments: {summary['total_treatment_in_significant_segments']}\n"
                output += f"  Total Effect Size: {summary['effect_calculation']}\n\n"

                output += f"POWER ANALYSIS:\n"
                output += f"  Segments with Adequate Power: {summary['segments_with_adequate_power']}\n"
                output += f"  Segments with Inadequate Power: {summary['segments_with_inadequate_power']}\n"
                output += f"  Power Adequacy Rate: {summary['power_adequacy_rate']:.1%}\n\n"

                output += f"SEGMENT DETAILS:\n"
                output += "-" * 100 + "\n"
                output += f"{'Segment':<20} {'Treat N':<10} {'Ctrl N':<10} {'Effect':<12} {'p-value':<12} {'Sig?':<6} {'Power':<8} {'Adequate?':<10}\n"
                output += "-" * 100 + "\n"

                for r in summary['detailed_results']:
                    sig = "YES" if r['significant'] else "NO"
                    adeq = "YES" if r['adequate_sample'] else "NO"
                    output += f"{r['segment']:<20} {r['treatment_n']:<10} {r['control_n']:<10} {r['effect']:<12.4f} {r['p_value']:<12.6f} {sig:<6} {r['power']:<8.2%} {adeq:<10}\n"

                output += "-" * 100 + "\n\n"

                output += f"RECOMMENDATIONS:\n"
                for i, rec in enumerate(summary['recommendations'], 1):
                    output += f"  {i}. {rec}\n\n"

                return output

            except Exception as e:
                return f"Error running full analysis: {str(e)}"

        full_analysis_tool = Tool(
            name="run_full_analysis",
            func=run_full_analysis,
            description="Run A/B tests for ALL segments and generate a comprehensive summary with recommendations. Use this for complete analysis."
        )

        # Tool: Query Data
        def query_data(query: str) -> str:
            """Query the data using pandas query syntax"""
            try:
                if self.analyzer.df is None:
                    return "No data loaded. Please load a CSV file first."

                result = self.analyzer.query_data(query)
                return f"Query result ({len(result)} rows):\n{result.head(20).to_string()}\n\n(Showing first 20 rows)"
            except Exception as e:
                return f"Error querying data: {str(e)}"

        query_tool = Tool(
            name="query_data",
            func=query_data,
            description="Query the loaded data using pandas query syntax. Example: 'segment == \"Premium\"' or 'effect_value > 100'"
        )

        # Tool: Get Data Summary
        def get_data_summary(_: str = "") -> str:
            """Get summary statistics of the data"""
            try:
                summary = self.analyzer.get_data_summary()

                output = f"DATA SUMMARY\n"
                output += f"{'='*50}\n\n"
                output += f"Shape: {summary['shape'][0]} rows x {summary['shape'][1]} columns\n\n"

                output += f"Columns and Types:\n"
                for col, dtype in summary['dtypes'].items():
                    missing = summary['missing_values'].get(col, 0)
                    output += f"  - {col}: {dtype} (missing: {missing})\n"

                output += f"\nNumeric Column Statistics:\n"
                numeric_stats = summary['numeric_summary']
                if numeric_stats:
                    for col in list(numeric_stats.get('mean', {}).keys())[:5]:
                        output += f"\n  {col}:\n"
                        output += f"    Mean: {numeric_stats['mean'].get(col, 'N/A'):.2f}\n"
                        output += f"    Std: {numeric_stats['std'].get(col, 'N/A'):.2f}\n"
                        output += f"    Min: {numeric_stats['min'].get(col, 'N/A'):.2f}\n"
                        output += f"    Max: {numeric_stats['max'].get(col, 'N/A'):.2f}\n"

                return output

            except Exception as e:
                return f"Error getting data summary: {str(e)}"

        summary_tool = Tool(
            name="get_data_summary",
            func=get_data_summary,
            description="Get summary statistics of the loaded data including column types, missing values, and descriptive statistics."
        )

        # Tool: Get Segment Distribution
        def get_segment_distribution(_: str = "") -> str:
            """Get distribution of customers across segments and groups"""
            try:
                dist = self.analyzer.get_segment_distribution()

                output = f"SEGMENT & GROUP DISTRIBUTION\n"
                output += f"{'='*50}\n\n"

                if 'group_distribution' in dist:
                    output += f"Group Distribution:\n"
                    for group, count in dist['group_distribution'].items():
                        output += f"  - {group}: {count}\n"
                    output += "\n"

                if 'segment_distribution' in dist:
                    output += f"Segment Distribution:\n"
                    for segment, count in dist['segment_distribution'].items():
                        output += f"  - {segment}: {count}\n"
                    output += "\n"

                if 'segment_by_group' in dist:
                    output += f"Segment x Group Cross-tabulation:\n"
                    for group, segments in dist['segment_by_group'].items():
                        output += f"\n  {group}:\n"
                        for segment, count in segments.items():
                            output += f"    - {segment}: {count}\n"

                return output

            except Exception as e:
                return f"Error getting distribution: {str(e)}"

        dist_tool = Tool(
            name="get_segment_distribution",
            func=get_segment_distribution,
            description="Get the distribution of customers across segments and treatment/control groups."
        )

        # Tool: Get Column Values
        def get_column_values(column_name: str) -> str:
            """Get unique values in a specific column"""
            try:
                if self.analyzer.df is None:
                    return "No data loaded."

                if column_name not in self.analyzer.df.columns:
                    return f"Column '{column_name}' not found. Available columns: {list(self.analyzer.df.columns)}"

                values = self.analyzer.df[column_name].unique()
                value_counts = self.analyzer.df[column_name].value_counts()

                output = f"Unique values in '{column_name}' ({len(values)} unique):\n"
                for val, count in value_counts.head(20).items():
                    output += f"  - {val}: {count}\n"

                if len(values) > 20:
                    output += f"  ... and {len(values) - 20} more values"

                return output

            except Exception as e:
                return f"Error: {str(e)}"

        column_values_tool = Tool(
            name="get_column_values",
            func=get_column_values,
            description="Get unique values and their counts for a specific column. Input: column name"
        )

        # Tool: Calculate Statistics
        def calculate_stats(column_name: str) -> str:
            """Calculate detailed statistics for a numeric column"""
            try:
                if self.analyzer.df is None:
                    return "No data loaded."

                if column_name not in self.analyzer.df.columns:
                    return f"Column '{column_name}' not found."

                col = self.analyzer.df[column_name]

                if not col.dtype in ['float64', 'int64', 'float32', 'int32']:
                    return f"Column '{column_name}' is not numeric (type: {col.dtype})"

                output = f"Statistics for '{column_name}':\n"
                output += f"  Count: {col.count()}\n"
                output += f"  Mean: {col.mean():.4f}\n"
                output += f"  Median: {col.median():.4f}\n"
                output += f"  Std Dev: {col.std():.4f}\n"
                output += f"  Min: {col.min():.4f}\n"
                output += f"  Max: {col.max():.4f}\n"
                output += f"  25th Percentile: {col.quantile(0.25):.4f}\n"
                output += f"  75th Percentile: {col.quantile(0.75):.4f}\n"
                output += f"  Missing Values: {col.isna().sum()}\n"

                return output

            except Exception as e:
                return f"Error: {str(e)}"

        stats_tool = Tool(
            name="calculate_statistics",
            func=calculate_stats,
            description="Calculate detailed statistics for a numeric column including mean, median, std dev, percentiles."
        )

        # Tool: Configure and Analyze (Combined one-step tool)
        class ConfigureAnalyzeInput(BaseModel):
            group_column: str = Field(..., description="Column name for treatment/control group")
            effect_column: str = Field(..., description="Column name for the metric/effect value")
            treatment_label: str = Field(..., description="Value that represents the treatment group")
            control_label: str = Field(..., description="Value that represents the control group")
            segment_column: Optional[str] = Field(None, description="Column for segments (optional)")
            customer_id_column: Optional[str] = Field(None, description="Column for customer ID (optional)")

        def configure_and_analyze(
            group_column: str,
            effect_column: str,
            treatment_label: str,
            control_label: str,
            segment_column: Optional[str] = None,
            customer_id_column: Optional[str] = None
        ) -> str:
            """Configure column mappings, set group labels, and run full analysis in one step"""
            try:
                # Build column mapping
                mapping = {
                    "group": group_column,
                    "effect_value": effect_column
                }
                if segment_column:
                    mapping["segment"] = segment_column
                if customer_id_column:
                    mapping["customer_id"] = customer_id_column

                # Set column mapping
                self.analyzer.set_column_mapping(mapping)

                # Set group labels
                self.analyzer.set_group_labels(treatment_label, control_label)

                # Run full segmented analysis
                results = self.analyzer.run_segmented_analysis()
                summary = self.analyzer.generate_summary(results)

                # Store for chart generation
                self._last_results = results
                self._last_summary = summary

                # Build output with markdown formatting
                output = "## Configuration Applied\n\n"
                output += "| Setting | Value |\n|---------|-------|\n"
                output += f"| Group Column | `{group_column}` |\n"
                output += f"| Effect Column | `{effect_column}` |\n"
                output += f"| Treatment Label | `{treatment_label}` |\n"
                output += f"| Control Label | `{control_label}` |\n"
                if segment_column:
                    output += f"| Segment Column | `{segment_column}` |\n"
                output += "\n---\n\n"

                output += "## A/B Test Results\n\n"

                output += "### Overview\n"
                output += f"- **Segments Analyzed:** {summary['total_segments_analyzed']}\n"
                output += f"- **T-test Significant:** {summary['t_test_significant_segments']} ({summary['t_test_significance_rate']:.1%})\n"
                output += f"- **Proportion Test Significant:** {summary['prop_test_significant_segments']} ({summary['prop_test_significance_rate']:.1%})\n\n"

                output += "### Sample Information\n"
                output += f"- **Total Treatment:** {summary['total_treatment_customers']:,}\n"
                output += f"- **Total Control:** {summary['total_control_customers']:,}\n\n"

                output += "### Effect Summary\n"
                output += f"- **T-test Effect:** {summary['t_test_effect_calculation']}\n"
                output += f"- **Proportion Effect:** {summary['prop_test_effect_calculation']}\n"
                output += f"- **Combined Total Effect:** {summary['combined_effect_calculation']}\n\n"

                output += "### Statistical Results Summary\n\n"
                output += "| Segment | Treat N | Ctrl N | T-test p-val | T-test Effect | Prop p-val | Prop Effect | Total Effect |\n"
                output += "|---------|---------|--------|--------------|---------------|------------|-------------|-------------|\n"

                for r in summary['detailed_results']:
                    t_effect = r['effect']
                    t_pval = r['p_value']
                    prop_pval = r['prop_p_value']
                    prop_effect_per_cust = r.get('prop_effect_per_customer', 0)

                    # Calculate total effect: t-test effect × treatment_n + prop effect × control_n
                    t_total = t_effect * r['treatment_n'] if r['significant'] else 0
                    prop_total = prop_effect_per_cust * r['control_n'] if r['prop_significant'] else 0
                    total_effect = t_total + prop_total

                    # Add significance markers
                    t_sig_marker = "*" if r['significant'] else ""
                    p_sig_marker = "*" if r['prop_significant'] else ""

                    output += f"| {r['segment']} | {r['treatment_n']:,} | {r['control_n']:,} | {t_pval:.4f}{t_sig_marker} | {t_effect:.4f} | {prop_pval:.4f}{p_sig_marker} | {prop_effect_per_cust:.4f} | {total_effect:.2f} |\n"

                output += "\n*\\* indicates statistical significance (p < 0.05)*\n"
                output += "*Total Effect = (T-test Effect × Treat N) + (Prop Effect × Ctrl N) for significant results*\n"

                output += "\n### Recommendations\n\n"
                for i, rec in enumerate(summary['recommendations'], 1):
                    output += f"{i}. {rec}\n"

                output += "\n---\n*Would you like to see the visualizations?*"

                return output

            except Exception as e:
                return f"Error in configure and analyze: {str(e)}"

        configure_analyze_tool = StructuredTool.from_function(
            func=configure_and_analyze,
            name="configure_and_analyze",
            description="""Configure column mappings, set treatment/control labels, and run full A/B test analysis in ONE step.
Use this tool to quickly set up and analyze data without multiple separate steps.
Required: group_column, effect_column, treatment_label, control_label
Optional: segment_column, customer_id_column"""
        )

        # Tool: Auto Configure and Analyze (Best Guess Mode)
        def auto_configure_and_analyze(_: str = "") -> str:
            """Automatically configure everything using best guesses and run full analysis"""
            try:
                if self.analyzer.df is None:
                    return "No data loaded. Please load a CSV file first."

                # Use auto_configure to set everything up
                config = self.analyzer.auto_configure()

                if not config["success"]:
                    return f"Auto-configuration failed: {config.get('error', 'Unknown error')}"

                # Run full analysis
                results = self.analyzer.run_segmented_analysis()
                summary = self.analyzer.generate_summary(results)

                # Store for chart generation
                self._last_results = results
                self._last_summary = summary

                # Build output with markdown formatting
                output = "## Best Guess Mode - Analysis Complete\n\n"

                output += "### Auto-Detected Configuration\n"
                output += "| Setting | Value |\n|---------|-------|\n"
                for key, value in config["mapping"].items():
                    output += f"| {key} | `{value}` |\n"
                output += f"| Treatment Label | `{config['labels'].get('treatment', 'N/A')}` |\n"
                output += f"| Control Label | `{config['labels'].get('control', 'N/A')}` |\n\n"

                if config["warnings"]:
                    output += "**Warnings:**\n"
                    for warning in config["warnings"]:
                        output += f"- {warning}\n"
                    output += "\n"

                output += "---\n\n"
                output += "## A/B Test Results\n\n"

                output += "### Overview\n"
                output += f"- **Segments Analyzed:** {summary['total_segments_analyzed']}\n"
                output += f"- **T-test Significant:** {summary['t_test_significant_segments']} ({summary['t_test_significance_rate']:.1%})\n"
                output += f"- **Proportion Test Significant:** {summary['prop_test_significant_segments']} ({summary['prop_test_significance_rate']:.1%})\n\n"

                output += "### Sample Information\n"
                output += f"- **Total Treatment:** {summary['total_treatment_customers']:,}\n"
                output += f"- **Total Control:** {summary['total_control_customers']:,}\n\n"

                output += "### Effect Summary\n"
                output += f"- **T-test Effect:** {summary['t_test_effect_calculation']}\n"
                output += f"- **Proportion Effect:** {summary['prop_test_effect_calculation']}\n"
                output += f"- **Combined Total Effect:** {summary['combined_effect_calculation']}\n\n"

                output += "### Statistical Results Summary\n\n"
                output += "| Segment | Treat N | Ctrl N | T-test p-val | T-test Effect | Prop p-val | Prop Effect | Total Effect |\n"
                output += "|---------|---------|--------|--------------|---------------|------------|-------------|-------------|\n"

                for r in summary['detailed_results']:
                    t_effect = r['effect']
                    t_pval = r['p_value']
                    prop_pval = r['prop_p_value']
                    prop_effect_per_cust = r.get('prop_effect_per_customer', 0)

                    # Calculate total effect: t-test effect × treatment_n + prop effect × control_n
                    t_total = t_effect * r['treatment_n'] if r['significant'] else 0
                    prop_total = prop_effect_per_cust * r['control_n'] if r['prop_significant'] else 0
                    total_effect = t_total + prop_total

                    # Add significance markers
                    t_sig_marker = "*" if r['significant'] else ""
                    p_sig_marker = "*" if r['prop_significant'] else ""

                    output += f"| {r['segment']} | {r['treatment_n']:,} | {r['control_n']:,} | {t_pval:.4f}{t_sig_marker} | {t_effect:.4f} | {prop_pval:.4f}{p_sig_marker} | {prop_effect_per_cust:.4f} | {total_effect:.2f} |\n"

                output += "\n*\\* indicates statistical significance (p < 0.05)*\n"
                output += "*Total Effect = (T-test Effect × Treat N) + (Prop Effect × Ctrl N) for significant results*\n"

                output += "\n### Recommendations\n\n"
                for i, rec in enumerate(summary['recommendations'], 1):
                    output += f"{i}. {rec}\n"

                output += "\n---\n*Would you like to see the visualizations?*"

                return output

            except Exception as e:
                return f"Error in auto-configure and analyze: {str(e)}"

        auto_analyze_tool = Tool(
            name="auto_configure_and_analyze",
            func=auto_configure_and_analyze,
            description="""Automatically detect column mappings and treatment/control labels using best guesses, then run full A/B test analysis.
Use this when the user says 'best guess', 'auto', 'automatic', or wants you to figure out the configuration yourself.
This is the fastest way to analyze data without manual configuration."""
        )

        # Visualization Tools
        def generate_charts(chart_type: str = "all") -> str:
            """Generate visualization charts for the A/B test results"""
            try:
                # Check if data is loaded and configured
                if self.analyzer.df is None:
                    return "No data loaded. Please load a CSV file first before generating charts."

                if "group" not in self.analyzer.column_mapping:
                    return "Column mapping not set. Please configure the group and effect_value columns first."

                if self.analyzer.treatment_label is None:
                    return "Group labels not set. Please specify which values represent treatment and control."

                # Run analysis if not done yet
                if self._last_results is None:
                    results = self.analyzer.run_segmented_analysis()
                    if not results:
                        return "No results from analysis. Please check your data configuration."
                    summary = self.analyzer.generate_summary(results)
                    self._last_results = results
                    self._last_summary = summary
                else:
                    results = self._last_results
                    summary = self._last_summary

                # Check for valid summary
                if summary is None or "error" in summary:
                    return "Could not generate summary. Please run the full analysis first."

                chart_type = chart_type.lower().strip()

                # Clear previous charts
                self._last_charts = {}

                # Statistical Summary - the main focused chart with table elements
                if chart_type in ["all", "summary", "statistical_summary", "stats", "table"]:
                    self._last_charts["statistical_summary"] = self.visualizer.plot_statistical_summary(results)

                if chart_type in ["all", "dashboard"]:
                    self._last_charts["dashboard"] = self.visualizer.plot_summary_dashboard(results, summary)

                if chart_type in ["all", "treatment_control", "comparison", "means"]:
                    self._last_charts["treatment_vs_control"] = self.visualizer.plot_treatment_vs_control(results)

                if chart_type in ["all", "effect", "effect_size", "effects"]:
                    self._last_charts["effect_sizes"] = self.visualizer.plot_effect_sizes(results)

                if chart_type in ["all", "pvalue", "p_value", "significance"]:
                    self._last_charts["p_values"] = self.visualizer.plot_p_values(results)

                if chart_type in ["all", "power", "power_analysis"]:
                    self._last_charts["power_analysis"] = self.visualizer.plot_power_analysis(results)

                if chart_type in ["all", "cohens_d", "cohen", "effect_magnitude"]:
                    self._last_charts["cohens_d"] = self.visualizer.plot_cohens_d(results)

                if chart_type in ["all", "sample", "sample_size", "samples"]:
                    self._last_charts["sample_sizes"] = self.visualizer.plot_sample_sizes(results)

                if chart_type in ["all", "waterfall", "contribution"]:
                    self._last_charts["effect_waterfall"] = self.visualizer.plot_effect_waterfall(results)

                chart_names = list(self._last_charts.keys())
                return f"Generated {len(chart_names)} chart(s): {', '.join(chart_names)}. The charts are now displayed in the UI."

            except Exception as e:
                return f"Error generating charts: {str(e)}. Please ensure you have loaded data, configured columns, and set group labels."

        generate_charts_tool = Tool(
            name="generate_charts",
            func=generate_charts,
            description="""Generate visualization charts for A/B test results.
Input options:
- 'all': Generate all charts
- 'summary' or 'statistical_summary': Statistical summary with T-test & Proportion test p-values, effects, and total effects (RECOMMENDED)
- 'dashboard': Summary dashboard
- 'treatment_control' or 'comparison': Treatment vs Control means chart
- 'effect' or 'effect_size': Effect sizes with confidence intervals
- 'pvalue' or 'significance': P-values chart
- 'power' or 'power_analysis': Statistical power chart
- 'cohens_d' or 'cohen': Cohen's d effect size chart
- 'sample' or 'sample_size': Sample sizes chart
- 'waterfall' or 'contribution': Effect contribution waterfall chart
Use this tool when the user asks to see charts, visualizations, or graphs."""
        )

        def show_distribution_chart(_: str = "") -> str:
            """Generate distribution chart showing treatment/control split"""
            try:
                if self.analyzer.df is None:
                    return "No data loaded. Please load a CSV file first."

                if "group" not in self.analyzer.column_mapping:
                    return "Group column not set. Please configure column mappings first."

                group_col = self.analyzer.column_mapping["group"]
                segment_col = self.analyzer.column_mapping.get("segment")

                self._last_charts["distribution"] = self.visualizer.plot_segment_distribution(
                    self.analyzer.df, group_col, segment_col
                )

                return "Generated distribution chart showing treatment/control split. The chart is now displayed in the UI."

            except Exception as e:
                return f"Error generating distribution chart: {str(e)}"

        distribution_chart_tool = Tool(
            name="show_distribution_chart",
            func=show_distribution_chart,
            description="Generate a pie/sunburst chart showing the distribution of customers across treatment and control groups. Use when user asks about group distribution or wants to visualize the experiment split."
        )

        return [
            load_csv_tool,
            load_auto_analyze_tool,  # Best guess: load + auto-analyze in one step
            set_mapping_tool,
            set_labels_tool,
            run_test_tool,
            full_analysis_tool,
            configure_analyze_tool,  # Combined one-step tool
            auto_analyze_tool,       # Best guess mode (data already loaded)
            query_tool,
            summary_tool,
            dist_tool,
            column_values_tool,
            stats_tool,
            generate_charts_tool,
            distribution_chart_tool
        ]

    def _create_agent(self):
        """Create the LangGraph agent with tools"""

        tools = self._create_tools()

        system_prompt = """You are an expert A/B Testing Analyst AI assistant. Your role is to help users analyze A/B test experiments from CSV data.

## CRITICAL - OUTPUT FORMATTING RULES:
**ALWAYS display the EXACT markdown tables returned by analysis tools. DO NOT summarize or rephrase the tables.**
When a tool returns markdown tables (like the Statistical Results Summary table), you MUST include them verbatim in your response.
The tables contain important statistical data that users need to see in tabular format.

## CRITICAL - Tool Selection Based on User Intent:

### BEST GUESS MODE (User wants automatic analysis):
If the user mentions ANY of these: "best guess", "auto", "automatic", "figure it out", "just analyze", "quick analysis", or similar:
=> Use `load_and_auto_analyze` tool with JUST the file path
=> This loads the file, auto-detects everything, and runs full analysis - NO QUESTIONS ASKED
=> Do NOT use load_csv, do NOT ask for confirmation

### MANUAL MODE (User wants to review/confirm settings):
If the user uploads a file WITHOUT mentioning auto/best guess:
=> Use `load_csv` to show columns
=> Then use `configure_and_analyze` with their confirmed settings

## Your Capabilities:
1. **Best Guess Analysis** - `load_and_auto_analyze`: Load file + auto-detect + run analysis in ONE step
2. **Manual Configuration** - `load_csv` then `configure_and_analyze`: For users who want control
3. **Generate visualizations** - Interactive charts after analysis

## Workflow Decision Tree:
1. User uploads file with "best guess"/"auto" keywords => `load_and_auto_analyze`
2. User uploads file without keywords => `load_csv`, then ask to confirm, then `configure_and_analyze`
3. Data already loaded + user wants best guess => `auto_configure_and_analyze`

## Important Guidelines:
- DEFAULT to full segmented analysis (all segments) unless user specifies otherwise
- Explain statistical concepts in accessible language
- Provide actionable recommendations based on results
- Offer to show charts after analysis completes

## Statistical Measures:
- Sample sizes, means, effect sizes
- Cohen's d, p-values, significance
- 95% confidence intervals
- Statistical power analysis

## Visualizations:
Dashboard, Treatment vs Control, Effect Sizes, P-values, Power Analysis, Cohen's d, Sample Sizes, Waterfall

Be efficient - minimize steps to get users their results."""

        return create_react_agent(self.llm, tools, prompt=system_prompt)

    async def arun(self, message: str) -> str:
        """Run the agent asynchronously"""
        try:
            self.chat_history.append(HumanMessage(content=message))
            result = await self.agent.ainvoke({"messages": self.chat_history})
            response = result["messages"][-1].content
            self.chat_history.append(AIMessage(content=response))
            return response
        except Exception as e:
            return f"Error processing request: {str(e)}"

    def run(self, message: str) -> str:
        """Run the agent synchronously"""
        try:
            self.chat_history.append(HumanMessage(content=message))
            result = self.agent.invoke({"messages": self.chat_history})
            response = result["messages"][-1].content
            self.chat_history.append(AIMessage(content=response))
            return response
        except Exception as e:
            return f"Error processing request: {str(e)}"

    def clear_memory(self):
        """Clear conversation memory"""
        self.chat_history = []


if __name__ == "__main__":
    agent = ABTestingAgent()
    print("A/B Testing Agent initialized. Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        response = agent.run(user_input)
        print(f"\nAgent: {response}")
