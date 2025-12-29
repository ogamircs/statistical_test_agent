"""
LangChain A/B Testing Agent

An intelligent agent that can:
- Load and analyze CSV data
- Perform comprehensive A/B testing
- Answer questions about the data
- Provide statistical insights and recommendations
"""

import os
from typing import Optional, List, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool, StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from .statistics import ABTestAnalyzer

load_dotenv()


class ABTestingAgent:
    """
    LangChain Agent for A/B Testing Analysis

    Provides conversational interface for:
    - Loading and exploring CSV data
    - Configuring column mappings
    - Running A/B tests
    - Answering data-related questions
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.analyzer = ABTestAnalyzer()
        self.chat_history: List[Any] = []
        self.agent = self._create_agent()
        self._pending_confirmation = None

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
            description="Load a CSV file for A/B test analysis. Input should be the file path."
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

        return [
            load_csv_tool,
            set_mapping_tool,
            set_labels_tool,
            run_test_tool,
            full_analysis_tool,
            query_tool,
            summary_tool,
            dist_tool,
            column_values_tool,
            stats_tool
        ]

    def _create_agent(self):
        """Create the LangGraph agent with tools"""

        tools = self._create_tools()

        system_prompt = """You are an expert A/B Testing Analyst AI assistant. Your role is to help users analyze A/B test experiments from CSV data.

## Your Capabilities:
1. **Load and explore CSV files** - Understand the data structure
2. **Configure column mappings** - Map columns to required fields (customer_id, group, effect_value, segment, duration)
3. **Run A/B tests** - Perform statistical analysis for individual segments or overall data
4. **Generate comprehensive reports** - Provide summaries with significance, effect sizes, and recommendations
5. **Answer data questions** - Query and analyze the data to answer user questions

## Workflow:
1. When a user provides a CSV file, load it and examine the columns
2. Attempt to auto-detect column mappings, but ASK THE USER TO CONFIRM if uncertain
3. Once columns are mapped, identify treatment/control labels in the group column
4. Run the requested analysis (individual segments or full analysis)
5. Provide clear interpretations and recommendations

## Important Guidelines:
- Always ask for clarification if column names are ambiguous
- Explain statistical concepts in accessible language
- Provide actionable recommendations based on results
- When sample sizes are inadequate, clearly communicate this limitation
- Calculate total effect size as: average significant effect x number of treatment customers in significant segments

## Statistical Measures You Report:
- Sample sizes (treatment and control)
- Mean values and their difference (effect size)
- Cohen's d (standardized effect size)
- p-value and statistical significance
- 95% confidence intervals
- Statistical power
- Required sample size for adequate power
- Recommendations for action

Be conversational, helpful, and thorough in your analysis."""

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
