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
import logging
from typing import Optional, List, Any, Dict, Tuple
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
import plotly.graph_objects as go

from .agent_reporting import render_tool_error
from .agent_tools import create_agent_tools
from .statistics import ABTestAnalyzer, ABTestVisualizer

# Try to import PySpark analyzer (optional dependency)
try:
    from .statistics.pyspark_analyzer import PySparkABTestAnalyzer
    PYSPARK_AVAILABLE = True
except (ImportError, AttributeError):
    # ImportError: pyspark not installed
    # AttributeError: pyspark installed but not compatible (e.g., Windows)
    PYSPARK_AVAILABLE = False
    PySparkABTestAnalyzer = None

load_dotenv()
logger = logging.getLogger(__name__)


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

    def __init__(self, model_name: str = "gpt-5.2", temperature: float = 0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.analyzer = ABTestAnalyzer()
        self.spark_analyzer = None  # Will be initialized if needed for large files
        self.visualizer = ABTestVisualizer()
        self.chat_history: List[Any] = []
        self.agent = self._create_agent()
        self._pending_confirmation = None
        self._last_charts: Dict[str, go.Figure] = {}
        self._last_results = None
        self._last_summary = None
        self._using_spark = False  # Track which backend is in use
        self.FILE_SIZE_THRESHOLD_MB = 2  # Auto-switch to PySpark for files >2MB
        logger.info(
            "ABTestingAgent initialized (model=%s, temperature=%s, spark_available=%s)",
            model_name,
            temperature,
            PYSPARK_AVAILABLE,
        )

    def get_charts(self) -> Dict[str, go.Figure]:
        """Get the last generated charts"""
        return self._last_charts

    def clear_charts(self):
        """Clear the stored charts"""
        self._last_charts = {}

    def _get_file_size_mb(self, filepath: str) -> float:
        """Get file size in megabytes"""
        try:
            file_size = os.path.getsize(filepath)
            return file_size / (1024 * 1024)
        except Exception:
            return 0

    def _should_use_spark(self, filepath: str) -> bool:
        """Determine if PySpark should be used based on file size"""
        if not PYSPARK_AVAILABLE:
            return False
        file_size_mb = self._get_file_size_mb(filepath)
        should_use = file_size_mb > self.FILE_SIZE_THRESHOLD_MB
        logger.info(
            "Backend selection evaluated (file=%s, size_mb=%.2f, threshold_mb=%.2f, use_spark=%s)",
            filepath,
            file_size_mb,
            self.FILE_SIZE_THRESHOLD_MB,
            should_use,
        )
        return should_use

    def _get_active_analyzer(self):
        """Get the currently active analyzer (pandas or PySpark)"""
        if self._using_spark and self.spark_analyzer is not None:
            return self.spark_analyzer
        return self.analyzer

    def _init_spark_analyzer(self):
        """Initialize the Spark analyzer lazily."""
        if self.spark_analyzer is not None:
            return self.spark_analyzer
        if not PYSPARK_AVAILABLE or PySparkABTestAnalyzer is None:
            raise RuntimeError("PySpark is not available in this environment")
        logger.info("Initializing PySpark analyzer")
        self.spark_analyzer = PySparkABTestAnalyzer()
        return self.spark_analyzer

    def _normalize_shape(self, info: Dict[str, Any]) -> Tuple[int, int]:
        """Normalize load_data metadata to (rows, columns)."""
        shape = info.get("shape")
        if isinstance(shape, (tuple, list)) and len(shape) >= 2:
            return int(shape[0]), int(shape[1])

        row_count = info.get("row_count")
        columns = info.get("columns")
        if row_count is not None and columns is not None:
            return int(row_count), len(columns)

        raise KeyError("shape")

    def _load_data_with_backend(self, filepath: str):
        """
        Load data using Spark when appropriate, with automatic pandas fallback.

        Returns:
            (analyzer, info, backend_name, file_size_mb, spark_selected, fallback_note)
        """
        file_size_mb = self._get_file_size_mb(filepath)
        spark_selected = self._should_use_spark(filepath)
        fallback_note = None
        logger.info(
            "Starting data load (file=%s, size_mb=%.2f, spark_selected=%s)",
            filepath,
            file_size_mb,
            spark_selected,
        )

        if spark_selected:
            try:
                analyzer = self._init_spark_analyzer()
            except Exception as e:
                self._using_spark = False
                fallback_note = f"PySpark initialization failed: {e}. Falling back to pandas."
                logger.warning("PySpark initialization failed; using pandas fallback", exc_info=e)
            else:
                try:
                    info = analyzer.load_data(filepath, format="csv")
                    self._using_spark = True
                    logger.info("Data load completed with spark backend")
                    return analyzer, info, "spark", file_size_mb, spark_selected, fallback_note
                except Exception as e:
                    self._using_spark = False
                    fallback_note = f"PySpark backend failed while loading data: {e}. Falling back to pandas."
                    logger.warning("PySpark load failed; using pandas fallback", exc_info=e)

        analyzer = self.analyzer
        info = analyzer.load_data(filepath)
        self._using_spark = False
        logger.info("Data load completed with pandas backend")
        return analyzer, info, "pandas", file_size_mb, spark_selected, fallback_note

    def _create_tools(self) -> List[Any]:
        """Create the tools for the agent."""
        return create_agent_tools(self)

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
            logger.info("Async agent run started (history_messages=%d)", len(self.chat_history))
            self.chat_history.append(HumanMessage(content=message))
            result = await self.agent.ainvoke({"messages": self.chat_history})
            response = result["messages"][-1].content
            self.chat_history.append(AIMessage(content=response))
            logger.info("Async agent run completed (response_chars=%d)", len(str(response)))
            return response
        except Exception as e:
            logger.exception("Async agent run failed")
            return render_tool_error(
                "Error processing request",
                e,
                default_code="AGENT_EXECUTION_FAILED",
                default_message="Unable to process your request right now.",
            )

    def run(self, message: str) -> str:
        """Run the agent synchronously"""
        try:
            logger.info("Agent run started (history_messages=%d)", len(self.chat_history))
            self.chat_history.append(HumanMessage(content=message))
            result = self.agent.invoke({"messages": self.chat_history})
            response = result["messages"][-1].content
            self.chat_history.append(AIMessage(content=response))
            logger.info("Agent run completed (response_chars=%d)", len(str(response)))
            return response
        except Exception as e:
            logger.exception("Agent run failed")
            return render_tool_error(
                "Error processing request",
                e,
                default_code="AGENT_EXECUTION_FAILED",
                default_message="Unable to process your request right now.",
            )

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
