"""
LangChain A/B Testing Agent

An intelligent agent that can:
- Load and analyze CSV data
- Perform comprehensive A/B testing
- Answer questions about the data
- Provide statistical insights and recommendations
- Generate interactive visualizations
"""

import asyncio
import logging
from typing import List, Any, Dict, Tuple
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
import plotly.graph_objects as go

from .agent_reporting import render_tool_error
from .agent_runtime import AgentRuntime
from .agent_session import AgentAnalysisSession
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

    @staticmethod
    def _create_spark_backend():
        """Create the optional Spark backend using the current module import state."""
        if PySparkABTestAnalyzer is None:
            raise RuntimeError("PySpark is not available in this environment")
        return PySparkABTestAnalyzer()

    def __init__(self, model_name: str = "gpt-5.2", temperature: float = 0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.runtime = AgentRuntime(
            analyzer=ABTestAnalyzer(),
            spark_factory=self._create_spark_backend,
            spark_available=lambda: PYSPARK_AVAILABLE,
            file_size_threshold_mb=2.0,
        )
        self.visualizer = ABTestVisualizer()
        self.session = AgentAnalysisSession(llm=self.llm)
        self.agent = self._create_agent()
        self._pending_confirmation = None
        logger.info(
            "ABTestingAgent initialized (model=%s, temperature=%s, spark_available=%s)",
            model_name,
            temperature,
            PYSPARK_AVAILABLE,
        )

    @property
    def analyzer(self) -> Any:
        return self.runtime.analyzer

    @analyzer.setter
    def analyzer(self, value: Any) -> None:
        self.runtime.analyzer = value

    @property
    def spark_analyzer(self) -> Any:
        return self.runtime.spark_analyzer

    @spark_analyzer.setter
    def spark_analyzer(self, value: Any) -> None:
        self.runtime.spark_analyzer = value

    @property
    def _using_spark(self) -> bool:
        return self.runtime.using_spark

    @_using_spark.setter
    def _using_spark(self, value: bool) -> None:
        self.runtime.using_spark = value

    @property
    def FILE_SIZE_THRESHOLD_MB(self) -> float:
        return self.runtime.file_size_threshold_mb

    @FILE_SIZE_THRESHOLD_MB.setter
    def FILE_SIZE_THRESHOLD_MB(self, value: float) -> None:
        self.runtime.file_size_threshold_mb = value

    @property
    def chat_history(self) -> List[Any]:
        return self.session.state.chat_history

    @chat_history.setter
    def chat_history(self, value: List[Any]) -> None:
        self.session.state.chat_history = value

    @property
    def _last_charts(self) -> Dict[str, go.Figure]:
        return self.session.state.last_charts

    @_last_charts.setter
    def _last_charts(self, value: Dict[str, go.Figure]) -> None:
        self.session.state.last_charts = value

    @property
    def _last_results(self) -> Any:
        return self.session.state.last_results

    @_last_results.setter
    def _last_results(self, value: Any) -> None:
        self.session.state.last_results = value

    @property
    def _last_summary(self) -> Any:
        return self.session.state.last_summary

    @_last_summary.setter
    def _last_summary(self, value: Any) -> None:
        self.session.state.last_summary = value

    @property
    def query_store(self) -> Any:
        return self.session.query_store

    @query_store.setter
    def query_store(self, value: Any) -> None:
        self.session.query_store = value
        if getattr(self.session.data_question_service, "query_store", None) is not None:
            self.session.data_question_service.query_store = value

    @property
    def data_question_service(self) -> Any:
        return self.session.data_question_service

    @data_question_service.setter
    def data_question_service(self, value: Any) -> None:
        self.session.data_question_service = value

    def persist_loaded_data(self, analyzer: Any) -> bool:
        """Persist the currently loaded raw dataframe to the session query store."""
        try:
            persisted = self.session.persist_loaded_data(analyzer)
        except Exception:
            logger.exception("Failed to persist raw dataframe to SQLite query store")
            return False

        if not persisted:
            logger.info("Skipping raw-data persistence for non-pandas backend")
            return False

        logger.info("Persisted raw dataframe to SQLite query store")
        return True

    def persist_analysis_outputs(self, results: Any, summary: Any) -> None:
        """Persist analysis outputs to the session query store."""
        try:
            self.session.persist_analysis_outputs(results, summary)
        except Exception:
            logger.exception("Failed to persist analysis outputs to SQLite query store")
            return

        logger.info("Persisted analysis outputs to SQLite query store")

    def get_charts(self) -> Dict[str, go.Figure]:
        """Get the last generated charts"""
        return self._last_charts

    def clear_charts(self):
        """Clear the stored charts"""
        self.session.state.last_charts = {}

    def _get_file_size_mb(self, filepath: str) -> float:
        """Get file size in megabytes."""
        return self.runtime.get_file_size_mb(filepath)

    def _should_use_spark(self, filepath: str) -> bool:
        """Determine if PySpark should be used based on file size."""
        return self.runtime.should_use_spark(filepath)

    def _get_active_analyzer(self):
        """Get the currently active analyzer (pandas or PySpark)."""
        return self.runtime.get_active_analyzer()

    def _init_spark_analyzer(self):
        """Initialize the Spark analyzer lazily."""
        return self.runtime.init_spark_analyzer()

    def _normalize_shape(self, info: Dict[str, Any]) -> Tuple[int, int]:
        """Normalize load_data metadata to (rows, columns)."""
        return self.runtime.normalize_shape(info)

    def _load_data_with_backend(self, filepath: str):
        """
        Load data using Spark when appropriate, with automatic pandas fallback.

        Returns:
            (analyzer, info, backend_name, file_size_mb, spark_selected, fallback_note)
        """
        return self.runtime.load_data_with_backend(filepath)

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
4. **Answer questions about loaded data and computed results** - Use `answer_data_question` for questions like counts, segment totals, effect sizes, or other lookups after data has been loaded
5. **Plan sample sizes BEFORE data exists** - Use `plan_sample_size` for "how many users do I need to detect a 5% lift at 80% power?" style questions. Pass JSON with `metric_type`, `mde`, and either `baseline_rate` (proportion) or `baseline_mean`+`baseline_std` (continuous).

## Workflow Decision Tree:
1. User uploads file with "best guess"/"auto" keywords => `load_and_auto_analyze`
2. User uploads file without keywords => `load_csv`, then ask to confirm, then `configure_and_analyze`
3. Data already loaded + user wants best guess => `auto_configure_and_analyze`
4. User asks a factual question about the loaded dataset or computed results => use `answer_data_question`

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

    def run(self, message: str) -> str:
        """Run the agent synchronously.

        Chainlit wraps this with ``cl.make_async(agent.run)`` for async dispatch.
        """
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

    async def arun(self, message: str) -> str:
        """Run the agent asynchronously.

        Thin wrapper kept for backward compatibility with notebooks, async
        tests, and background workers that call ``await agent.arun(...)``.
        """
        return await asyncio.to_thread(self.run, message)

    def clear_memory(self):
        """Clear conversation memory"""
        self.session.state.clear_chat_history()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    agent = ABTestingAgent()
    logger.info("A/B Testing Agent CLI initialized. Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        response = agent.run(user_input)
        print(f"\nAgent: {response}")
