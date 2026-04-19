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
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .agent_reporting import render_tool_error
from .agent_runtime import AgentRuntime
from .agent_session import AgentAnalysisSession
from .agent_tools import create_agent_tools
from .config import Config
from .observability import TokenUsageCallback
from .prompts import PROMPT_VERSION, load_system_prompt
from .query_store import SQLiteQueryStore
from .statistics import ABTestAnalyzer, ABTestVisualizer
from .statistics.analyzer_protocol import ABAnalyzerProtocol
from .statistics.models import ABTestResult, ABTestSummary

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

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        config: Config | None = None,
        query_store_path: str | None = None,
    ):
        self.config = config or Config.from_env()
        resolved_model = model_name if model_name is not None else self.config.llm_model
        resolved_temperature = (
            temperature if temperature is not None else self.config.llm_temperature
        )

        self.token_usage = TokenUsageCallback()
        self.llm = ChatOpenAI(
            model=resolved_model,
            temperature=resolved_temperature,
            callbacks=[self.token_usage],
        )
        self.runtime = AgentRuntime(
            analyzer=ABTestAnalyzer(),
            spark_factory=self._create_spark_backend,
            spark_available=lambda: PYSPARK_AVAILABLE,
            file_size_threshold_mb=self.config.file_size_threshold_mb,
        )
        self.visualizer = ABTestVisualizer()
        session_kwargs: dict[str, Any] = {"llm": self.llm}
        if query_store_path is not None:
            session_kwargs["query_store_path"] = query_store_path
        self.session = AgentAnalysisSession(**session_kwargs)
        self._restore_chat_history_from_store()
        self.agent = self._create_agent()
        self._pending_confirmation = None
        logger.info(
            "ABTestingAgent initialized (model=%s, temperature=%s, spark_available=%s, "
            "file_size_threshold_mb=%.2f, restored_history=%d)",
            resolved_model,
            resolved_temperature,
            PYSPARK_AVAILABLE,
            self.config.file_size_threshold_mb,
            len(self.session.state.chat_history),
        )

    def _restore_chat_history_from_store(self) -> None:
        try:
            persisted = self.session.query_store.load_chat_messages()
        except Exception:
            logger.exception("Failed to load persisted chat history; starting fresh")
            return
        for entry in persisted:
            role = entry.get("role")
            content = entry.get("content", "")
            if role == "human":
                self.session.state.chat_history.append(HumanMessage(content=content))
            elif role == "ai":
                self.session.state.chat_history.append(AIMessage(content=content))

    @property
    def analyzer(self) -> ABAnalyzerProtocol:
        return self.runtime.analyzer

    @analyzer.setter
    def analyzer(self, value: ABAnalyzerProtocol) -> None:
        self.runtime.analyzer = value

    @property
    def spark_analyzer(self) -> Optional[ABAnalyzerProtocol]:
        return self.runtime.spark_analyzer

    @spark_analyzer.setter
    def spark_analyzer(self, value: Optional[ABAnalyzerProtocol]) -> None:
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
    def chat_history(self) -> List[BaseMessage]:
        return self.session.state.chat_history

    @chat_history.setter
    def chat_history(self, value: List[BaseMessage]) -> None:
        self.session.state.chat_history = value

    @property
    def _last_charts(self) -> Dict[str, go.Figure]:
        return self.session.state.last_charts

    @_last_charts.setter
    def _last_charts(self, value: Dict[str, go.Figure]) -> None:
        self.session.state.last_charts = value

    @property
    def _last_results(self) -> Optional[List[ABTestResult]]:
        return self.session.state.last_results

    @_last_results.setter
    def _last_results(self, value: Optional[List[ABTestResult]]) -> None:
        self.session.state.last_results = value

    @property
    def _last_summary(self) -> Optional[ABTestSummary]:
        return self.session.state.last_summary

    @_last_summary.setter
    def _last_summary(self, value: Optional[ABTestSummary]) -> None:
        self.session.state.last_summary = value

    @property
    def query_store(self) -> SQLiteQueryStore:
        return self.session.query_store

    @query_store.setter
    def query_store(self, value: SQLiteQueryStore) -> None:
        self.session.query_store = value
        if getattr(self.session.data_question_service, "query_store", None) is not None:
            self.session.data_question_service.query_store = value

    @property
    def data_question_service(self) -> Optional[Any]:
        return self.session.data_question_service

    @data_question_service.setter
    def data_question_service(self, value: Optional[Any]) -> None:
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
        system_prompt = load_system_prompt()
        logger.info("System prompt loaded (version=%s)", PROMPT_VERSION)
        return create_react_agent(self.llm, tools, prompt=system_prompt)

    def run(self, message: str) -> str:
        """Run the agent synchronously.

        Chainlit wraps this with ``cl.make_async(agent.run)`` for async dispatch.
        """
        try:
            self.token_usage.reset()
            logger.info("Agent run started (history_messages=%d)", len(self.chat_history))
            self.chat_history.append(HumanMessage(content=message))
            self.session.query_store.save_chat_message("human", message)
            result = self.agent.invoke({"messages": self.chat_history})
            response = result["messages"][-1].content
            self.chat_history.append(AIMessage(content=response))
            self.session.query_store.save_chat_message("ai", str(response))
            usage = self.token_usage.snapshot()
            logger.info(
                "Agent run completed (response_chars=%d, llm_calls=%d, "
                "prompt_tokens=%d, completion_tokens=%d, total_tokens=%d)",
                len(str(response)),
                usage["calls"],
                usage["prompt_tokens"],
                usage["completion_tokens"],
                usage["total_tokens"],
            )
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
        logger.info("Agent: %s", response)
