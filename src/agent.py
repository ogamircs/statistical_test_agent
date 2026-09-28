"""
LangChain A/B Testing Agent

An intelligent agent that can:
- Load and analyze CSV data
- Perform comprehensive A/B testing
- Answer questions about the data
- Provide statistical insights and recommendations
- Generate interactive visualizations
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import openai
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

from .agent_reporting import AgentUserFacingError, render_tool_error
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

load_dotenv()


@wrap_model_call
def _sequential_tool_calls(request: ModelRequest, handler) -> ModelResponse:
    """Ask the model for one tool call at a time.

    Tools share mutable analyzer state (column mapping, labels, last results),
    so parallel calls race: e.g. generate_charts reading stale results while
    run_full_analysis is still re-running. Set per agent model call only; the
    SQL planner reuses the same LLM without tools, where OpenAI rejects it.
    """
    settings = {**request.model_settings, "parallel_tool_calls": False}
    return handler(request.override(model_settings=settings))
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
            timeout=self.config.llm_request_timeout_seconds,
            max_retries=self.config.llm_max_retries,
            callbacks=[self.token_usage],
        )
        self.runtime = AgentRuntime(analyzer=ABTestAnalyzer())
        self.visualizer = ABTestVisualizer()
        session_kwargs: dict[str, Any] = {
            "llm": self.llm,
            "query_timeout_seconds": self.config.query_timeout_seconds,
            "sql_default_row_limit": self.config.sql_default_row_limit,
        }
        session_kwargs["query_store_path"] = (
            query_store_path
            if query_store_path is not None
            else Path(self.config.query_store_dir) / f"session-{uuid4().hex}.sqlite"
        )
        self.session = AgentAnalysisSession(**session_kwargs)
        self._restore_chat_history_from_store()
        # Analysis state from a previous process, rebuilt lazily on first use
        # (TODO.md #104) so opening or listing a session stays cheap.
        self._pending_analysis_state: Optional[Dict[str, Any]] = self._load_analysis_state()
        self.agent = self._create_agent()
        self._pending_confirmation = None
        logger.info(
            "ABTestingAgent initialized (model=%s, temperature=%s, restored_history=%d)",
            resolved_model,
            resolved_temperature,
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

    # -- restart recovery (TODO.md #104) -------------------------------------

    _ANALYSIS_STATE_KEY = "analysis"

    def _load_analysis_state(self) -> Optional[Dict[str, Any]]:
        try:
            state = self.session.query_store.load_state(self._ANALYSIS_STATE_KEY)
        except Exception:
            logger.exception("Failed to read persisted analysis state; chat-only restore")
            return None
        if not isinstance(state, dict) or not isinstance(state.get("column_mapping"), dict):
            return None
        return state

    def _persist_analysis_state(self) -> None:
        analyzer: Any = self.runtime.analyzer
        mapping = getattr(analyzer, "column_mapping", None)
        if not mapping or getattr(analyzer, "treatment_label", None) is None:
            return
        self.session.query_store.save_state(
            self._ANALYSIS_STATE_KEY,
            {
                "column_mapping": dict(mapping),
                "treatment_label": analyzer.treatment_label,
                "control_label": analyzer.control_label,
            },
        )

    def _ensure_analysis_restored(self) -> None:
        """Rebuild data, mapping, labels and results saved by a previous process."""
        state = self._pending_analysis_state
        if state is None:
            return
        self._pending_analysis_state = None  # one attempt, even if it fails
        try:
            df = self.session.query_store.load_raw_dataframe()
            if df is None:
                return
            analyzer: Any = self.runtime.analyzer
            analyzer.set_dataframe(df)
            analyzer.set_column_mapping(dict(state["column_mapping"]))
            analyzer.set_group_labels(state["treatment_label"], state["control_label"])
            results = analyzer.run_segmented_analysis()
            self.session.state.last_results = results
            self.session.state.last_summary = analyzer.generate_summary(results)
            logger.info("Restored analysis state after restart (segments=%d)", len(results))
        except Exception:
            logger.exception("Failed to restore analysis state; continuing chat-only")

    @property
    def analyzer(self) -> ABAnalyzerProtocol:
        return self.runtime.analyzer

    @analyzer.setter
    def analyzer(self, value: ABAnalyzerProtocol) -> None:
        self.runtime.analyzer = value

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
        self._ensure_analysis_restored()
        return self.session.state.last_results

    @_last_results.setter
    def _last_results(self, value: Optional[List[ABTestResult]]) -> None:
        self._pending_analysis_state = None  # fresh results supersede a restore
        self.session.state.last_results = value

    @property
    def _last_summary(self) -> Optional[ABTestSummary]:
        self._ensure_analysis_restored()
        return self.session.state.last_summary

    @_last_summary.setter
    def _last_summary(self, value: Optional[ABTestSummary]) -> None:
        self._pending_analysis_state = None
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
        """Persist analysis outputs (and the state to rebuild them) to the query store."""
        try:
            self.session.persist_analysis_outputs(results, summary)
        except Exception:
            logger.exception("Failed to persist analysis outputs to SQLite query store")
        try:
            self._persist_analysis_state()
        except Exception:
            logger.exception("Failed to persist analysis state; restart recovery will be chat-only")
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

    def _get_active_analyzer(self):
        """Get the active analyzer."""
        self._ensure_analysis_restored()
        return self.runtime.get_active_analyzer()

    def _normalize_shape(self, info: Dict[str, Any]) -> Tuple[int, int]:
        """Normalize load_data metadata to (rows, columns)."""
        return self.runtime.normalize_shape(info)

    def _load_data(self, filepath: str):
        """Load a CSV into the pandas analyzer.

        Returns:
            (analyzer, info, file_size_mb)
        """
        # New data supersedes anything a restart would have restored.
        self._pending_analysis_state = None
        return self.runtime.load_data(filepath)

    def _create_tools(self) -> List[Any]:
        """Create the tools for the agent."""
        return create_agent_tools(self)

    def _create_agent(self):
        """Create the LangGraph agent with tools"""

        tools = self._create_tools()
        system_prompt = load_system_prompt()
        logger.info("System prompt loaded (version=%s)", PROMPT_VERSION)
        return create_agent(
            self.llm, tools, system_prompt=system_prompt, middleware=[_sequential_tool_calls]
        )

    def _model_bound_history(self) -> List[BaseMessage]:
        """Return the most recent slice of chat history to send to the model.

        The full history stays in session state and SQLite; only the window
        re-sent each turn is bounded so long sessions don't grow cost
        linearly or overflow the context window. The window always starts on
        a human turn so the model never sees a dangling AI reply.
        """
        history = self.chat_history
        limit = self.config.max_history_messages
        if len(history) <= limit:
            return list(history)
        window = history[-limit:]
        for index, message in enumerate(window):
            if isinstance(message, HumanMessage):
                return list(window[index:])
        return list(history[-1:])

    @staticmethod
    def _classify_run_error(error: Exception) -> Exception:
        """Map LLM/graph failures to user-facing errors with distinct codes."""
        if isinstance(error, GraphRecursionError):
            return AgentUserFacingError(
                "AGENT_STEP_LIMIT_REACHED",
                "The request needed more reasoning/tool steps than allowed and was "
                "stopped. Try a narrower question or break it into smaller steps.",
            )
        if isinstance(error, openai.RateLimitError):
            return AgentUserFacingError(
                "LLM_RATE_LIMITED",
                "The language model provider is rate-limiting requests or the API "
                "quota is exhausted. Please wait a moment and try again.",
            )
        if isinstance(error, openai.APITimeoutError):
            return AgentUserFacingError(
                "LLM_TIMEOUT",
                "The language model did not respond in time. Please try again.",
            )
        if isinstance(error, openai.AuthenticationError):
            return AgentUserFacingError(
                "LLM_AUTH_FAILED",
                "The language model API key was rejected. Check OPENAI_API_KEY.",
            )
        if isinstance(error, openai.APIConnectionError):
            return AgentUserFacingError(
                "LLM_UNAVAILABLE",
                "Could not reach the language model provider. Please try again.",
            )
        if isinstance(error, openai.InternalServerError):
            return AgentUserFacingError(
                "LLM_UNAVAILABLE",
                "The language model provider returned a server error. Please try again.",
            )
        return error

    def run(self, message: str, callbacks: Optional[Sequence[Any]] = None) -> str:
        """Run the agent synchronously.

        The web API runs this in a worker thread. ``callbacks`` are extra
        LangChain callback handlers for this run only (the API uses one to
        stream tool progress to the browser).
        """
        try:
            self.token_usage.reset()
            logger.info("Agent run started (history_messages=%d)", len(self.chat_history))
            self.chat_history.append(HumanMessage(content=message))
            self.session.query_store.save_chat_message("human", message)
            run_config: Dict[str, Any] = {"recursion_limit": self.config.agent_recursion_limit}
            if callbacks:
                run_config["callbacks"] = list(callbacks)
            result = self.agent.invoke(
                {"messages": self._model_bound_history()},
                config=run_config,
            )
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
                self._classify_run_error(e),
                default_code="AGENT_EXECUTION_FAILED",
                default_message="Unable to process your request right now.",
            )

    def clear_memory(self):
        """Clear conversation memory (both in-memory and persisted SQLite).

        Without the persisted wipe, a reconnect would re-hydrate the old
        turns through _restore_chat_history_from_store and resurrect the
        conversation the user just asked to clear.
        """
        self.session.state.clear_chat_history()
        try:
            self.session.query_store.clear_chat_messages()
        except Exception:
            logger.exception("Failed to wipe persisted chat history")

