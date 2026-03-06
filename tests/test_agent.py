"""Unit tests for ABTestingAgent orchestration without live OpenAI calls."""

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

import src.agent as agent_module
from src.agent import ABTestingAgent
from src.agent_reporting import AgentUserFacingError
from src.statistics.data_manager import DataQueryError
from src.statistics.models import ABTestSummary


class _DummyGraphAgent:
    def invoke(self, _payload):
        return {"messages": [AIMessage(content="dummy response")]}

    async def ainvoke(self, _payload):
        return {"messages": [AIMessage(content="dummy async response")]}


class _FailingGraphAgent:
    def invoke(self, _payload):
        raise RuntimeError("boom")

    async def ainvoke(self, _payload):
        raise RuntimeError("boom")


class _FakeAnalyzer:
    def __init__(self):
        self.df = None
        self.column_mapping = {}
        self.treatment_label = None
        self.control_label = None
        self.load_calls = []

    def load_data(self, filepath, **kwargs):
        self.load_calls.append((filepath, kwargs))
        self.df = pd.DataFrame(
            {
                "customer_id": ["A", "B"],
                "experiment_group": ["treatment", "control"],
                "post_effect": [1.2, 1.0],
                "segment": ["Premium", "Premium"],
            }
        )
        return {
            "columns": ["customer_id", "experiment_group", "post_effect", "segment"],
            "shape": (100, 4),
        }

    def detect_columns(self):
        return {
            "customer_id": ["customer_id"],
            "group": ["experiment_group"],
            "pre_effect": [],
            "post_effect": ["post_effect"],
            "effect_value": ["post_effect"],
            "segment": ["segment"],
            "duration": [],
        }

    def auto_configure(self):
        self.column_mapping = {
            "group": "experiment_group",
            "effect_value": "post_effect",
            "post_effect": "post_effect",
            "segment": "segment",
        }
        self.treatment_label = "treatment"
        self.control_label = "control"
        return {
            "success": True,
            "warnings": [],
            "mapping": self.column_mapping,
            "labels": {"treatment": "treatment", "control": "control"},
        }

    def run_segmented_analysis(self):
        return ["dummy"]

    def set_column_mapping(self, mapping):
        self.column_mapping.update(mapping)

    def get_group_values(self):
        return {"unique_values": ["treatment", "control"]}

    def set_group_labels(self, treatment_label, control_label):
        self.treatment_label = treatment_label
        self.control_label = control_label

    def generate_summary(self, _results):
        return {
            "total_segments_analyzed": 1,
            "aa_test_passed_segments": 1,
            "aa_test_failed_segments": 0,
            "bootstrapped_segments": 0,
            "t_test_significant_segments": 1,
            "t_test_significance_rate": 1.0,
            "prop_test_significant_segments": 1,
            "prop_test_significance_rate": 1.0,
            "bayesian_significant_segments": 1,
            "bayesian_significance_rate": 1.0,
            "total_treatment_customers": 50,
            "total_control_customers": 50,
            "did_avg_effect": 1.2,
            "did_total_effect": 60.0,
            "t_test_effect_calculation": "1.2 x 50",
            "prop_test_effect_calculation": "0.1 x 50",
            "combined_effect_calculation": "1.2 x 50 + 0.1 x 50",
            "bayesian_total_effect": 62.0,
            "bayesian_avg_prob_treatment_better": 0.97,
            "bayesian_avg_expected_loss": 0.01,
            "detailed_results": [
                {
                    "segment": "Overall",
                    "treatment_n": 50,
                    "control_n": 50,
                    "p_value": 0.01,
                    "effect": 1.2,
                    "power": 0.91,
                    "adequate_sample": True,
                    "prop_p_value": 0.02,
                    "prop_effect_per_customer": 0.1,
                    "significant": True,
                    "prop_significant": True,
                    "aa_test_passed": True,
                    "bootstrapping_applied": False,
                    "treatment_pre_mean": 10.0,
                    "control_pre_mean": 10.0,
                    "treatment_post_mean": 11.2,
                    "control_post_mean": 10.0,
                    "did_effect": 1.2,
                    "bayesian_prob": 0.97,
                    "bayesian_credible_lower": 0.5,
                    "bayesian_credible_upper": 1.8,
                    "bayesian_expected_loss": 0.01,
                    "bayesian_total_effect": 62.0,
                    "bayesian_significant": True,
                }
            ],
            "recommendations": ["Proceed with treatment rollout."],
        }

    def query_data(self, query):
        raise DataQueryError(
            "INVALID_QUERY",
            "Invalid query syntax. Use pandas query syntax, e.g. segment == 'Premium'.",
            query=query,
        )


class _TrackingAnalyzer(_FakeAnalyzer):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.df = object()
        self.column_mapping = {"group": "experiment_group", "effect_value": "post_effect"}
        self.treatment_label = "treatment"
        self.control_label = "control"
        self.run_segmented_analysis_calls = 0
        self.get_data_summary_calls = 0

    def run_segmented_analysis(self):
        self.run_segmented_analysis_calls += 1
        return ["dummy"]

    def generate_summary(self, _results):
        summary = super().generate_summary(_results)
        summary.update(
            {
                "significant_segments": 1,
                "non_significant_segments": 0,
                "significance_rate": 1.0,
                "average_significant_effect": 1.2,
                "total_treatment_in_significant_segments": 50,
                "total_effect_size": 60.0,
                "effect_calculation": "1.2 x 50",
                "segments_with_adequate_power": 1,
                "segments_with_inadequate_power": 0,
                "power_adequacy_rate": 1.0,
                "treatment_control_ratio": 1.0,
            }
        )
        return summary

    def get_data_summary(self):
        self.get_data_summary_calls += 1
        return {
            "shape": (10, 2),
            "dtypes": {"experiment_group": "object", "post_effect": "float64"},
            "missing_values": {"experiment_group": 0, "post_effect": 0},
            "numeric_summary": {
                "mean": {"post_effect": 1.0},
                "std": {"post_effect": 0.2},
                "min": {"post_effect": 0.5},
                "max": {"post_effect": 1.5},
            },
        }


class _BackendWithoutQuery(_TrackingAnalyzer):
    def query_data(self, query):
        raise AgentUserFacingError(
            "BACKEND_OPERATION_UNSUPPORTED",
            f"Querying data is not supported for the active backend ({self.name}).",
        )


class _FakeQueryStore:
    def __init__(self):
        self.raw_saved = False
        self.segment_results_saved = False
        self.summary_saved = False

    def save_raw_dataframe(self, _df):
        self.raw_saved = True

    def save_segment_results(self, _results):
        self.segment_results_saved = True

    def save_summary(self, _summary):
        self.summary_saved = True


class _FakeDataQuestionService:
    def __init__(self):
        self.questions = []

    def answer_question(self, question):
        self.questions.append(question)
        return {
            "answer_text": "The Premium total effect is 60.0.",
            "row_count": 1,
            "source_tables": ["analysis_segment_results"],
            "sql": "SELECT segment, total_effect FROM analysis_segment_results WHERE segment = 'Premium'",
            "data": None,
        }


@pytest.fixture
def stubbed_agent(monkeypatch):
    monkeypatch.setattr(agent_module, "ChatOpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(
        agent_module,
        "create_react_agent",
        lambda _llm, _tools, prompt=None: _DummyGraphAgent(),
    )
    return ABTestingAgent()


def _get_tool(agent: ABTestingAgent, name: str):
    return next(tool for tool in agent._create_tools() if tool.name == name)


def test_agent_run_tracks_history(stubbed_agent):
    response = stubbed_agent.run("hello")

    assert response == "dummy response"
    assert len(stubbed_agent.chat_history) == 2


def test_agent_has_expected_tools(stubbed_agent):
    tools = stubbed_agent._create_tools()
    names = {tool.name for tool in tools}

    expected = {
        "load_csv",
        "load_and_auto_analyze",
        "configure_and_analyze",
        "auto_configure_and_analyze",
        "generate_charts",
    }
    assert expected.issubset(names)


def test_clear_memory(stubbed_agent):
    stubbed_agent.run("hello")
    stubbed_agent.clear_memory()

    assert stubbed_agent.chat_history == []


def test_load_csv_uses_spark_for_large_files(stubbed_agent, monkeypatch, tmp_path):
    fake_spark_analyzer = _FakeAnalyzer()
    fake_pandas_analyzer = _FakeAnalyzer()
    stubbed_agent.analyzer = fake_pandas_analyzer

    monkeypatch.setattr(stubbed_agent, "_get_file_size_mb", lambda _filepath: 10.0)
    monkeypatch.setattr(agent_module, "PYSPARK_AVAILABLE", True)
    monkeypatch.setattr(agent_module, "PySparkABTestAnalyzer", lambda: fake_spark_analyzer)

    csv_path = tmp_path / "large.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    load_csv = _get_tool(stubbed_agent, "load_csv")
    result = load_csv.func(str(csv_path))

    assert "Using PySpark for distributed processing" in result
    assert stubbed_agent._using_spark is True
    assert len(fake_spark_analyzer.load_calls) == 1
    assert len(fake_pandas_analyzer.load_calls) == 0


def test_load_csv_falls_back_to_pandas_when_spark_init_fails(
    stubbed_agent, monkeypatch, tmp_path
):
    fake_pandas_analyzer = _FakeAnalyzer()
    stubbed_agent.analyzer = fake_pandas_analyzer

    monkeypatch.setattr(stubbed_agent, "_get_file_size_mb", lambda _filepath: 10.0)
    monkeypatch.setattr(agent_module, "PYSPARK_AVAILABLE", True)

    def _raise_spark_init_error():
        raise RuntimeError("Java gateway failed")

    monkeypatch.setattr(agent_module, "PySparkABTestAnalyzer", _raise_spark_init_error)

    csv_path = tmp_path / "large.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    load_csv = _get_tool(stubbed_agent, "load_csv")
    result = load_csv.func(str(csv_path))

    assert "Using pandas for in-memory processing" in result
    assert "PySpark initialization failed" in result
    assert stubbed_agent._using_spark is False
    assert len(fake_pandas_analyzer.load_calls) == 1


def test_load_and_auto_analyze_reports_actual_backend_after_fallback(
    stubbed_agent, monkeypatch, tmp_path
):
    fake_pandas_analyzer = _FakeAnalyzer()
    stubbed_agent.analyzer = fake_pandas_analyzer

    monkeypatch.setattr(stubbed_agent, "_get_file_size_mb", lambda _filepath: 10.0)
    monkeypatch.setattr(agent_module, "PYSPARK_AVAILABLE", True)

    def _raise_spark_init_error():
        raise RuntimeError("Java gateway failed")

    monkeypatch.setattr(agent_module, "PySparkABTestAnalyzer", _raise_spark_init_error)

    csv_path = tmp_path / "large.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    load_and_auto_analyze = _get_tool(stubbed_agent, "load_and_auto_analyze")
    result = load_and_auto_analyze.func(str(csv_path))

    assert "**Backend:** pandas (in-memory processing)" in result
    assert "PySpark initialization failed" in result
    assert stubbed_agent._using_spark is False


def test_load_and_auto_analyze_persists_query_state(stubbed_agent, tmp_path):
    fake_analyzer = _FakeAnalyzer()
    stubbed_agent.analyzer = fake_analyzer
    stubbed_agent.query_store = _FakeQueryStore()

    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    load_and_auto_analyze = _get_tool(stubbed_agent, "load_and_auto_analyze")
    load_and_auto_analyze.func(str(csv_path))

    assert stubbed_agent.query_store.raw_saved is True
    assert stubbed_agent.query_store.segment_results_saved is True
    assert stubbed_agent.query_store.summary_saved is True


def test_load_csv_missing_file_returns_structured_error(stubbed_agent):
    load_csv = _get_tool(stubbed_agent, "load_csv")
    result = load_csv.func("/tmp/this/file/does/not/exist.csv")

    assert result.startswith("Error loading file:")
    assert "[error_code=FILE_NOT_FOUND]" in result


def test_load_csv_failure_is_logged(stubbed_agent, caplog):
    load_csv = _get_tool(stubbed_agent, "load_csv")

    with caplog.at_level("ERROR"):
        load_csv.func("/tmp/this/file/does/not/exist.csv")

    assert any("Tool failure: load_csv" in record.message for record in caplog.records)


def test_configure_and_analyze_still_returns_markdown_report(stubbed_agent):
    fake_analyzer = _FakeAnalyzer()
    stubbed_agent.analyzer = fake_analyzer

    configure_and_analyze = _get_tool(stubbed_agent, "configure_and_analyze")
    result = configure_and_analyze.func(
        group_column="experiment_group",
        effect_column="post_effect",
        treatment_label="treatment",
        control_label="control",
        segment_column="segment",
    )

    assert "## Configuration Applied" in result
    assert "| Group Column | `experiment_group` |" in result
    assert "## A/B Test Results" in result
    assert "Proceed with treatment rollout." in result
    assert fake_analyzer.column_mapping["group"] == "experiment_group"
    assert fake_analyzer.treatment_label == "treatment"
    assert fake_analyzer.control_label == "control"


def test_auto_configure_and_analyze_requires_loaded_data(stubbed_agent):
    fake_analyzer = _FakeAnalyzer()
    stubbed_agent.analyzer = fake_analyzer

    auto_configure_and_analyze = _get_tool(stubbed_agent, "auto_configure_and_analyze")
    result = auto_configure_and_analyze.func("")

    assert result == "No data loaded. Please load a CSV file first."


def test_query_data_tool_returns_structured_error_for_invalid_query(stubbed_agent):
    fake_analyzer = _FakeAnalyzer()
    fake_analyzer.df = object()
    stubbed_agent.analyzer = fake_analyzer

    query_data = _get_tool(stubbed_agent, "query_data")
    result = query_data.func("bad query")

    assert result.startswith("Error querying data:")
    assert "[error_code=INVALID_QUERY]" in result


def test_agent_run_failure_returns_structured_error(stubbed_agent):
    stubbed_agent.agent = _FailingGraphAgent()

    result = stubbed_agent.run("hello")

    assert result.startswith("Error processing request:")
    assert "[error_code=AGENT_EXECUTION_FAILED]" in result


def test_run_full_analysis_uses_active_backend(stubbed_agent):
    fake_pandas = _TrackingAnalyzer("pandas")
    fake_spark = _TrackingAnalyzer("spark")
    stubbed_agent.analyzer = fake_pandas
    stubbed_agent.spark_analyzer = fake_spark
    stubbed_agent._using_spark = True

    run_full_analysis = _get_tool(stubbed_agent, "run_full_analysis")
    result = run_full_analysis.func("")

    assert "FULL A/B TEST ANALYSIS SUMMARY" in result
    assert fake_spark.run_segmented_analysis_calls == 1
    assert fake_pandas.run_segmented_analysis_calls == 0


def test_get_data_summary_uses_active_backend(stubbed_agent):
    fake_pandas = _TrackingAnalyzer("pandas")
    fake_spark = _TrackingAnalyzer("spark")
    stubbed_agent.analyzer = fake_pandas
    stubbed_agent.spark_analyzer = fake_spark
    stubbed_agent._using_spark = True

    get_data_summary = _get_tool(stubbed_agent, "get_data_summary")
    result = get_data_summary.func("")

    assert "DATA SUMMARY" in result
    assert fake_spark.get_data_summary_calls == 1
    assert fake_pandas.get_data_summary_calls == 0


def test_generate_charts_uses_active_backend_state(stubbed_agent):
    fake_pandas = _TrackingAnalyzer("pandas")
    fake_pandas.df = None
    fake_pandas.column_mapping = {}
    fake_pandas.treatment_label = None

    fake_spark = _TrackingAnalyzer("spark")
    fake_spark.column_mapping = {
        "group": "experiment_group",
        "effect_value": "post_effect",
    }
    stubbed_agent.analyzer = fake_pandas
    stubbed_agent.spark_analyzer = fake_spark
    stubbed_agent._using_spark = True
    stubbed_agent._last_results = ["dummy"]
    stubbed_agent._last_summary = fake_spark.generate_summary(["dummy"])

    class _FakeVisualizer:
        def plot_statistical_summary(self, _results):
            return "figure"

    stubbed_agent.visualizer = _FakeVisualizer()

    generate_charts = _get_tool(stubbed_agent, "generate_charts")
    result = generate_charts.func("summary")

    assert "Generated 1 chart(s): statistical_summary" in result
    assert stubbed_agent._last_charts["statistical_summary"] == "figure"


def test_generate_charts_accepts_typed_summary_state(stubbed_agent):
    fake_analyzer = _TrackingAnalyzer("pandas")
    stubbed_agent.analyzer = fake_analyzer
    stubbed_agent._using_spark = False
    stubbed_agent._last_results = ["dummy"]
    stubbed_agent._last_summary = ABTestSummary(total_segments_analyzed=1)

    class _FakeVisualizer:
        def plot_statistical_summary(self, _results):
            return "figure"

    stubbed_agent.visualizer = _FakeVisualizer()

    generate_charts = _get_tool(stubbed_agent, "generate_charts")
    result = generate_charts.func("summary")

    assert "Generated 1 chart(s): statistical_summary" in result
    assert stubbed_agent._last_charts["statistical_summary"] == "figure"


def test_query_data_reports_unsupported_active_backend(stubbed_agent):
    fake_pandas = _TrackingAnalyzer("pandas")
    fake_pandas.df = None
    fake_spark = _BackendWithoutQuery("spark")
    stubbed_agent.analyzer = fake_pandas
    stubbed_agent.spark_analyzer = fake_spark
    stubbed_agent._using_spark = True

    query_data = _get_tool(stubbed_agent, "query_data")
    result = query_data.func("experiment_group == 'treatment'")

    assert result.startswith("Error querying data:")
    assert "[error_code=BACKEND_OPERATION_UNSUPPORTED]" in result


def test_answer_data_question_requires_loaded_data(stubbed_agent):
    answer_data_question = _get_tool(stubbed_agent, "answer_data_question")

    result = answer_data_question.func("What is the total effect size for Premium?")

    assert result == "No data loaded. Please load a CSV file first."


def test_answer_data_question_uses_query_service(stubbed_agent):
    fake_analyzer = _TrackingAnalyzer("pandas")
    fake_service = _FakeDataQuestionService()
    stubbed_agent.analyzer = fake_analyzer
    stubbed_agent.data_question_service = fake_service

    answer_data_question = _get_tool(stubbed_agent, "answer_data_question")
    result = answer_data_question.func("What is the total effect size for Premium?")

    assert "The Premium total effect is 60.0." in result
    assert "analysis_segment_results" in result
    assert fake_service.questions == ["What is the total effect size for Premium?"]
