"""Unit tests for ABTestingAgent orchestration without live OpenAI calls."""

import pytest
from langchain_core.messages import AIMessage

import src.agent as agent_module
from src.agent import ABTestingAgent
from src.statistics.data_manager import DataQueryError


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
        self.df = object()
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
