"""Visualization smoke tests for A/B charts."""

from pathlib import Path

from src.statistics import ABTestAnalyzer, ABTestVisualizer


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_ab_data.csv"


def _build_results():
    analyzer = ABTestAnalyzer()
    analyzer.load_data(str(DATA_PATH))
    config = analyzer.auto_configure()
    assert config["success"] is True

    results = analyzer.run_segmented_analysis()
    summary = analyzer.generate_summary(results)
    return analyzer, results, summary


def test_create_all_charts() -> None:
    analyzer, results, summary = _build_results()
    visualizer = ABTestVisualizer()

    charts = visualizer.create_all_charts(
        results,
        summary,
        df=analyzer.df,
        group_col=analyzer.column_mapping["group"],
        segment_col=analyzer.column_mapping.get("segment"),
    )

    assert len(charts) >= 10
    assert "statistical_summary" in charts
    assert "dashboard" in charts


def test_statistical_summary_chart_has_traces() -> None:
    _, results, _ = _build_results()
    visualizer = ABTestVisualizer()

    fig = visualizer.plot_statistical_summary(results)

    assert fig is not None
    assert len(fig.data) >= 3


def test_plot_statistical_summary_semantics() -> None:
    _, results, _ = _build_results()
    visualizer = ABTestVisualizer()

    fig = visualizer.plot_statistical_summary(results)

    assert fig.layout.title.text == "<b>Statistical Results Summary</b>"
    assert [trace.name for trace in fig.data] == [
        "T-test p-value",
        "Proportion p-value",
        "T-test effect",
        "Proportion effect",
        "Total effect",
    ]
    assert fig.layout.yaxis.title.text == "P-value"
    assert fig.layout.yaxis2.title.text == "Effect"
    assert fig.layout.showlegend is False
    assert fig.data[0].y[0] == results[0].p_value
    assert fig.data[2].y[0] == results[0].effect_size
    assert any(shape["y0"] == shape["y1"] == 0.05 for shape in fig.layout.shapes)
    assert any(shape["y0"] == shape["y1"] == 0 for shape in fig.layout.shapes)


def test_plot_summary_dashboard_semantics() -> None:
    _, results, summary = _build_results()
    visualizer = ABTestVisualizer()

    fig = visualizer.plot_summary_dashboard(results, summary)

    title = fig.layout.title.text
    assert "A/B Test Analysis Dashboard" in title
    assert "T-test sig:" in title
    assert "Prop sig:" in title
    assert "Combined effect:" in title
    assert {trace.name for trace in fig.data if trace.name} >= {
        "Treatment",
        "Control",
        "T-test Effect",
        "Proportion Effect",
    }
    assert {
        fig.layout.yaxis.title.text,
        fig.layout.yaxis2.title.text,
        fig.layout.yaxis3.title.text,
        fig.layout.yaxis4.title.text,
    } == {"Mean", "Total Effect", "Conv Rate %", "P-Value"}
    assert fig.data[0].y[0] == results[0].treatment_mean
    assert fig.data[2].y[0] == (
        results[0].effect_size * results[0].treatment_size
        if results[0].is_significant
        else 0
    )
    assert any(shape["y0"] == shape["y1"] == 0 for shape in fig.layout.shapes)
    assert any(shape["y0"] == shape["y1"] == 0.05 for shape in fig.layout.shapes)
