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
