"""Integration tests for the pandas analyzer facade."""

from pathlib import Path

from src.statistics import ABTestAnalyzer


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_ab_data.csv"


def _configure(analyzer: ABTestAnalyzer) -> None:
    config = analyzer.auto_configure()
    assert config["success"] is True


def test_ab_analyzer_end_to_end() -> None:
    analyzer = ABTestAnalyzer()
    info = analyzer.load_data(str(DATA_PATH))

    assert info["shape"][0] > 0
    assert "experiment_group" in info["columns"]

    _configure(analyzer)

    results = analyzer.run_segmented_analysis()
    summary = analyzer.generate_summary(results)

    assert len(results) > 0
    assert summary["total_segments_analyzed"] == len(results)
    assert "recommendations" in summary
    assert isinstance(summary["recommendations"], list)


def test_data_summary_and_distribution() -> None:
    analyzer = ABTestAnalyzer()
    analyzer.load_data(str(DATA_PATH))
    _configure(analyzer)

    data_summary = analyzer.get_data_summary()
    distribution = analyzer.get_segment_distribution()

    assert "shape" in data_summary
    assert "columns" in data_summary
    assert "group_distribution" in distribution
