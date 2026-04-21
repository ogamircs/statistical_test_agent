"""End-to-end pipeline test: load -> configure -> analyze -> summarize -> chart -> persist."""

import os
import tempfile

import plotly.graph_objects as go
import pytest

from src.query_store import SQLiteQueryStore
from src.statistics.analyzer import ABTestAnalyzer
from src.statistics.models import ABTestResult, ABTestSummary
from src.statistics.visualizer import ABTestVisualizer

SAMPLE_DATA = os.path.join("data", "sample_ab_data.csv")


@pytest.fixture
def analyzer():
    a = ABTestAnalyzer()
    a.load_data(SAMPLE_DATA)
    a.auto_configure()
    return a


@pytest.mark.skipif(not os.path.exists(SAMPLE_DATA), reason="Sample data not generated")
class TestPipelineE2E:
    def test_load_and_configure(self, analyzer):
        assert analyzer.df is not None
        assert len(analyzer.df) > 0
        assert analyzer.column_mapping is not None
        # Verify essential columns were detected
        assert "group" in analyzer.column_mapping
        assert "post_effect" in analyzer.column_mapping or "effect" in analyzer.column_mapping

    def test_segmented_analysis_returns_results(self, analyzer):
        results = analyzer.run_segmented_analysis()
        assert len(results) > 0
        for r in results:
            assert isinstance(r, ABTestResult)
            assert r.treatment_size > 0
            assert r.control_size > 0
            assert r.segment  # non-empty segment name

    def test_generate_summary(self, analyzer):
        results = analyzer.run_segmented_analysis()
        summary = analyzer.generate_summary(results)
        assert isinstance(summary, ABTestSummary)
        assert summary.total_segments_analyzed > 0
        assert summary.total_segments_analyzed == len(results)
        assert isinstance(summary.recommendations, list)

    def test_chart_generation(self, analyzer):
        results = analyzer.run_segmented_analysis()
        summary = analyzer.generate_summary(results)
        vis = ABTestVisualizer()
        group_col = analyzer.column_mapping.get("group")
        segment_col = analyzer.column_mapping.get("segment")
        charts = vis.create_all_charts(
            results, summary,
            df=analyzer.df, group_col=group_col, segment_col=segment_col,
        )
        # The chart catalog defines 10+ chart types; with distribution that's 11+
        assert len(charts) >= 10, f"Expected at least 10 charts, got {len(charts)}: {list(charts.keys())}"
        for name, fig in charts.items():
            assert isinstance(fig, go.Figure), f"Chart '{name}' is not a Figure"
            assert len(fig.data) >= 1, f"Chart '{name}' has no traces"

    def test_persistence_round_trip(self, analyzer):
        results = analyzer.run_segmented_analysis()
        summary = analyzer.generate_summary(results)
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SQLiteQueryStore(db_path)
            store.save_raw_dataframe(analyzer.df)
            store.save_segment_results(results)
            store.save_summary(summary)
            # Query back and verify row counts
            count_df = store.execute_query("SELECT count(*) as cnt FROM raw_data")
            assert count_df.iloc[0]["cnt"] == len(analyzer.df)
            segments_df = store.execute_query(
                "SELECT count(*) as cnt FROM analysis_segment_results"
            )
            assert segments_df.iloc[0]["cnt"] == len(results)
            summary_df = store.execute_query(
                "SELECT count(*) as cnt FROM analysis_summary"
            )
            assert summary_df.iloc[0]["cnt"] == 1

    def test_full_pipeline_coherence(self, analyzer):
        """Verify that data flows coherently across all pipeline stages."""
        # Stage 1: Analyze
        results = analyzer.run_segmented_analysis()
        # Stage 2: Summarize
        summary = analyzer.generate_summary(results)
        # Stage 3: Chart
        vis = ABTestVisualizer()
        charts = vis.create_all_charts(results, summary)
        # Stage 4: Persist
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "coherence.db")
            store = SQLiteQueryStore(db_path)
            store.save_segment_results(results)
            store.save_summary(summary)
            # Cross-check: persisted segment count matches summary
            seg_df = store.execute_query(
                "SELECT count(*) as cnt FROM analysis_segment_results"
            )
            assert seg_df.iloc[0]["cnt"] == summary.total_segments_analyzed
        # Cross-check: charts were generated for analyzed segments
        assert len(charts) > 0
        assert summary.total_segments_analyzed == len(results)
