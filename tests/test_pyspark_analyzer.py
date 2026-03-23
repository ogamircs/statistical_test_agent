"""
Comprehensive Unit Tests for PySparkABTestAnalyzer

Tests distributed statistical analysis using PySpark including:
- Data loading from various formats
- Column detection using Spark schema
- Statistical aggregations via Spark DataFrame operations
- All statistical tests (t-test, proportion, AA, Bayesian, DiD)
- Segmented analysis with parallel processing
- Result persistence (Parquet, Delta)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

pytest.importorskip("pyspark", reason="PySpark not installed; skipping PySpark analyzer tests.")

from src.statistics.analyzer import ABTestAnalyzer
from src.statistics.models import canonical_result_as_dict
from src.statistics.pyspark_analyzer import (
    PySparkABTestAnalyzer,
    SparkABTestResult,
    create_spark_session,
)


def _create_spark_or_skip(**kwargs):
    try:
        return create_spark_session(**kwargs)
    except Exception as exc:
        pytest.skip(f"Spark runtime unavailable; skipping PySpark tests. Details: {exc}")


@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing"""
    spark = _create_spark_or_skip(
        app_name="ABTestAnalyzerTests",
        master="local[2]",
        config={
            "spark.sql.shuffle.partitions": "4",
            "spark.driver.memory": "1g"
        }
    )
    yield spark
    try:
        spark.stop()
    except Exception:
        pass


@pytest.fixture
def sample_pandas_df():
    """Create sample pandas DataFrame"""
    np.random.seed(42)
    n = 1000

    data = {
        'customer_id': [f'CUST_{i:04d}' for i in range(n)],
        'experiment_group': ['treatment' if i % 2 == 0 else 'control' for i in range(n)],
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n),
        'pre_effect': np.random.normal(50, 10, n),
        'post_effect': np.random.normal(55, 12, n),
        'experiment_duration_days': np.random.randint(7, 30, n)
    }

    df = pd.DataFrame(data)
    # Add treatment effect
    df.loc[df['experiment_group'] == 'treatment', 'post_effect'] += 3

    return df


@pytest.fixture
def sample_spark_df(spark, sample_pandas_df):
    """Convert pandas DataFrame to Spark DataFrame"""
    return spark.createDataFrame(sample_pandas_df)


@pytest.fixture
def temp_csv_path(tmp_path, sample_pandas_df):
    """Create temporary CSV file"""
    csv_path = tmp_path / "test_data.csv"
    sample_pandas_df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def temp_parquet_path(tmp_path, sample_pandas_df, spark):
    """Create temporary Parquet file using Spark to avoid pandas parquet engine deps."""
    parquet_path = tmp_path / "test_data.parquet"
    spark.createDataFrame(sample_pandas_df).write.mode("overwrite").parquet(str(parquet_path))
    return str(parquet_path)


@pytest.fixture
def analyzer(spark):
    """Create fresh analyzer instance"""
    return PySparkABTestAnalyzer(spark, significance_level=0.05, power_threshold=0.8)


class TestSparkSessionCreation:
    """Test Spark session creation and configuration"""

    def test_create_spark_session_default(self):
        """Test creating Spark session with defaults"""
        spark = _create_spark_or_skip()
        assert spark is not None
        assert spark.sparkContext.appName == "ABTestingAnalyzer"
        spark.stop()

    def test_create_spark_session_custom_config(self):
        """Test creating Spark session with custom config"""
        spark = _create_spark_or_skip(
            app_name="CustomTest",
            config={"spark.sql.shuffle.partitions": "8"}
        )
        assert spark is not None
        assert spark.conf.get("spark.sql.shuffle.partitions") == "8"
        spark.stop()


class TestDataLoading:
    """Test data loading from various sources"""

    def test_load_csv(self, analyzer, temp_csv_path):
        """Test loading CSV file"""
        info = analyzer.load_data(temp_csv_path, format="csv")

        assert 'columns' in info
        assert 'row_count' in info
        assert 'partitions' in info
        assert info['row_count'] == 1000
        assert len(info['columns']) == 6

    def test_load_parquet(self, analyzer, temp_parquet_path):
        """Test loading Parquet file"""
        info = analyzer.load_data(temp_parquet_path, format="parquet")

        assert info['row_count'] == 1000
        assert 'columns' in info

    def test_set_dataframe(self, analyzer, sample_spark_df):
        """Test setting DataFrame directly"""
        analyzer.set_dataframe(sample_spark_df)
        assert analyzer.df is not None
        assert analyzer.df.count() == 1000

    def test_dataframe_cached(self, analyzer, sample_spark_df):
        """Test that DataFrame is cached"""
        analyzer.set_dataframe(sample_spark_df)
        # Check cache status
        assert analyzer.df.is_cached


class TestColumnDetection:
    """Test automatic column detection using Spark schema"""

    def test_detect_columns(self, analyzer, sample_spark_df):
        """Test column detection"""
        analyzer.set_dataframe(sample_spark_df)
        suggestions = analyzer.detect_columns()

        assert 'customer_id' in suggestions['customer_id']
        assert 'experiment_group' in suggestions['group']
        assert 'pre_effect' in suggestions['pre_effect']
        assert 'post_effect' in suggestions['post_effect']
        assert 'customer_segment' in suggestions['segment']

    def test_detect_numeric_columns(self, analyzer, spark):
        """Test detection of numeric columns only"""
        data = spark.createDataFrame([
            ('treatment', 'text', 10.5),
            ('control', 'text', 12.3)
        ], ['group', 'text_col', 'numeric_col'])

        analyzer.set_dataframe(data)
        suggestions = analyzer.detect_columns()

        # Numeric column should be detected for effect
        assert any('numeric_col' in vals for vals in suggestions.values())

    def test_auto_configure(self, analyzer, sample_spark_df):
        """Test automatic configuration"""
        analyzer.set_dataframe(sample_spark_df)
        config = analyzer.auto_configure()

        assert config['success'] is True
        assert config['mapping']['group'] == 'experiment_group'
        assert 'post_effect' in config['mapping']
        assert config['labels']['treatment'] == 'treatment'
        assert config['labels']['control'] == 'control'


class TestStatisticalAggregations:
    """Test Spark aggregation operations"""

    def test_calculate_segment_statistics(self, analyzer, sample_spark_df):
        """Test segment statistics calculation via Spark"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        treatment_stats, control_stats = analyzer._calculate_segment_statistics()

        # Collect to verify
        t_row = treatment_stats.first()
        c_row = control_stats.first()

        assert t_row is not None
        assert c_row is not None
        assert t_row['n'] > 0
        assert c_row['n'] > 0
        assert 'mean' in t_row.asDict()
        assert 'variance' in t_row.asDict()
        assert 'conversions' in t_row.asDict()

    def test_calculate_segment_statistics_with_pre_effect(self, analyzer, sample_spark_df):
        """Test statistics with pre-effect data"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        treatment_stats, control_stats = analyzer._calculate_segment_statistics()

        t_row = treatment_stats.first()

        # Should have pre-effect statistics
        assert 'pre_mean' in t_row.asDict()
        assert 'pre_variance' in t_row.asDict()

    def test_calculate_segment_statistics_filtered(self, analyzer, sample_spark_df):
        """Test statistics for specific segment"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        treatment_stats, control_stats = analyzer._calculate_segment_statistics(
            segment_filter="Premium"
        )

        t_row = treatment_stats.first()
        assert t_row is not None
        # Sample size should be smaller than overall
        assert t_row['n'] < 500


class TestStatisticalCalculations:
    """Test statistical calculation methods"""

    def test_calculate_cohens_d(self, analyzer):
        """Test Cohen's d calculation"""
        cohens_d = analyzer._calculate_cohens_d(
            mean1=55, mean2=50,
            var1=100, var2=100,
            n1=100, n2=100
        )

        assert cohens_d > 0
        assert 0.3 < cohens_d < 0.7  # Should be around 0.5

    def test_calculate_t_test(self, analyzer):
        """Test t-test calculation"""
        t_stat, p_value, (ci_lower, ci_upper) = analyzer._calculate_t_test(
            mean1=55, mean2=50,
            var1=100, var2=100,
            n1=100, n2=100
        )

        assert t_stat > 0
        assert 0 <= p_value <= 1
        assert ci_lower < ci_upper

    def test_calculate_proportion_test(self, analyzer):
        """Test proportion test calculation"""
        z_stat, p_value, prop_diff, pooled = analyzer._calculate_proportion_test(
            conversions1=50, n1=100,
            conversions2=30, n2=100
        )

        assert z_stat > 0  # Treatment has higher proportion
        assert 0 <= p_value <= 1
        assert prop_diff == 0.2  # 50% - 30%

    def test_calculate_power(self, analyzer):
        """Test power calculation"""
        power = analyzer._calculate_power(
            effect_size=0.5,
            n1=100, n2=100
        )

        assert 0 <= power <= 1
        assert power > 0.5

    def test_calculate_required_sample_size(self, analyzer):
        """Test required sample size"""
        n_required = analyzer._calculate_required_sample_size(effect_size=0.5)

        assert n_required > 0
        assert 50 < n_required < 200


class TestBayesianAnalysis:
    """Test Bayesian Monte Carlo methods"""

    def test_run_bayesian_test_montecarlo(self, analyzer):
        """Test Bayesian test with Monte Carlo"""
        result = analyzer._run_bayesian_test_montecarlo(
            mean_t=55, var_t=100, n_t=100,
            mean_c=50, var_c=100, n_c=100
        )

        assert 'prob_treatment_better' in result
        assert 'expected_loss_treatment' in result
        assert 'credible_interval' in result
        assert 0 <= result['prob_treatment_better'] <= 1

    def test_bayesian_credible_intervals(self, analyzer):
        """Test Bayesian credible interval calculation"""
        result = analyzer._run_bayesian_test_montecarlo(
            mean_t=55, var_t=100, n_t=100,
            mean_c=50, var_c=100, n_c=100
        )

        ci_lower, ci_upper = result['credible_interval']
        assert ci_lower < ci_upper
        # Should capture the true difference of 5
        assert ci_lower < 5 < ci_upper


class TestABTestExecution:
    """Test full A/B test execution"""

    def test_run_ab_test_overall(self, analyzer, sample_spark_df):
        """Test running A/B test on overall data"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        result = analyzer.run_ab_test()

        assert isinstance(result, SparkABTestResult)
        assert result.segment == "Overall"
        assert result.treatment_size > 0
        assert result.control_size > 0
        assert result.effect_size != 0

    def test_run_ab_test_segment(self, analyzer, sample_spark_df):
        """Test A/B test on specific segment"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        result = analyzer.run_ab_test(segment_filter="Premium")

        assert result.segment == "Premium"
        assert result.treatment_size > 0
        assert result.control_size > 0

    def test_ab_test_includes_all_metrics(self, analyzer, sample_spark_df):
        """Test that A/B test includes all metric types"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        result = analyzer.run_ab_test()

        # Frequentist metrics
        assert result.t_statistic != 0
        assert 0 <= result.p_value <= 1
        assert result.cohens_d != 0

        # Power analysis
        assert 0 <= result.power <= 1
        assert result.required_sample_size > 0

        # AA test (we have pre_effect)
        assert 0 <= result.aa_p_value <= 1

        # DiD
        assert hasattr(result, 'did_effect')

        # Proportion test
        assert 0 <= result.treatment_proportion <= 1
        assert 0 <= result.control_proportion <= 1

        # Bayesian
        assert 0 <= result.bayesian_prob_treatment_better <= 1

    def test_ab_test_with_did_analysis(self, analyzer, sample_spark_df):
        """Test DiD calculation in A/B test"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        result = analyzer.run_ab_test()

        # Should have DiD metrics
        assert result.did_treatment_change != 0
        assert result.did_control_change != 0
        # DiD effect = (post_t - pre_t) - (post_c - pre_c)
        expected_did = result.did_treatment_change - result.did_control_change
        assert abs(result.did_effect - expected_did) < 0.001


class TestSchemaParity:
    """Validate canonical schema parity across pandas and Spark analyzers."""

    def test_segment_outputs_share_canonical_schema(self, analyzer, sample_spark_df, sample_pandas_df):
        """Equivalent data should normalize to the same canonical result shape."""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()
        spark_results = analyzer.run_segmented_analysis()

        pandas_analyzer = ABTestAnalyzer(significance_level=0.05, power_threshold=0.8)
        pandas_analyzer.set_dataframe(sample_pandas_df)
        pandas_analyzer.auto_configure()
        pandas_results = pandas_analyzer.run_segmented_analysis()

        spark_by_segment = {
            result.segment: canonical_result_as_dict(result)
            for result in spark_results
        }
        pandas_by_segment = {
            result.segment: canonical_result_as_dict(result)
            for result in pandas_results
        }

        assert set(spark_by_segment) == set(pandas_by_segment)

        for segment in pandas_by_segment:
            spark_payload = spark_by_segment[segment]
            pandas_payload = pandas_by_segment[segment]

            assert set(spark_payload.keys()) == set(pandas_payload.keys())
            assert spark_payload["segment"] == pandas_payload["segment"]
            assert spark_payload["treatment_size"] == pandas_payload["treatment_size"]
            assert spark_payload["control_size"] == pandas_payload["control_size"]

    def test_summary_detailed_results_keys_match_between_backends(self, analyzer, sample_spark_df, sample_pandas_df):
        """Summary detailed rows should expose the same keys for visualization compatibility."""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()
        spark_summary = analyzer.generate_summary(analyzer.run_segmented_analysis())

        pandas_analyzer = ABTestAnalyzer(significance_level=0.05, power_threshold=0.8)
        pandas_analyzer.set_dataframe(sample_pandas_df)
        pandas_analyzer.auto_configure()
        pandas_summary = pandas_analyzer.generate_summary(pandas_analyzer.run_segmented_analysis())

        from dataclasses import asdict
        spark_detail_keys = set(asdict(spark_summary.detailed_results[0]).keys())
        pandas_detail_keys = set(asdict(pandas_summary.detailed_results[0]).keys())

        assert spark_detail_keys == pandas_detail_keys


class TestSegmentedAnalysis:
    """Test distributed segmented analysis"""

    def test_run_segmented_analysis(self, analyzer, sample_spark_df):
        """Test running analysis across all segments"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        results = analyzer.run_segmented_analysis()

        assert len(results) == 3  # Premium, Standard, Basic
        assert all(isinstance(r, SparkABTestResult) for r in results)

        segments = {r.segment for r in results}
        assert 'Premium' in segments
        assert 'Standard' in segments
        assert 'Basic' in segments

    def test_segmented_analysis_no_segments(self, analyzer, spark):
        """Test analysis without segmentation"""
        # Create data without segments
        data = spark.createDataFrame([
            ('treatment', 55.0),
            ('control', 50.0)
        ] * 100, ['group', 'effect'])

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({
            'group': 'group',
            'effect_value': 'effect',
            'post_effect': 'effect'
        })
        analyzer.set_group_labels('treatment', 'control')

        results = analyzer.run_segmented_analysis()

        assert len(results) == 1
        assert results[0].segment == "Overall"


class TestSummaryGeneration:
    """Test summary statistics generation"""

    def test_generate_summary(self, analyzer, sample_spark_df):
        """Test generating summary from results"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        results = analyzer.run_segmented_analysis()
        summary = analyzer.generate_summary(results)

        assert hasattr(summary, 'total_segments_analyzed')
        assert hasattr(summary, 't_test_significant_segments')
        assert hasattr(summary, 'prop_test_significant_segments')
        assert hasattr(summary, 'bayesian_significant_segments')
        assert hasattr(summary, 'combined_total_effect')
        assert hasattr(summary, 'recommendations')

        assert summary.total_segments_analyzed == 3

    def test_summary_aggregations(self, analyzer, sample_spark_df):
        """Test summary aggregation calculations"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        results = analyzer.run_segmented_analysis()
        summary = analyzer.generate_summary(results)

        # Total customers should match sum across segments
        total_t = sum(r.treatment_size for r in results)
        total_c = sum(r.control_size for r in results)

        assert summary.total_treatment_customers == total_t
        assert summary.total_control_customers == total_c

    def test_generate_recommendations(self, analyzer, sample_spark_df):
        """Test recommendation generation"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        results = analyzer.run_segmented_analysis()
        summary = analyzer.generate_summary(results)

        recommendations = summary.recommendations

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


class TestResultPersistence:
    """Test saving results to various formats"""

    def test_save_results_to_parquet(self, analyzer, sample_spark_df, tmp_path):
        """Test saving results to Parquet"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        results = analyzer.run_segmented_analysis()

        output_path = str(tmp_path / "results.parquet")
        analyzer.save_results_to_parquet(results, output_path)

        # Verify file was created
        assert Path(output_path).exists()

        # Verify can be read back
        spark = analyzer.spark
        loaded_df = spark.read.parquet(output_path)
        assert loaded_df.count() == len(results)

    def test_parquet_result_schema(self, analyzer, sample_spark_df, tmp_path):
        """Test Parquet result schema contains all fields"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        results = analyzer.run_segmented_analysis()

        output_path = str(tmp_path / "results_schema.parquet")
        analyzer.save_results_to_parquet(results, output_path)

        spark = analyzer.spark
        loaded_df = spark.read.parquet(output_path)

        # Check key columns exist
        columns = loaded_df.columns
        assert 'segment' in columns
        assert 'treatment_size' in columns
        assert 'effect_size' in columns
        assert 'p_value' in columns
        assert 'bayesian_prob_treatment_better' in columns


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_insufficient_data(self, analyzer, spark):
        """Test with insufficient data"""
        data = spark.createDataFrame([
            ('treatment', 10.0)
        ], ['group', 'effect'])

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({
            'group': 'group',
            'effect_value': 'effect',
            'post_effect': 'effect'
        })
        analyzer.set_group_labels('treatment', 'control')

        with pytest.raises(ValueError, match="Insufficient"):
            analyzer.run_ab_test()

    def test_empty_segment(self, analyzer, sample_spark_df):
        """Test with non-existent segment"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        with pytest.raises(ValueError):
            analyzer.run_ab_test(segment_filter="NonExistent")

    def test_zero_variance_handling(self, analyzer, spark):
        """Test handling of zero variance data"""
        # All same values
        data = spark.createDataFrame(
            [('treatment', 10.0)] * 50 + [('control', 10.0)] * 50,
            ['group', 'effect']
        )

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({
            'group': 'group',
            'effect_value': 'effect',
            'post_effect': 'effect'
        })
        analyzer.set_group_labels('treatment', 'control')

        result = analyzer.run_ab_test()

        # Should handle gracefully
        assert result.effect_size == 0
        assert result.p_value == 1.0


class TestPerformanceAndScalability:
    """Test performance characteristics"""

    def test_dataframe_caching(self, analyzer, sample_spark_df):
        """Test that DataFrame is cached for performance"""
        analyzer.set_dataframe(sample_spark_df)

        # DataFrame should be cached
        assert analyzer.df.is_cached

    def test_aggregation_pushdown(self, analyzer, sample_spark_df):
        """Test that aggregations are pushed down to Spark"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        # Get statistics - should use Spark aggregations, not UDFs
        treatment_stats, control_stats = analyzer._calculate_segment_statistics()

        # Verify these are Spark DataFrames (not collected)
        from pyspark.sql import DataFrame
        assert isinstance(treatment_stats, DataFrame)
        assert isinstance(control_stats, DataFrame)


class TestCompleteWorkflow:
    """Test complete end-to-end workflows"""

    def test_complete_workflow_csv(self, analyzer, temp_csv_path):
        """Test complete workflow from CSV"""
        # Load
        analyzer.load_data(temp_csv_path)

        # Auto-configure
        config = analyzer.auto_configure()
        assert config['success'] is True

        # Run analysis
        results = analyzer.run_segmented_analysis()
        assert len(results) > 0

        # Generate summary
        summary = analyzer.generate_summary(results)
        assert hasattr(summary, 'total_segments_analyzed')

    def test_complete_workflow_with_persistence(self, analyzer, sample_spark_df, tmp_path):
        """Test workflow with result persistence"""
        analyzer.set_dataframe(sample_spark_df)
        analyzer.auto_configure()

        # Run analysis
        results = analyzer.run_segmented_analysis()

        # Save results
        output_path = str(tmp_path / "final_results.parquet")
        analyzer.save_results_to_parquet(results, output_path)

        # Verify saved
        assert Path(output_path).exists()

        # Generate summary
        summary = analyzer.generate_summary(results)
        assert summary.total_segments_analyzed == 3
