"""
Comprehensive Unit Tests for ABTestAnalyzer

Tests all statistical functions including:
- Data loading and column detection
- Auto-configuration
- T-tests and effect size calculations
- Proportion tests
- AA tests and bootstrapping
- Bayesian analysis
- DiD (Difference-in-Differences)
- Power analysis
- Summary generation
"""

import pytest
import pandas as pd
import numpy as np
import warnings

from src.statistics.analyzer import ABTestAnalyzer
from src.agent_reporting import render_full_analysis_output
from src.statistics.models import AATestResult, ABTestResult, ABTestSummary


@pytest.fixture
def sample_data():
    """Create sample A/B test data for testing"""
    np.random.seed(42)
    n = 1000

    data = {
        'customer_id': [f'CUST_{i:04d}' for i in range(n)],
        'experiment_group': ['treatment' if i % 2 == 0 else 'control' for i in range(n)],
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n),
        'pre_effect': np.random.normal(50, 10, n),
        'post_effect': np.random.normal(55, 12, n),  # Slight increase for treatment
        'experiment_duration_days': np.random.randint(7, 30, n)
    }

    # Add treatment effect to post_effect for treatment group
    df = pd.DataFrame(data)
    df.loc[df['experiment_group'] == 'treatment', 'post_effect'] += 3

    return df


@pytest.fixture
def sample_csv(tmp_path, sample_data):
    """Save sample data to CSV and return path"""
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def analyzer():
    """Create a fresh analyzer instance"""
    return ABTestAnalyzer(significance_level=0.05, power_threshold=0.8)


class TestDataLoading:
    """Test data loading and initial setup"""

    def test_load_data(self, analyzer, sample_csv):
        """Test loading CSV data"""
        info = analyzer.load_data(sample_csv)

        assert 'columns' in info
        assert 'shape' in info
        assert len(info['columns']) == 6
        assert info['shape'][0] == 1000

    def test_set_dataframe(self, analyzer, sample_data):
        """Test setting DataFrame directly"""
        analyzer.set_dataframe(sample_data)
        assert analyzer.df is not None
        assert len(analyzer.df) == 1000


class TestColumnDetection:
    """Test automatic column detection"""

    def test_detect_columns(self, analyzer, sample_data):
        """Test column detection with standard naming"""
        analyzer.set_dataframe(sample_data)
        suggestions = analyzer.detect_columns()

        assert 'customer_id' in suggestions['customer_id']
        assert 'experiment_group' in suggestions['group']
        assert 'pre_effect' in suggestions['pre_effect']
        assert 'post_effect' in suggestions['post_effect']
        assert 'customer_segment' in suggestions['segment']

    def test_auto_configure(self, analyzer, sample_data):
        """Test automatic configuration"""
        analyzer.set_dataframe(sample_data)
        config = analyzer.auto_configure()

        assert config['success'] is True
        assert 'group' in config['mapping']
        assert 'post_effect' in config['mapping']
        assert config['labels']['treatment'] == 'treatment'
        assert config['labels']['control'] == 'control'

    def test_auto_configure_does_not_pick_pre_metric_as_post(self, analyzer):
        """Auto-config should prefer post-period metric over pre-period similarly named columns."""
        df = pd.DataFrame({
            'customer_id': [f'C{i}' for i in range(10)],
            'experiment_group': ['treatment'] * 5 + ['control'] * 5,
            'pre_revenue': [100.0] * 10,
            'post_revenue': [110.0] * 5 + [100.0] * 5,
            'segment': ['A'] * 10,
        })

        analyzer.set_dataframe(df)
        config = analyzer.auto_configure()

        assert config['success'] is True
        assert config['mapping']['pre_effect'] == 'pre_revenue'
        assert config['mapping']['post_effect'] == 'post_revenue'

    def test_auto_configure_ambiguous_variant_labels_use_low_confidence_fallback(self, analyzer):
        """Variant A/B labels should use a low-confidence fallback instead of substring guesses."""
        df = pd.DataFrame({
            'ab_variant': ['variant_a', 'variant_b'] * 5,
            'post_effect': [1.0, 0.0] * 5,
            'segment': ['All'] * 10,
        })

        analyzer.set_dataframe(df)
        config = analyzer.auto_configure()

        assert config['success'] is True
        assert config['labels']['control'] == 'variant_a'
        assert config['labels']['treatment'] == 'variant_b'
        assert any('low-confidence' in warning.lower() for warning in config['warnings'])


class TestStatisticalCalculations:
    """Test core statistical calculation methods"""

    def test_calculate_cohens_d(self, analyzer):
        """Test Cohen's d calculation"""
        treatment = np.array([10, 12, 14, 16, 18])
        control = np.array([8, 9, 10, 11, 12])

        cohens_d = analyzer.calculate_cohens_d(treatment, control)

        assert cohens_d > 0  # Treatment has higher mean
        assert 1.0 < cohens_d < 3.0  # Reasonable range

    def test_calculate_power(self, analyzer):
        """Test statistical power calculation"""
        power = analyzer.calculate_power(
            effect_size=0.5,
            n_treatment=100,
            n_control=100
        )

        assert 0 <= power <= 1
        assert power > 0.5  # Should have decent power with n=100 and d=0.5

    def test_calculate_required_sample_size(self, analyzer):
        """Test required sample size calculation"""
        n_required = analyzer.calculate_required_sample_size(effect_size=0.5)

        assert n_required > 0
        assert 50 < n_required < 200  # Typical range for d=0.5


class TestAATest:
    """Test AA test for balance checking"""

    def test_run_aa_test_balanced(self, analyzer):
        """Test AA test with balanced groups"""
        np.random.seed(42)
        treatment_pre = np.random.normal(50, 10, 100)
        control_pre = np.random.normal(50, 10, 100)

        result = analyzer.run_aa_test(treatment_pre, control_pre, "Test")

        assert isinstance(result, AATestResult)
        assert result.is_balanced == True  # Should be balanced
        assert result.aa_p_value > 0.05

    def test_run_aa_test_imbalanced(self, analyzer):
        """Test AA test with imbalanced groups"""
        np.random.seed(42)
        treatment_pre = np.random.normal(50, 10, 100)
        control_pre = np.random.normal(60, 10, 100)  # Different mean

        result = analyzer.run_aa_test(treatment_pre, control_pre, "Test")

        assert isinstance(result, AATestResult)
        assert result.is_balanced == False  # Should be imbalanced
        assert result.aa_p_value < 0.05


class TestProportionTest:
    """Test proportion test calculations"""

    def test_run_proportion_test(self, analyzer):
        """Test two-proportion z-test"""
        # Treatment has higher conversion
        treatment_data = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # 50% conversion
        control_data = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])    # 20% conversion

        result = analyzer.run_proportion_test(treatment_data, control_data)

        assert 'treatment_proportion' in result
        assert 'control_proportion' in result
        assert 'proportion_diff' in result
        assert result['treatment_proportion'] > result['control_proportion']

    def test_run_proportion_test_edge_cases(self, analyzer):
        """Test proportion test with edge cases"""
        # All zeros
        treatment_data = np.zeros(10)
        control_data = np.zeros(10)

        result = analyzer.run_proportion_test(treatment_data, control_data)

        assert result['p_value'] == 1.0  # No difference

    def test_run_proportion_test_invalid_inputs_guardrail(self, analyzer):
        """Non-finite inputs should trigger guardrails and diagnostics."""
        treatment_data = np.array([1.0, np.nan, np.inf, 0.0, 1.0, 0.0, 1.0, 0.0])
        control_data = np.array([0.0, 0.0, 1.0, np.nan, 1.0, 0.0, 0.0, 1.0])

        result = analyzer.run_proportion_test(treatment_data, control_data)

        assert "diagnostics" in result
        diagnostics = result["diagnostics"]
        assert diagnostics["guardrail_triggered"] is True
        assert diagnostics["non_finite_values_removed"] > 0
        assert diagnostics["blocks_significance"] is True

    def test_run_proportion_test_extreme_boundary_case_emits_no_runtime_warnings(self, analyzer):
        """Extreme 0/1 boundary inputs should short-circuit guardrails without library warnings."""
        treatment_data = np.ones(50)
        control_data = np.zeros(50)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyzer.run_proportion_test(treatment_data, control_data)

        assert caught == []
        assert result["p_value"] == 1.0
        assert result["diagnostics"]["blocks_significance"] is True
        assert result["diagnostics"]["expected_counts_too_small"] is True


class TestBayesianTest:
    """Test Bayesian A/B test"""

    def test_run_bayesian_test_basic(self, analyzer):
        """Test basic Bayesian test"""
        np.random.seed(42)
        treatment_post = np.random.normal(55, 10, 100)
        control_post = np.random.normal(50, 10, 100)

        result = analyzer.run_bayesian_test(treatment_post, control_post)

        assert 'prob_treatment_better' in result
        assert 'expected_loss_treatment' in result
        assert 'expected_loss_control' in result
        assert 'credible_interval' in result
        assert 0 <= result['prob_treatment_better'] <= 1

    def test_run_bayesian_test_with_did(self, analyzer):
        """Test Bayesian test with DiD"""
        np.random.seed(42)
        treatment_pre = np.random.normal(50, 10, 100)
        treatment_post = np.random.normal(55, 10, 100)
        control_pre = np.random.normal(50, 10, 100)
        control_post = np.random.normal(51, 10, 100)

        result = analyzer.run_bayesian_test(
            treatment_post, control_post,
            treatment_pre, control_pre
        )

        assert 'did_effect' in result
        assert 'treatment_change' in result
        assert 'control_change' in result


class TestABTest:
    """Test full A/B test analysis"""

    def test_run_ab_test_overall(self, analyzer, sample_data):
        """Test running A/B test on overall data"""
        analyzer.set_dataframe(sample_data)
        analyzer.auto_configure()

        result = analyzer.run_ab_test()

        assert isinstance(result, ABTestResult)
        assert result.segment == "Overall"
        assert result.treatment_size > 0
        assert result.control_size > 0
        assert result.effect_size != 0

    def test_run_ab_test_segment(self, analyzer, sample_data):
        """Test running A/B test on a specific segment"""
        analyzer.set_dataframe(sample_data)
        analyzer.auto_configure()

        result = analyzer.run_ab_test(segment_filter="Premium")

        assert result.segment == "Premium"
        assert result.treatment_size > 0
        assert result.control_size > 0

    def test_run_ab_test_with_aa_test(self, analyzer, sample_data):
        """Test A/B test includes AA test results"""
        analyzer.set_dataframe(sample_data)
        analyzer.auto_configure()

        result = analyzer.run_ab_test()

        # Should have AA test results since we have pre_effect
        assert hasattr(result, 'aa_test_passed')
        assert hasattr(result, 'aa_p_value')

    def test_run_ab_test_with_did(self, analyzer, sample_data):
        """Test A/B test includes DiD calculations"""
        analyzer.set_dataframe(sample_data)
        analyzer.auto_configure()

        result = analyzer.run_ab_test()

        # Should have DiD results
        assert hasattr(result, 'did_effect')
        assert hasattr(result, 'did_treatment_change')
        assert hasattr(result, 'did_control_change')

    def test_run_ab_test_includes_srm_diagnostics(self, analyzer):
        """Segment result should include SRM diagnostics with expected split p-value."""
        np.random.seed(42)
        treatment_n = 180
        control_n = 20
        data = pd.DataFrame({
            "group": ["treatment"] * treatment_n + ["control"] * control_n,
            "effect": np.concatenate(
                [
                    np.random.normal(10.0, 1.0, treatment_n),
                    np.random.normal(10.0, 1.0, control_n),
                ]
            ),
            "segment": ["Overall"] * (treatment_n + control_n),
        })

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({"group": "group", "effect_value": "effect", "segment": "segment"})
        analyzer.set_group_labels("treatment", "control")

        result = analyzer.run_ab_test()

        srm_diag = result.diagnostics["experiment_quality"]["srm"]
        assert srm_diag["expected_treatment_ratio"] == 0.5
        assert 0.0 <= srm_diag["p_value"] <= 1.0
        assert srm_diag["is_sample_ratio_mismatch"] is True

    def test_run_ab_test_includes_assumption_diagnostics(self, analyzer):
        """Segment result should include normality and variance assumption checks."""
        np.random.seed(123)
        n = 60
        data = pd.DataFrame({
            "group": ["treatment"] * n + ["control"] * n,
            "effect": np.concatenate(
                [
                    np.random.normal(101.0, 8.0, n),
                    np.random.normal(100.0, 8.0, n),
                ]
            ),
        })

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({"group": "group", "effect_value": "effect"})
        analyzer.set_group_labels("treatment", "control")

        result = analyzer.run_ab_test()

        assumptions = result.diagnostics["experiment_quality"]["assumptions"]
        assert assumptions["normality_test"] in {"shapiro", "dagostino_k2", "not_applicable"}
        assert assumptions["variance_test"] in {"levene", "not_applicable"}
        assert "treatment_normality_p_value" in assumptions
        assert "control_normality_p_value" in assumptions
        assert "equal_variance_p_value" in assumptions

    def test_run_ab_test_includes_outlier_sensitivity_diagnostics(self, analyzer):
        """Outlier diagnostics should expose trimmed/winsorized deltas per segment."""
        np.random.seed(7)
        n = 120
        treatment = np.random.normal(10.0, 1.0, n)
        control = np.random.normal(10.0, 1.0, n)
        treatment[0] = 250.0  # strong outlier to force sensitivity

        data = pd.DataFrame({
            "group": ["treatment"] * n + ["control"] * n,
            "effect": np.concatenate([treatment, control]),
            "segment": ["S1"] * (2 * n),
        })

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({"group": "group", "effect_value": "effect", "segment": "segment"})
        analyzer.set_group_labels("treatment", "control")

        result = analyzer.run_ab_test()

        outlier_diag = result.diagnostics["experiment_quality"]["outlier_sensitivity"]
        assert "winsorized_delta" in outlier_diag
        assert "trimmed_delta" in outlier_diag
        assert outlier_diag["max_abs_delta"] >= 0.0
        assert outlier_diag["max_abs_delta"] > 0.0

    def test_run_ab_test_selects_binary_glm_for_binary_metric(self, analyzer):
        """Binary outcomes should use binomial/logit inference metadata."""
        np.random.seed(100)
        n = 500
        data = pd.DataFrame({
            "group": ["treatment"] * n + ["control"] * n,
            "effect": np.concatenate(
                [
                    np.random.binomial(1, 0.62, n),
                    np.random.binomial(1, 0.50, n),
                ]
            ),
        })

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({"group": "group", "effect_value": "effect"})
        analyzer.set_group_labels("treatment", "control")

        result = analyzer.run_ab_test()

        assert result.metric_type == "binary"
        assert result.model_type == "glm_binomial"
        assert result.model_effect_scale == "log_odds"
        assert result.model_effect_exponentiated > 1.0

    def test_run_ab_test_selects_count_glm_for_count_metric(self, analyzer):
        """Count outcomes should use Poisson/NegBin inference metadata."""
        np.random.seed(101)
        n = 400
        data = pd.DataFrame({
            "group": ["treatment"] * n + ["control"] * n,
            "effect": np.concatenate(
                [
                    np.random.poisson(3.2, n),
                    np.random.poisson(2.4, n),
                ]
            ),
        })

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({"group": "group", "effect_value": "effect"})
        analyzer.set_group_labels("treatment", "control")

        result = analyzer.run_ab_test()

        assert result.metric_type == "count"
        assert result.model_type in {"glm_poisson", "glm_negative_binomial"}
        assert result.model_effect_scale == "log_rate"
        assert result.model_effect_exponentiated > 1.0

    def test_run_ab_test_selects_robust_model_for_heavy_tail_metric(self, analyzer):
        """Heavy-tailed outcomes should use robust inference metadata."""
        np.random.seed(102)
        n = 300
        treatment = np.random.lognormal(mean=2.15, sigma=1.35, size=n)
        control = np.random.lognormal(mean=2.0, sigma=1.35, size=n)
        treatment[:5] = treatment[:5] * 40.0

        data = pd.DataFrame({
            "group": ["treatment"] * n + ["control"] * n,
            "effect": np.concatenate([treatment, control]),
        })

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({"group": "group", "effect_value": "effect"})
        analyzer.set_group_labels("treatment", "control")

        result = analyzer.run_ab_test()

        assert result.metric_type == "heavy_tail"
        assert result.model_type in {"rlm_huber", "ols_log1p_hc3"}
        assert result.model_effect_scale in {"location_shift", "log_mean_difference"}

    def test_run_ab_test_reports_covariate_adjusted_effect_when_covariate_available(self, analyzer):
        """Analyzer should expose covariate-adjusted treatment effect metadata."""
        np.random.seed(103)
        n = 350
        treatment_pre = np.random.normal(72.0, 8.0, n)
        control_pre = np.random.normal(52.0, 8.0, n)
        treatment_post = 2.0 + 0.85 * treatment_pre + np.random.normal(0.0, 2.0, n) + 1.0
        control_post = 2.0 + 0.85 * control_pre + np.random.normal(0.0, 2.0, n)

        data = pd.DataFrame({
            "group": ["treatment"] * n + ["control"] * n,
            "pre_metric": np.concatenate([treatment_pre, control_pre]),
            "post_metric": np.concatenate([treatment_post, control_post]),
        })

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping(
            {
                "group": "group",
                "effect_value": "post_metric",
                "post_effect": "post_metric",
                "covariates": ["pre_metric"],
            }
        )
        analyzer.set_group_labels("treatment", "control")

        result = analyzer.run_ab_test()

        assert result.covariate_adjustment_applied is True
        assert "pre_metric" in result.covariates_used
        assert result.covariate_adjusted_model_type != "none"
        assert abs(result.covariate_adjusted_effect) < abs(result.effect_size)
        assert 0.0 <= result.covariate_adjusted_p_value <= 1.0


class TestSequentialDecisionSupport:
    """Test opt-in sequential testing recommendations."""

    def test_run_ab_test_sequential_recommends_stop_for_efficacy(self, analyzer):
        """Strong early evidence should recommend stopping for efficacy."""
        np.random.seed(220)
        n = 400
        data = pd.DataFrame({
            "group": ["treatment"] * n + ["control"] * n,
            "effect": np.concatenate(
                [
                    np.random.normal(6.0, 1.0, n),
                    np.random.normal(4.5, 1.0, n),
                ]
            ),
        })

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({"group": "group", "effect_value": "effect"})
        analyzer.set_group_labels("treatment", "control")

        result = analyzer.run_ab_test(
            sequential_config={
                "enabled": True,
                "look_index": 1,
                "max_looks": 4,
                "spending_method": "obrien_fleming",
            }
        )

        assert result.sequential_mode_enabled is True
        assert result.sequential_stop_recommended is True
        assert result.sequential_decision == "stop_efficacy"
        assert result.sequential_method == "obrien_fleming"
        assert 0.0 < result.sequential_alpha_spent < analyzer.significance_level
        assert result.sequential_thresholds["information_fraction"] == pytest.approx(0.25)

    def test_run_ab_test_sequential_recommends_continue_midstream(self, analyzer):
        """Weak midstream evidence should recommend continuing when futility gate is high."""
        np.random.seed(221)
        n = 250
        data = pd.DataFrame({
            "group": ["treatment"] * n + ["control"] * n,
            "effect": np.concatenate(
                [
                    np.random.normal(5.0, 1.5, n),
                    np.random.normal(5.0, 1.5, n),
                ]
            ),
        })

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({"group": "group", "effect_value": "effect"})
        analyzer.set_group_labels("treatment", "control")

        result = analyzer.run_ab_test(
            sequential_config={
                "enabled": True,
                "look_index": 2,
                "max_looks": 4,
                "spending_method": "obrien_fleming",
                "futility_min_information_fraction": 0.9,
                "futility_p_value_threshold": 0.7,
            }
        )

        assert result.sequential_mode_enabled is True
        assert result.sequential_stop_recommended is False
        assert result.sequential_decision == "continue"
        assert "continue" in result.sequential_rationale.lower()
        assert result.sequential_alpha_spent < analyzer.significance_level

    def test_run_ab_test_default_sequential_fields_when_not_requested(self, analyzer):
        """Default behavior remains unchanged when sequential mode is not requested."""
        np.random.seed(222)
        n = 120
        data = pd.DataFrame({
            "group": ["treatment"] * n + ["control"] * n,
            "effect": np.concatenate(
                [
                    np.random.normal(5.3, 1.4, n),
                    np.random.normal(5.0, 1.4, n),
                ]
            ),
        })

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({"group": "group", "effect_value": "effect"})
        analyzer.set_group_labels("treatment", "control")

        result = analyzer.run_ab_test()

        assert result.sequential_mode_enabled is False
        assert result.sequential_stop_recommended is False
        assert result.sequential_decision == "not_requested"
        assert result.sequential_method == "none"
        assert result.sequential_thresholds == {}


class TestSegmentedAnalysis:
    """Test segmented analysis"""

    def test_run_segmented_analysis(self, analyzer, sample_data):
        """Test running analysis across all segments"""
        analyzer.set_dataframe(sample_data)
        analyzer.auto_configure()

        results = analyzer.run_segmented_analysis()

        assert len(results) == 3  # Premium, Standard, Basic
        assert all(isinstance(r, ABTestResult) for r in results)

        segments = {r.segment for r in results}
        assert 'Premium' in segments
        assert 'Standard' in segments
        assert 'Basic' in segments
        assert all(r.multiple_testing_applied for r in results)
        assert all(r.multiple_testing_method == "fdr_bh" for r in results)
        assert all(0.0 <= r.p_value_adjusted <= 1.0 for r in results)
        assert all(0.0 <= r.proportion_p_value_adjusted <= 1.0 for r in results)

    def test_run_segmented_analysis_no_segment(self, analyzer, sample_data):
        """Test analysis with no segmentation"""
        # Remove segment column
        df = sample_data.drop(columns=['customer_segment'])
        analyzer.set_dataframe(df)
        analyzer.auto_configure()

        results = analyzer.run_segmented_analysis()

        assert len(results) == 1
        assert results[0].segment == "Overall"
        assert results[0].multiple_testing_applied is False
        assert results[0].multiple_testing_method == "none"

    def test_segmented_analysis_surfaces_failed_segments_in_summary(self, analyzer):
        """Failed segments should be captured and exposed in the generated summary."""
        df = pd.DataFrame({
            'experiment_group': [
                'treatment', 'treatment', 'control', 'control',
                'treatment', 'control',
            ],
            'post_effect': [10.0, 11.0, 9.5, 10.5, 12.0, 8.0],
            'customer_segment': ['Good', 'Good', 'Good', 'Good', 'Bad', 'Bad'],
        })

        analyzer.set_dataframe(df)
        analyzer.set_column_mapping({
            'group': 'experiment_group',
            'effect_value': 'post_effect',
            'post_effect': 'post_effect',
            'segment': 'customer_segment',
        })
        analyzer.set_group_labels('treatment', 'control')

        results = analyzer.run_segmented_analysis()
        summary = analyzer.generate_summary(results)

        assert [result.segment for result in results] == ['Good']
        assert len(summary['segment_failures']) == 1
        assert summary['segment_failures'][0]['segment'] == 'Bad'
        assert 'Skipped 1 segment' in summary['analysis_warnings'][0]


class TestSummaryGeneration:
    """Test summary generation and recommendations"""

    def test_generate_summary(self, analyzer, sample_data):
        """Test generating summary from results"""
        analyzer.set_dataframe(sample_data)
        analyzer.auto_configure()

        results = analyzer.run_segmented_analysis()
        summary = analyzer.generate_summary(results)

        assert isinstance(summary, ABTestSummary)
        assert summary.total_segments_analyzed == 3
        assert summary.t_test_significant_segments >= 0
        assert summary.t_test_significant_segments_adjusted >= 0
        assert summary.prop_test_significant_segments >= 0
        assert summary.prop_test_significant_segments_adjusted >= 0
        assert summary.multiple_testing_method in {"none", "fdr_bh"}
        assert isinstance(summary.guardrail_segment_names, list)
        assert summary.bayesian_significant_segments >= 0
        assert isinstance(summary.combined_total_effect, float)
        assert isinstance(summary.recommendations, list)
        assert summary["total_segments_analyzed"] == summary.total_segments_analyzed

    def test_generate_recommendations(self, analyzer, sample_data):
        """Test recommendation generation"""
        analyzer.set_dataframe(sample_data)
        analyzer.auto_configure()

        results = analyzer.run_segmented_analysis()
        summary = analyzer.generate_summary(results)

        recommendations = summary.recommendations

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_summary_exposes_experiment_quality_diagnostics(self, analyzer, sample_data):
        """Summary should expose SRM, assumptions, and outlier diagnostics per segment."""
        analyzer.set_dataframe(sample_data)
        analyzer.auto_configure()

        results = analyzer.run_segmented_analysis()
        summary = analyzer.generate_summary(results)

        detail = summary.detailed_results[0]
        detail_payload = detail.to_legacy_dict()
        assert "diagnostics" in detail
        assert "experiment_quality" in detail["diagnostics"]
        assert "srm_p_value" in detail_payload
        assert "assumption_diagnostics" in detail_payload
        assert "outlier_sensitivity_diagnostics" in detail_payload

    def test_summary_exposes_sequential_fields_when_enabled(self, analyzer, sample_data):
        """Summary should include additive sequential decision metadata when configured."""
        analyzer.set_dataframe(sample_data)
        analyzer.auto_configure()

        results = analyzer.run_segmented_analysis(
            sequential_config={"enabled": True, "look_index": 2, "max_looks": 4}
        )
        summary = analyzer.generate_summary(results)

        assert summary.sequential_mode_segments == len(results)
        assert summary.sequential_stop_recommended_segments >= 0
        assert isinstance(summary.sequential_decision_breakdown, dict)

        detail = summary.detailed_results[0]
        assert "sequential_mode_enabled" in detail
        assert "sequential_decision" in detail
        assert "sequential_thresholds" in detail

    def test_render_full_analysis_accepts_typed_summary(self, analyzer, sample_data):
        """Report renderers should consume the typed summary model directly."""
        analyzer.set_dataframe(sample_data)
        analyzer.auto_configure()

        results = analyzer.run_segmented_analysis()
        summary = analyzer.generate_summary(results)

        output = render_full_analysis_output(summary)

        assert "FULL A/B TEST ANALYSIS SUMMARY" in output
        assert f"Segments Analyzed: {summary.total_segments_analyzed}" in output


class TestAnalysisStages:
    """Targeted tests for the staged run_ab_test orchestration helpers."""

    def test_prepare_segment_data_falls_back_to_post_only_when_pre_alignment_collapses(self, analyzer):
        """Pre-period alignment should gracefully fall back to post-only analysis when too sparse."""
        data = pd.DataFrame(
            {
                "group": ["treatment", "treatment", "control", "control"],
                "pre_metric": [1.0, np.nan, np.nan, np.nan],
                "post_metric": [10.0, 12.0, 9.0, 11.0],
            }
        )

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping(
            {
                "group": "group",
                "effect_value": "post_metric",
                "pre_effect": "pre_metric",
                "post_effect": "post_metric",
            }
        )
        analyzer.set_group_labels("treatment", "control")

        selection = analyzer._resolve_analysis_selection(None)
        prepared = analyzer._prepare_segment_data(selection)

        assert prepared.has_pre_effect is False
        assert prepared.treatment_pre_aligned is None
        assert prepared.control_pre_aligned is None
        assert len(prepared.treatment_post_aligned) == 2
        assert len(prepared.control_post_aligned) == 2

    def test_apply_covariate_alignment_reuses_covariate_complete_rows(self, analyzer):
        """Covariate alignment should trim the modeled arrays to rows with complete covariates."""
        data = pd.DataFrame(
            {
                "group": ["treatment"] * 3 + ["control"] * 3,
                "pre_metric": [5.0, 6.0, np.nan, 4.0, 4.5, 5.0],
                "post_metric": [7.0, 8.5, 9.0, 4.5, 5.0, 5.5],
                "aux_covariate": [1.0, 2.0, 3.0, 1.5, np.nan, 2.5],
            }
        )

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping(
            {
                "group": "group",
                "effect_value": "post_metric",
                "pre_effect": "pre_metric",
                "post_effect": "post_metric",
                "covariates": ["aux_covariate"],
            }
        )
        analyzer.set_group_labels("treatment", "control")

        selection = analyzer._resolve_analysis_selection(None)
        prepared = analyzer._prepare_segment_data(selection)
        covariates, treatment_model_df, control_model_df = analyzer._apply_covariate_alignment(
            selection,
            prepared,
        )

        assert covariates == ["aux_covariate", "pre_metric"]
        assert len(treatment_model_df) == 2
        assert len(control_model_df) == 2
        assert len(prepared.treatment_post_aligned) == 2
        assert len(prepared.control_post_aligned) == 2


class TestBootstrapping:
    """Test bootstrap balancing for imbalanced groups"""

    def test_bootstrap_balanced_control(self, analyzer):
        """Test bootstrapping to find balanced control group"""
        np.random.seed(42)

        # Create imbalanced groups
        treatment_pre = np.random.normal(50, 10, 100)
        control_df = pd.DataFrame({
            'pre_effect': np.random.normal(60, 10, 200),  # Different mean
            'post_effect': np.random.normal(60, 10, 200)
        })

        balanced_df, aa_result = analyzer.bootstrap_balanced_control(
            treatment_pre,
            control_df,
            'pre_effect',
            max_iterations=100
        )

        assert isinstance(balanced_df, pd.DataFrame)
        assert isinstance(aa_result, AATestResult)
        assert aa_result.bootstrapping_applied is True
        assert len(balanced_df) <= len(control_df)


class TestDataQueries:
    """Test data querying and summary methods"""

    def test_get_data_summary(self, analyzer, sample_data):
        """Test getting data summary statistics"""
        analyzer.set_dataframe(sample_data)

        summary = analyzer.get_data_summary()

        assert 'shape' in summary
        assert 'columns' in summary
        assert 'dtypes' in summary
        assert 'missing_values' in summary
        assert 'numeric_summary' in summary

    def test_get_segment_distribution(self, analyzer, sample_data):
        """Test getting segment distribution"""
        analyzer.set_dataframe(sample_data)
        analyzer.auto_configure()

        dist = analyzer.get_segment_distribution()

        assert 'group_distribution' in dist
        assert 'segment_distribution' in dist
        assert 'segment_by_group' in dist

    def test_query_data(self, analyzer, sample_data):
        """Test querying data"""
        analyzer.set_dataframe(sample_data)

        result = analyzer.query_data("customer_segment == 'Premium'")

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert all(result['customer_segment'] == 'Premium')


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_insufficient_data(self, analyzer):
        """Test with insufficient data"""
        small_data = pd.DataFrame({
            'group': ['treatment'],
            'effect': [10]
        })

        analyzer.set_dataframe(small_data)
        analyzer.set_column_mapping({'group': 'group', 'effect_value': 'effect'})
        analyzer.set_group_labels('treatment', 'control')

        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.run_ab_test()

    def test_missing_required_columns(self, analyzer, sample_data):
        """Test with missing required columns"""
        analyzer.set_dataframe(sample_data)
        analyzer.set_column_mapping({'group': 'experiment_group'})  # Missing effect_value
        analyzer.set_group_labels('treatment', 'control')

        with pytest.raises(ValueError, match="Column mapping"):
            analyzer.run_ab_test()

    def test_zero_variance(self, analyzer):
        """Test with zero variance data"""
        data = pd.DataFrame({
            'group': ['treatment'] * 50 + ['control'] * 50,
            'effect': [10] * 100  # All same value
        })

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({'group': 'group', 'effect_value': 'effect'})
        analyzer.set_group_labels('treatment', 'control')

        result = analyzer.run_ab_test()

        # Should handle gracefully - effect is zero
        assert result.effect_size == 0
        # P-value might be NaN when variance is zero
        assert np.isnan(result.p_value) or result.p_value == 1.0
        assert result.inference_guardrail_triggered is True
        assert result.diagnostics["frequentist"]["t_test"]["degenerate_variance"] is True

    def test_zero_variance_ab_test_emits_no_runtime_warnings(self, analyzer):
        """Deterministic zero-variance inputs should be handled without SciPy runtime warnings."""
        data = pd.DataFrame({
            'group': ['treatment'] * 50 + ['control'] * 50,
            'effect': [10] * 100,
        })

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({'group': 'group', 'effect_value': 'effect'})
        analyzer.set_group_labels('treatment', 'control')

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyzer.run_ab_test()

        assert caught == []
        assumptions = result.diagnostics["experiment_quality"]["assumptions"]
        assert assumptions["variance_test"] == "not_applicable"
        assert assumptions["variance_reason"] == "degenerate_variance"

    def test_zero_variance_with_constant_difference_is_significant(self, analyzer):
        """Deterministic group separation should remain significant even with zero variance."""
        data = pd.DataFrame({
            'group': ['treatment'] * 50 + ['control'] * 50,
            'effect': [1.0] * 50 + [0.0] * 50,
        })

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({'group': 'group', 'effect_value': 'effect'})
        analyzer.set_group_labels('treatment', 'control')

        result = analyzer.run_ab_test()

        assert result.effect_size == 1.0
        assert result.p_value == 0.0
        assert result.is_significant is True
        assert result.inference_guardrail_triggered is True

    def test_small_sample_guardrail_blocks_significance(self, analyzer):
        """Very small samples should trigger guardrails and suppress frequentist significance."""
        data = pd.DataFrame({
            "group": ["treatment"] * 3 + ["control"] * 3,
            "effect": [10.0, 11.0, 12.0, 1.0, 2.0, 3.0],
        })

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({"group": "group", "effect_value": "effect"})
        analyzer.set_group_labels("treatment", "control")

        result = analyzer.run_ab_test()

        assert result.inference_guardrail_triggered is True
        assert result.diagnostics["frequentist"]["t_test"]["small_n"] is True
        assert result.is_significant is False
        assert result.proportion_guardrail_triggered is True
        assert result.proportion_is_significant is False

    def test_small_sample_ab_test_emits_no_runtime_warnings(self, analyzer):
        """Small-sample guardrails should avoid leaking power-analysis warnings."""
        data = pd.DataFrame({
            "group": ["treatment"] * 3 + ["control"] * 3,
            "effect": [10.0, 11.0, 12.0, 1.0, 2.0, 3.0],
        })

        analyzer.set_dataframe(data)
        analyzer.set_column_mapping({"group": "group", "effect_value": "effect"})
        analyzer.set_group_labels("treatment", "control")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyzer.run_ab_test()

        assert caught == []
        assert result.inference_guardrail_triggered is True


class TestCompleteWorkflow:
    """Test complete end-to-end workflows"""

    def test_complete_workflow_with_auto_config(self, analyzer, sample_csv):
        """Test complete workflow with auto-configuration"""
        # Load data
        analyzer.load_data(sample_csv)

        # Auto-configure
        config = analyzer.auto_configure()
        assert config['success'] is True

        # Run segmented analysis
        results = analyzer.run_segmented_analysis()
        assert len(results) > 0

        # Generate summary
        summary = analyzer.generate_summary(results)
        assert 'total_segments_analyzed' in summary

        # Check all statistical tests ran
        for result in results:
            assert result.p_value is not None
            assert result.cohens_d is not None
            assert result.power is not None
            assert result.bayesian_prob_treatment_better is not None

    def test_complete_workflow_manual_config(self, analyzer, sample_data):
        """Test complete workflow with manual configuration"""
        analyzer.set_dataframe(sample_data)

        # Manual configuration
        analyzer.set_column_mapping({
            'group': 'experiment_group',
            'effect_value': 'post_effect',
            'post_effect': 'post_effect',
            'pre_effect': 'pre_effect',
            'segment': 'customer_segment'
        })
        analyzer.set_group_labels('treatment', 'control')

        # Run analysis
        results = analyzer.run_segmented_analysis()
        summary = analyzer.generate_summary(results)

        assert summary['total_segments_analyzed'] == 3
        assert 'recommendations' in summary
