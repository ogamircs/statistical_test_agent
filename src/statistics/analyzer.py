"""
A/B Test Statistical Analyzer

Provides comprehensive statistical analysis for A/B experiments including:
- T-tests for continuous metrics
- Effect size calculations (Cohen's d)
- Power analysis and sample size validation
- Segment-level analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.weightstats import ttest_ind as sm_ttest_ind, CompareMeans, DescrStatsW
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from typing import Dict, List, Optional, Any

from .models import ABTestResult


class ABTestAnalyzer:
    """
    Comprehensive A/B Test Analyzer

    Handles data loading, column detection, and statistical analysis
    for A/B experiments with customer segmentation.
    """

    def __init__(self, significance_level: float = 0.05, power_threshold: float = 0.8):
        self.significance_level = significance_level
        self.power_threshold = power_threshold
        self.df: Optional[pd.DataFrame] = None
        self.column_mapping: Dict[str, str] = {}
        self.treatment_label = None
        self.control_label = None

    def load_data(self, filepath: str) -> Dict[str, Any]:
        """Load CSV data and return column information"""
        self.df = pd.read_csv(filepath)
        return {
            "columns": list(self.df.columns),
            "shape": self.df.shape,
            "dtypes": self.df.dtypes.to_dict(),
            "sample": self.df.head(3).to_dict()
        }

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Set dataframe directly"""
        self.df = df

    def detect_columns(self) -> Dict[str, List[str]]:
        """
        Attempt to auto-detect column types based on common naming patterns
        Returns suggestions for each required column type
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        columns = [col.lower() for col in self.df.columns]
        original_columns = list(self.df.columns)

        suggestions = {
            "customer_id": [],
            "group": [],
            "effect_value": [],
            "segment": [],
            "duration": []
        }

        # Customer ID patterns
        id_patterns = ['customer_id', 'customerid', 'user_id', 'userid', 'id', 'customer']
        for i, col in enumerate(columns):
            if any(pattern in col for pattern in id_patterns):
                suggestions["customer_id"].append(original_columns[i])

        # Group/Treatment indicator patterns
        group_patterns = ['group', 'treatment', 'control', 'variant', 'test_group', 'experiment_group', 'ab_group']
        for i, col in enumerate(columns):
            if any(pattern in col for pattern in group_patterns):
                suggestions["group"].append(original_columns[i])

        # Effect value patterns
        effect_patterns = ['effect', 'value', 'metric', 'outcome', 'result', 'conversion', 'revenue', 'amount', 'score']
        for i, col in enumerate(columns):
            if any(pattern in col for pattern in effect_patterns):
                if self.df[original_columns[i]].dtype in ['float64', 'int64', 'float32', 'int32']:
                    suggestions["effect_value"].append(original_columns[i])

        # Segment patterns
        segment_patterns = ['segment', 'category', 'tier', 'type', 'cohort', 'cluster', 'group_name']
        for i, col in enumerate(columns):
            if any(pattern in col for pattern in segment_patterns):
                if col not in [c.lower() for c in suggestions["group"]]:
                    suggestions["segment"].append(original_columns[i])

        # Duration patterns
        duration_patterns = ['duration', 'days', 'period', 'time', 'length', 'exposure']
        for i, col in enumerate(columns):
            if any(pattern in col for pattern in duration_patterns):
                suggestions["duration"].append(original_columns[i])

        return suggestions

    def set_column_mapping(self, mapping: Dict[str, str]) -> None:
        """
        Set the column mapping for analysis

        mapping should contain:
        - customer_id: column name for customer identifier
        - group: column name for treatment/control indicator
        - effect_value: column name for the metric to analyze
        - segment: column name for customer segments (optional)
        - duration: column name for experiment duration (optional)
        """
        self.column_mapping = mapping

    def get_group_values(self) -> Dict[str, List[Any]]:
        """Get unique values in the group column to identify treatment/control labels"""
        if self.df is None or "group" not in self.column_mapping:
            raise ValueError("Data not loaded or group column not set")

        group_col = self.column_mapping["group"]
        unique_values = self.df[group_col].unique().tolist()
        return {"group_column": group_col, "unique_values": unique_values}

    def set_group_labels(self, treatment_label: Any, control_label: Any) -> None:
        """Set the labels used for treatment and control groups"""
        self.treatment_label = treatment_label
        self.control_label = control_label

    def auto_configure(self) -> Dict[str, Any]:
        """
        Automatically configure column mappings and group labels based on best guesses.
        Returns a dict with the configuration used and any warnings.
        """
        if self.df is None:
            return {"success": False, "error": "No data loaded"}

        config = {"success": True, "warnings": [], "mapping": {}, "labels": {}}

        # Auto-detect columns
        suggestions = self.detect_columns()

        # Set group column (required)
        if suggestions["group"]:
            config["mapping"]["group"] = suggestions["group"][0]
        else:
            # Try to find any column with exactly 2 unique values
            for col in self.df.columns:
                if self.df[col].nunique() == 2:
                    config["mapping"]["group"] = col
                    config["warnings"].append(f"Guessed '{col}' as group column (has 2 unique values)")
                    break

        if "group" not in config["mapping"]:
            return {"success": False, "error": "Could not detect group column"}

        # Set effect value column (required)
        if suggestions["effect_value"]:
            config["mapping"]["effect_value"] = suggestions["effect_value"][0]
        else:
            # Try to find any numeric column
            numeric_cols = self.df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
            for col in numeric_cols:
                if col != config["mapping"].get("group"):
                    config["mapping"]["effect_value"] = col
                    config["warnings"].append(f"Guessed '{col}' as effect value column (numeric)")
                    break

        if "effect_value" not in config["mapping"]:
            return {"success": False, "error": "Could not detect effect value column"}

        # Set optional columns
        if suggestions["segment"]:
            config["mapping"]["segment"] = suggestions["segment"][0]
        if suggestions["customer_id"]:
            config["mapping"]["customer_id"] = suggestions["customer_id"][0]

        # Apply column mapping
        self.set_column_mapping(config["mapping"])

        # Auto-detect treatment/control labels
        group_col = config["mapping"]["group"]
        unique_values = self.df[group_col].unique().tolist()

        treatment_patterns = ['treatment', 'treat', 'test', 'experiment', 'variant', 'exposed', '1', 'true', 'yes', 'a']
        control_patterns = ['control', 'ctrl', 'baseline', 'placebo', 'unexposed', '0', 'false', 'no', 'b']

        treatment_label = None
        control_label = None

        for val in unique_values:
            val_lower = str(val).lower().strip()
            if any(p in val_lower for p in treatment_patterns):
                treatment_label = val
            elif any(p in val_lower for p in control_patterns):
                control_label = val

        # If we couldn't detect, use first two values
        if treatment_label is None or control_label is None:
            if len(unique_values) >= 2:
                # Assume alphabetically/numerically first is control, second is treatment
                sorted_vals = sorted(unique_values, key=lambda x: str(x).lower())
                control_label = sorted_vals[0]
                treatment_label = sorted_vals[1] if len(sorted_vals) > 1 else sorted_vals[0]
                config["warnings"].append(f"Guessed treatment='{treatment_label}', control='{control_label}' based on order")

        if treatment_label is None or control_label is None:
            return {"success": False, "error": "Could not detect treatment/control labels"}

        config["labels"]["treatment"] = treatment_label
        config["labels"]["control"] = control_label
        self.set_group_labels(treatment_label, control_label)

        return config

    def calculate_cohens_d(self, treatment_data: np.ndarray, control_data: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(treatment_data), len(control_data)
        var1, var2 = treatment_data.var(), control_data.var()

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (treatment_data.mean() - control_data.mean()) / pooled_std

    def calculate_power(self, effect_size: float, n_treatment: int, n_control: int) -> float:
        """Calculate statistical power for the given effect size and sample sizes"""
        if effect_size == 0:
            return 0.0

        power_analysis = TTestIndPower()
        ratio = n_control / n_treatment if n_treatment > 0 else 1

        try:
            power = power_analysis.solve_power(
                effect_size=abs(effect_size),
                nobs1=n_treatment,
                ratio=ratio,
                alpha=self.significance_level
            )
            return min(power, 1.0)
        except Exception:
            return 0.0

    def calculate_required_sample_size(self, effect_size: float, ratio: float = 1.0) -> int:
        """Calculate required sample size per group for desired power"""
        if effect_size == 0:
            return float('inf')

        power_analysis = TTestIndPower()
        try:
            n = power_analysis.solve_power(
                effect_size=abs(effect_size),
                power=self.power_threshold,
                ratio=ratio,
                alpha=self.significance_level
            )
            return int(np.ceil(n))
        except Exception:
            return float('inf')

    def run_proportion_test(self, treatment_data: np.ndarray, control_data: np.ndarray) -> Dict[str, float]:
        """
        Run a two-proportion z-test using statsmodels to compare conversion rates.
        Conversion = having a non-zero effect value.

        Returns proportion test statistics.
        """
        # Count non-zero values (conversions/activations)
        treatment_conversions = int(np.sum(treatment_data != 0))
        control_conversions = int(np.sum(control_data != 0))

        n_treatment = len(treatment_data)
        n_control = len(control_data)

        # Proportions
        p_treatment = treatment_conversions / n_treatment if n_treatment > 0 else 0
        p_control = control_conversions / n_control if n_control > 0 else 0

        # Handle edge cases where there's no variation
        if (treatment_conversions == 0 and control_conversions == 0) or \
           (treatment_conversions == n_treatment and control_conversions == n_control):
            return {
                "treatment_proportion": p_treatment,
                "control_proportion": p_control,
                "proportion_diff": p_treatment - p_control,
                "z_stat": 0.0,
                "p_value": 1.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0
            }

        try:
            # Use statsmodels proportions_ztest for two-proportion z-test
            count = np.array([treatment_conversions, control_conversions])
            nobs = np.array([n_treatment, n_control])

            z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')

            # Calculate confidence interval for difference in proportions
            ci_lower, ci_upper = confint_proportions_2indep(
                treatment_conversions, n_treatment,
                control_conversions, n_control,
                method='wald'
            )

            return {
                "treatment_proportion": p_treatment,
                "control_proportion": p_control,
                "proportion_diff": p_treatment - p_control,
                "z_stat": float(z_stat),
                "p_value": float(p_value),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper)
            }
        except Exception:
            # Fallback if statsmodels fails
            return {
                "treatment_proportion": p_treatment,
                "control_proportion": p_control,
                "proportion_diff": p_treatment - p_control,
                "z_stat": 0.0,
                "p_value": 1.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0
            }

    def run_bayesian_test(self, treatment_data: np.ndarray, control_data: np.ndarray,
                          n_samples: int = 10000) -> Dict[str, float]:
        """
        Run Bayesian A/B test using Monte Carlo simulation with conjugate priors.

        For continuous data, uses Normal-Inverse-Gamma conjugate prior.
        Estimates posterior distribution of difference in means.

        Returns:
            Dict with:
            - prob_treatment_better: P(treatment > control)
            - expected_loss_treatment: Expected loss if choosing treatment when control is better
            - expected_loss_control: Expected loss if choosing control when treatment is better
            - credible_interval: 95% credible interval for difference (treatment - control)
            - relative_uplift: Relative improvement estimate
        """
        n_treatment = len(treatment_data)
        n_control = len(control_data)

        # Handle edge cases
        if n_treatment < 2 or n_control < 2:
            return {
                "prob_treatment_better": 0.5,
                "expected_loss_treatment": 0.0,
                "expected_loss_control": 0.0,
                "credible_interval": (0.0, 0.0),
                "relative_uplift": 0.0
            }

        # Sample statistics
        mean_t = treatment_data.mean()
        mean_c = control_data.mean()
        var_t = treatment_data.var(ddof=1)
        var_c = control_data.var(ddof=1)

        # Handle zero variance cases
        if var_t == 0 and var_c == 0:
            # Both groups have identical values
            diff = mean_t - mean_c
            return {
                "prob_treatment_better": 1.0 if diff > 0 else (0.0 if diff < 0 else 0.5),
                "expected_loss_treatment": 0.0 if diff >= 0 else abs(diff),
                "expected_loss_control": 0.0 if diff <= 0 else abs(diff),
                "credible_interval": (diff, diff),
                "relative_uplift": diff / mean_c if mean_c != 0 else 0.0
            }

        # Use uninformative priors (Jeffreys prior for Normal-Inverse-Gamma)
        # Posterior for mean given variance is Normal
        # Posterior for variance is Inverse-Gamma (approximated as scaled inverse chi-squared)

        # For computational efficiency, use Normal approximation for posterior of means
        # Standard error for each group
        se_t = np.sqrt(var_t / n_treatment)
        se_c = np.sqrt(var_c / n_control)

        # Draw samples from posterior distributions of means
        # Using t-distribution would be more accurate for small samples,
        # but Normal is a good approximation for n > 30
        np.random.seed(42)  # For reproducibility

        if n_treatment > 30 and n_control > 30:
            # Normal approximation
            treatment_samples = np.random.normal(mean_t, se_t, n_samples)
            control_samples = np.random.normal(mean_c, se_c, n_samples)
        else:
            # Use t-distribution for small samples
            treatment_samples = mean_t + se_t * np.random.standard_t(n_treatment - 1, n_samples)
            control_samples = mean_c + se_c * np.random.standard_t(n_control - 1, n_samples)

        # Compute difference samples
        diff_samples = treatment_samples - control_samples

        # Probability that treatment is better
        prob_treatment_better = np.mean(diff_samples > 0)

        # Expected loss calculations
        # Loss if we choose treatment but control is actually better
        treatment_worse = diff_samples < 0
        expected_loss_treatment = np.mean(np.abs(diff_samples) * treatment_worse)

        # Loss if we choose control but treatment is actually better
        control_worse = diff_samples > 0
        expected_loss_control = np.mean(np.abs(diff_samples) * control_worse)

        # 95% credible interval (highest density interval approximated by percentiles)
        ci_lower = np.percentile(diff_samples, 2.5)
        ci_upper = np.percentile(diff_samples, 97.5)

        # Relative uplift (using posterior mean of difference / control mean)
        mean_diff = np.mean(diff_samples)
        relative_uplift = mean_diff / mean_c if mean_c != 0 else 0.0

        return {
            "prob_treatment_better": float(prob_treatment_better),
            "expected_loss_treatment": float(expected_loss_treatment),
            "expected_loss_control": float(expected_loss_control),
            "credible_interval": (float(ci_lower), float(ci_upper)),
            "relative_uplift": float(relative_uplift)
        }

    def run_ab_test(self, segment_filter: Optional[str] = None) -> ABTestResult:
        """
        Run A/B test for overall data or a specific segment

        Returns comprehensive test results including:
        - T-test for continuous metric (effect size, significance)
        - Proportion test for conversion rates
        - Combined effect calculation
        - Power analysis
        - Sample size adequacy
        """
        if self.df is None:
            raise ValueError("No data loaded")

        required_cols = ["group", "effect_value"]
        for col in required_cols:
            if col not in self.column_mapping:
                raise ValueError(f"Column mapping for '{col}' not set")

        group_col = self.column_mapping["group"]
        effect_col = self.column_mapping["effect_value"]

        # Filter by segment if specified
        df = self.df.copy()
        segment_name = "Overall"

        if segment_filter and "segment" in self.column_mapping:
            segment_col = self.column_mapping["segment"]
            df = df[df[segment_col] == segment_filter]
            segment_name = str(segment_filter)

        # Split into treatment and control
        treatment_data = df[df[group_col] == self.treatment_label][effect_col].dropna()
        control_data = df[df[group_col] == self.control_label][effect_col].dropna()

        if len(treatment_data) < 2 or len(control_data) < 2:
            raise ValueError(f"Insufficient data for segment '{segment_name}': "
                           f"Treatment n={len(treatment_data)}, Control n={len(control_data)}")

        # ============ T-TEST (continuous metric) using statsmodels ============
        treatment_mean = treatment_data.mean()
        control_mean = control_data.mean()
        effect_size = treatment_mean - control_mean

        # Cohen's d
        cohens_d = self.calculate_cohens_d(treatment_data.values, control_data.values)

        # T-test using statsmodels
        # Create DescrStatsW objects for each group
        d1 = DescrStatsW(treatment_data.values)
        d2 = DescrStatsW(control_data.values)

        # CompareMeans for the two groups
        cm = CompareMeans(d1, d2)

        # Get t-test results (using Welch's t-test for unequal variances)
        t_stat, p_value, df = cm.ttest_ind(usevar='unequal')

        # Get confidence interval for the difference using statsmodels
        ci_low, ci_high = cm.tconfint_diff(alpha=self.significance_level, usevar='unequal')
        ci = (ci_low, ci_high)

        # Power analysis
        power = self.calculate_power(cohens_d, len(treatment_data), len(control_data))
        required_n = self.calculate_required_sample_size(
            cohens_d,
            ratio=len(control_data) / len(treatment_data) if len(treatment_data) > 0 else 1
        )

        is_significant = p_value < self.significance_level

        # ============ PROPORTION TEST (conversion rate) ============
        prop_results = self.run_proportion_test(treatment_data.values, control_data.values)

        proportion_is_significant = prop_results["p_value"] < self.significance_level

        # Calculate proportion-based effect
        # If treatment has higher conversion rate, the additional converters
        # would generate value at the control mean rate (since they wouldn't exist without treatment)
        proportion_diff = prop_results["proportion_diff"]

        # Proportion effect: additional proportion × control mean × treatment N
        # This represents value from customers who converted ONLY because of treatment
        if proportion_is_significant and proportion_diff > 0:
            # These are NEW conversions that wouldn't exist without treatment
            # Their value is estimated at control_mean (what a typical converter generates)
            proportion_effect_per_customer = proportion_diff * control_mean
            proportion_effect = proportion_effect_per_customer * len(treatment_data)
        else:
            proportion_effect_per_customer = 0.0
            proportion_effect = 0.0

        # ============ COMBINED EFFECT ============
        # T-test effect: difference in value for existing converters
        # Proportion effect: value from new converters
        t_test_total_effect = effect_size * len(treatment_data) if is_significant else 0
        total_effect = t_test_total_effect + proportion_effect
        total_effect_per_customer = (effect_size if is_significant else 0) + proportion_effect_per_customer

        # ============ BAYESIAN TEST ============
        bayesian_results = self.run_bayesian_test(treatment_data.values, control_data.values)
        bayesian_is_significant = (bayesian_results["prob_treatment_better"] > 0.95 or
                                   bayesian_results["prob_treatment_better"] < 0.05)

        return ABTestResult(
            segment=segment_name,
            treatment_size=len(treatment_data),
            control_size=len(control_data),
            # T-test results
            treatment_mean=treatment_mean,
            control_mean=control_mean,
            effect_size=effect_size,
            cohens_d=cohens_d,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=ci,
            power=power,
            required_sample_size=required_n,
            is_sample_adequate=power >= self.power_threshold,
            # Proportion test results
            treatment_proportion=prop_results["treatment_proportion"],
            control_proportion=prop_results["control_proportion"],
            proportion_diff=proportion_diff,
            proportion_z_stat=prop_results["z_stat"],
            proportion_p_value=prop_results["p_value"],
            proportion_is_significant=proportion_is_significant,
            proportion_effect=proportion_effect,
            proportion_effect_per_customer=proportion_effect_per_customer,
            # Combined effects
            total_effect=total_effect,
            total_effect_per_customer=total_effect_per_customer,
            # Bayesian test results
            bayesian_prob_treatment_better=bayesian_results["prob_treatment_better"],
            bayesian_expected_loss_treatment=bayesian_results["expected_loss_treatment"],
            bayesian_expected_loss_control=bayesian_results["expected_loss_control"],
            bayesian_credible_interval=bayesian_results["credible_interval"],
            bayesian_relative_uplift=bayesian_results["relative_uplift"],
            bayesian_is_significant=bayesian_is_significant
        )

    def run_segmented_analysis(self) -> List[ABTestResult]:
        """Run A/B tests for all segments"""
        if "segment" not in self.column_mapping:
            return [self.run_ab_test()]

        segment_col = self.column_mapping["segment"]
        segments = self.df[segment_col].unique()

        results = []
        for segment in segments:
            try:
                result = self.run_ab_test(segment_filter=segment)
                results.append(result)
            except ValueError as e:
                print(f"Skipping segment '{segment}': {e}")

        return results

    def generate_summary(self, results: List[ABTestResult]) -> Dict[str, Any]:
        """
        Generate comprehensive summary of A/B test results

        Includes:
        - T-test significance summary
        - Proportion test significance summary
        - Combined effect size calculation
        - Sample size adequacy assessment
        - Recommendations
        """
        if not results:
            return {"error": "No results to summarize"}

        # T-test significant results
        t_significant_results = [r for r in results if r.is_significant]

        # Proportion test significant results
        prop_significant_results = [r for r in results if r.proportion_is_significant]

        # Bayesian significant results (prob > 0.95 or prob < 0.05)
        bayesian_significant_results = [r for r in results if r.bayesian_is_significant]

        # Calculate T-test total effect
        t_test_total_effect = sum(r.effect_size * r.treatment_size for r in t_significant_results)
        avg_t_test_effect = (
            np.mean([r.effect_size for r in t_significant_results])
            if t_significant_results else 0
        )
        total_treatment_in_t_significant = sum(r.treatment_size for r in t_significant_results)

        # Calculate Proportion test total effect
        prop_total_effect = sum(r.proportion_effect for r in prop_significant_results)
        avg_prop_effect = (
            np.mean([r.proportion_effect_per_customer for r in prop_significant_results])
            if prop_significant_results else 0
        )
        total_treatment_in_prop_significant = sum(r.treatment_size for r in prop_significant_results)

        # Combined total effect
        combined_total_effect = sum(r.total_effect for r in results)

        # Bayesian statistics
        avg_bayesian_prob = np.mean([r.bayesian_prob_treatment_better for r in results])
        avg_expected_loss = np.mean([min(r.bayesian_expected_loss_treatment, r.bayesian_expected_loss_control) for r in results])

        # Sample adequacy
        adequate_samples = [r for r in results if r.is_sample_adequate]
        inadequate_samples = [r for r in results if not r.is_sample_adequate]

        # Overall statistics
        total_treatment = sum(r.treatment_size for r in results)
        total_control = sum(r.control_size for r in results)

        summary = {
            "total_segments_analyzed": len(results),

            # T-test summary
            "t_test_significant_segments": len(t_significant_results),
            "t_test_significance_rate": len(t_significant_results) / len(results) if results else 0,
            "t_test_avg_effect": avg_t_test_effect,
            "t_test_total_effect": t_test_total_effect,
            "t_test_effect_calculation": f"{avg_t_test_effect:.4f} × {total_treatment_in_t_significant} = {t_test_total_effect:.2f}",

            # Proportion test summary
            "prop_test_significant_segments": len(prop_significant_results),
            "prop_test_significance_rate": len(prop_significant_results) / len(results) if results else 0,
            "prop_test_avg_effect": avg_prop_effect,
            "prop_test_total_effect": prop_total_effect,
            "prop_test_effect_calculation": f"{avg_prop_effect:.4f} × {total_treatment_in_prop_significant} = {prop_total_effect:.2f}",

            # Combined totals
            "combined_total_effect": combined_total_effect,
            "combined_effect_calculation": f"T-test ({t_test_total_effect:.2f}) + Proportion ({prop_total_effect:.2f}) = {combined_total_effect:.2f}",

            # Bayesian test summary
            "bayesian_significant_segments": len(bayesian_significant_results),
            "bayesian_significance_rate": len(bayesian_significant_results) / len(results) if results else 0,
            "bayesian_avg_prob_treatment_better": avg_bayesian_prob,
            "bayesian_avg_expected_loss": avg_expected_loss,

            # Legacy fields for backward compatibility
            "significant_segments": len(t_significant_results),
            "non_significant_segments": len(results) - len(t_significant_results),
            "significance_rate": len(t_significant_results) / len(results) if results else 0,
            "average_significant_effect": avg_t_test_effect,
            "total_treatment_in_significant_segments": total_treatment_in_t_significant,
            "total_effect_size": t_test_total_effect,
            "effect_calculation": f"{avg_t_test_effect:.4f} × {total_treatment_in_t_significant} = {t_test_total_effect:.2f}",

            "total_treatment_customers": total_treatment,
            "total_control_customers": total_control,
            "treatment_control_ratio": total_treatment / total_control if total_control > 0 else None,

            "segments_with_adequate_power": len(adequate_samples),
            "segments_with_inadequate_power": len(inadequate_samples),
            "power_adequacy_rate": len(adequate_samples) / len(results) if results else 0,

            "detailed_results": [
                {
                    "segment": r.segment,
                    "treatment_n": r.treatment_size,
                    "control_n": r.control_size,
                    # T-test results
                    "effect": r.effect_size,
                    "cohens_d": r.cohens_d,
                    "p_value": r.p_value,
                    "significant": r.is_significant,
                    "power": r.power,
                    "adequate_sample": r.is_sample_adequate,
                    "ci_lower": r.confidence_interval[0],
                    "ci_upper": r.confidence_interval[1],
                    # Proportion test results
                    "treatment_prop": r.treatment_proportion,
                    "control_prop": r.control_proportion,
                    "prop_diff": r.proportion_diff,
                    "prop_p_value": r.proportion_p_value,
                    "prop_significant": r.proportion_is_significant,
                    "prop_effect": r.proportion_effect,
                    "prop_effect_per_customer": r.proportion_effect_per_customer,
                    # Combined
                    "total_effect": r.total_effect,
                    "total_effect_per_customer": r.total_effect_per_customer,
                    # Bayesian results
                    "bayesian_prob": r.bayesian_prob_treatment_better,
                    "bayesian_credible_lower": r.bayesian_credible_interval[0],
                    "bayesian_credible_upper": r.bayesian_credible_interval[1],
                    "bayesian_expected_loss": min(r.bayesian_expected_loss_treatment, r.bayesian_expected_loss_control),
                    "bayesian_relative_uplift": r.bayesian_relative_uplift,
                    "bayesian_significant": r.bayesian_is_significant
                }
                for r in results
            ],

            "recommendations": self._generate_recommendations(results)
        }

        return summary

    def _generate_recommendations(self, results: List[ABTestResult]) -> List[str]:
        """Generate actionable recommendations based on results"""
        recommendations = []

        significant_results = [r for r in results if r.is_significant]
        inadequate_samples = [r for r in results if not r.is_sample_adequate]

        if significant_results:
            positive_effects = [r for r in significant_results if r.effect_size > 0]
            negative_effects = [r for r in significant_results if r.effect_size < 0]

            if positive_effects:
                segments = [r.segment for r in positive_effects]
                recommendations.append(
                    f"POSITIVE IMPACT: Treatment shows significant positive effect in {len(positive_effects)} segment(s): {', '.join(segments)}. "
                    f"Consider rolling out treatment to these segments."
                )

            if negative_effects:
                segments = [r.segment for r in negative_effects]
                recommendations.append(
                    f"NEGATIVE IMPACT: Treatment shows significant negative effect in {len(negative_effects)} segment(s): {', '.join(segments)}. "
                    f"Investigate root cause before broader rollout."
                )
        else:
            recommendations.append(
                "NO SIGNIFICANT EFFECTS: No segments showed statistically significant differences. "
                "Consider extending experiment duration or increasing sample size."
            )

        if inadequate_samples:
            segments = [f"{r.segment} (needs ~{r.required_sample_size} per group)"
                       for r in inadequate_samples[:3]]
            recommendations.append(
                f"SAMPLE SIZE: {len(inadequate_samples)} segment(s) have insufficient statistical power. "
                f"Examples: {'; '.join(segments)}"
            )

        # Check for imbalanced groups
        imbalanced = [r for r in results if r.treatment_size / r.control_size > 2 or
                     r.control_size / r.treatment_size > 2]
        if imbalanced:
            recommendations.append(
                f"GROUP IMBALANCE: {len(imbalanced)} segment(s) have imbalanced treatment/control ratios. "
                f"Consider using stratified randomization in future experiments."
            )

        return recommendations

    def query_data(self, query: str) -> pd.DataFrame:
        """
        Execute a pandas query on the data
        Supports both query strings and column operations
        """
        if self.df is None:
            raise ValueError("No data loaded")

        try:
            result = self.df.query(query)
            return result
        except Exception:
            return self.df

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the loaded data"""
        if self.df is None:
            raise ValueError("No data loaded")

        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_summary": self.df.describe().to_dict(),
            "sample_rows": self.df.head(5).to_dict()
        }

    def get_segment_distribution(self) -> Dict[str, Any]:
        """Get distribution of customers across segments and groups"""
        if self.df is None:
            raise ValueError("No data loaded")

        result = {"columns_used": self.column_mapping}

        if "group" in self.column_mapping:
            group_col = self.column_mapping["group"]
            result["group_distribution"] = self.df[group_col].value_counts().to_dict()

        if "segment" in self.column_mapping:
            segment_col = self.column_mapping["segment"]
            result["segment_distribution"] = self.df[segment_col].value_counts().to_dict()

            if "group" in self.column_mapping:
                cross_tab = pd.crosstab(
                    self.df[segment_col],
                    self.df[group_col]
                ).to_dict()
                result["segment_by_group"] = cross_tab

        return result
