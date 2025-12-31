"""
Data Models for A/B Testing Results

Contains dataclasses and type definitions for statistical analysis results.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List


@dataclass
class AATestResult:
    """Container for AA test results to check treatment/control balance"""

    segment: str
    treatment_size: int
    control_size: int

    # Pre-effect statistics
    treatment_pre_mean: float
    control_pre_mean: float
    pre_effect_diff: float  # Difference in pre-effect means

    # AA test statistics (t-test on pre_effect)
    aa_t_statistic: float
    aa_p_value: float
    is_balanced: bool  # True if p > 0.05 (no significant difference)

    # Bootstrapping info (if applied)
    bootstrapping_applied: bool = False
    original_control_size: int = 0
    balanced_control_size: int = 0
    bootstrap_iterations: int = 0


@dataclass
class ABTestResult:
    """Container for A/B test results for a single segment"""

    # Basic info
    segment: str
    treatment_size: int
    control_size: int

    # Pre/Post effect values
    treatment_pre_mean: float = 0.0
    treatment_post_mean: float = 0.0
    control_pre_mean: float = 0.0
    control_post_mean: float = 0.0

    # T-test results (continuous metric - on post_effect)
    treatment_mean: float = 0.0  # Same as treatment_post_mean for backward compatibility
    control_mean: float = 0.0    # Same as control_post_mean for backward compatibility
    effect_size: float = 0.0     # Absolute difference in post means
    cohens_d: float = 0.0
    t_statistic: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    power: float = 0.0
    required_sample_size: int = 0
    is_sample_adequate: bool = False

    # Difference-in-Differences (DiD) effect
    # True causal effect = (post_treatment - pre_treatment) - (post_control - pre_control)
    did_treatment_change: float = 0.0  # post_treatment - pre_treatment
    did_control_change: float = 0.0    # post_control - pre_control
    did_effect: float = 0.0            # Difference-in-differences effect

    # AA test results (pre-experiment balance check)
    aa_test_passed: bool = True
    aa_p_value: float = 1.0
    bootstrapping_applied: bool = False
    original_control_size: int = 0

    # Proportion test results (conversion/activation rate)
    treatment_proportion: float = 0.0  # Proportion with non-zero effect in treatment
    control_proportion: float = 0.0    # Proportion with non-zero effect in control
    proportion_diff: float = 0.0       # Treatment proportion - Control proportion
    proportion_z_stat: float = 0.0     # Z-statistic for proportion test
    proportion_p_value: float = 1.0    # P-value for proportion test
    proportion_is_significant: bool = False

    # Proportion-based effect (incremental conversions * control mean)
    # This represents value from customers who converted ONLY because of treatment
    proportion_effect: float = 0.0     # Additional proportion × control mean × treatment N
    proportion_effect_per_customer: float = 0.0  # proportion_diff × control_mean

    # Combined effects
    total_effect: float = 0.0          # T-test effect + proportion effect
    total_effect_per_customer: float = 0.0  # Combined per-customer effect

    # Bayesian test results (now using DiD for total effect calculation)
    bayesian_prob_treatment_better: float = 0.5  # P(treatment > control)
    bayesian_expected_loss_treatment: float = 0.0  # Expected loss if choosing treatment
    bayesian_expected_loss_control: float = 0.0    # Expected loss if choosing control
    bayesian_credible_interval: Tuple[float, float] = (0.0, 0.0)  # 95% credible interval for DiD effect
    bayesian_relative_uplift: float = 0.0  # Relative improvement based on DiD
    bayesian_is_significant: bool = False  # True if prob > 0.95 or prob < 0.05

    # Bayesian total effect (using pre/post difference)
    bayesian_total_effect: float = 0.0  # Expected total effect from Bayesian DiD
    bayesian_total_effect_per_customer: float = 0.0
