"""
Data Models for A/B Testing Results

Contains dataclasses and type definitions for statistical analysis results.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class ABTestResult:
    """Container for A/B test results for a single segment"""

    # Basic info
    segment: str
    treatment_size: int
    control_size: int

    # T-test results (continuous metric)
    treatment_mean: float
    control_mean: float
    effect_size: float  # Absolute difference in means
    cohens_d: float
    t_statistic: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    power: float
    required_sample_size: int
    is_sample_adequate: bool

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
