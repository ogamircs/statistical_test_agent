"""
Data Models for A/B Testing Results

Contains dataclasses and type definitions for statistical analysis results.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ABTestResult:
    """Container for A/B test results for a single segment"""

    segment: str
    treatment_size: int
    control_size: int
    treatment_mean: float
    control_mean: float
    effect_size: float
    cohens_d: float
    t_statistic: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    power: float
    required_sample_size: int
    is_sample_adequate: bool
