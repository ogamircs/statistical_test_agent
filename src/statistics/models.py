"""
Data Models for A/B Testing Results

Contains dataclasses and type definitions for statistical analysis results.
"""

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


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
    p_value_adjusted: float = 1.0  # BH/FDR-adjusted p-value across segments
    is_significant_adjusted: bool = False
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    metric_type: str = "continuous"  # continuous | binary | count | heavy_tail
    model_type: str = "ols_hc3"
    model_effect: float = 0.0
    model_confidence_interval: Tuple[float, float] = (0.0, 0.0)
    model_effect_scale: str = "mean_difference"
    model_effect_exponentiated: float = 1.0
    covariate_adjustment_applied: bool = False
    covariates_used: List[str] = field(default_factory=list)
    covariate_adjusted_effect: float = 0.0
    covariate_adjusted_p_value: float = 1.0
    covariate_adjusted_confidence_interval: Tuple[float, float] = (0.0, 0.0)
    covariate_adjusted_model_type: str = "none"
    covariate_adjusted_effect_scale: str = "mean_difference"
    covariate_adjusted_effect_exponentiated: float = 1.0
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
    proportion_p_value_adjusted: float = 1.0
    proportion_is_significant_adjusted: bool = False
    multiple_testing_method: str = "none"  # e.g., fdr_bh
    multiple_testing_applied: bool = False

    # Frequentist guardrails and diagnostics
    inference_guardrail_triggered: bool = False
    proportion_guardrail_triggered: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Sequential testing decision support (optional; additive metadata)
    sequential_mode_enabled: bool = False
    sequential_method: str = "none"
    sequential_look_index: int = 0
    sequential_max_looks: int = 0
    sequential_information_fraction: float = 0.0
    sequential_alpha_spent: float = 0.0
    sequential_stop_recommended: bool = False
    sequential_decision: str = "not_requested"
    sequential_rationale: str = ""
    sequential_thresholds: Dict[str, Any] = field(default_factory=dict)

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


_CANONICAL_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "treatment_size": ("treatment_n",),
    "control_size": ("control_n",),
    "effect_size": ("effect",),
    "is_significant": ("significant",),
    "p_value_adjusted": ("adjusted_p_value", "p_adj"),
    "is_significant_adjusted": ("significant_adjusted", "significant_fdr"),
    "is_sample_adequate": ("adequate_sample",),
    "proportion_p_value": ("prop_p_value",),
    "proportion_is_significant": ("prop_significant",),
    "proportion_p_value_adjusted": ("prop_p_value_adjusted", "prop_p_adj"),
    "proportion_is_significant_adjusted": ("prop_significant_adjusted",),
    "bayesian_prob_treatment_better": ("bayesian_prob",),
}

_AB_RESULT_FIELDS = tuple(fields(ABTestResult))
_AB_RESULT_FIELD_DEFAULTS = {field_def.name: field_def for field_def in _AB_RESULT_FIELDS}


def _field_default(field_name: str) -> Any:
    field_def = _AB_RESULT_FIELD_DEFAULTS[field_name]
    if field_def.default is not MISSING:
        return field_def.default
    if field_def.default_factory is not MISSING:
        return field_def.default_factory()
    raise ValueError(f"Missing required canonical field '{field_name}'")


def _result_to_mapping(result: Any) -> Dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "__dict__"):
        return dict(vars(result))
    raise TypeError(f"Unsupported result type: {type(result)!r}")


def _extract_value(data: Mapping[str, Any], field_name: str) -> Any:
    keys = (field_name, *_CANONICAL_FIELD_ALIASES.get(field_name, ()))
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _extract_interval(
    data: Mapping[str, Any],
    interval_key: str,
    lower_keys: Tuple[str, ...],
    upper_keys: Tuple[str, ...],
) -> Optional[Tuple[float, float]]:
    interval = data.get(interval_key)
    if isinstance(interval, (tuple, list)) and len(interval) == 2:
        return (float(interval[0]), float(interval[1]))

    lower = next((data[k] for k in lower_keys if k in data and data[k] is not None), None)
    upper = next((data[k] for k in upper_keys if k in data and data[k] is not None), None)
    if lower is None or upper is None:
        return None
    return (float(lower), float(upper))


def to_canonical_ab_test_result(result: Any) -> ABTestResult:
    """
    Normalize analyzer output into the canonical ABTestResult schema.

    Supports:
    - Native ABTestResult objects
    - Spark result objects/dataclasses with lower/upper interval keys
    - Dict payloads using either canonical or legacy field aliases
    """
    if isinstance(result, ABTestResult) and type(result) is ABTestResult:
        return result

    to_canonical = getattr(result, "to_canonical_result", None)
    if callable(to_canonical):
        converted = to_canonical()
        if isinstance(converted, ABTestResult):
            return converted
        result = converted

    data = _result_to_mapping(result)
    payload: Dict[str, Any] = {}

    confidence_interval = _extract_interval(
        data,
        interval_key="confidence_interval",
        lower_keys=("confidence_interval_lower", "ci_lower"),
        upper_keys=("confidence_interval_upper", "ci_upper"),
    )
    if confidence_interval is not None:
        payload["confidence_interval"] = confidence_interval

    bayesian_interval = _extract_interval(
        data,
        interval_key="bayesian_credible_interval",
        lower_keys=("bayesian_credible_interval_lower", "bayesian_credible_lower"),
        upper_keys=("bayesian_credible_interval_upper", "bayesian_credible_upper"),
    )
    if bayesian_interval is not None:
        payload["bayesian_credible_interval"] = bayesian_interval

    for field_def in _AB_RESULT_FIELDS:
        if field_def.name in payload:
            continue
        value = _extract_value(data, field_def.name)
        payload[field_def.name] = _field_default(field_def.name) if value is None else value

    return ABTestResult(**payload)


def normalize_ab_test_results(results: Iterable[Any]) -> List[ABTestResult]:
    """Normalize a sequence of results into canonical ABTestResult objects."""
    return [to_canonical_ab_test_result(result) for result in results]


def canonical_result_as_dict(result: Any, include_legacy_aliases: bool = False) -> Dict[str, Any]:
    """Serialize any backend result object to canonical dict shape."""
    canonical = to_canonical_ab_test_result(result)
    payload = asdict(canonical)

    if include_legacy_aliases:
        payload.update(
            {
                "treatment_n": canonical.treatment_size,
                "control_n": canonical.control_size,
                "effect": canonical.effect_size,
                "significant": canonical.is_significant,
                "adjusted_p_value": canonical.p_value_adjusted,
                "significant_adjusted": canonical.is_significant_adjusted,
                "adequate_sample": canonical.is_sample_adequate,
                "ci_lower": canonical.confidence_interval[0],
                "ci_upper": canonical.confidence_interval[1],
                "prop_p_value": canonical.proportion_p_value,
                "prop_significant": canonical.proportion_is_significant,
                "prop_p_value_adjusted": canonical.proportion_p_value_adjusted,
                "prop_significant_adjusted": canonical.proportion_is_significant_adjusted,
                "bayesian_prob": canonical.bayesian_prob_treatment_better,
                "bayesian_credible_lower": canonical.bayesian_credible_interval[0],
                "bayesian_credible_upper": canonical.bayesian_credible_interval[1],
            }
        )

    return payload
