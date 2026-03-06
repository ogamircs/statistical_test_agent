"""
Data Models for A/B Testing Results

Contains dataclasses and type definitions for statistical analysis results.
"""

from collections.abc import Mapping as MappingABC
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


class LegacyMappingMixin(MappingABC):
    """Expose typed dataclasses through the legacy mapping interface during migration."""

    def to_legacy_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        return self.to_legacy_dict()[key]

    def __iter__(self):
        return iter(self.to_legacy_dict())

    def __len__(self) -> int:
        return len(self.to_legacy_dict())

    def __contains__(self, key: object) -> bool:
        return key in self.to_legacy_dict()

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_legacy_dict().get(key, default)

    def keys(self):
        return self.to_legacy_dict().keys()

    def items(self):
        return self.to_legacy_dict().items()

    def values(self):
        return self.to_legacy_dict().values()


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
class ABTestResult(LegacyMappingMixin):
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

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Expose the summary detail payload used by reports and charts."""
        experiment_quality = self.diagnostics.get("experiment_quality", {})
        srm = experiment_quality.get("srm", {})
        assumptions = experiment_quality.get("assumptions", {})
        outlier_sensitivity = experiment_quality.get("outlier_sensitivity", {})

        return {
            "segment": self.segment,
            "treatment_n": self.treatment_size,
            "control_n": self.control_size,
            "treatment_pre_mean": self.treatment_pre_mean,
            "treatment_post_mean": self.treatment_post_mean,
            "control_pre_mean": self.control_pre_mean,
            "control_post_mean": self.control_post_mean,
            "aa_test_passed": self.aa_test_passed,
            "aa_p_value": self.aa_p_value,
            "bootstrapping_applied": self.bootstrapping_applied,
            "original_control_size": self.original_control_size,
            "did_treatment_change": self.did_treatment_change,
            "did_control_change": self.did_control_change,
            "did_effect": self.did_effect,
            "effect": self.effect_size,
            "cohens_d": self.cohens_d,
            "p_value": self.p_value,
            "significant": self.is_significant,
            "p_value_adjusted": self.p_value_adjusted,
            "significant_adjusted": self.is_significant_adjusted,
            "power": self.power,
            "adequate_sample": self.is_sample_adequate,
            "ci_lower": self.confidence_interval[0],
            "ci_upper": self.confidence_interval[1],
            "metric_type": self.metric_type,
            "model_type": self.model_type,
            "model_effect": self.model_effect,
            "model_ci_lower": self.model_confidence_interval[0],
            "model_ci_upper": self.model_confidence_interval[1],
            "model_effect_scale": self.model_effect_scale,
            "model_effect_exponentiated": self.model_effect_exponentiated,
            "covariate_adjustment_applied": self.covariate_adjustment_applied,
            "covariates_used": list(self.covariates_used),
            "covariate_adjusted_effect": self.covariate_adjusted_effect,
            "covariate_adjusted_p_value": self.covariate_adjusted_p_value,
            "covariate_adjusted_ci_lower": self.covariate_adjusted_confidence_interval[0],
            "covariate_adjusted_ci_upper": self.covariate_adjusted_confidence_interval[1],
            "covariate_adjusted_model_type": self.covariate_adjusted_model_type,
            "covariate_adjusted_effect_scale": self.covariate_adjusted_effect_scale,
            "covariate_adjusted_effect_exponentiated": self.covariate_adjusted_effect_exponentiated,
            "treatment_prop": self.treatment_proportion,
            "control_prop": self.control_proportion,
            "prop_diff": self.proportion_diff,
            "prop_p_value": self.proportion_p_value,
            "prop_significant": self.proportion_is_significant,
            "prop_p_value_adjusted": self.proportion_p_value_adjusted,
            "prop_significant_adjusted": self.proportion_is_significant_adjusted,
            "multiple_testing_method": self.multiple_testing_method,
            "multiple_testing_applied": self.multiple_testing_applied,
            "inference_guardrail_triggered": self.inference_guardrail_triggered,
            "proportion_guardrail_triggered": self.proportion_guardrail_triggered,
            "diagnostics": self.diagnostics,
            "sequential_mode_enabled": self.sequential_mode_enabled,
            "sequential_method": self.sequential_method,
            "sequential_look_index": self.sequential_look_index,
            "sequential_max_looks": self.sequential_max_looks,
            "sequential_information_fraction": self.sequential_information_fraction,
            "sequential_alpha_spent": self.sequential_alpha_spent,
            "sequential_stop_recommended": self.sequential_stop_recommended,
            "sequential_decision": self.sequential_decision,
            "sequential_rationale": self.sequential_rationale,
            "sequential_thresholds": self.sequential_thresholds,
            "srm_p_value": srm.get("p_value"),
            "srm_is_mismatch": srm.get("is_sample_ratio_mismatch", False),
            "assumption_diagnostics": assumptions,
            "outlier_sensitivity_diagnostics": outlier_sensitivity,
            "prop_effect": self.proportion_effect,
            "prop_effect_per_customer": self.proportion_effect_per_customer,
            "total_effect": self.total_effect,
            "total_effect_per_customer": self.total_effect_per_customer,
            "bayesian_prob": self.bayesian_prob_treatment_better,
            "bayesian_credible_lower": self.bayesian_credible_interval[0],
            "bayesian_credible_upper": self.bayesian_credible_interval[1],
            "bayesian_expected_loss": min(
                self.bayesian_expected_loss_treatment,
                self.bayesian_expected_loss_control,
            ),
            "bayesian_relative_uplift": self.bayesian_relative_uplift,
            "bayesian_significant": self.bayesian_is_significant,
            "bayesian_total_effect": self.bayesian_total_effect,
            "bayesian_total_effect_per_customer": self.bayesian_total_effect_per_customer,
        }
@dataclass(frozen=True)
class SegmentAnalysisFailure(LegacyMappingMixin):
    """Structured record for a segment skipped during analysis."""

    segment: str
    error: str
    error_type: str = "ValueError"

    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            "segment": self.segment,
            "error": self.error,
            "error_type": self.error_type,
        }


@dataclass
class ABTestSummary(LegacyMappingMixin):
    """Typed summary payload shared by the analyzer, reports, and charts."""

    total_segments_analyzed: int = 0
    segment_failures: List[SegmentAnalysisFailure] = field(default_factory=list)
    analysis_warnings: List[str] = field(default_factory=list)
    aa_test_passed_segments: int = 0
    aa_test_failed_segments: int = 0
    bootstrapped_segments: int = 0
    aa_failed_segment_names: List[str] = field(default_factory=list)
    did_avg_effect: float = 0.0
    did_total_effect: float = 0.0
    t_test_significant_segments: int = 0
    t_test_significant_segments_adjusted: int = 0
    t_test_significance_rate: float = 0.0
    t_test_significance_rate_adjusted: float = 0.0
    t_test_avg_effect: float = 0.0
    t_test_total_effect: float = 0.0
    t_test_effect_calculation: str = "0.0000 × 0 = 0.00"
    prop_test_significant_segments: int = 0
    prop_test_significant_segments_adjusted: int = 0
    prop_test_significance_rate: float = 0.0
    prop_test_significance_rate_adjusted: float = 0.0
    prop_test_avg_effect: float = 0.0
    prop_test_total_effect: float = 0.0
    prop_test_effect_calculation: str = "0.0000 × 0 = 0.00"
    multiple_testing_method: str = "none"
    multiple_testing_applied_segments: int = 0
    covariate_adjusted_segments: int = 0
    sequential_mode_segments: int = 0
    sequential_stop_recommended_segments: int = 0
    sequential_continue_segments: int = 0
    sequential_stop_segment_names: List[str] = field(default_factory=list)
    sequential_decision_breakdown: Dict[str, int] = field(default_factory=dict)
    metric_type_breakdown: Dict[str, int] = field(default_factory=dict)
    model_type_breakdown: Dict[str, int] = field(default_factory=dict)
    inference_guardrail_segments: int = 0
    proportion_guardrail_segments: int = 0
    srm_mismatch_segments: int = 0
    srm_mismatch_segment_names: List[str] = field(default_factory=list)
    assumption_warning_segments: int = 0
    assumption_warning_segment_names: List[str] = field(default_factory=list)
    outlier_sensitive_segments: int = 0
    outlier_sensitive_segment_names: List[str] = field(default_factory=list)
    guardrail_segment_names: List[str] = field(default_factory=list)
    combined_total_effect: float = 0.0
    combined_effect_calculation: str = "T-test (0.00) + Proportion (0.00) = 0.00"
    bayesian_significant_segments: int = 0
    bayesian_significance_rate: float = 0.0
    bayesian_avg_prob_treatment_better: float = 0.0
    bayesian_avg_expected_loss: float = 0.0
    bayesian_total_effect: float = 0.0
    significant_segments: int = 0
    non_significant_segments: int = 0
    significance_rate: float = 0.0
    average_significant_effect: float = 0.0
    total_treatment_in_significant_segments: int = 0
    total_effect_size: float = 0.0
    effect_calculation: str = "0.0000 × 0 = 0.00"
    total_treatment_customers: int = 0
    total_control_customers: int = 0
    treatment_control_ratio: Optional[float] = None
    segments_with_adequate_power: int = 0
    segments_with_inadequate_power: int = 0
    power_adequacy_rate: float = 0.0
    detailed_results: List[ABTestResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_legacy_dict(self) -> Dict[str, Any]:
        payload = {
            field_def.name: getattr(self, field_def.name)
            for field_def in fields(self)
            if field_def.name not in {"segment_failures", "detailed_results"}
        }
        payload["segment_failures"] = [failure.to_legacy_dict() for failure in self.segment_failures]
        payload["detailed_results"] = [result.to_legacy_dict() for result in self.detailed_results]
        return payload


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
_SUMMARY_FIELDS = tuple(fields(ABTestSummary))
_SUMMARY_FIELD_DEFAULTS = {field_def.name: field_def for field_def in _SUMMARY_FIELDS}


def _field_default(field_name: str) -> Any:
    field_def = _AB_RESULT_FIELD_DEFAULTS[field_name]
    if field_def.default is not MISSING:
        return field_def.default
    if field_def.default_factory is not MISSING:
        return field_def.default_factory()
    raise ValueError(f"Missing required canonical field '{field_name}'")


def _summary_field_default(field_name: str) -> Any:
    field_def = _SUMMARY_FIELD_DEFAULTS[field_name]
    if field_def.default is not MISSING:
        return field_def.default
    if field_def.default_factory is not MISSING:
        return field_def.default_factory()
    raise ValueError(f"Missing required summary field '{field_name}'")


def _result_to_mapping(result: Any) -> Dict[str, Any]:
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, Mapping):
        return dict(result)
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


def to_segment_analysis_failure(failure: Any) -> SegmentAnalysisFailure:
    """Normalize failure payloads to the typed segment failure record."""
    if isinstance(failure, SegmentAnalysisFailure):
        return failure

    data = _result_to_mapping(failure)
    return SegmentAnalysisFailure(
        segment=str(data.get("segment", "Unknown")),
        error=str(data.get("error", "Unknown error")),
        error_type=str(data.get("error_type", "ValueError")),
    )


def to_ab_test_summary(summary: Any) -> ABTestSummary:
    """Normalize summary payloads into the typed ABTestSummary model."""
    if isinstance(summary, ABTestSummary):
        return summary

    if not isinstance(summary, MappingABC):
        data = _result_to_mapping(summary)
    else:
        data = dict(summary)

    payload: Dict[str, Any] = {}
    for field_def in _SUMMARY_FIELDS:
        field_name = field_def.name
        if field_name == "detailed_results":
            raw_results = data.get(field_name, [])
            payload[field_name] = normalize_ab_test_results(raw_results)
            continue
        if field_name == "segment_failures":
            raw_failures = data.get(field_name, [])
            payload[field_name] = [to_segment_analysis_failure(failure) for failure in raw_failures]
            continue
        value = data.get(field_name, None)
        payload[field_name] = _summary_field_default(field_name) if value is None else value

    return ABTestSummary(**payload)
