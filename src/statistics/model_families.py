"""Metric-family inference models used by the statsmodels analysis engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kurtosis, skew
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from .engine_helpers import build_diagnostics, sanitize_p_value, zero_if_tiny


def coerce_covariate_frame(
    covariates: pd.DataFrame | np.ndarray | None,
    *,
    n_rows: int,
    names: Sequence[str] | None = None,
    prefix: str = "covariate",
) -> pd.DataFrame:
    """Coerce covariates to a numeric DataFrame with stable column names."""
    if covariates is None:
        return pd.DataFrame(index=np.arange(n_rows))

    if isinstance(covariates, pd.DataFrame):
        frame = covariates.copy()
    else:
        array = np.asarray(covariates)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            return pd.DataFrame(index=np.arange(n_rows))
        frame = pd.DataFrame(array)

    if len(frame) != n_rows:
        return pd.DataFrame(index=np.arange(n_rows))

    frame = frame.reset_index(drop=True)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if names is not None and len(names) == frame.shape[1]:
        frame.columns = [str(name) for name in names]
    else:
        frame.columns = [f"{prefix}_{idx + 1}" for idx in range(frame.shape[1])]
    return frame


def is_binary_metric(values: np.ndarray) -> bool:
    finite = np.asarray(values, dtype=float)
    if finite.size == 0:
        return False
    unique_vals = np.unique(finite)
    return bool(np.all(np.isin(unique_vals, [0.0, 1.0])))


def is_count_metric(values: np.ndarray) -> bool:
    finite = np.asarray(values, dtype=float)
    if finite.size == 0 or np.any(finite < 0) or is_binary_metric(finite):
        return False
    return bool(np.all(np.isclose(finite, np.round(finite), atol=1e-8)))


def is_heavy_tail_metric(values: np.ndarray, *, variance_epsilon: float) -> bool:
    finite = np.asarray(values, dtype=float)
    if finite.size < 20:
        return False
    try:
        abs_skew = abs(float(skew(finite, bias=False)))
        excess_kurtosis = float(kurtosis(finite, fisher=True, bias=False))
    except Exception:
        return False

    if not np.isfinite(abs_skew) or not np.isfinite(excess_kurtosis):
        return False
    if abs_skew > 2.0 or excess_kurtosis > 7.0:
        return True

    p95 = float(np.percentile(finite, 95))
    p99 = float(np.percentile(finite, 99))
    p50 = float(np.percentile(finite, 50))
    spread = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    if abs(p50) <= variance_epsilon:
        return (p99 - p95) > max(5.0 * spread, 1.0)
    return abs(p99 / p50) > 15.0


def infer_metric_type(
    values: np.ndarray,
    *,
    requested_metric_type: str = "auto",
    variance_epsilon: float,
) -> str:
    """Infer a supported metric family unless explicitly specified."""
    requested = str(requested_metric_type or "auto").strip().lower()
    supported = {"auto", "continuous", "binary", "count", "heavy_tail"}
    if requested in supported and requested != "auto":
        return requested

    if is_binary_metric(values):
        return "binary"
    if is_count_metric(values):
        return "count"
    if is_heavy_tail_metric(values, variance_epsilon=variance_epsilon):
        return "heavy_tail"
    return "continuous"


def extract_term_inference(
    model_result: Any,
    *,
    term: str = "treatment",
    significance_level: float,
) -> Dict[str, float]:
    """Extract coefficient, statistic, p-value, and confidence interval for a term."""
    params = model_result.params
    try:
        coef = float(params[term])
        term_index = list(getattr(params, "index", [])).index(term)
    except Exception:
        term_index = 1
        coef = float(np.asarray(params)[term_index])

    if hasattr(model_result, "tvalues"):
        try:
            stat_value = float(model_result.tvalues[term])
        except Exception:
            stat_value = float(np.asarray(model_result.tvalues)[term_index])
    elif hasattr(model_result, "zvalues"):
        try:
            stat_value = float(model_result.zvalues[term])
        except Exception:
            stat_value = float(np.asarray(model_result.zvalues)[term_index])
    else:
        stat_value = 0.0

    try:
        raw_p = model_result.pvalues[term]
    except Exception:
        raw_p = (
            np.asarray(model_result.pvalues)[term_index]
            if hasattr(model_result, "pvalues")
            else 1.0
        )
    p_value = sanitize_p_value(raw_p)

    try:
        conf = model_result.conf_int(alpha=significance_level)
        if hasattr(conf, "loc"):
            ci_low = float(conf.loc[term][0])
            ci_high = float(conf.loc[term][1])
        else:
            row = np.asarray(conf)[term_index]
            ci_low = float(row[0])
            ci_high = float(row[1])
    except Exception:
        ci_low, ci_high = coef, coef

    return {
        "coef": float(coef),
        "statistic": float(stat_value),
        "p_value": float(p_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def estimate_treatment_effect(
    *,
    treatment_data: np.ndarray,
    control_data: np.ndarray,
    significance_level: float,
    min_recommended_sample_size: int,
    variance_epsilon: float,
    metric_type: str = "auto",
    treatment_covariates: pd.DataFrame | np.ndarray | None = None,
    control_covariates: pd.DataFrame | np.ndarray | None = None,
    covariate_names: Sequence[str] | None = None,
    count_model: str = "auto",
    heavy_tail_strategy: str = "robust",
) -> Dict[str, Any]:
    """Estimate treatment effect using a metric-aware model family."""
    treatment_array = np.asarray(treatment_data, dtype=float).reshape(-1)
    control_array = np.asarray(control_data, dtype=float).reshape(-1)

    treatment_covariate_frame = coerce_covariate_frame(
        treatment_covariates,
        n_rows=len(treatment_array),
        names=covariate_names,
        prefix="covariate",
    )
    control_covariate_frame = coerce_covariate_frame(
        control_covariates,
        n_rows=len(control_array),
        names=covariate_names,
        prefix="covariate",
    )
    if treatment_covariate_frame.shape[1] != control_covariate_frame.shape[1]:
        treatment_covariate_frame = pd.DataFrame(index=np.arange(len(treatment_array)))
        control_covariate_frame = pd.DataFrame(index=np.arange(len(control_array)))

    covariate_columns = list(treatment_covariate_frame.columns)
    treatment_frame = pd.DataFrame({"y": treatment_array, "treatment": 1.0})
    control_frame = pd.DataFrame({"y": control_array, "treatment": 0.0})
    for column in covariate_columns:
        treatment_frame[column] = treatment_covariate_frame[column]
        control_frame[column] = control_covariate_frame[column]

    model_frame = pd.concat([treatment_frame, control_frame], ignore_index=True)
    numeric_columns = ["y", "treatment", *covariate_columns]
    finite_mask = np.isfinite(model_frame[numeric_columns].to_numpy(dtype=float)).all(axis=1)
    non_finite_removed = int(len(model_frame) - np.count_nonzero(finite_mask))
    model_frame = model_frame.loc[finite_mask].reset_index(drop=True)

    if covariate_columns:
        covariate_columns = [
            column
            for column in covariate_columns
            if model_frame[column].nunique(dropna=True) > 1
        ]

    treatment_values = model_frame.loc[model_frame["treatment"] == 1.0, "y"].to_numpy()
    control_values = model_frame.loc[model_frame["treatment"] == 0.0, "y"].to_numpy()

    n_treatment = len(treatment_values)
    n_control = len(control_values)
    treatment_mean = float(np.mean(treatment_values)) if n_treatment > 0 else 0.0
    control_mean = float(np.mean(control_values)) if n_control > 0 else 0.0
    raw_effect = zero_if_tiny(treatment_mean - control_mean)

    reasons = []
    if non_finite_removed > 0:
        reasons.append("non_finite_values_removed")

    small_n = n_treatment < min_recommended_sample_size or n_control < min_recommended_sample_size
    if small_n:
        reasons.append("small_sample_size")

    if n_treatment < 2 or n_control < 2:
        diagnostics = build_diagnostics(
            [*reasons, "insufficient_sample_size"],
            blocks_significance=True,
            small_n=True,
            degenerate_variance=False,
            non_finite_values_removed=non_finite_removed,
        )
        return {
            "treatment_mean": treatment_mean,
            "control_mean": control_mean,
            "effect_size": 0.0,
            "t_statistic": 0.0,
            "p_value": 1.0,
            "confidence_interval": (0.0, 0.0),
            "metric_type": "continuous",
            "model_type": "insufficient_data",
            "model_effect": 0.0,
            "model_confidence_interval": (0.0, 0.0),
            "model_effect_scale": "mean_difference",
            "model_effect_exponentiated": 1.0,
            "covariate_adjusted": bool(covariate_columns),
            "covariates_used": covariate_columns,
            "diagnostics": diagnostics,
        }

    var_treatment = float(np.var(treatment_values, ddof=1))
    var_control = float(np.var(control_values, ddof=1))
    degenerate_variance = var_treatment <= variance_epsilon or var_control <= variance_epsilon
    if degenerate_variance:
        reasons.append("degenerate_variance")

    selected_metric_type = infer_metric_type(
        model_frame["y"].to_numpy(dtype=float),
        requested_metric_type=metric_type,
        variance_epsilon=variance_epsilon,
    )

    if var_treatment <= variance_epsilon and var_control <= variance_epsilon:
        if raw_effect == 0.0:
            t_statistic = 0.0
            p_value = 1.0
        else:
            t_statistic = float(np.sign(raw_effect) * np.inf)
            p_value = 0.0

        diagnostics = build_diagnostics(
            reasons,
            blocks_significance=False,
            small_n=small_n,
            degenerate_variance=True,
            non_finite_values_removed=non_finite_removed,
            metric_type=selected_metric_type,
            model_type="deterministic",
        )
        return {
            "treatment_mean": treatment_mean,
            "control_mean": control_mean,
            "effect_size": raw_effect,
            "t_statistic": t_statistic,
            "p_value": p_value,
            "confidence_interval": (raw_effect, raw_effect),
            "metric_type": selected_metric_type,
            "model_type": "deterministic",
            "model_effect": raw_effect,
            "model_confidence_interval": (raw_effect, raw_effect),
            "model_effect_scale": "mean_difference",
            "model_effect_exponentiated": 1.0,
            "covariate_adjusted": bool(covariate_columns),
            "covariates_used": covariate_columns,
            "diagnostics": diagnostics,
        }

    if (
        selected_metric_type in {"binary", "count"}
        and np.unique(model_frame["y"].to_numpy(dtype=float)).size <= 1
    ):
        diagnostics = build_diagnostics(
            [*reasons, "degenerate_outcome"],
            blocks_significance=True,
            small_n=small_n,
            degenerate_variance=degenerate_variance,
            non_finite_values_removed=non_finite_removed,
            metric_type=selected_metric_type,
            model_type=f"glm_{selected_metric_type}",
        )
        return {
            "treatment_mean": treatment_mean,
            "control_mean": control_mean,
            "effect_size": raw_effect,
            "t_statistic": 0.0,
            "p_value": 1.0,
            "confidence_interval": (0.0, 0.0),
            "metric_type": selected_metric_type,
            "model_type": f"glm_{selected_metric_type}",
            "model_effect": 0.0,
            "model_confidence_interval": (0.0, 0.0),
            "model_effect_scale": "log_odds" if selected_metric_type == "binary" else "log_rate",
            "model_effect_exponentiated": 1.0,
            "covariate_adjusted": bool(covariate_columns),
            "covariates_used": covariate_columns,
            "diagnostics": diagnostics,
        }

    X = sm.add_constant(model_frame[["treatment", *covariate_columns]], has_constant="add")
    y = model_frame["y"].to_numpy(dtype=float)

    model_type = "ols_hc3"
    model_effect_scale = "mean_difference"
    model_effect_exponentiated = 1.0
    model_fit_reason: str | None = None

    try:
        if selected_metric_type == "binary":
            model_type = "glm_binomial"
            model_effect_scale = "log_odds"
            fitted = sm.GLM(y, X, family=sm.families.Binomial()).fit()
            term_stats = extract_term_inference(
                fitted,
                term="treatment",
                significance_level=significance_level,
            )
            model_effect_exponentiated = float(np.exp(term_stats["coef"]))
        elif selected_metric_type == "count":
            requested_count_model = str(count_model or "auto").strip().lower()
            mean_y = float(np.mean(y)) if y.size else 0.0
            var_y = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
            overdispersed = mean_y > variance_epsilon and var_y > 1.5 * mean_y
            use_negative_binomial = (
                requested_count_model == "negative_binomial"
                or (requested_count_model == "auto" and overdispersed)
            )
            family = (
                sm.families.NegativeBinomial(alpha=1.0)
                if use_negative_binomial
                else sm.families.Poisson()
            )
            model_type = "glm_negative_binomial" if use_negative_binomial else "glm_poisson"
            model_effect_scale = "log_rate"
            fitted = sm.GLM(y, X, family=family).fit()
            term_stats = extract_term_inference(
                fitted,
                term="treatment",
                significance_level=significance_level,
            )
            model_effect_exponentiated = float(np.exp(term_stats["coef"]))
        elif selected_metric_type == "heavy_tail":
            strategy = str(heavy_tail_strategy or "robust").strip().lower()
            if strategy == "log_transform":
                if np.min(y) <= -1.0:
                    shift = abs(float(np.min(y))) + 1.0
                    transformed_y = np.log(y + shift)
                else:
                    transformed_y = np.log1p(y)
                model_type = "ols_log1p_hc3"
                model_effect_scale = "log_mean_difference"
                fitted = sm.OLS(transformed_y, X).fit(cov_type="HC3")
                term_stats = extract_term_inference(
                    fitted,
                    term="treatment",
                    significance_level=significance_level,
                )
                model_effect_exponentiated = float(np.exp(term_stats["coef"]))
            else:
                model_type = "rlm_huber"
                model_effect_scale = "location_shift"
                fitted = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
                term_stats = extract_term_inference(
                    fitted,
                    term="treatment",
                    significance_level=significance_level,
                )
        else:
            fitted = sm.OLS(y, X).fit(cov_type="HC3")
            term_stats = extract_term_inference(
                fitted,
                term="treatment",
                significance_level=significance_level,
            )
    except Exception:
        model_fit_reason = "model_fit_failed_fallback_to_ols"
        try:
            model_type = "ols_hc3_fallback"
            fitted = sm.OLS(y, X).fit(cov_type="HC3")
            term_stats = extract_term_inference(
                fitted,
                term="treatment",
                significance_level=significance_level,
            )
        except Exception:
            d_treatment = DescrStatsW(treatment_values)
            d_control = DescrStatsW(control_values)
            compare = CompareMeans(d_treatment, d_control)
            t_stat, p_val, _df = compare.ttest_ind(usevar="unequal")
            ci_low, ci_high = compare.tconfint_diff(
                alpha=significance_level,
                usevar="unequal",
            )
            model_type = "compare_means_fallback"
            term_stats = {
                "coef": raw_effect,
                "statistic": float(t_stat),
                "p_value": sanitize_p_value(p_val),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }

    model_effect = zero_if_tiny(float(term_stats["coef"]))
    raw_p_value = float(term_stats["p_value"])
    p_value = sanitize_p_value(raw_p_value)
    if p_value != raw_p_value:
        reasons.append("invalid_p_value")
    if model_fit_reason is not None:
        reasons.append(model_fit_reason)

    diagnostics = build_diagnostics(
        reasons,
        blocks_significance=small_n,
        small_n=small_n,
        degenerate_variance=degenerate_variance,
        non_finite_values_removed=non_finite_removed,
        metric_type=selected_metric_type,
        model_type=model_type,
        covariate_adjusted=bool(covariate_columns),
        covariates_used=covariate_columns,
    )

    return {
        "treatment_mean": treatment_mean,
        "control_mean": control_mean,
        "effect_size": raw_effect,
        "t_statistic": float(term_stats["statistic"]),
        "p_value": p_value,
        "confidence_interval": (float(term_stats["ci_low"]), float(term_stats["ci_high"])),
        "metric_type": selected_metric_type,
        "model_type": model_type,
        "model_effect": model_effect,
        "model_confidence_interval": (
            float(term_stats["ci_low"]),
            float(term_stats["ci_high"]),
        ),
        "model_effect_scale": model_effect_scale,
        "model_effect_exponentiated": float(model_effect_exponentiated),
        "covariate_adjusted": bool(covariate_columns),
        "covariates_used": covariate_columns,
        "diagnostics": diagnostics,
    }


def estimate_did_effect(
    *,
    treatment_pre: np.ndarray,
    treatment_post: np.ndarray,
    control_pre: np.ndarray,
    control_post: np.ndarray,
) -> Dict[str, float]:
    """Estimate difference-in-differences with OLS fallback to raw deltas."""
    treatment_change = treatment_post - treatment_pre
    control_change = control_post - control_pre

    if len(treatment_change) < 2 or len(control_change) < 2:
        did_effect = zero_if_tiny(float(np.mean(treatment_change) - np.mean(control_change)))
        return {
            "did_effect": did_effect,
            "treatment_change": float(np.mean(treatment_change)),
            "control_change": float(np.mean(control_change)),
        }

    delta = np.concatenate([treatment_change, control_change])
    treatment_indicator = np.concatenate(
        [np.ones(len(treatment_change)), np.zeros(len(control_change))]
    )

    try:
        X = sm.add_constant(treatment_indicator, has_constant="add")
        ols_result = sm.OLS(delta, X).fit(cov_type="HC3")
        did_effect = zero_if_tiny(float(ols_result.params[1]))
    except Exception:
        did_effect = zero_if_tiny(float(np.mean(treatment_change) - np.mean(control_change)))

    return {
        "did_effect": did_effect,
        "treatment_change": float(np.mean(treatment_change)),
        "control_change": float(np.mean(control_change)),
    }
