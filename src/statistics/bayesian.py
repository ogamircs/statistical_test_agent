"""Bayesian Monte Carlo effect estimation for A/B testing."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def run_bayesian_test(
    *,
    treatment_post: np.ndarray,
    control_post: np.ndarray,
    treatment_pre: np.ndarray | None = None,
    control_pre: np.ndarray | None = None,
    n_samples: int,
) -> Dict[str, Any]:
    """Estimate treatment-vs-control uncertainty with a Monte Carlo posterior."""
    n_treatment = len(treatment_post)
    n_control = len(control_post)

    if n_treatment < 2 or n_control < 2:
        return {
            "prob_treatment_better": 0.5,
            "expected_loss_treatment": 0.0,
            "expected_loss_control": 0.0,
            "credible_interval": (0.0, 0.0),
            "relative_uplift": 0.0,
            "total_effect": 0.0,
            "did_effect": 0.0,
            "treatment_change": 0.0,
            "control_change": 0.0,
        }

    use_did = (
        treatment_pre is not None
        and control_pre is not None
        and len(treatment_pre) == n_treatment
        and len(control_pre) == n_control
    )

    if use_did:
        treatment_change = treatment_post - treatment_pre
        control_change = control_post - control_pre
        mean_t = float(np.mean(treatment_change))
        mean_c = float(np.mean(control_change))
        var_t = float(np.var(treatment_change, ddof=1)) if n_treatment > 1 else 0.0
        var_c = float(np.var(control_change, ddof=1)) if n_control > 1 else 0.0
        did_effect = mean_t - mean_c
        treatment_change_mean = mean_t
        control_change_mean = mean_c
    else:
        mean_t = float(np.mean(treatment_post))
        mean_c = float(np.mean(control_post))
        var_t = float(np.var(treatment_post, ddof=1)) if n_treatment > 1 else 0.0
        var_c = float(np.var(control_post, ddof=1)) if n_control > 1 else 0.0
        did_effect = 0.0
        treatment_change_mean = 0.0
        control_change_mean = 0.0

    if var_t == 0 and var_c == 0:
        diff = mean_t - mean_c
        total_effect = did_effect * n_treatment if use_did else diff * n_treatment
        return {
            "prob_treatment_better": 1.0 if diff > 0 else (0.0 if diff < 0 else 0.5),
            "expected_loss_treatment": 0.0 if diff >= 0 else abs(diff),
            "expected_loss_control": 0.0 if diff <= 0 else abs(diff),
            "credible_interval": (diff, diff),
            "relative_uplift": (diff / mean_c) if mean_c != 0 else 0.0,
            "total_effect": float(total_effect),
            "did_effect": float(did_effect),
            "treatment_change": float(treatment_change_mean),
            "control_change": float(control_change_mean),
        }

    se_t = np.sqrt(max(var_t, 1e-10) / n_treatment)
    se_c = np.sqrt(max(var_c, 1e-10) / n_control)

    np.random.seed(42)
    if n_treatment > 30 and n_control > 30:
        treatment_samples = np.random.normal(mean_t, se_t, n_samples)
        control_samples = np.random.normal(mean_c, se_c, n_samples)
    else:
        treatment_samples = mean_t + se_t * np.random.standard_t(max(n_treatment - 1, 1), n_samples)
        control_samples = mean_c + se_c * np.random.standard_t(max(n_control - 1, 1), n_samples)

    diff_samples = treatment_samples - control_samples
    prob_treatment_better = float(np.mean(diff_samples > 0))
    expected_loss_treatment = float(np.mean(np.abs(diff_samples) * (diff_samples < 0)))
    expected_loss_control = float(np.mean(np.abs(diff_samples) * (diff_samples > 0)))
    ci_lower = float(np.percentile(diff_samples, 2.5))
    ci_upper = float(np.percentile(diff_samples, 97.5))

    mean_diff = float(np.mean(diff_samples))
    if use_did:
        baseline = float(np.mean(control_post)) if abs(mean_c) < 1e-10 else mean_c
        relative_uplift = mean_diff / abs(baseline) if baseline != 0 else 0.0
    else:
        relative_uplift = mean_diff / mean_c if mean_c != 0 else 0.0

    total_effect = mean_diff * n_treatment
    return {
        "prob_treatment_better": prob_treatment_better,
        "expected_loss_treatment": expected_loss_treatment,
        "expected_loss_control": expected_loss_control,
        "credible_interval": (ci_lower, ci_upper),
        "relative_uplift": float(relative_uplift),
        "total_effect": float(total_effect),
        "did_effect": float(did_effect),
        "treatment_change": float(treatment_change_mean),
        "control_change": float(control_change_mean),
    }
