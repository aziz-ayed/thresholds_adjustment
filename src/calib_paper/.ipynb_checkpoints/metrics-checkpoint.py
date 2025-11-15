# src/calib_paper/metrics.py

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.utils import resample


def get_calibrated_value_at_threshold(
    golds,
    probs,
    threshold: float,
    n_bins: int = 10,
    strategy: str = "quantile",
):
    """
    Compute calibration curve and return observed incidence at a given
    decision threshold via interpolation.

    Returns
    -------
    float
        Estimated E[Y | score ~ threshold].
    """
    golds = np.asarray(golds)
    probs = np.asarray(probs)

    prob_true, prob_pred = calibration_curve(
        golds, probs, n_bins=n_bins, strategy=strategy
    )

    # Ensure sorted by prob_pred for interpolation
    order = np.argsort(prob_pred)
    prob_pred_sorted = prob_pred[order]
    prob_true_sorted = prob_true[order]

    return float(np.interp(threshold, prob_pred_sorted, prob_true_sorted))


def get_calibrated_value_with_ci(
    golds,
    probs,
    threshold: float,
    n_bins: int = 10,
    strategy: str = "quantile",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
):
    """
    Bootstrap CI around the calibrated incidence at a given threshold.

    Returns a dict with:
        - mean
        - ci_lower
        - ci_upper
        - original
        - bootstrap_values
    """
    golds = np.asarray(golds)
    probs = np.asarray(probs)

    rng = np.random.default_rng(random_state)

    # Original (no bootstrap)
    original_value = get_calibrated_value_at_threshold(
        golds, probs, threshold, n_bins, strategy
    )

    boot_vals = []
    n = len(golds)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        g_boot = golds[idx]
        p_boot = probs[idx]
        try:
            val = get_calibrated_value_at_threshold(
                g_boot, p_boot, threshold, n_bins, strategy
            )
            boot_vals.append(val)
        except Exception:
            continue

    if len(boot_vals) == 0:
        return {
            "mean": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "original": original_value,
            "bootstrap_values": np.array([]),
        }

    boot_vals = np.asarray(boot_vals)
    alpha = 1 - confidence_level

    return {
        "mean": float(np.mean(boot_vals)),
        "ci_lower": float(np.percentile(boot_vals, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_vals, 100 * (1 - alpha / 2))),
        "original": float(original_value),
        #"bootstrap_values": boot_vals,
    }


def find_threshold_for_target_incidence(
    golds,
    probs,
    target_incidence: float,
    n_bins: int = 10,
):
    """
    Invert the calibration curve: find the score threshold that achieves
    a given target incidence.
    """
    golds = np.asarray(golds)
    probs = np.asarray(probs)

    prob_true, prob_pred = calibration_curve(
        golds, probs, n_bins=n_bins, strategy="quantile"
    )

    order = np.argsort(prob_true)
    prob_true_sorted = prob_true[order]
    prob_pred_sorted = prob_pred[order]

    if target_incidence <= prob_true_sorted[0]:
        return float(prob_pred_sorted[0])
    if target_incidence >= prob_true_sorted[-1]:
        return float(prob_pred_sorted[-1])

    from scipy.interpolate import interp1d

    f_inv = interp1d(
        prob_true_sorted,
        prob_pred_sorted,
        kind="linear",
        fill_value="extrapolate",
    )
    return float(f_inv(target_incidence))


def compute_eligibility_change_with_ci(
    golds,
    probs,
    guideline_threshold: float,
    target_incidence: float,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
):
    """
    Compute relative change (%) in eligible population when moving from
    a guideline threshold to the "oracle" (calibration-based) threshold,
    with bootstrap CI.

    Returns dict with:
        - pct_change_mean
        - pct_change_ci_lower
        - pct_change_ci_upper
        - pct_change_original
        - oracle_threshold_mean
        - oracle_threshold_ci [lower, upper]
    """
    golds = np.asarray(golds)
    probs = np.asarray(probs)

    rng = np.random.default_rng(random_state)

    # Original
    n_guideline = np.sum(probs >= guideline_threshold)
    oracle_thr = find_threshold_for_target_incidence(
        golds, probs, target_incidence, n_bins
    )
    n_oracle = np.sum(probs >= oracle_thr)
    original_pct_change = (n_oracle - n_guideline) / n_guideline * 100.0

    pct_changes = []
    oracle_thresholds = []

    # Stratified bootstrap by outcome
    pos_idx = np.where(golds == 1)[0]
    neg_idx = np.where(golds == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    for i in range(n_bootstrap):
        boot_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        boot_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])

        g_boot = golds[boot_idx]
        p_boot = probs[boot_idx]

        try:
            thr_boot = find_threshold_for_target_incidence(
                g_boot, p_boot, target_incidence, n_bins
            )
        except Exception:
            continue

        n_guid_boot = np.sum(p_boot >= guideline_threshold)
        n_oracle_boot = np.sum(p_boot >= thr_boot)
        if n_guid_boot == 0:
            continue

        pct_change = (n_oracle_boot - n_guid_boot) / n_guid_boot * 100.0
        pct_changes.append(pct_change)
        oracle_thresholds.append(thr_boot)

    pct_changes = np.asarray(pct_changes)
    oracle_thresholds = np.asarray(oracle_thresholds)
    alpha = 1 - confidence_level

    return {
        "pct_change_mean": float(np.mean(pct_changes)),
        "pct_change_ci_lower": float(np.percentile(pct_changes, 100 * alpha / 2)),
        "pct_change_ci_upper": float(
            np.percentile(pct_changes, 100 * (1 - alpha / 2))
        ),
        "pct_change_original": float(original_pct_change),
        "oracle_threshold_mean": float(np.mean(oracle_thresholds)),
        "oracle_threshold_ci": [
            float(np.percentile(oracle_thresholds, 100 * alpha / 2)),
            float(np.percentile(oracle_thresholds, 100 * (1 - alpha / 2))),
        ],
    }


# ---- simple confusion metrics & a tiny bootstrap helper --------------------


def compute_confusion_metrics(golds, probs, threshold: float):
    """
    Return (sensitivity, specificity, ppv) at a given threshold.
    """
    golds = np.asarray(golds)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)

    tp = np.sum((preds == 1) & (golds == 1))
    fp = np.sum((preds == 1) & (golds == 0))
    tn = np.sum((preds == 0) & (golds == 0))
    fn = np.sum((preds == 0) & (golds == 1))

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    return sens, spec, ppv


def direct_incidence_around_threshold(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    threshold: float,
    margin: float = 0.001,
):
    """
    Observed incidence in [threshold - margin, threshold + margin].
    """
    lower = threshold - margin
    upper = threshold + margin
    subset = df[(df[score_col] >= lower) & (df[score_col] <= upper)]
    if len(subset) == 0:
        return np.nan, 0
    return float(subset[label_col].mean()), int(len(subset))


def bootstrap_ci_1d(values: np.ndarray, confidence_level: float = 0.95):
    """
    Utility to compute percentile bootstrap CI given a 1D array.
    """
    values = np.asarray(values, dtype=float)
    alpha = 1 - confidence_level
    return (
        float(np.nanpercentile(values, 100 * alpha / 2)),
        float(np.nanpercentile(values, 100 * (1 - alpha / 2))),
    )
