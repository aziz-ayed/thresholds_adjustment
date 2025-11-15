# src/calib_paper/bootstrap.py

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logit, expit as sigmoid
from scipy.stats import binomtest
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .metrics import direct_incidence_around_threshold


def _stratified_bootstrap_indices(
    df: pd.DataFrame,
    label_col: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Row indices of a stratified bootstrap sample."""
    idxs = []
    for _, rows in df.groupby(label_col).groups.items():
        rows = np.asarray(rows, dtype=int)
        resampled = rng.choice(rows, size=len(rows), replace=True)
        idxs.append(resampled)
    return np.concatenate(idxs)


def _fit_intercept_logistic(
    dev_df: pd.DataFrame,
    score_col: str,
    label_col: str,
) -> float:
    """
    Intercept-only logistic recalibration:
        p_cal = sigmoid(a + logit(p))
    such that mean(p_cal) matches mean outcome in dev set.
    """
    probs = dev_df[score_col].clip(1e-7, 1 - 1e-7).to_numpy()
    labels = dev_df[label_col].to_numpy()
    if probs.size == 0:
        return 0.0

    s_i = logit(probs)
    mean_golds = labels.mean()

    if np.allclose(s_i, s_i[0]):
        clipped = float(np.clip(mean_golds, 1e-7, 1 - 1e-7))
        return math.log(clipped / (1 - clipped)) - float(s_i[0])

    def f(a_param: float):
        return sigmoid(a_param + s_i).mean() - mean_golds

    try:
        return float(brentq(f, -30, 30))
    except ValueError:
        res = minimize_scalar(lambda a_val: abs(f(a_val)), bounds=(-30, 30), method="bounded")
        return float(res.x) if res.success else 0.0


def _apply_intercept_logistic(df: pd.DataFrame, score_col: str, a: float) -> np.ndarray:
    s_i = logit(df[score_col].clip(1e-7, 1 - 1e-7))
    return sigmoid(a + s_i)


def _get_threshold_from_cal_curve(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    target_incidence: float,
    n_bins: int = 15,
    strategy: str = "quantile",
) -> float:
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy=strategy)
    idx = np.argsort(prob_true)
    from scipy.interpolate import interp1d

    f_inv = interp1d(prob_true[idx], prob_pred[idx], bounds_error=False, fill_value="extrapolate")
    return float(f_inv(target_incidence))


def paired_sign_test(
    err_method,
    err_baseline,
    *,
    alternative: str = "less",
) -> float:
    """
    Exact paired sign test on two error vectors.

    alternative:
      - "less"    → H1: method error < baseline error
      - "greater" → H1: method error > baseline error
      - "two-sided"
    """
    err_m = np.asarray(err_method, dtype=float)
    err_b = np.asarray(err_baseline, dtype=float)

    if err_m.shape != err_b.shape:
        raise ValueError("Input vectors must have the same length.")

    mask_valid = ~np.isnan(err_m) & ~np.isnan(err_b) & (err_m != err_b)
    diff = err_m[mask_valid] - err_b[mask_valid]
    if diff.size == 0:
        return np.nan

    improved = np.sum(diff < 0)
    n = diff.size

    alt_map = {
        "less": "greater",   # successes > n/2 = method better
        "greater": "less",
        "two-sided": "two-sided",
    }
    alt_kw = alt_map.get(alternative)
    if alt_kw is None:
        raise ValueError("alternative must be 'less', 'greater', or 'two-sided'.")

    p_val = binomtest(improved, n, 0.5, alternative=alt_kw).pvalue
    return float(p_val)


def out_of_fold_bootstrap_calibration_comparison_top_centers(
    df: pd.DataFrame,
    cen_col: str,
    score_col: str,
    label_col: str,
    target_incidence: float,
    get_calibrated_value_at_threshold: Callable,
    *,
    top_k: int = 10,
    nb_recal: float | int = 0.2,
    n_bins: int = 15,
    n_outer: int = 200,
    ci_width: float = 0.95,
    random_state: int | None = 42,
    use_direct_incidence: bool = False,
    direct_margin: float = 0.001,
) -> pd.DataFrame:
    """
    Out-of-fold (OOF) nested bootstrap calibration comparison across top centres.

    Returns one row per centre with:
      - obs_incidence_*_* statistics
      - threshold summaries
      - n_above_*_* (counts ≥ threshold)
      - prop_wins_*_vs_at_op
      - paired sign test p-values vs at_op
    """
    if get_calibrated_value_at_threshold is None and not use_direct_incidence:
        raise ValueError("Provide get_calibrated_value_at_threshold or set use_direct_incidence=True.")

    rng = np.random.default_rng(random_state)

    lower_q = (1 - ci_width) / 2 * 100
    upper_q = (1 + ci_width) / 2 * 100

    results = []
    top_centres = df[cen_col].value_counts().head(top_k).index.tolist()

    for centre in top_centres:
        centre_df = df[df[cen_col] == centre].reset_index(drop=True)
        n_centre = len(centre_df)
        if centre_df[label_col].nunique() < 2 or n_centre < 20:
            continue

        methods_all = ["at_op", "uncalib", "intercept", "platt", "iso", "oracle"]
        boot_stats = {m: [] for m in methods_all}
        boot_counts = {m: [] for m in methods_all}
        boot_thresh = {m: [] for m in methods_all}

        # ------------------------ outer bootstrap ---------------------------
        for _ in tqdm(range(n_outer), desc=f"Center {centre} (OOF nested)", leave=False):
            boot_idx = _stratified_bootstrap_indices(centre_df, label_col, rng)
            boot_df = centre_df.loc[boot_idx].reset_index(drop=True)

            # dev/eval via K-folds to achieve desired dev share
            if isinstance(nb_recal, float):
                if not 0 < nb_recal < 1:
                    raise ValueError("nb_recal float must be in (0,1)")
                n_folds = max(2, round(1 / (1 - nb_recal)))
            elif isinstance(nb_recal, int):
                if nb_recal < 1:
                    raise ValueError("nb_recal int must be ≥1")
                n_folds = max(2, round(len(boot_df) / nb_recal))
            else:
                raise TypeError("nb_recal must be float or int")

            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=_)

            oof_rows = []
            for train_idx, eval_idx in skf.split(boot_df, boot_df[label_col]):
                dev_df = boot_df.iloc[train_idx].reset_index(drop=True)
                eval_df = boot_df.iloc[eval_idx].reset_index(drop=True)

                eval_df = eval_df.copy()
                eval_df["probs_at_op"] = eval_df[score_col]

                # uncalib threshold from dev
                try:
                    thresh_uncal = _get_threshold_from_cal_curve(
                        dev_df[label_col], dev_df[score_col], target_incidence, n_bins=n_bins
                    )
                    eval_df["probs_uncalib"] = eval_df[score_col]
                    eval_df["thresh_uncalib"] = thresh_uncal
                except Exception:
                    eval_df["probs_uncalib"] = np.nan
                    eval_df["thresh_uncalib"] = np.nan

                # intercept-only logistic
                try:
                    a_hat = _fit_intercept_logistic(dev_df, score_col, label_col)
                    eval_df["probs_intercept"] = _apply_intercept_logistic(eval_df, score_col, a_hat)
                except Exception:
                    eval_df["probs_intercept"] = np.nan

                # Platt scaling
                try:
                    dev_logits = logit(dev_df[score_col].clip(1e-7, 1 - 1e-7))
                    lr = LogisticRegression(solver="lbfgs", penalty="none")
                    lr.fit(dev_logits.values.reshape(-1, 1), dev_df[label_col].values)

                    eval_logits = logit(eval_df[score_col].clip(1e-7, 1 - 1e-7))
                    eval_df["probs_platt"] = lr.predict_proba(eval_logits.values.reshape(-1, 1))[:, 1]
                except Exception:
                    eval_df["probs_platt"] = np.nan

                # Isotonic
                try:
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(dev_df[score_col].to_numpy(), dev_df[label_col].to_numpy())
                    eval_df["probs_iso"] = iso.transform(eval_df[score_col].to_numpy())
                except Exception:
                    eval_df["probs_iso"] = np.nan

                oof_rows.append(eval_df)

            oof_df = pd.concat(oof_rows, ignore_index=True)

            # ORACLE threshold based on OOF preds
            try:
                thresh_oracle = _get_threshold_from_cal_curve(
                    oof_df[label_col], oof_df[score_col], target_incidence, n_bins=n_bins
                )
            except Exception:
                scores = np.sort(oof_df[score_col].to_numpy())
                best_thr, best_gap = np.nan, np.inf
                for t in scores:
                    inc = oof_df.loc[oof_df[score_col] >= t, label_col].mean()
                    gap = abs(inc - target_incidence)
                    if gap < best_gap:
                        best_gap, best_thr = gap, t
                thresh_oracle = best_thr

            method_cols = {
                "at_op": "probs_at_op",
                "uncalib": "probs_uncalib",
                "intercept": "probs_intercept",
                "platt": "probs_platt",
                "iso": "probs_iso",
                "oracle": "probs_at_op",  # same probs, different threshold
            }

            for m, col in method_cols.items():
                # choose threshold
                try:
                    if m == "uncalib":
                        thresh = oof_df["thresh_uncalib"].dropna().iloc[0]
                    elif m == "oracle":
                        thresh = thresh_oracle
                    else:
                        thresh = target_incidence
                except Exception:
                    thresh = np.nan

                # incidence estimate
                try:
                    if use_direct_incidence:
                        val, _subset_len = direct_incidence_around_threshold(
                            oof_df, score_col=col, label_col=label_col,
                            threshold=thresh, margin=direct_margin
                        )
                    else:
                        val = get_calibrated_value_at_threshold(
                            golds=oof_df[label_col],
                            probs=oof_df[col],
                            threshold=thresh,
                            n_bins=n_bins,
                            strategy="quantile",
                        )
                except Exception:
                    val = np.nan

                boot_stats[m].append(val)
                boot_thresh[m].append(thresh)

                try:
                    n_above = float(np.sum(oof_df[col] >= thresh))
                except Exception:
                    n_above = np.nan
                boot_counts[m].append(n_above)

        # ------------------------ summarise per-centre ----------------------
        methods = list(boot_stats.keys())
        abs_errors = {
            m: np.abs(np.asarray(v, dtype=float) - target_incidence) for m, v in boot_stats.items()
        }
        baseline_err = abs_errors["at_op"]
        counts_at_op = np.asarray(boot_counts["at_op"], dtype=float)

        summary = {"center": centre, "n": n_centre}

        for m in methods:
            vals = np.asarray(boot_stats[m], dtype=float)
            diffs = vals - target_incidence
            abs_vals = abs_errors[m]

            summary[f"obs_incidence_{m}_mean"] = float(np.nanmean(vals))
            summary[f"obs_incidence_{m}_ci_lower"] = float(np.nanpercentile(vals, lower_q))
            summary[f"obs_incidence_{m}_ci_upper"] = float(np.nanpercentile(vals, upper_q))

            summary[f"obs_incidence_{m}_bias_mean"] = float(np.nanmean(diffs))
            summary[f"obs_incidence_{m}_bias_ci_lower"] = float(np.nanpercentile(diffs, lower_q))
            summary[f"obs_incidence_{m}_bias_ci_upper"] = float(np.nanpercentile(diffs, upper_q))

            summary[f"obs_incidence_{m}_abs_bias_mean"] = float(np.nanmean(abs_vals))
            summary[f"obs_incidence_{m}_abs_bias_ci_lower"] = float(np.nanpercentile(abs_vals, lower_q))
            summary[f"obs_incidence_{m}_abs_bias_ci_upper"] = float(np.nanpercentile(abs_vals, upper_q))

            thr_vals = np.asarray(boot_thresh[m], dtype=float)
            summary[f"thresh_{m}_mean"] = float(np.nanmean(thr_vals))
            summary[f"thresh_{m}_ci_lower"] = float(np.nanpercentile(thr_vals, lower_q))
            summary[f"thresh_{m}_ci_upper"] = float(np.nanpercentile(thr_vals, upper_q))

            c_vals = np.asarray(boot_counts[m], dtype=float)
            summary[f"n_above_{m}_mean"] = float(np.nanmean(c_vals))
            summary[f"n_above_{m}_ci_lower"] = float(np.nanpercentile(c_vals, lower_q))
            summary[f"n_above_{m}_ci_upper"] = float(np.nanpercentile(c_vals, upper_q))

            boot_wins = np.sum(abs_errors[m] < baseline_err) / len(abs_errors[m])
            summary[f"prop_wins_{m}_vs_at_op"] = float(boot_wins)

            if m != "at_op":
                with np.errstate(divide="ignore", invalid="ignore"):
                    pct_delta = (c_vals - counts_at_op) / counts_at_op * 100.0
                summary[f"pct_delta_n_above_{m}_vs_at_op_mean"] = float(np.nanmean(pct_delta))
                summary[f"pct_delta_n_above_{m}_vs_at_op_ci_lower"] = float(
                    np.nanpercentile(pct_delta, lower_q)
                )
                summary[f"pct_delta_n_above_{m}_vs_at_op_ci_upper"] = float(
                    np.nanpercentile(pct_delta, upper_q)
                )

        summary["oracle_shift_abs"] = abs(summary.get("thresh_oracle_mean", np.nan) - target_incidence)

        for m in methods:
            if m == "at_op":
                continue
            p_sign = paired_sign_test(abs_errors[m], baseline_err, alternative="less")
            summary[f"p_sign_abs_error_{m}_vs_at_op"] = p_sign

        results.append(summary)

    return pd.DataFrame(results)
