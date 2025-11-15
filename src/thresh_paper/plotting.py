# src/calib_paper/plotting.py

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_absolute_bias_summary(
    results_df: pd.DataFrame,
    methods=None,
    method_map=None,
    plot_title=None,
    figsize=(10, 5),
    ci_width: float = 0.95,
    ylim=None,
    show_pvalues: bool = True,
    plot_type: str = "abs_error",  # "abs_error" or "threshold"
    target_incidence: float | None = None,
):
    """
    Bar plot of mean absolute bias (or observed incidence) with bootstrap CIs
    across centres, for selected calibration methods.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Helvetica", "sans-serif"],
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 10,
            "figure.dpi": 300,
        }
    )

    if methods is None:
        methods = ["at_op", "intercept"]

    if method_map is None:
        method_map = {
            "at_op": "No Adjustment\n(Guideline)",
            "uncalib": "CalCurve Root",
            "intercept": "Threshold\nAdjustment",
            "platt": "Platt Scaling",
            "iso": "Full Calibration\n(Isotonic)",
        }

    if plot_title is None:
        if plot_type == "threshold":
            plot_title = f"Observed Incidence by Center (Bootstrap {int(ci_width * 100)}% CI)"
        else:
            plot_title = f"Absolute Error to Target Incidence (Bootstrap {int(ci_width * 100)}% CI)"

    if plot_type == "threshold":
        y_label = f"Observed Incidence (mean, {int(ci_width * 100)}% CI)"
    else:
        y_label = f"Absolute Error (Observed − Target) (mean, {int(ci_width * 100)}% CI)"

    df = results_df.copy()
    if "center" in df.columns:
        df["center"] = df["center"].astype(str)
    centers = df["center"].tolist()
    n_centers = len(centers)
    n_methods = len(methods)

    jco_colors = {
        "at_op": "#2E5A87",
        "intercept": "#B85A3E",
        "uncalib": "#5A7A4A",
        "platt": "#7A5A8A",
        "iso": "#8A7A5A",
    }

    x = np.arange(n_centers) * 1.2
    bar_group_width = 0.6
    width = bar_group_width / n_methods

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    for i, m in enumerate(methods):
        if plot_type == "threshold":
            mean_col = f"obs_incidence_{m}_mean"
            lower_col = f"obs_incidence_{m}_ci_lower"
            upper_col = f"obs_incidence_{m}_ci_upper"
        else:
            mean_col = f"obs_incidence_{m}_abs_bias_mean"
            lower_col = f"obs_incidence_{m}_abs_bias_ci_lower"
            upper_col = f"obs_incidence_{m}_abs_bias_ci_upper"

        if not all(c in df.columns for c in (mean_col, lower_col, upper_col)):
            continue

        means = df[mean_col].values
        lowers = df[lower_col].values
        uppers = df[upper_col].values

        lower_err = np.maximum(0, means - lowers)
        upper_err = np.maximum(0, uppers - means)
        err = [lower_err, upper_err]

        bar_pos = x + width * (i - (n_methods - 1) / 2)
        color = jco_colors.get(m, "#6B6B6B")

        ax.bar(
            bar_pos,
            means,
            width,
            label=method_map.get(m, m),
            yerr=err,
            capsize=3,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            error_kw={"linewidth": 0.8, "capthick": 0.8, "alpha": 0.6},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(centers)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Center")
    ax.set_title(plot_title, pad=15, fontweight="normal")

    if plot_type == "threshold" and target_incidence is not None:
        ax.axhline(
            target_incidence,
            color="#666666",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Target Incidence",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    if ylim is not None:
        ax.set_ylim(ylim)

    if show_pvalues and "p_sign_abs_error_intercept_vs_at_op" in df.columns:
        p_vals = df["p_sign_abs_error_intercept_vs_at_op"].values
        for i, p_val in enumerate(p_vals):
            if not np.isfinite(p_val):
                continue
            # highest mean across methods for this center
            max_height = 0.0
            for m in methods:
                if plot_type == "threshold":
                    col = f"obs_incidence_{m}_mean"
                else:
                    col = f"obs_incidence_{m}_abs_bias_mean"
                if col in df.columns:
                    max_height = max(max_height, df[col].iloc[i])

            p_text = "p<0.01" if p_val < 0.01 else f"p={p_val:.2f}"
            x_pos = x[i]
            y_pos = max_height * 1.10

            ax.text(
                x_pos,
                y_pos,
                p_text,
                ha="center",
                va="bottom",
                fontsize=6.5,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="#DDDDDD",
                    linewidth=0.3,
                    alpha=0.95,
                ),
            )

    legend = ax.legend(
        title="Method",
        bbox_to_anchor=(0.98, 0.98),
        loc="upper right",
        frameon=True,
        edgecolor="#CCCCCC",
        facecolor="white",
    )
    legend.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    return fig, ax


def create_three_panel_figure(
    df: pd.DataFrame,
    suptitle: str = "Performance Metrics at Guideline Threshold (Bootstrap CI)",
    plot_ci: bool = False,
    hline_value: float | None = None,
    hline_label: str | None = None,
):
    """
    1×3 panel for PPV, Sensitivity, Specificity across centers.
    Expects columns: 'center', 'ppv', 'sensitivity', 'specificity'
    and optionally *_lower, *_upper if plot_ci=True.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Helvetica", "sans-serif"],
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
        }
    )

    jco_color = "#2E5A87"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def _panel(ax, metric: str, title: str, ylim=None):
        df_sorted = df.copy()
        df_sorted["center"] = df_sorted["center"].astype(str)

        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        x = df_sorted["center"]
        y = df_sorted[metric].values

        bars = ax.bar(
            x,
            y,
            color=jco_color,
            edgecolor="white",
            linewidth=0.5,
            width=0.6,
        )

        if plot_ci:
            lower_col = f"{metric}_lower"
            upper_col = f"{metric}_upper"
            if lower_col in df_sorted.columns and upper_col in df_sorted.columns:
                lower_err = np.maximum(0, y - df_sorted[lower_col].values)
                upper_err = np.maximum(0, df_sorted[upper_col].values - y)
                ax.errorbar(
                    x,
                    y,
                    yerr=[lower_err, upper_err],
                    fmt="none",
                    capsize=3,
                    linewidth=0.8,
                    capthick=0.8,
                    alpha=0.6,
                    color="#333333",
                )

        if hline_value is not None:
            ax.axhline(
                hline_value,
                color="#666666",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=hline_label,
            )
            if hline_label:
                ax.legend(frameon=True, edgecolor="#CCCCCC", facecolor="white")

        ax.set_xlabel("Center")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title, pad=15, fontweight="normal")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.tick_params(axis="x", rotation=0)

    _panel(axes[0], "ppv", "a) Positive Predictive Value")
    _panel(axes[1], "sensitivity", "b) Sensitivity", ylim=(0.0, 1.0))
    _panel(axes[2], "specificity", "c) Specificity", ylim=(0.0, 1.0))

    fig.suptitle(suptitle, fontsize=12, fontweight="normal", y=0.98)
    plt.tight_layout()
    return fig, axes
