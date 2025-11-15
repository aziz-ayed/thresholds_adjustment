from __future__ import annotations

import os
import numpy as np
import pandas as pd

from thresh_paper.data import make_toy_dataset
from thresh_paper.metrics import (
    compute_eligibility_change_with_ci,
    get_calibrated_value_with_ci,
    compute_confusion_metrics,
    direct_incidence_around_threshold,
)
from thresh_paper.bootstrap import (
    out_of_fold_bootstrap_calibration_comparison_top_centers,
)
from thresh_paper.plotting import (
    plot_absolute_bias_summary,
    create_three_panel_figure,
)


def main():
    os.makedirs("figs", exist_ok=True)

    # 1) Toy data
    df = make_toy_dataset(n_centers=10, n_per_center=800, random_state=0)
    print(df.head())

    # global prevalence as "guideline threshold" / target incidence
    target_incidence = df["golds"].mean()
    guideline_threshold = target_incidence

    print(f"Global event rate (target_incidence) = {target_incidence:.4f}")

    # 2) Eligibility change with CI
    elig_stats = compute_eligibility_change_with_ci(
        golds=df["golds"],
        probs=df["probs"],
        guideline_threshold=guideline_threshold,
        target_incidence=target_incidence,
        n_bins=10,
        n_bootstrap=500,
        confidence_level=0.90,
        random_state=0,
    )
    print("\nEligibility change stats (oracle vs guideline):")
    for k, v in elig_stats.items():
        print(f"  {k}: {v}")

    # 3) Calibrated value with CI in a subpopulation
    younger = df[df["age"] <= 65]
    cal_stats = get_calibrated_value_with_ci(
        golds=younger["golds"],
        probs=younger["probs"],
        threshold=guideline_threshold,
        n_bins=15,
        strategy="quantile",
        n_bootstrap=500,
        confidence_level=0.95,
        random_state=0,
    )
    print("\nCalibrated value stats in age ≤ 65:")
    for k, v in cal_stats.items():
        if k == "bootstrap_values":
            print(f"  {k}: array of length {len(v)}")
        else:
            print(f"  {k}: {v}")

    # 4) OOF nested bootstrap comparison across centres
    summary_df = out_of_fold_bootstrap_calibration_comparison_top_centers(
        df=df.rename(columns={"center": "cen"}),  # function expects 'cen_col'
        cen_col="cen",
        score_col="probs",
        label_col="golds",
        target_incidence=target_incidence,
        top_k=10,
        nb_recal=500,        # roughly like your original
        n_bins=20,
        n_outer=300,         # smaller for demo; paper version can use 2000
        random_state=42,
        get_calibrated_value_at_threshold=get_calibrated_value_with_ci.__globals__[
            "get_calibrated_value_at_threshold"
        ],
        ci_width=0.90,
        use_direct_incidence=False,
    )

    # rename back 'cen' -> 'center' for plotting
    summary_df = summary_df.rename(columns={"center": "center"})
    print("\nSummary_df columns:")
    print(summary_df.columns.tolist())

    # 5) Plot absolute bias summary
    fig, ax = plot_absolute_bias_summary(
        summary_df,
        methods=["at_op", "iso", "intercept"],
        ci_width=0.95,
        show_pvalues=True,
        plot_type="abs_error",
        target_incidence=target_incidence,
        plot_title="Toy Data: Absolute Error by Center (Bootstrap 95% CI)",
    )
    fig.savefig("figs/toy_absolute_error_by_center.png", dpi=300, bbox_inches="tight")

    # 6) Plot observed incidence by center for at_op only
    fig, ax = plot_absolute_bias_summary(
        summary_df,
        methods=["at_op"],
        ci_width=0.95,
        show_pvalues=False,
        plot_type="threshold",
        target_incidence=target_incidence,
        plot_title="Toy Data: Observed Incidence at Guideline Threshold (Bootstrap 95% CI)",
    )
    fig.savefig("figs/toy_incidence_by_center.png", dpi=300, bbox_inches="tight")

    # 7) Simple per-center PPV / Sens / Spec for create_three_panel_figure
    center_rows = []
    for center, sub in df.groupby("center"):
        sens, spec, ppv = compute_confusion_metrics(
            sub["golds"].values,
            sub["probs"].values,
            threshold=guideline_threshold,
        )
        center_rows.append(
            {
                "center": center,
                "ppv": ppv,
                "sensitivity": sens,
                "specificity": spec,
            }
        )
    metrics_df = pd.DataFrame(center_rows).sort_values("center")

    fig, axes = create_three_panel_figure(
        df=metrics_df,
        suptitle="Toy Data: PPV, Sensitivity, Specificity at Guideline Threshold",
        plot_ci=False,
    )
    fig.savefig("figs/toy_three_panel_metrics.png", dpi=300, bbox_inches="tight")

    print("\nFigures written to figs/ directory.")


if __name__ == "__main__":
    main()
