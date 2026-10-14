"""Plot Possession Creation Index against NBA draft surplus value."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("data/draft_with_college.csv")
OUTPUT_PATH = Path("plots/pci_vs_surplus_value.png")
SCORING_OUTPUT_PATH = Path("plots/scoring_profile_vs_surplus_value.png")
GRANULAR_OUTPUT_PATH = Path("plots/granular_index_vs_surplus_value.png")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run 02_collect_college_stats.py first."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    required = ["possession_creation_index", "surplus_value"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    plot_df = df.copy()
    for column in required:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=required)

    if len(plot_df) < 2:
        raise ValueError("At least two matched players are required for the plot.")

    x = plot_df["possession_creation_index"]
    y = plot_df["surplus_value"]
    correlation = x.corr(y)
    slope, intercept = np.polyfit(x, y, 1)
    trend_x = np.linspace(x.min(), x.max(), 200)
    plot_df["regression_residual"] = y - (slope * x + intercept)
    plot_df["absolute_residual"] = plot_df["regression_residual"].abs()
    outliers = plot_df.nlargest(20, "absolute_residual", keep="first").copy()

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(
        x,
        y,
        s=34,
        alpha=0.55,
        color="#4f46e5",
        edgecolors="white",
        linewidths=0.35,
        label=f"Drafted players (n={len(plot_df)})",
    )
    ax.plot(
        trend_x,
        slope * trend_x + intercept,
        color="#f97316",
        linewidth=2.5,
        label="Linear trend",
    )
    ax.scatter(
        outliers["possession_creation_index"],
        outliers["surplus_value"],
        s=68,
        color="#dc2626",
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
        label="20 largest regression outliers",
    )

    label_offsets = [
        (8, 9), (8, -17), (-8, 10), (-8, -18), (12, 18),
        (12, -27), (-12, 20), (-12, -28), (18, 7), (18, -17),
        (-18, 8), (-18, -18), (23, 16), (23, -27), (-23, 17),
        (-23, -28), (30, 7), (30, -18), (-30, 9), (-30, -19),
    ]
    for (_, row), offset in zip(outliers.iterrows(), label_offsets):
        ax.annotate(
            str(row.get("player", "Unknown")),
            xy=(row["possession_creation_index"], row["surplus_value"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            color="#991b1b",
            ha="left" if offset[0] >= 0 else "right",
            arrowprops={"arrowstyle": "-", "color": "#dc2626", "alpha": 0.55},
            zorder=5,
        )
    ax.axhline(0, color="#64748b", linewidth=1, linestyle="--", alpha=0.7)

    ax.text(
        0.03,
        0.96,
        f"Pearson correlation: r = {correlation:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9},
    )
    ax.set_title(
        "Possession Creation Index vs. NBA Draft Surplus Value",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("College Possession Creation Index", fontsize=12)
    ax.set_ylabel("Peak 3-Year VORP Surplus Value", fontsize=12)
    ax.grid(True, alpha=0.18)
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Threshold-based scoring profile (0-6) versus surplus value.
    scoring_df = df.copy()
    scoring_df["scoring_profile_score"] = pd.to_numeric(
        scoring_df.get("scoring_profile_score"), errors="coerce"
    )
    scoring_df["surplus_value"] = pd.to_numeric(
        scoring_df["surplus_value"], errors="coerce"
    )
    scoring_df = scoring_df.dropna(subset=["scoring_profile_score", "surplus_value"])
    scoring_correlation = scoring_df["scoring_profile_score"].corr(
        scoring_df["surplus_value"]
    )

    rng = np.random.default_rng(42)
    jittered_score = scoring_df["scoring_profile_score"] + rng.normal(
        0, 0.07, len(scoring_df)
    )
    elite = scoring_df["scoring_profile_score"] == 6
    score_means = scoring_df.groupby("scoring_profile_score")["surplus_value"].mean()
    high_outliers = scoring_df.nlargest(15, "surplus_value").copy()
    high_outliers["jittered_score"] = (
        high_outliers["scoring_profile_score"]
        + rng.normal(0, 0.035, len(high_outliers))
    )

    fig2, ax2 = plt.subplots(figsize=(11, 7))
    ax2.scatter(
        jittered_score[~elite],
        scoring_df.loc[~elite, "surplus_value"],
        s=28,
        alpha=0.42,
        color="#4f46e5",
        edgecolors="none",
        label="Players",
    )
    ax2.scatter(
        jittered_score[elite],
        scoring_df.loc[elite, "surplus_value"],
        s=38,
        alpha=0.65,
        color="#dc2626",
        edgecolors="white",
        linewidths=0.4,
        label=f"Elite profile: 6/6 (n={int(elite.sum())})",
    )
    ax2.plot(
        score_means.index,
        score_means.values,
        color="#f97316",
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Mean surplus by score",
    )
    ax2.scatter(
        high_outliers["jittered_score"],
        high_outliers["surplus_value"],
        s=78,
        color="#f59e0b",
        edgecolors="#78350f",
        linewidths=0.7,
        zorder=5,
        label="15 highest-surplus outliers",
    )
    high_label_offsets = [
        (8, 10), (-8, 10), (10, -18), (-10, -18), (15, 12),
        (-15, 12), (18, -20), (-18, -20), (22, 10), (-22, 10),
        (26, -18), (-26, -18), (30, 9), (-30, 9), (10, 20),
    ]
    for (_, row), offset in zip(high_outliers.iterrows(), high_label_offsets):
        ax2.annotate(
            str(row.get("player", "Unknown")),
            xy=(row["jittered_score"], row["surplus_value"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            color="#92400e",
            ha="left" if offset[0] >= 0 else "right",
            arrowprops={"arrowstyle": "-", "color": "#f59e0b", "alpha": 0.65},
            zorder=6,
        )
    ax2.axhline(0, color="#64748b", linewidth=1, linestyle="--", alpha=0.7)
    ax2.text(
        0.03,
        0.96,
        f"Pearson correlation: r = {scoring_correlation:.3f}",
        transform=ax2.transAxes,
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9},
    )
    ax2.set_title(
        "College Scoring Profile vs. NBA Draft Surplus Value",
        fontsize=15,
        fontweight="bold",
    )
    ax2.set_xlabel("Scoring Profile Score (thresholds met out of 6)", fontsize=12)
    ax2.set_ylabel("Peak 3-Year VORP Surplus Value", fontsize=12)
    ax2.set_xticks(range(7))
    ax2.set_xlim(-0.35, 6.35)
    ax2.grid(True, alpha=0.18)
    ax2.legend(loc="upper left", bbox_to_anchor=(0, 0.88))
    fig2.tight_layout()
    fig2.savefig(SCORING_OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    # Fitted granular college index versus surplus value.  The index contains
    # no BPM-family, VORP, NBA-career, or draft-position inputs.
    granular_required = ["granular_prospect_index", "surplus_value"]
    granular_missing = [
        column for column in granular_required if column not in df.columns
    ]
    if granular_missing:
        raise KeyError(
            "Missing granular index. Re-run 02_collect_college_stats.py; "
            f"missing columns: {granular_missing}"
        )

    granular_df = df.copy()
    for column in granular_required:
        granular_df[column] = pd.to_numeric(
            granular_df[column], errors="coerce"
        )
    granular_df = granular_df.dropna(subset=granular_required)
    granular_x = granular_df["granular_prospect_index"]
    granular_y = granular_df["surplus_value"]
    granular_correlation = granular_x.corr(granular_y)
    granular_slope, granular_intercept = np.polyfit(
        granular_x, granular_y, 1
    )
    granular_trend_x = np.linspace(
        granular_x.min(), granular_x.max(), 200
    )
    granular_df["regression_residual"] = granular_y - (
        granular_slope * granular_x + granular_intercept
    )
    granular_df["absolute_residual"] = granular_df[
        "regression_residual"
    ].abs()
    granular_outliers = granular_df.nlargest(
        20, "absolute_residual", keep="first"
    ).copy()

    fig3, (ax3, outlier_ax) = plt.subplots(
        1,
        2,
        figsize=(15, 8.5),
        gridspec_kw={"width_ratios": [4.8, 1.5]},
    )
    ax3.scatter(
        granular_x,
        granular_y,
        s=34,
        alpha=0.52,
        color="#2563eb",
        edgecolors="white",
        linewidths=0.35,
        label=f"College-matched players (n={len(granular_df)})",
    )
    ax3.plot(
        granular_trend_x,
        granular_slope * granular_trend_x + granular_intercept,
        color="#f97316",
        linewidth=2.5,
        label="Linear trend",
    )
    ax3.scatter(
        granular_outliers["granular_prospect_index"],
        granular_outliers["surplus_value"],
        s=72,
        color="#dc2626",
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
        label="20 numbered regression outliers",
    )
    outlier_lines = []
    number_offsets = {
        12: (-7, 7),
        13: (7, -7),
        15: (-7, 7),
        20: (7, -7),
    }
    for rank, (_, row) in enumerate(granular_outliers.iterrows(), start=1):
        number_offset = number_offsets.get(rank, (0, 0))
        ax3.annotate(
            str(rank),
            xy=(row["granular_prospect_index"], row["surplus_value"]),
            xytext=number_offset,
            textcoords="offset points",
            fontsize=6.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "circle,pad=0.12",
                "facecolor": "#dc2626",
                "edgecolor": "white",
                "linewidth": 0.35,
            },
            arrowprops=(
                {"arrowstyle": "-", "color": "#dc2626", "linewidth": 0.6}
                if number_offset != (0, 0)
                else None
            ),
            zorder=5,
        )
        outlier_lines.append(
            f"{rank:>2}. {str(row.get('player', 'Unknown')):<24} "
            f"{row['regression_residual']:+.2f}"
        )

    outlier_ax.axis("off")
    outlier_ax.text(
        0,
        0.98,
        "Largest residuals\n(player: actual minus predicted)",
        transform=outlier_ax.transAxes,
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )
    outlier_ax.text(
        0,
        0.89,
        "\n".join(outlier_lines),
        transform=outlier_ax.transAxes,
        va="top",
        fontsize=8.2,
        family="monospace",
        linespacing=1.42,
    )
    ax3.axhline(0, color="#64748b", linewidth=1, linestyle="--", alpha=0.7)
    ax3.text(
        0.03,
        0.96,
        f"Pearson correlation: r = {granular_correlation:.3f}",
        transform=ax3.transAxes,
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9},
    )
    ax3.set_title(
        "Granular College Prospect Index vs. NBA Draft Surplus Value",
        fontsize=15,
        fontweight="bold",
    )
    ax3.set_xlabel(
        "Granular Prospect Index (no BPM/VORP/NBA inputs)", fontsize=12
    )
    ax3.set_ylabel("Peak 3-Year VORP Surplus Value", fontsize=12)
    ax3.grid(True, alpha=0.18)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(GRANULAR_OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig3)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {SCORING_OUTPUT_PATH}")
    print(f"Saved: {GRANULAR_OUTPUT_PATH}")
    print(f"Players plotted: {len(plot_df)}")
    print(f"Pearson correlation: {correlation:.3f}")
    print(f"Scoring-profile correlation: {scoring_correlation:.3f}")
    print(f"Granular-index in-sample correlation: {granular_correlation:.3f}")
    print("\n20 largest Granular Prospect Index regression outliers:")
    print(
        granular_outliers[
            [
                "player", "Draft_Year", "granular_prospect_index",
                "surplus_value", "regression_residual",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
