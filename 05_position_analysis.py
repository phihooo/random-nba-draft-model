"""Create position-stratified datasets, correlations, and diagnostic plots.

This script is the canonical downstream position step.  It enriches the main
derived draft tables with the shared PG/SG/SF/PF/C assignment, writes one full
dataset per position, calculates within-position Pearson/Spearman correlations,
and creates position-faceted versions of the three prospect-index plots.

Run:
    python3 05_position_analysis.py

Outputs:
    data/position_assignments.csv
    data/position_summary.csv
    data/position_feature_correlations.csv
    data/position_top_correlations.csv
    data/by_position/{PG,SG,SF,PF,C}.csv
    data/by_position/unassigned.csv
    plots/by_position/*.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from position_utils import (
    POSITIONS,
    POSITION_COLORS,
    add_primary_position,
    position_coverage,
)


MASTER_PATH = ROOT / "data/draft_with_college.csv"
CLEAN_DRAFT_PATH = ROOT / "data/nba_draft_clean.csv"
RAW_DRAFT_PATH = ROOT / "data/nba_draft_raw.csv"
ASSIGNMENTS_PATH = ROOT / "data/position_assignments.csv"
SUMMARY_PATH = ROOT / "data/position_summary.csv"
CORRELATIONS_PATH = ROOT / "data/position_feature_correlations.csv"
TOP_CORRELATIONS_PATH = ROOT / "data/position_top_correlations.csv"
POSITION_DATA_DIR = ROOT / "data/by_position"
PLOT_DIR = ROOT / "plots/by_position"

MIN_CORRELATION_SAMPLE = 20

FEATURE_FAMILIES = {
    "possession": [
        "ORB_per", "AST_per", "TO_per", "stl_per", "ast/tov", "oreb",
        "ast", "stl", "possession_creation_index",
    ],
    "scoring": [
        "usg", "eFG", "TS_per", "FTM", "FTA", "FT_per", "twoP_per",
        "TP_per", "ftr", "pts", "scoring_profile_score",
    ],
    "impact_offense": [
        "ORtg", "porpag", "adjoe", "bpm", "obpm", "gbpm", "ogbpm",
        "granular_prospect_index",
    ],
    "defense_size": [
        "DRB_per", "blk_per", "drtg", "adrtg", "dporpag", "stops",
        "dbpm", "dgbpm", "dreb", "blk", "height_in",
    ],
    "context_age": ["GP", "Min_per", "age_at_draft"],
    "market": ["pick"],
}

TARGETS = {
    "peak_3yr_vorp": "Peak 3-year VORP",
    "surplus_value": "Draft surplus value",
}

SCATTER_SPECS = {
    "possession_creation_index": (
        "Possession Creation Index", "pci_vs_surplus_value_by_position.png"
    ),
    "scoring_profile_score": (
        "Scoring Profile Score", "scoring_profile_vs_surplus_value_by_position.png"
    ),
    "granular_prospect_index": (
        "Granular Prospect Index", "granular_index_vs_surplus_value_by_position.png"
    ),
}


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().astype(float)
    if valid.empty:
        return result
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty(len(adjusted))
    restored[order] = np.minimum(adjusted, 1.0)
    result.loc[valid.index] = restored
    return result


def fisher_interval(r_value: float, n: int) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r_value):
        return np.nan, np.nan
    clipped = np.clip(r_value, -0.999999, 0.999999)
    center = np.arctanh(clipped)
    half_width = 1.96 / np.sqrt(n - 3)
    return float(np.tanh(center - half_width)), float(np.tanh(center + half_width))


def partial_correlation(
    feature: np.ndarray, target: np.ndarray, pick: np.ndarray
) -> tuple[float, float]:
    """Correlation after removing a quadratic draft-pick trend from both."""
    design = np.column_stack([np.ones(len(pick)), pick, pick ** 2])
    feature_residual = feature - design @ np.linalg.lstsq(
        design, feature, rcond=None
    )[0]
    target_residual = target - design @ np.linalg.lstsq(
        design, target, rcond=None
    )[0]
    if np.std(feature_residual) == 0 or np.std(target_residual) == 0:
        return np.nan, np.nan
    result = pearsonr(feature_residual, target_residual)
    return float(result.statistic), float(result.pvalue)


def add_age_at_draft(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "age_at_draft" in out:
        out["age_at_draft"] = pd.to_numeric(out["age_at_draft"], errors="coerce")
        return out
    dob = pd.to_datetime(out.get("dob"), errors="coerce")
    draft_day = pd.to_datetime(
        out["Draft_Year"].astype("Int64").astype(str) + "-06-25",
        errors="coerce",
    )
    out["age_at_draft"] = (draft_day - dob).dt.days / 365.25
    return out


def remove_position_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "nba_entry_position", "primary_position", "position_source", "position_confidence",
        "position_observed_season",
    ]
    return df.drop(columns=[column for column in columns if column in df])


def propagate_assignments(
    master: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    """Add the canonical assignment to derived tables; raw caches stay immutable."""
    assignment_columns = [
        "player", "Draft_Year", "nba_entry_position", "primary_position", "position_source",
        "position_confidence", "position_observed_season",
    ]
    lookup = assignments[assignment_columns].copy()

    if CLEAN_DRAFT_PATH.exists():
        clean = remove_position_columns(pd.read_csv(CLEAN_DRAFT_PATH, low_memory=False))
        clean = clean.merge(lookup, on=["player", "Draft_Year"], how="left")
        clean.to_csv(CLEAN_DRAFT_PATH, index=False)

    master.to_csv(MASTER_PATH, index=False)


def write_position_datasets(df: pd.DataFrame) -> None:
    POSITION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for position in POSITIONS:
        df[df["primary_position"].eq(position)].to_csv(
            POSITION_DATA_DIR / f"{position}.csv", index=False
        )
    df[df["primary_position"].isna()].to_csv(
        POSITION_DATA_DIR / "unassigned.csv", index=False
    )


def make_position_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for position in POSITIONS:
        group = df[df["primary_position"].eq(position)].copy()
        rows.append({
            "primary_position": position,
            "players": len(group),
            "college_matched_players": int(group["player_name"].notna().sum()),
            "mean_draft_pick": pd.to_numeric(group["pick"], errors="coerce").mean(),
            "mean_peak_3yr_vorp": pd.to_numeric(
                group["peak_3yr_vorp"], errors="coerce"
            ).mean(),
            "mean_surplus_value": pd.to_numeric(
                group["surplus_value"], errors="coerce"
            ).mean(),
            "rookie_position_labels": int(
                group["position_source"].eq("nba_rookie_season").sum()
            ),
            "college_role_labels": int(
                group["position_source"].eq("college_role").sum()
            ),
            "college_model_labels": int(
                group["position_source"].eq("college_position_model").sum()
            ),
        })
    rows.append({
        "primary_position": "Unassigned",
        "players": int(df["primary_position"].isna().sum()),
        "college_matched_players": int(
            (df["primary_position"].isna() & df["player_name"].notna()).sum()
        ),
    })
    return pd.DataFrame(rows)


def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    features = list(dict.fromkeys(
        feature for family in FEATURE_FAMILIES.values() for feature in family
    ))
    rows = []
    matched = df[df["player_name"].notna() & df["primary_position"].notna()].copy()
    scopes = list(POSITIONS) + ["ALL", "WITHIN_POSITION"]
    for position in scopes:
        if position in POSITIONS:
            group = matched[matched["primary_position"].eq(position)].copy()
        else:
            group = matched.copy()
        for target in TARGETS:
            for family, family_features in FEATURE_FAMILIES.items():
                for feature in family_features:
                    if feature not in group or target not in group:
                        continue
                    numeric = pd.DataFrame({
                        "feature": pd.to_numeric(group[feature], errors="coerce"),
                        "target": pd.to_numeric(group[target], errors="coerce"),
                        "pick": pd.to_numeric(group["pick"], errors="coerce"),
                        "position": group["primary_position"].astype("string"),
                    }).dropna()
                    n = len(numeric)
                    if n < MIN_CORRELATION_SAMPLE or numeric["feature"].nunique() < 2:
                        continue
                    if position == "WITHIN_POSITION":
                        for column in ("feature", "target", "pick"):
                            numeric[column] = numeric[column] - numeric.groupby(
                                "position"
                            )[column].transform("mean")
                    pearson = pearsonr(numeric["feature"], numeric["target"])
                    spearman = spearmanr(numeric["feature"], numeric["target"])
                    partial_r, partial_p = partial_correlation(
                        numeric["feature"].to_numpy(dtype=float),
                        numeric["target"].to_numpy(dtype=float),
                        numeric["pick"].to_numpy(dtype=float),
                    )
                    ci_low, ci_high = fisher_interval(float(pearson.statistic), n)
                    rows.append({
                        "primary_position": position,
                        "target": target,
                        "feature_family": family,
                        "feature": feature,
                        "n": n,
                        "pearson_r": float(pearson.statistic),
                        "pearson_ci_low": ci_low,
                        "pearson_ci_high": ci_high,
                        "pearson_p_value": float(pearson.pvalue),
                        "spearman_r": float(spearman.statistic),
                        "spearman_p_value": float(spearman.pvalue),
                        "partial_r_controlling_pick": partial_r,
                        "partial_p_value": partial_p,
                    })
    result = pd.DataFrame(rows)
    result["pearson_q_value"] = np.nan
    result["partial_q_value"] = np.nan
    for (_, _), indices in result.groupby(
        ["primary_position", "target"], observed=True
    ).groups.items():
        result.loc[indices, "pearson_q_value"] = benjamini_hochberg(
            result.loc[indices, "pearson_p_value"]
        )
        result.loc[indices, "partial_q_value"] = benjamini_hochberg(
            result.loc[indices, "partial_p_value"]
        )
    return result.sort_values(
        ["target", "primary_position", "pearson_r"],
        key=lambda values: values.abs() if values.name == "pearson_r" else values,
        ascending=[True, True, False],
    ).reset_index(drop=True)


def draw_correlation_heatmap(
    correlations: pd.DataFrame, target: str, partial: bool = False
) -> Path:
    value_column = "partial_r_controlling_pick" if partial else "pearson_r"
    q_column = "partial_q_value" if partial else "pearson_q_value"
    subset = correlations[correlations["target"].eq(target)].copy()
    matrix = subset.pivot(
        index="feature", columns="primary_position", values=value_column
    ).reindex(columns=POSITIONS)
    q_values = subset.pivot(
        index="feature", columns="primary_position", values=q_column
    ).reindex(columns=POSITIONS)
    order = matrix.abs().max(axis=1).sort_values(ascending=False).index
    matrix = matrix.loc[order]
    q_values = q_values.loc[order]

    fig_height = max(10, 0.31 * len(matrix) + 2.5)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(POSITIONS)), POSITIONS, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=8)
    for row in range(len(matrix)):
        for column in range(len(POSITIONS)):
            value = matrix.iloc[row, column]
            if pd.isna(value):
                label = "—"
            else:
                star = "*" if q_values.iloc[row, column] < 0.05 else ""
                label = f"{value:+.2f}{star}"
            ax.text(column, row, label, ha="center", va="center", fontsize=7.4)
    method = "Partial r controlling for draft pick" if partial else "Pearson r"
    ax.set_title(
        f"{TARGETS[target]} correlations within primary position\n"
        f"{method}; * Benjamini-Hochberg q < 0.05",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label(method)
    ax.set_xlabel("Primary position")
    ax.set_ylabel("Pre-NBA feature")
    fig.tight_layout()
    suffix = "partial" if partial else "pearson"
    output = PLOT_DIR / f"{target}_correlations_by_position_{suffix}.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def draw_position_scatter(
    df: pd.DataFrame, feature: str, label: str, filename: str
) -> Path:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8), sharey=True)
    rng = np.random.default_rng(42)
    for ax, position in zip(axes, POSITIONS):
        group = df[
            df["primary_position"].eq(position) & df["player_name"].notna()
        ].copy()
        x = pd.to_numeric(group[feature], errors="coerce")
        y = pd.to_numeric(group["surplus_value"], errors="coerce")
        valid = x.notna() & y.notna()
        x = x[valid]
        y = y[valid]
        display_x = x.to_numpy(dtype=float)
        if feature == "scoring_profile_score":
            display_x = display_x + rng.normal(0, 0.055, len(display_x))
        ax.scatter(
            display_x, y, s=25, alpha=0.58,
            color=POSITION_COLORS[position], edgecolors="white", linewidths=0.25,
        )
        if len(x) >= 3 and x.nunique() >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            trend_x = np.linspace(x.min(), x.max(), 100)
            ax.plot(trend_x, slope * trend_x + intercept, color="#111827", linewidth=1.7)
            r_value = x.corr(y)
        else:
            r_value = np.nan
        ax.axhline(0, color="#64748b", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"{position}  n={len(x)}  r={r_value:+.3f}", fontweight="bold")
        ax.set_xlabel(label, fontsize=9)
        ax.grid(True, alpha=0.14)
    axes[0].set_ylabel("Peak 3-year VORP surplus value")
    fig.suptitle(
        f"{label} vs. draft surplus within primary position",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output = PLOT_DIR / filename
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def draw_position_counts(summary: pd.DataFrame) -> Path:
    assigned = summary[summary["primary_position"].isin(POSITIONS)].copy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        assigned["primary_position"], assigned["players"],
        color=[POSITION_COLORS[position] for position in assigned["primary_position"]],
    )
    for index, row in assigned.reset_index(drop=True).iterrows():
        ax.text(index, row["players"] + 4, str(int(row["players"])), ha="center", fontweight="bold")
    ax.set_title("Drafted players by assigned primary position", fontsize=14, fontweight="bold")
    ax.set_ylabel("Players")
    ax.grid(axis="y", alpha=0.16)
    fig.tight_layout()
    output = PLOT_DIR / "position_assignment_counts.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"{MASTER_PATH} not found. Run 02_collect_college_stats.py first."
        )
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    source = remove_position_columns(pd.read_csv(MASTER_PATH, low_memory=False))
    positioned = add_primary_position(add_age_at_draft(source))

    assignment_columns = [
        "player", "Draft_Year", "pick", "nba_entry_position", "primary_position", "position_source",
        "position_confidence", "position_observed_season", "role", "ht",
        "height_in", "college",
    ]
    assignments = positioned[assignment_columns].copy()
    assignments.to_csv(ASSIGNMENTS_PATH, index=False)
    propagate_assignments(positioned, assignments)
    write_position_datasets(positioned)

    summary = make_position_summary(positioned)
    summary.to_csv(SUMMARY_PATH, index=False)
    correlations = calculate_correlations(positioned)
    correlations.to_csv(CORRELATIONS_PATH, index=False)
    top_correlations = (
        correlations[correlations["primary_position"].isin(POSITIONS)]
        .assign(_absolute_r=lambda frame: frame["pearson_r"].abs())
        .sort_values(
            ["target", "primary_position", "_absolute_r"],
            ascending=[True, True, False],
        )
        .groupby(["target", "primary_position"], observed=True)
        .head(10)
        .drop(columns="_absolute_r")
    )
    top_correlations.to_csv(TOP_CORRELATIONS_PATH, index=False)

    outputs = [draw_position_counts(summary)]
    for target in TARGETS:
        outputs.append(draw_correlation_heatmap(correlations, target, partial=False))
        outputs.append(draw_correlation_heatmap(correlations, target, partial=True))
    for feature, (label, filename) in SCATTER_SPECS.items():
        outputs.append(draw_position_scatter(positioned, feature, label, filename))

    audit = position_coverage(positioned)
    assigned = int(positioned["primary_position"].notna().sum())
    matched = positioned["player_name"].notna()
    matched_assigned = int(positioned.loc[matched, "primary_position"].notna().sum())
    print("Position assignment coverage:")
    print(audit.to_string(index=False))
    print(f"\nAssigned drafted players: {assigned}/{len(positioned)}")
    print(f"Assigned college-matched analysis players: {matched_assigned}/{int(matched.sum())}")
    print(f"Saved: {ASSIGNMENTS_PATH.relative_to(ROOT)}")
    print(f"Saved: {SUMMARY_PATH.relative_to(ROOT)}")
    print(f"Saved: {CORRELATIONS_PATH.relative_to(ROOT)}")
    print(f"Saved: {TOP_CORRELATIONS_PATH.relative_to(ROOT)}")
    print(f"Saved position datasets: {POSITION_DATA_DIR.relative_to(ROOT)}/")
    for output in outputs:
        print(f"Saved plot: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
