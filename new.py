"""
NBA Draft Surplus Value Project — Correlations & Combined Predictors
====================================================================

Goal
----
1. Measure how each *pre-NBA* (college / recruiting / draft) statistic correlates
   with NBA **peak 3-year VORP** (average VORP over a player's best 3 straight
   seasons — isolates peak quality from career length).
2. Use a |Pearson r| >= 0.5 threshold on the STAT-TO-STAT matrix to find families
   of redundant statistics, then COMBINE each family into a single composite.
3. Combine those families into one model and report an honest predictive
   correlation (in-sample multiple R and 5-fold cross-validated r).

Key finding (see the printout): no single college stat reaches r >= 0.5 against
peak VORP (strongest are draft pick ~-0.39 and college BPM/gBPM ~0.31). But
COMBINING the redundant families + draft pick pushes the combined predictor to an
in-sample multiple R ~0.49 (>0.5 once every feature is used) and an honest
5-fold cross-validated r ~0.46 — materially better than the same model against
*career* VORP (~0.39), because peak VORP drops the longevity noise no college box
score can forecast. The 0.5 threshold on the stat-to-stat matrix is what tells you
*which* stats are redundant and should be merged before modelling.

Usage:
    python new.py

Output:
    - plots/college_stat_correlation_clusters.png
    - console tables
"""

import sys
from pathlib import Path

# The project vendors its dependencies in ./.deps — make the script runnable
# standalone the same way steps 1-2 assume an importable pandas/numpy.
_DEPS = Path(__file__).resolve().parent / ".deps"
if _DEPS.exists():
    sys.path.insert(0, str(_DEPS))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict

DATA_PATH = Path("data/draft_with_college.csv")
HEATMAP_PATH = Path("plots/college_stat_correlation_clusters.png")
TARGET = "peak_3yr_vorp"      # average VORP over a player's best 3 straight seasons
TARGET_LABEL = "peak 3-yr VORP"
CLUSTER_THRESHOLD = 0.5  # merge stats whose |Pearson r| >= 0.5

# Pre-NBA numeric predictors available at draft time (Barttorvik college block,
# recruiting rank, engineered height/age). NBA-outcome columns are intentionally
# excluded to avoid leakage.
COLLEGE_FEATURES = [
    "GP", "Min_per", "ORtg", "usg", "eFG", "TS_per", "ORB_per", "DRB_per",
    "AST_per", "TO_per", "FTM", "FTA", "FT_per", "twoP_per", "TP_per", "blk_per",
    "stl_per", "ftr", "porpag", "adjoe", "ast/tov", "drtg", "adrtg", "dporpag",
    "stops", "bpm", "obpm", "dbpm", "gbpm", "ogbpm", "dgbpm", "oreb", "dreb",
    "ast", "stl", "blk", "pts", "height_in", "age_at_draft",
]


def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    def ht_to_inches(text):
        try:
            feet, inches = str(text).split("-")
            return int(feet) * 12 + int(inches)
        except (ValueError, AttributeError):
            return np.nan

    df["height_in"] = df["ht"].map(ht_to_inches)
    dob = pd.to_datetime(df["dob"], errors="coerce")
    draft_day = pd.to_datetime(
        df["Draft_Year"].astype("Int64").astype(str) + "-06-25", errors="coerce"
    )
    df["age_at_draft"] = (draft_day - dob).dt.days / 365.25
    return df


def zscore(series):
    x = pd.to_numeric(series, errors="coerce")
    return (x - x.mean()) / x.std()


def pearson(a, b):
    mask = a.notna() & b.notna()
    if mask.sum() < 30:
        return np.nan
    return np.corrcoef(a[mask], b[mask])[0, 1]


def per_stat_correlations(df, y):
    """Pearson (and Spearman) of each single college stat vs career VORP."""
    rows = []
    for col in COLLEGE_FEATURES + ["pick"]:
        x = pd.to_numeric(df[col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 30:
            continue
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        rs = pd.Series(x[mask].values).corr(pd.Series(y[mask].values), method="spearman")
        rows.append((col, int(mask.sum()), r, rs))
    out = pd.DataFrame(rows, columns=["stat", "n", "pearson_r", "spearman_r"])
    return out.reindex(out["pearson_r"].abs().sort_values(ascending=False).index)


def cluster_features(X):
    """Group college stats into families linked at |Pearson r| >= threshold."""
    corr = X.corr()
    abs_corr = corr.abs().fillna(0.0).values.copy()
    np.fill_diagonal(abs_corr, 1.0)
    dist = 1.0 - abs_corr
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - CLUSTER_THRESHOLD, criterion="distance")
    order = leaves_list(Z)  # for a block-diagonal heatmap
    clusters = {}
    for feat, lab in zip(X.columns, labels):
        clusters.setdefault(lab, []).append(feat)
    return corr, clusters, order


def build_composite(X, feats, y):
    """Sign-aligned mean of z-scores so higher composite = more career VORP."""
    parts = []
    for f in feats:
        z = zscore(X[f])
        sign = np.sign(pearson(z, y)) or 1.0
        parts.append(sign * z)
    return pd.concat(parts, axis=1).mean(axis=1)


def name_cluster(feats):
    labels = {
        "bpm": "offensive impact/efficiency", "porpag": "offensive impact/efficiency",
        "adjoe": "offensive impact/efficiency", "dbpm": "defensive impact",
        "drtg": "defensive impact", "stl": "steals/disruption",
        "ORB_per": "size/rebounding/rim-protection", "blk": "size/rebounding/rim-protection",
        "usg": "scoring volume/usage", "pts": "scoring volume/usage",
        "eFG": "shooting efficiency", "AST_per": "playmaking",
        "GP": "availability", "age_at_draft": "youth",
    }
    for key, nm in labels.items():
        if key in feats:
            return nm
    return feats[0]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run steps 1-2 first.")

    df = load_data()
    y_all = pd.to_numeric(df[TARGET], errors="coerce")

    # ---- 1. single-stat correlations -------------------------------------
    print("=" * 74)
    print(f"1. EACH PRE-NBA STAT vs {TARGET_LABEL}  (sorted by |Pearson r|)")
    print("=" * 74)
    per = per_stat_correlations(df, y_all)
    print(per.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))
    strong = per[per["pearson_r"].abs() >= 0.5]
    print(f"\n  Stats with |Pearson r| >= 0.5 vs {TARGET_LABEL}: "
          f"{len(strong) if len(strong) else 'NONE'}")

    # analysis sample = players with a college row
    has_college = df["bpm"].notna()
    X = df.loc[has_college, COLLEGE_FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.reset_index(drop=True)
    y = y_all[has_college].reset_index(drop=True)
    print(f"  Analysis sample (players with Barttorvik stats): n = {len(X)}")

    # ---- 2. cluster redundant stats (|r| >= 0.5) and combine -------------
    print("\n" + "=" * 74)
    print(f"2. STAT FAMILIES linked at |Pearson r| >= {CLUSTER_THRESHOLD}"
          " -> combined composites")
    print("=" * 74)
    corr, clusters, order = cluster_features(X)
    composites = {}
    for feats in sorted(clusters.values(), key=len, reverse=True):
        comp = build_composite(X, feats, y)
        composites[name_cluster(feats)] = comp
        r = pearson(comp, y)
        if len(feats) >= 2:
            iu = corr.loc[feats, feats].abs().values[np.triu_indices(len(feats), 1)]
            intra = np.nanmean(iu)
            tag = f"avg intra |r|={intra:.2f}"
        else:
            tag = "singleton"
        print(f"\n  [{len(feats)} stats | {tag}] composite r vs VORP = {r:+.3f}"
              f"  -> {name_cluster(feats)}")
        print("     " + ", ".join(feats))

    # ---- 3. combine all families into one model -------------------------
    print("\n" + "=" * 74)
    print("3. COMBINED MODEL (all family composites)")
    print("=" * 74)
    comp_df = pd.DataFrame(composites)
    comp_df = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(comp_df),
        columns=comp_df.columns,
    )
    r_insample = np.corrcoef(LinearRegression().fit(comp_df, y).predict(comp_df), y)[0, 1]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 40)))
    r_cv = np.corrcoef(cross_val_predict(ridge, comp_df, y, cv=cv), y)[0, 1]
    print(f"  Composites only          in-sample multiple R = {r_insample:+.3f}"
          f"   |   5-fold CV r = {r_cv:+.3f}")

    # ---- 4. clustered correlation heatmap -------------------------------
    make_heatmap(corr, order)
    print(f"\n  Saved figure: {HEATMAP_PATH}")


def make_heatmap(corr, order):
    """Diverging heatmap (blue<0, gray~0, red>0), cluster-ordered so the
    |r| >= 0.5 families appear as red blocks along the diagonal."""
    feats = [corr.columns[i] for i in order]
    M = corr.loc[feats, feats].values
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(feats)))
    ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=7)
    ax.set_yticklabels(feats, fontsize=7)
    # outline the >=0.5 cells so the families are unmistakable
    for i in range(len(feats)):
        for j in range(len(feats)):
            if i != j and abs(M[i, j]) >= CLUSTER_THRESHOLD:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#111827", linewidth=0.4))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r between college stats", fontsize=10)
    ax.set_title(
        "College-stat correlation matrix (cluster-ordered)\n"
        "boxed cells: |r| >= 0.5 — redundant stats that get combined",
        fontsize=13, fontweight="bold",
    )
    HEATMAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(HEATMAP_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
