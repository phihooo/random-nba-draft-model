"""Map college-player similarities and clusters with two-dimensional t-SNE.

The embedding uses pre-NBA information only.  For each basketball-defined
feature family (and all families combined), the script:

* median-imputes and standardizes the features;
* chooses 2-8 KMeans clusters using the best silhouette score in the original
  standardized feature space;
* projects the players to two dimensions with t-SNE for visualization;
* measures local-neighborhood preservation with t-SNE trustworthiness; and
* exports each player's nearest statistical comparisons in the original
  standardized feature space.

Run:
    python3 04_tsne_similarity_clusters.py

Outputs:
    plots/tsne_player_similarity_clusters.png
    plots/tsne/*.png
    plots/tsne/by_position/{PG,SG,SF,PF,C}.png
    data/tsne_player_embeddings.csv
    data/tsne_nearest_neighbors.csv
    data/tsne_cluster_metrics.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

# Keep matplotlib's cache inside the project when the user cache is read-only.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from position_utils import POSITIONS, add_primary_position


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = ROOT / "data/draft_with_college.csv"
COMBINED_PLOT_PATH = ROOT / "plots/tsne_player_similarity_clusters.png"
INDIVIDUAL_PLOT_DIR = ROOT / "plots/tsne"
POSITION_PLOT_DIR = INDIVIDUAL_PLOT_DIR / "by_position"
EMBEDDINGS_PATH = ROOT / "data/tsne_player_embeddings.csv"
NEIGHBORS_PATH = ROOT / "data/tsne_nearest_neighbors.csv"
METRICS_PATH = ROOT / "data/tsne_cluster_metrics.csv"

RANDOM_SEED = 42
DEFAULT_PERPLEXITY = 30.0
DEFAULT_MAX_CLUSTERS = 8
DEFAULT_NEIGHBORS = 5

# Basketball-defined groups are fixed before looking at the embedding.  NBA
# outcomes (VORP, BPM, win shares, etc.) are deliberately absent.
FEATURE_FAMILIES = {
    "possession": [
        "ORB_per", "AST_per", "TO_per", "stl_per", "ast/tov", "oreb",
        "ast", "stl",
    ],
    "scoring": [
        "usg", "eFG", "TS_per", "FTM", "FTA", "FT_per", "twoP_per",
        "TP_per", "ftr", "pts",
    ],
    "impact_offense": [
        "ORtg", "porpag", "adjoe", "bpm", "obpm", "gbpm", "ogbpm",
    ],
    "defense_size": [
        "DRB_per", "blk_per", "drtg", "adrtg", "dporpag", "stops",
        "dbpm", "dgbpm", "dreb", "blk", "height_in",
    ],
    "context_age": ["GP", "Min_per", "age_at_draft"],
}

PANEL_LABELS = {
    "possession": "Possession creation",
    "scoring": "Scoring profile",
    "impact_offense": "Offensive impact",
    "defense_size": "Defense and size",
    "context_age": "Context and age",
    "all_features": "All college features",
}


def height_to_inches(value: object) -> float:
    try:
        feet, inches = str(value).split("-")
        return int(feet) * 12 + int(inches)
    except (TypeError, ValueError):
        return np.nan


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run 02_collect_college_stats.py first."
        )

    df = add_primary_position(pd.read_csv(DATA_PATH, low_memory=False))
    required = {"player", "Draft_Year", "pick", "player_name", "ht", "dob"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df["height_in"] = df["ht"].map(height_to_inches)
    dob = pd.to_datetime(df["dob"], errors="coerce")
    draft_day = pd.to_datetime(
        df["Draft_Year"].astype("Int64").astype(str) + "-06-25",
        errors="coerce",
    )
    df["age_at_draft"] = (draft_day - dob).dt.days / 365.25

    # player_name is populated only when the draft row matched a college row.
    df = df[df["player_name"].notna()].copy().reset_index(drop=True)
    if len(df) < 30:
        raise ValueError("At least 30 college-matched players are required.")
    return df


def usable_features(df: pd.DataFrame, requested: list[str]) -> list[str]:
    """Return numeric, non-constant features usable for this sample."""
    missing = [feature for feature in requested if feature not in df.columns]
    if missing:
        raise KeyError(f"Missing requested feature columns: {missing}")
    usable = []
    for feature in requested:
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.notna().sum() >= 2 and values.nunique(dropna=True) >= 2:
            usable.append(feature)
    if len(usable) < 2:
        raise ValueError(f"Fewer than two usable features among: {requested}")
    return usable


def standardized_matrix(
    df: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, list[str]]:
    selected = usable_features(df, features)
    numeric = df[selected].apply(pd.to_numeric, errors="coerce")
    pipeline = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
    )
    values = pipeline.fit_transform(numeric)
    return np.asarray(values, dtype=float), selected


def select_clusters(
    values: np.ndarray, max_clusters: int, seed: int
) -> tuple[np.ndarray, int, float, float]:
    """Select K in source feature space, not from potentially distorted t-SNE."""
    upper = min(max_clusters, len(values) - 1)
    if upper < 2:
        raise ValueError("At least three observations are required for clustering.")

    best: tuple[float, int, np.ndarray] | None = None
    for k in range(2, upper + 1):
        labels = KMeans(n_clusters=k, n_init=25, random_state=seed).fit_predict(values)
        score = float(silhouette_score(values, labels))
        candidate = (score, -k, labels)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    assert best is not None
    silhouette, negative_k, labels = best
    k = -negative_k
    db_score = float(davies_bouldin_score(values, labels))
    return labels, k, silhouette, db_score


def embed(
    values: np.ndarray, perplexity: float, seed: int, max_iter: int
) -> tuple[np.ndarray, float, float]:
    # t-SNE requires perplexity < n_samples.  The one-third cap follows the
    # practical constraint that each effective neighborhood needs enough data.
    effective_perplexity = min(perplexity, max(5.0, (len(values) - 1) / 3.0))
    coordinates = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=max_iter,
        random_state=seed,
        method="barnes_hut",
    ).fit_transform(values)
    local_neighbors = min(15, max(5, (len(values) - 1) // 2))
    preservation = float(
        trustworthiness(values, coordinates, n_neighbors=local_neighbors)
    )
    return coordinates, effective_perplexity, preservation


def relabel_clusters(labels: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Give clusters deterministic left-to-right display labels starting at 1."""
    ordered = sorted(
        np.unique(labels), key=lambda label: coordinates[labels == label, 0].mean()
    )
    mapping = {old: new for new, old in enumerate(ordered, start=1)}
    return np.array([mapping[label] for label in labels], dtype=int)


def cluster_representatives(
    df: pd.DataFrame, values: np.ndarray, labels: np.ndarray
) -> dict[int, tuple[int, str]]:
    representatives = {}
    for label in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        centroid = values[indices].mean(axis=0)
        representative = indices[np.argmin(np.linalg.norm(values[indices] - centroid, axis=1))]
        representatives[int(label)] = (int(representative), str(df.iloc[representative]["player"]))
    return representatives


def nearest_neighbor_rows(
    df: pd.DataFrame,
    values: np.ndarray,
    labels: np.ndarray,
    panel: str,
    count: int,
    analysis_scope: str,
) -> list[dict]:
    count = min(count, len(values) - 1)
    model = NearestNeighbors(n_neighbors=count + 1, metric="euclidean")
    distances, indices = model.fit(values).kneighbors(values)
    scale = np.sqrt(values.shape[1])
    rows = []
    for player_index in range(len(df)):
        for rank, (neighbor_index, distance) in enumerate(
            zip(indices[player_index, 1:], distances[player_index, 1:]), start=1
        ):
            normalized_distance = float(distance / scale)
            rows.append({
                "analysis_scope": analysis_scope,
                "panel": panel,
                "player": df.iloc[player_index]["player"],
                "draft_year": int(df.iloc[player_index]["Draft_Year"]),
                "primary_position": str(
                    df.iloc[player_index]["primary_position"]
                ),
                "neighbor_rank": rank,
                "similar_player": df.iloc[neighbor_index]["player"],
                "similar_player_draft_year": int(df.iloc[neighbor_index]["Draft_Year"]),
                "similar_player_position": str(
                    df.iloc[neighbor_index]["primary_position"]
                ),
                "standardized_distance": normalized_distance,
                "similarity_score": 1.0 / (1.0 + normalized_distance),
                "same_cluster": bool(labels[player_index] == labels[neighbor_index]),
            })
    return rows


def draw_panel(
    ax,
    df: pd.DataFrame,
    values: np.ndarray,
    coordinates: np.ndarray,
    labels: np.ndarray,
    panel: str,
    silhouette: float,
    preservation: float,
    show_legend: bool,
) -> None:
    palette = plt.get_cmap("tab10")
    for label in sorted(np.unique(labels)):
        mask = labels == label
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=24,
            alpha=0.68,
            color=palette((label - 1) % 10),
            edgecolors="white",
            linewidths=0.25,
            label=f"Cluster {label} (n={mask.sum()})",
        )

    for label, (index, player) in cluster_representatives(df, values, labels).items():
        ax.annotate(
            f"C{label}: {player}",
            xy=coordinates[index],
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            color="#111827",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "linewidth": 0.3},
        )

    ax.set_title(
        f"{PANEL_LABELS[panel]}\n"
        f"k={len(np.unique(labels))}  silhouette={silhouette:.3f}  "
        f"trust={preservation:.3f}",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_xlabel("t-SNE dimension 1", fontsize=8)
    ax.set_ylabel("t-SNE dimension 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.12)
    if show_legend:
        ax.legend(fontsize=7, frameon=True, ncol=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perplexity", type=float, default=DEFAULT_PERPLEXITY)
    parser.add_argument("--max-clusters", type=int, default=DEFAULT_MAX_CLUSTERS)
    parser.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    if args.perplexity <= 0:
        parser.error("--perplexity must be positive")
    if args.max_clusters < 2:
        parser.error("--max-clusters must be at least 2")
    if args.neighbors < 1:
        parser.error("--neighbors must be at least 1")
    if args.max_iter < 250:
        parser.error("--max-iter must be at least 250")
    return args


def main() -> None:
    args = parse_args()
    df = load_data()
    all_features = list(dict.fromkeys(
        feature
        for family in FEATURE_FAMILIES.values()
        for feature in family
    ))
    panels = {**FEATURE_FAMILIES, "all_features": all_features}

    results = {}
    embedding_rows = []
    neighbor_rows = []
    metric_rows = []

    for panel_index, (panel, requested_features) in enumerate(panels.items()):
        values, features = standardized_matrix(df, requested_features)
        labels, k, silhouette, db_score = select_clusters(
            values, args.max_clusters, args.seed
        )
        coordinates, effective_perplexity, preservation = embed(
            values, args.perplexity, args.seed + panel_index, args.max_iter
        )
        labels = relabel_clusters(labels, coordinates)
        results[panel] = (values, coordinates, labels, silhouette, preservation)

        for row_index, row in df.iterrows():
            embedding_rows.append({
                "analysis_scope": "all_players",
                "panel": panel,
                "player": row["player"],
                "draft_year": int(row["Draft_Year"]),
                "pick": int(row["pick"]),
                "primary_position": str(row["primary_position"]),
                "tsne_1": float(coordinates[row_index, 0]),
                "tsne_2": float(coordinates[row_index, 1]),
                "cluster": int(labels[row_index]),
            })
        neighbor_rows.extend(
            nearest_neighbor_rows(
                df, values, labels, panel, args.neighbors, "all_players"
            )
        )
        metric_rows.append({
            "analysis_scope": "all_players",
            "primary_position": "ALL",
            "panel": panel,
            "players": len(df),
            "source_features": len(features),
            "transformed_dimensions": values.shape[1],
            "selected_clusters": k,
            "silhouette_score": silhouette,
            "davies_bouldin_score": db_score,
            "tsne_trustworthiness": preservation,
            "perplexity": effective_perplexity,
            "features": "|".join(features),
        })
        print(
            f"{PANEL_LABELS[panel]:<24} k={k}  "
            f"silhouette={silhouette:.3f}  trust={preservation:.3f}"
        )

    # Refit every embedding and cluster model within a single primary
    # position.  This prevents size/role separation from dominating player
    # similarities and guarantees every exported neighbor is position-matched.
    POSITION_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for position_index, position in enumerate(POSITIONS):
        position_df = df[df["primary_position"].eq(position)].reset_index(drop=True)
        if len(position_df) < 30:
            print(f"Skipping {position}: only {len(position_df)} players")
            continue
        position_results = {}
        print(f"\n{position} position-only maps (n={len(position_df)})")
        for panel_index, (panel, requested_features) in enumerate(panels.items()):
            values, features = standardized_matrix(position_df, requested_features)
            labels, k, silhouette, db_score = select_clusters(
                values, args.max_clusters, args.seed
            )
            coordinates, effective_perplexity, preservation = embed(
                values,
                args.perplexity,
                args.seed + 100 + position_index * len(panels) + panel_index,
                args.max_iter,
            )
            labels = relabel_clusters(labels, coordinates)
            position_results[panel] = (
                values, coordinates, labels, silhouette, preservation
            )
            for row_index, row in position_df.iterrows():
                embedding_rows.append({
                    "analysis_scope": "position_only",
                    "panel": panel,
                    "player": row["player"],
                    "draft_year": int(row["Draft_Year"]),
                    "pick": int(row["pick"]),
                    "primary_position": position,
                    "tsne_1": float(coordinates[row_index, 0]),
                    "tsne_2": float(coordinates[row_index, 1]),
                    "cluster": int(labels[row_index]),
                })
            neighbor_rows.extend(nearest_neighbor_rows(
                position_df,
                values,
                labels,
                panel,
                args.neighbors,
                "position_only",
            ))
            metric_rows.append({
                "analysis_scope": "position_only",
                "primary_position": position,
                "panel": panel,
                "players": len(position_df),
                "source_features": len(features),
                "transformed_dimensions": values.shape[1],
                "selected_clusters": k,
                "silhouette_score": silhouette,
                "davies_bouldin_score": db_score,
                "tsne_trustworthiness": preservation,
                "perplexity": effective_perplexity,
                "features": "|".join(features),
            })
            print(
                f"  {PANEL_LABELS[panel]:<22} k={k}  "
                f"silhouette={silhouette:.3f}  trust={preservation:.3f}"
            )

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        for ax, panel in zip(axes.flat, panels):
            values, coordinates, labels, silhouette, preservation = (
                position_results[panel]
            )
            draw_panel(
                ax,
                position_df,
                values,
                coordinates,
                labels,
                panel,
                silhouette,
                preservation,
                show_legend=False,
            )
        fig.suptitle(
            f"{position} prospects: position-only t-SNE similarities and clusters",
            fontsize=16,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.015,
            "Every comparison is restricted to the same primary position; "
            "clusters use standardized pre-NBA features.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.96))
        fig.savefig(
            POSITION_PLOT_DIR / f"{position}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(embedding_rows).to_csv(EMBEDDINGS_PATH, index=False)
    pd.DataFrame(neighbor_rows).to_csv(NEIGHBORS_PATH, index=False)
    pd.DataFrame(metric_rows).to_csv(METRICS_PATH, index=False)

    COMBINED_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, panel in zip(axes.flat, panels):
        values, coordinates, labels, silhouette, preservation = results[panel]
        draw_panel(
            ax, df, values, coordinates, labels, panel,
            silhouette, preservation, show_legend=False,
        )
    fig.suptitle(
        "NBA Draft Prospects: t-SNE Local Similarities and Statistical Clusters",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Clusters and nearest neighbors are calculated in standardized pre-NBA "
        "feature space; t-SNE is the 2D display only. Labeled players are cluster representatives.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(COMBINED_PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    INDIVIDUAL_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for panel in panels:
        values, coordinates, labels, silhouette, preservation = results[panel]
        fig, ax = plt.subplots(figsize=(11, 8))
        draw_panel(
            ax, df, values, coordinates, labels, panel,
            silhouette, preservation, show_legend=True,
        )
        fig.tight_layout()
        fig.savefig(INDIVIDUAL_PLOT_DIR / f"{panel}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"\nSaved combined graph: {COMBINED_PLOT_PATH.relative_to(ROOT)}")
    print(f"Saved individual graphs: {INDIVIDUAL_PLOT_DIR.relative_to(ROOT)}/")
    print(f"Saved position-only graphs: {POSITION_PLOT_DIR.relative_to(ROOT)}/")
    print(f"Saved embeddings: {EMBEDDINGS_PATH.relative_to(ROOT)}")
    print(f"Saved nearest neighbors: {NEIGHBORS_PATH.relative_to(ROOT)}")
    print(f"Saved metrics: {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
