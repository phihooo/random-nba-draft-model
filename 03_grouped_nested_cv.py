"""Leakage-safe evaluation of college-stat families beyond draft position.

This script answers a narrower and more useful question than a table of
single-stat p-values:

    Does a family of college statistics improve predictions for an entirely
    unseen draft class after draft position is already known?

Design
------
* The outer split leaves out one complete draft year at a time.
* Ridge regularization is selected by a five-fold GroupKFold using only the
  remaining draft years (nested cross-validation).
* Draft position is represented by a smooth spline fitted inside each fold.
* Imputation, scaling, and the fixed PCI standardization are fitted inside each
  fold.
* The target is reconstructed from season-level files so every player receives
  the same outcome horizon.  No full-sample draft-slot average is used.
* Statistical comparisons use paired, draft-year-level loss differences.

Outputs
-------
* data/grouped_nested_cv_model_metrics.csv
* data/grouped_nested_cv_comparisons.csv
* data/grouped_nested_cv_oof_predictions.csv
* data/grouped_nested_cv_position_metrics.csv

Run:
    python3 03_grouped_nested_cv.py
"""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path


DEPS = Path(__file__).resolve().parent / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from position_utils import POSITIONS, add_primary_position


DATA_PATH = Path("data/draft_with_college.csv")
SEASON_DIR = Path("data/nba_seasons")
METRICS_PATH = Path("data/grouped_nested_cv_model_metrics.csv")
COMPARISONS_PATH = Path("data/grouped_nested_cv_comparisons.csv")
PREDICTIONS_PATH = Path("data/grouped_nested_cv_oof_predictions.csv")
POSITION_METRICS_PATH = Path("data/grouped_nested_cv_position_metrics.csv")

RANDOM_SEED = 42
ALPHAS = np.logspace(-2, 3, 10)
INNER_SPLITS = 5
FIRST_FIVE_MAX_DRAFT_YEAR = 2021


# Mutually exclusive, basketball-defined families.  Keeping these definitions
# independent of the outcome avoids discovering convenient groups on test data.
FEATURE_FAMILIES = {
    "possession": [
        "ORB_per", "AST_per", "TO_per", "stl_per",
        "ast/tov", "oreb", "ast", "stl",
    ],
    "scoring": [
        "usg", "eFG", "TS_per", "FTM", "FTA", "FT_per",
        "twoP_per", "TP_per", "ftr", "pts",
    ],
    "impact_offense": [
        "ORtg", "porpag", "adjoe", "bpm", "obpm", "gbpm", "ogbpm",
    ],
    "defense_size": [
        "DRB_per", "blk_per", "drtg", "adrtg", "dporpag", "stops",
        "dbpm", "dgbpm", "dreb", "blk",
    ],
    "context_age": ["GP", "Min_per", "height_in", "age_at_draft"],
}

ALL_FEATURES = list(itertools.chain.from_iterable(FEATURE_FAMILIES.values()))
PCI_COLUMNS = ["stl_per", "ORB_per", "TO_per", "AST_per"]
PCI_WEIGHTS = np.array([1.0, 0.8, -1.0, 0.6])


def normalize_name(value: object) -> str:
    """Match the normalization used by the data-collection scripts."""
    return re.sub(r"[^a-z\s]", "", str(value).lower().strip())


def height_to_inches(value: object) -> float:
    try:
        feet, inches = str(value).split("-")
        return int(feet) * 12 + int(inches)
    except (TypeError, ValueError):
        return np.nan


def load_season_vorp() -> dict[tuple[str, int], float]:
    """Return one VORP value per Basketball Reference player ID and season."""
    rows = []
    for path in sorted(SEASON_DIR.glob("*.csv")):
        try:
            season = int(path.stem)
        except ValueError:
            continue
        frame = pd.read_csv(path, low_memory=False)
        required = {"Player", "VORP"}
        if not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["match_name"] = frame["Player"].map(normalize_name)
        frame["nba_player_id"] = (
            frame["Player-additional"].astype("string")
            if "Player-additional" in frame.columns
            else frame["match_name"]
        )
        frame["VORP"] = pd.to_numeric(frame["VORP"], errors="coerce")
        if "Team" in frame.columns:
            # Basketball Reference includes team rows plus a combined 2TM/3TM
            # row for traded players.  Prefer the combined row.
            frame["combined_priority"] = (
                frame["Team"].astype(str).str.fullmatch(r"(?:TOT|\d+TM)")
            ).astype(int)
        else:
            frame["combined_priority"] = 0
        frame = (
            frame.sort_values("combined_priority", ascending=False)
            .drop_duplicates("nba_player_id")
        )
        frame["season_end"] = season
        rows.append(frame[["nba_player_id", "season_end", "VORP"]])

    if not rows:
        raise FileNotFoundError(f"No usable season CSV files found in {SEASON_DIR}")

    seasons = pd.concat(rows, ignore_index=True).dropna(subset=["VORP"])
    return seasons.set_index(["nba_player_id", "season_end"])["VORP"].to_dict()


def add_common_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create comparable early-career and peak-within-five-year outcomes."""
    vorp_lookup = load_season_vorp()
    out = df.copy()
    first_three = []
    best_three_of_first_five = []

    for row in out.itertuples(index=False):
        player_id = getattr(row, "nba_player_id", None)
        if pd.isna(player_id):
            player_id = normalize_name(row.player)
        player_id = str(player_id)
        draft_year = int(row.Draft_Year)
        values = np.array(
            [
                vorp_lookup.get((player_id, draft_year + offset), 0.0)
                for offset in range(1, 6)
            ],
            dtype=float,
        )
        first_three.append(values[:3].mean())
        windows = [values[start:start + 3].mean() for start in range(3)]
        best_three_of_first_five.append(max(windows))

    out["first_3yr_vorp"] = first_three
    out["best_3yr_vorp_first5"] = best_three_of_first_five
    return out


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found")
    df = add_primary_position(pd.read_csv(DATA_PATH, low_memory=False))
    df["height_in"] = df["ht"].map(height_to_inches)
    dob = pd.to_datetime(df["dob"], errors="coerce")
    draft_day = pd.to_datetime(
        df["Draft_Year"].astype("Int64").astype(str) + "-06-25",
        errors="coerce",
    )
    df["age_at_draft"] = (draft_day - dob).dt.days / 365.25
    df = add_common_horizon_targets(df)

    missing = [column for column in ALL_FEATURES if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required college columns: {missing}")
    if len(ALL_FEATURES) != len(set(ALL_FEATURES)):
        raise ValueError("Feature-family definitions overlap")
    return df


class FixedWeightedZScore(BaseEstimator, TransformerMixin):
    """Training-fold z-scores followed by a fixed weighted sum."""

    def __init__(self, weights: tuple[float, ...] = tuple(PCI_WEIGHTS)):
        self.weights = weights

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.medians_ = np.nanmedian(values, axis=0)
        filled = np.where(np.isnan(values), self.medians_, values)
        self.means_ = filled.mean(axis=0)
        self.scales_ = filled.std(axis=0, ddof=1)
        self.scales_[self.scales_ == 0] = 1.0
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        filled = np.where(np.isnan(values), self.medians_, values)
        z_scores = (filled - self.means_) / self.scales_
        return (z_scores @ np.asarray(self.weights)).reshape(-1, 1)


def pick_transformer() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=5, degree=2, include_bias=False)),
        ("scale", StandardScaler()),
    ])


def college_transformer() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])


def make_model(feature_columns: list[str], fixed_pci: bool = False) -> Pipeline:
    transformers = [("pick", pick_transformer(), ["pick"])]
    if fixed_pci:
        transformers.append(("fixed_pci", FixedWeightedZScore(), PCI_COLUMNS))
    elif feature_columns:
        transformers.append(("college", college_transformer(), feature_columns))
    return Pipeline([
        ("features", ColumnTransformer(transformers)),
        ("ridge", Ridge()),
    ])


def nested_logo_predictions(
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    fixed_pci: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """Leave-one-draft-year-out predictions with grouped inner tuning."""
    y = pd.to_numeric(df[target], errors="coerce").to_numpy()
    groups = df["Draft_Year"].astype(int).to_numpy()
    predictors = list(dict.fromkeys(["pick"] + feature_columns + (
        PCI_COLUMNS if fixed_pci else []
    )))
    predictions = np.full(len(df), np.nan)
    selected_alphas = []

    for train_index, test_index in LeaveOneGroupOut().split(df, y, groups):
        inner_groups = groups[train_index]
        n_splits = min(INNER_SPLITS, len(np.unique(inner_groups)))
        search = GridSearchCV(
            make_model(feature_columns, fixed_pci=fixed_pci),
            param_grid={"ridge__alpha": ALPHAS},
            scoring="neg_mean_absolute_error",
            cv=GroupKFold(n_splits=n_splits),
            n_jobs=1,
            refit=True,
        )
        search.fit(
            df.iloc[train_index][predictors],
            y[train_index],
            groups=inner_groups,
        )
        predictions[test_index] = search.predict(df.iloc[test_index][predictors])
        selected_alphas.append(float(search.best_params_["ridge__alpha"]))

    return predictions, selected_alphas


def model_metrics(y: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "n": len(y),
        "mae": mean_absolute_error(y, predictions),
        "rmse": mean_squared_error(y, predictions) ** 0.5,
        "r2": r2_score(y, predictions),
        "pearson_r": np.corrcoef(y, predictions)[0, 1],
    }


def exact_sign_flip_pvalue(year_differences: np.ndarray) -> float:
    """One-sided exact test: is the candidate's MAE smaller?"""
    observed = year_differences.mean()
    n_years = len(year_differences)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n_years)))
    null_means = signs @ year_differences / n_years
    return float((np.sum(null_means >= observed) + 1) / (len(null_means) + 1))


def cluster_bootstrap_interval(
    year_differences: np.ndarray,
    repetitions: int = 20_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    samples = rng.choice(
        year_differences,
        size=(repetitions, len(year_differences)),
        replace=True,
    ).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def paired_comparison(
    df: pd.DataFrame,
    target: str,
    reference_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> dict[str, float]:
    y = df[target].to_numpy(dtype=float)
    groups = df["Draft_Year"].astype(int).to_numpy()
    years = np.unique(groups)
    year_differences = np.array([
        np.mean(np.abs(y[groups == year] - reference_predictions[groups == year]))
        - np.mean(np.abs(y[groups == year] - candidate_predictions[groups == year]))
        for year in years
    ])
    ci_low, ci_high = cluster_bootstrap_interval(year_differences)
    return {
        "delta_mae_pooled": (
            mean_absolute_error(y, reference_predictions)
            - mean_absolute_error(y, candidate_predictions)
        ),
        "delta_rmse_pooled": (
            mean_squared_error(y, reference_predictions) ** 0.5
            - mean_squared_error(y, candidate_predictions) ** 0.5
        ),
        "mean_year_delta_mae": year_differences.mean(),
        "year_delta_mae_ci_low": ci_low,
        "year_delta_mae_ci_high": ci_high,
        "years_improved": int(np.sum(year_differences > 0)),
        "years_tested": len(years),
        "p_value": exact_sign_flip_pvalue(year_differences),
    }


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def evaluate_target(
    source: pd.DataFrame,
    target: str,
    maximum_draft_year: int | None,
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    df = source[source["bpm"].notna()].copy()
    if maximum_draft_year is not None:
        df = df[df["Draft_Year"] <= maximum_draft_year].copy()
    df = df.dropna(subset=[target, "pick"]).reset_index(drop=True)

    model_specs = {
        "pick_only": ([], False),
        "pick_plus_fixed_pci": ([], True),
        **{
            f"pick_plus_{family}": (columns, False)
            for family, columns in FEATURE_FAMILIES.items()
        },
        "pick_plus_all": (ALL_FEATURES, False),
        **{
            f"full_minus_{family}": (
                [column for column in ALL_FEATURES if column not in columns],
                False,
            )
            for family, columns in FEATURE_FAMILIES.items()
        },
    }

    predictions = {}
    metrics_rows = []
    print(f"\n{target}: n={len(df)}, draft years={df.Draft_Year.nunique()}")
    for model_name, (columns, fixed_pci) in model_specs.items():
        pred, alphas = nested_logo_predictions(
            df, target, columns, fixed_pci=fixed_pci
        )
        predictions[model_name] = pred
        row = {"target": target, "model": model_name}
        row.update(model_metrics(df[target].to_numpy(dtype=float), pred))
        row["median_selected_alpha"] = float(np.median(alphas))
        metrics_rows.append(row)
        print(
            f"  {model_name:<32} MAE={row['mae']:.4f}  "
            f"RMSE={row['rmse']:.4f}  r={row['pearson_r']:.3f}"
        )

    comparisons = []
    add_candidates = ["pick_plus_fixed_pci"] + [
        f"pick_plus_{family}" for family in FEATURE_FAMILIES
    ] + ["pick_plus_all"]
    for candidate in add_candidates:
        row = {
            "target": target,
            "comparison": "adds_beyond_pick",
            "reference_model": "pick_only",
            "candidate_model": candidate,
        }
        row.update(paired_comparison(
            df, target, predictions["pick_only"], predictions[candidate]
        ))
        comparisons.append(row)

    for family in FEATURE_FAMILIES:
        reference = f"full_minus_{family}"
        row = {
            "target": target,
            "comparison": "unique_in_full_model",
            "reference_model": reference,
            "candidate_model": "pick_plus_all",
        }
        row.update(paired_comparison(
            df, target, predictions[reference], predictions["pick_plus_all"]
        ))
        comparisons.append(row)

    prediction_frame = df[
        ["player", "Draft_Year", "pick", "primary_position", target]
    ].copy()
    prediction_frame.insert(0, "target", target)
    prediction_frame = prediction_frame.rename(columns={target: "actual"})
    for model_name, values in predictions.items():
        prediction_frame[model_name] = values
    return metrics_rows, comparisons, prediction_frame


def position_oof_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the pooled leakage-safe OOF predictions within each position."""
    identifier_columns = {
        "target", "player", "Draft_Year", "pick", "primary_position", "actual"
    }
    model_columns = [
        column for column in predictions.columns if column not in identifier_columns
    ]
    rows = []
    for target, target_frame in predictions.groupby("target"):
        for position in POSITIONS:
            group = target_frame[
                target_frame["primary_position"].astype(str).eq(position)
            ]
            for model in model_columns:
                usable = group[["actual", model]].dropna()
                if len(usable) < 3:
                    continue
                row = {
                    "target": target,
                    "primary_position": position,
                    "model": model,
                }
                row.update(model_metrics(
                    usable["actual"].to_numpy(dtype=float),
                    usable[model].to_numpy(dtype=float),
                ))
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = load_data()
    all_metrics = []
    all_comparisons = []
    all_predictions = []

    evaluations = [
        ("first_3yr_vorp", None),
        ("best_3yr_vorp_first5", FIRST_FIVE_MAX_DRAFT_YEAR),
    ]
    for target, maximum_year in evaluations:
        metrics, comparisons, predictions = evaluate_target(
            df, target, maximum_year
        )
        all_metrics.extend(metrics)
        all_comparisons.extend(comparisons)
        all_predictions.append(predictions)

    metrics_df = pd.DataFrame(all_metrics)
    comparisons_df = pd.DataFrame(all_comparisons)
    comparisons_df["q_value"] = np.nan
    for (_, comparison), index in comparisons_df.groupby(
        ["target", "comparison"]
    ).groups.items():
        comparisons_df.loc[index, "q_value"] = benjamini_hochberg(
            comparisons_df.loc[index, "p_value"]
        )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)
    comparisons_df.to_csv(COMPARISONS_PATH, index=False)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    position_oof_metrics(predictions_df).to_csv(POSITION_METRICS_PATH, index=False)

    print("\nPaired draft-year comparisons (positive delta = improvement):")
    columns = [
        "target", "comparison", "candidate_model", "delta_mae_pooled",
        "delta_rmse_pooled", "mean_year_delta_mae", "year_delta_mae_ci_low",
        "year_delta_mae_ci_high", "years_improved", "years_tested",
        "p_value", "q_value",
    ]
    print(comparisons_df[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {METRICS_PATH}")
    print(f"Saved: {COMPARISONS_PATH}")
    print(f"Saved: {PREDICTIONS_PATH}")
    print(f"Saved: {POSITION_METRICS_PATH}")


if __name__ == "__main__":
    main()
