"""Shared, auditable PG/SG/SF/PF/C assignment utilities.

Position precedence
-------------------
1. The player's first recorded NBA-season ``Pos`` value after his draft year.
2. A regularized classifier estimate from college height, role, and statistics.
3. Barttorvik's pre-draft college role when too few statistics are available.

The first source is the clearest conventional primary position, while the
fallbacks retain drafted players who never appeared in an NBA season file.
Rows without enough evidence remain unassigned rather than receiving a
fabricated label.  ``position_source`` and ``position_confidence`` make the
provenance explicit in every downstream output.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


POSITIONS = ("PG", "SG", "SF", "PF", "C")

POSITION_COLORS = {
    "PG": "#2563eb",
    "SG": "#7c3aed",
    "SF": "#db2777",
    "PF": "#ea580c",
    "C": "#16a34a",
}

POSITION_MARKERS = {
    "PG": "o",
    "SG": "s",
    "SF": "D",
    "PF": "^",
    "C": "P",
}

ROLE_POSITION_MAP = {
    "Pure PG": "PG",
    "Scoring PG": "PG",
    "Combo G": "SG",
    "Wing G": "SG",
    "Wing F": "SF",
    "Stretch 4": "PF",
    "C": "C",
}

# Only pre-NBA college variables are used for the final statistical fallback.
INFERENCE_FEATURES = [
    "height_in", "AST_per", "ORB_per", "DRB_per", "blk_per", "stl_per",
    "usg", "ast/tov", "oreb", "dreb", "ast", "stl", "blk",
]


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z\s]", "", str(value).lower().strip())


def height_to_inches(value: object) -> float:
    try:
        feet, inches = str(value).split("-")
        return float(int(feet) * 12 + int(inches))
    except (TypeError, ValueError):
        return np.nan


def _player_column(df: pd.DataFrame) -> str:
    for candidate in ("player", "Player", "player_name"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No player-name column found")


def _draft_year_column(df: pd.DataFrame) -> str:
    for candidate in ("Draft_Year", "draft_year", "year"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No draft-year column found")


def load_nba_position_history(
    season_dir: Path = ROOT / "data/nba_seasons",
) -> pd.DataFrame:
    """Load one player-id/season/position record from local NBA files."""
    frames = []
    for path in sorted(season_dir.glob("*.csv")):
        try:
            season_end = int(path.stem)
        except ValueError:
            continue
        frame = pd.read_csv(path, low_memory=False)
        if not {"Player", "Pos"}.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["match_name"] = frame["Player"].map(normalize_name)
        frame["season_end"] = season_end
        frame["games_numeric"] = pd.to_numeric(frame.get("G"), errors="coerce")
        frame["minutes_numeric"] = pd.to_numeric(frame.get("MP"), errors="coerce")
        frame["position"] = frame["Pos"].astype("string").str.extract(
            r"\b(PG|SG|SF|PF|C)\b", expand=False
        )
        frame["player_id"] = (
            frame["Player-additional"].astype("string")
            if "Player-additional" in frame
            else frame["match_name"]
        )
        if "Team" in frame:
            team = frame["Team"]
        elif "Tm" in frame:
            team = frame["Tm"]
        else:
            team = pd.Series("", index=frame.index)
        frame["combined_priority"] = team.astype(str).str.fullmatch(
            r"(?:TOT|\d+TM)"
        ).astype(int)
        frame = frame[frame["games_numeric"].gt(0)].copy()
        frame = (
            frame.sort_values(
                [
                    "match_name", "player_id", "season_end",
                    "combined_priority", "minutes_numeric",
                ],
                ascending=[True, True, True, False, False],
            )
            .drop_duplicates(["match_name", "player_id", "season_end"])
        )
        frames.append(
            frame[
                [
                    "match_name", "player_id", "season_end", "position",
                    "minutes_numeric",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "match_name", "player_id", "season_end", "position",
                "minutes_numeric",
            ]
        )
    return pd.concat(frames, ignore_index=True).dropna(subset=["position"])


def rookie_position_lookup(
    df: pd.DataFrame,
    season_dir: Path = ROOT / "data/nba_seasons",
) -> pd.DataFrame:
    """Return the first observed NBA position after each player's draft year."""
    player_column = _player_column(df)
    year_column = _draft_year_column(df)
    history = load_nba_position_history(season_dir)
    keys = df[[player_column, year_column]].copy()
    keys["match_name"] = keys[player_column].map(normalize_name)
    keys["draft_year_numeric"] = pd.to_numeric(keys[year_column], errors="coerce")
    keys["_position_row_id"] = np.arange(len(keys))

    if history.empty:
        result = pd.DataFrame(index=np.arange(len(keys)))
        result["rookie_position"] = pd.NA
        result["position_observed_season"] = np.nan
        return result.reset_index(drop=True)

    history = history.copy()
    history["career_debut"] = history.groupby(
        ["match_name", "player_id"]
    )["season_end"].transform("min")
    history["career_minutes"] = history.groupby(
        ["match_name", "player_id"]
    )["minutes_numeric"].transform("sum")
    candidates = keys.merge(history, on="match_name", how="left")
    # A matching NBA career must begin after this draft year.  This prevents
    # a later same-name draftee from inheriting an older player's position.
    candidates = candidates[
        (candidates["career_debut"] > candidates["draft_year_numeric"])
        & (candidates["season_end"] > candidates["draft_year_numeric"])
    ].copy()
    first = (
        candidates.sort_values(
            [
                "_position_row_id", "career_debut", "career_minutes",
                "season_end",
            ],
            ascending=[True, True, False, True],
        )
        .drop_duplicates("_position_row_id")
        .set_index("_position_row_id")
    )
    result = pd.DataFrame(index=np.arange(len(keys)))
    result["rookie_position"] = first["position"]
    result["position_observed_season"] = first["season_end"]
    return result.reset_index(drop=True)


def college_role_position(role: object, height_in: float = np.nan) -> str | None:
    """Map Barttorvik roles to a conventional primary position."""
    if pd.isna(role):
        return None
    role_text = str(role).strip()
    if role_text in ROLE_POSITION_MAP:
        return ROLE_POSITION_MAP[role_text]
    if role_text == "PF/C":
        # Barttorvik intentionally combines these roles.  Height provides a
        # reproducible primary-position split for players without NBA records.
        return "C" if pd.notna(height_in) and height_in >= 82 else "PF"
    return None


def _add_height(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "height_in" not in out.columns:
        if "ht" in out.columns:
            out["height_in"] = out["ht"].map(height_to_inches)
        else:
            out["height_in"] = np.nan
    else:
        out["height_in"] = pd.to_numeric(out["height_in"], errors="coerce")
        if "ht" in out.columns:
            out["height_in"] = out["height_in"].fillna(
                out["ht"].map(height_to_inches)
            )
    return out


def add_primary_position(
    df: pd.DataFrame,
    season_dir: Path = ROOT / "data/nba_seasons",
    allow_statistical_fallback: bool = True,
) -> pd.DataFrame:
    """Add primary position, source, confidence, and observed-season columns."""
    out = _add_height(df)
    rookie = rookie_position_lookup(out, season_dir)
    out["nba_entry_position"] = rookie["rookie_position"].to_numpy()
    out["primary_position"] = out["nba_entry_position"]
    out["position_observed_season"] = rookie[
        "position_observed_season"
    ].to_numpy()
    out["position_source"] = np.where(
        out["primary_position"].notna(), "nba_rookie_season", pd.NA
    )
    out["position_confidence"] = np.where(
        out["primary_position"].notna(), 1.0, np.nan
    )

    if allow_statistical_fallback:
        available = [feature for feature in INFERENCE_FEATURES if feature in out]
        train = out["primary_position"].isin(POSITIONS)
        has_role = (
            out["role"].notna()
            if "role" in out
            else pd.Series(False, index=out.index)
        )
        predict = (
            out["primary_position"].isna()
            & (out[available].notna().sum(axis=1).ge(3) | has_role)
        )
        if len(available) >= 3 and train.sum() >= 50 and predict.any():
            model_frame = out[available].apply(pd.to_numeric, errors="coerce")
            transformers = [(
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler(),
                ),
                available,
            )]
            if "role" in out:
                model_frame["role"] = (
                    out["role"].astype(object).where(out["role"].notna(), np.nan)
                )
                transformers.append((
                    "role",
                    make_pipeline(
                        SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                    ["role"],
                ))
            classifier = make_pipeline(
                ColumnTransformer(transformers),
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            )
            classifier.fit(
                model_frame.loc[train], out.loc[train, "primary_position"]
            )
            predicted = classifier.predict(model_frame.loc[predict])
            probabilities = classifier.predict_proba(
                model_frame.loc[predict]
            ).max(axis=1)
            out.loc[predict, "primary_position"] = predicted
            out.loc[predict, "position_source"] = "college_position_model"
            out.loc[predict, "position_confidence"] = probabilities

    # Retain a role-only fallback for future rows with too little numeric data
    # for the classifier.  Its confidence is empirical within this dataset.
    if "role" in out.columns:
        mapped_roles = pd.Series(
            [
                college_role_position(role, height)
                for role, height in zip(out["role"], out["height_in"])
            ],
            index=out.index,
            dtype="string",
        )
        observed = (
            out["position_source"].eq("nba_rookie_season")
            & mapped_roles.notna()
        )
        accuracy_by_role = (
            pd.DataFrame({
                "role": out.loc[observed, "role"],
                "correct": (
                    mapped_roles.loc[observed].astype(str)
                    == out.loc[observed, "primary_position"].astype(str)
                ),
            })
            .groupby("role")["correct"]
            .mean()
            .to_dict()
        )
        role_fallback = out["primary_position"].isna() & mapped_roles.notna()
        out.loc[role_fallback, "primary_position"] = mapped_roles[role_fallback]
        out.loc[role_fallback, "position_source"] = "college_role"
        out.loc[role_fallback, "position_confidence"] = (
            out.loc[role_fallback, "role"]
            .map(accuracy_by_role)
            .fillna(0.50)
        )

    out["primary_position"] = pd.Categorical(
        out["primary_position"], categories=POSITIONS, ordered=True
    )
    return out


def position_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Compact assignment audit by position and provenance."""
    if "primary_position" not in df:
        raise KeyError("primary_position not found; call add_primary_position first")
    position = df["primary_position"].astype("string").fillna("Unassigned")
    source = df["position_source"].astype("string").fillna("unassigned")
    return (
        pd.DataFrame({"primary_position": position, "position_source": source})
        .value_counts(sort=False)
        .rename("players")
        .reset_index()
        .sort_values(["primary_position", "position_source"])
        .reset_index(drop=True)
    )
