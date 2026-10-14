"""
NBA Draft Surplus Value Project — Step 2: College Stats from Barttorvik
=======================================================================

This script pulls college stats for each drafted player from Barttorvik,
focusing on the POSSESSION-BASED features that form your unique hypothesis.

Prerequisites:
    - Run 01_collect_draft_data.py first (needs data/nba_draft_clean.csv)

Setup:
    pip install pandas requests

Usage:
    python 02_collect_college_stats.py

Output:
    - data/college_stats.csv           (raw college stats)
    - data/draft_with_college.csv      (merged: draft data + college stats)

NOTE ON BARTTORVIK:
    Barttorvik (barttorvik.com) is the best source for college basketball
    advanced stats. It provides per-possession and pace-adjusted numbers.
    However, it doesn't have a formal API, so we'll use their data export
    URL patterns. If Barttorvik's structure changes, you may need to adjust.
    
    Alternative: you can manually download season CSVs from Barttorvik
    and place them in data/barttorvik/ — the script will detect and use them.

Author: [Your name]
Project: NBA Draft Surplus Value Model
"""

import pandas as pd
import requests
import time
import os
from io import StringIO

from position_utils import add_primary_position

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
WAIT_SECONDS = 10
BARTTORVIK_START_YEAR = 2009
os.makedirs("data/barttorvik", exist_ok=True)
os.makedirs("data/barttorvik_cache", exist_ok=True)

BARTTORVIK_COLUMNS = [
    "player_name", "team", "conf", "GP", "Min_per", "ORtg", "usg",
    "eFG", "TS_per", "ORB_per", "DRB_per", "AST_per", "TO_per",
    "FTM", "FTA", "FT_per", "twoPM", "twoPA", "twoP_per", "TPM",
    "TPA", "TP_per", "blk_per", "stl_per", "ftr", "yr", "ht", "num",
    "porpag", "adjoe", "pfr", "year", "pid", "type", "Rec Rank",
    "ast/tov", "rimmade", "rimmade+rimmiss", "midmade",
    "midmade+midmiss", "rimmade/(rimmade+rimmiss)",
    "midmade/(midmade+midmiss)", "dunksmade", "dunksmiss+dunksmade",
    "dunksmade/(dunksmade+dunksmiss)", "pick", "drtg", "adrtg",
    "dporpag", "stops", "bpm", "obpm", "dbpm", "gbpm", "mp", "ogbpm",
    "dgbpm", "oreb", "dreb", "treb", "ast", "stl", "blk", "pts",
    "role", "3p/100?", "dob",
]

# Reference moments used by the Granular Prospect Index (GPI).  These were
# calculated from the 682 matched 2009-2023 prospects.  Keeping the reference
# fixed makes a future prospect's score comparable with the historical sample.
# No NBA statistics, draft position, BPM-family statistic, VORP, or PORPAG is
# used in this index.
GRANULAR_INDEX_REFERENCE = {
    "oreb": (1.656823, 0.983401),
    "ast/tov": (1.175335, 0.639134),
    "ftr": (40.275073, 14.079196),
    "stl": (1.128416, 0.504881),
    "FT_per": (0.741587, 0.090401),
    "AST_per": (16.014663, 9.168372),
    "TS_per": (57.905337, 4.692281),
    "blk_per": (3.265396, 3.150605),
    "dunksmade/(dunksmade+dunksmiss)": (0.906408, 0.113341),
    "dunksmade": (22.743307, 19.177408),
    "rimmade/(rimmade+rimmiss)": (0.661243, 0.073116),
    "TP_per": (0.319777, 0.137128),
    "twoPA": (254.703812, 87.341681),
}


class BarttorvikBlockedError(RuntimeError):
    """Raised when Barttorvik refuses a request with HTTP 403."""


class NetworkUnavailableError(RuntimeError):
    """Raised when Barttorvik cannot be reached or resolved."""


def fetch_barttorvik_season(season_year):
    """
    Fetch player stats for a college basketball season from Barttorvik.
    
    season_year: the year the season ENDS in (e.g., 2023 for 2022-23 season)
    
    Barttorvik provides data via a URL pattern. We request the full player
    stats table for that season.
    
    Returns a DataFrame or None if request fails.
    """
    url = f"https://barttorvik.com/getadvstats.php?year={season_year}&csv=1"
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/134.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 403:
            raise BarttorvikBlockedError(f"Barttorvik returned 403 for {season_year}.")
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text), header=None, names=BARTTORVIK_COLUMNS)
        if df.empty:
            print(f"  ❌ Empty CSV returned for {season_year}")
            return None

        returned_years = set(pd.to_numeric(df["year"], errors="coerce").dropna().astype(int))
        if returned_years != {season_year}:
            print(f"  ❌ Barttorvik returned {sorted(returned_years)} instead of {season_year}")
            return None

        df["college_season"] = season_year
        return df

    except BarttorvikBlockedError:
        raise
    except requests.ConnectionError as e:
        raise NetworkUnavailableError(str(e)) from e
    except Exception as e:
        print(f"  ❌ Failed to fetch Barttorvik {season_year}: {e}")
        return None


def load_manual_barttorvik():
    """
    If the automatic scraping doesn't work (Barttorvik may block),
    you can manually download CSVs from barttorvik.com and put them
    in data/barttorvik/. This function loads those files.
    
    To download manually:
    1. Go to barttorvik.com → Player Stats
    2. Select the season
    3. Copy/paste the table into a spreadsheet
    4. Save as CSV in data/barttorvik/season_YYYY.csv
    """
    files = [f for f in os.listdir("data/barttorvik") if f.endswith(".csv")]
    
    if not files:
        return None
    
    dfs = []
    for f in files:
        df = pd.read_csv(f"data/barttorvik/{f}")
        # Extract year from filename
        year = int(f.replace("season_", "").replace(".csv", ""))
        df["college_season"] = year
        dfs.append(df)
        print(f"  📂 Loaded {f} ({len(df)} players)")
    
    return pd.concat(dfs, ignore_index=True)


def match_players(draft_df, college_df):
    """
    Match drafted players to their college stats.
    
    This is trickier than it sounds because:
    - Names may be spelled differently across sources
    - Some players transfer and have stats at multiple schools
    - International players may not have college stats
    
    Strategy: match on player name (fuzzy) + college name.
    We'll start with exact matching and flag unmatched players
    for manual review.
    """
    # Normalise names for matching (lowercase, strip whitespace)
    draft_df = draft_df.copy()
    college_df = college_df.copy()
    draft_df["_draft_row_id"] = range(len(draft_df))
    
    # Create a normalised name column for matching
    if "player" in draft_df.columns:
        draft_df["match_name"] = (
            draft_df["player"]
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"[^a-z\s]", "", regex=True)  # Remove special chars
        )
    
    # Try to find the player name column in college data
    # Barttorvik uses "Player" or "Name" typically
    name_col = None
    for col in college_df.columns:
        if col.lower() in ["player", "name", "player_name"]:
            name_col = col
            break
    
    if name_col is None:
        print("  ⚠️  Could not find player name column in college data")
        print(f"  Available columns: {list(college_df.columns)}")
        return draft_df
    
    college_df["match_name"] = (
        college_df[name_col]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"[^a-z\s]", "", regex=True)
    )
    draft_df["_college_key"] = (
        draft_df.get("college", pd.Series(index=draft_df.index, dtype=str))
        .fillna("").astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)
    )
    college_df["_college_key"] = (
        college_df.get("team", pd.Series(index=college_df.index, dtype=str))
        .fillna("").astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)
    )
    
    # Match the player's final pre-draft season, not every college season in
    # which the same name appears.
    merged = draft_df.merge(
        college_df,
        left_on=["match_name", "Draft_Year"],
        right_on=["match_name", "college_season"],
        how="left",
        suffixes=("", "_college")
    )
    merged["_college_match"] = (
        (merged["_college_key"] != "")
        & (merged["_college_key"] == merged["_college_key_college"])
    )
    merged["_minutes_sort"] = pd.to_numeric(merged.get("Min_per"), errors="coerce").fillna(-1)
    merged = (
        merged.sort_values(
            ["_draft_row_id", "_college_match", "_minutes_sort"],
            ascending=[True, False, False],
        )
        .drop_duplicates("_draft_row_id")
        .sort_values("_draft_row_id")
        .drop(columns=[
            "_draft_row_id", "_college_key", "_college_key_college",
            "_college_match", "_minutes_sort",
        ])
    )
    
    # Count matches
    n_matched = merged[name_col].notna().sum() if name_col in merged.columns else 0
    n_total = len(draft_df)
    print(f"\n  Matched {n_matched}/{n_total} players ({n_matched/n_total*100:.1f}%)")
    
    if n_matched < n_total * 0.7:
        print("  ⚠️  Low match rate. You may need to review unmatched players.")
        unmatched = merged[merged[name_col].isna()]["player"].tolist()[:10]
        print(f"  Examples of unmatched: {unmatched}")
    
    return merged


def create_granular_prospect_index(df):
    """Create a fitted index exclusively from granular college statistics.

    The formula combines offensive rebounding, assist-to-turnover ratio,
    free-throw rate, steals, shooting, playmaking, and finishing/defensive
    interactions.  Its weights were fit on the current 2009-2023 matched
    sample, so its correlation with ``surplus_value`` is descriptive and must
    not be treated as an out-of-sample model score.

    Shot-location fields are not available for every matched season.  Missing
    values for those fields receive the historical mean (a z-score of zero).
    Rows without the core Barttorvik statistics remain missing rather than
    receiving a fabricated average score.
    """
    df = df.copy()
    missing = [
        column for column in GRANULAR_INDEX_REFERENCE
        if column not in df.columns
    ]
    if missing:
        print(f"  ⚠️ Could not create Granular Prospect Index; missing: {missing}")
        return df

    for column in GRANULAR_INDEX_REFERENCE:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    core_columns = [
        "oreb", "ast/tov", "ftr", "stl", "FT_per", "AST_per",
        "TS_per", "blk_per", "TP_per", "twoPA",
    ]
    has_college_data = df[core_columns].notna().all(axis=1)

    z = {}
    for column, (mean, std) in GRANULAR_INDEX_REFERENCE.items():
        # Mean imputation is only used inside the formula.  The final mask below
        # prevents unmatched prospects from being assigned an index value.
        z[column] = (df[column].fillna(mean) - mean) / std

    positive = lambda values: values.clip(lower=0)
    indicator = lambda condition: condition.astype(float)

    raw_index = (
        -0.082033
        + 0.202343 * z["oreb"]
        + 0.268865 * z["ast/tov"]
        + 0.076392 * positive(z["ftr"])
        + 0.370708 * indicator(z["stl"] > 0.5)
        + 0.658424 * indicator(z["FT_per"] > 1.5)
        + 0.134471 * z["AST_per"] * z["FT_per"]
        - 0.636863 * indicator(z["ast/tov"] > 1.5)
        + 0.133875 * z["TS_per"] * z["AST_per"]
        + 0.201869
        * z["blk_per"]
        * z["dunksmade/(dunksmade+dunksmiss)"]
        - 0.344106 * indicator(z["dunksmade"] > 0)
        + 0.176337
        * positive(z["rimmade/(rimmade+rimmiss)"] + 0.5)
        + 0.167776
        * z["TP_per"]
        * z["dunksmade/(dunksmade+dunksmiss)"]
        - 0.400862 * indicator(z["twoPA"] > 1.5)
    )
    df["granular_prospect_index"] = raw_index.where(has_college_data)
    print(
        "  ✅ Created Granular Prospect Index from 13 granular college "
        f"statistics for {int(has_college_data.sum())} players"
    )
    return df


def create_possession_features(df):
    """
    Create the Possession Creation Index and related features.
    
    These are the features that encode YOUR theory of basketball value:
    players who create extra possessions (steals, offensive rebounds)
    and preserve possessions (low turnovers) are undervalued.
    
    The specific column names depend on what Barttorvik provides.
    Common Barttorvik column names:
    - stl%  or STL%  (steal rate)
    - tov%  or TOV%  (turnover rate) 
    - orb%  or ORB%  (offensive rebound rate)
    - drb%  or DRB%  (defensive rebound rate)
    - ast%  or AST%  (assist rate)
    - usg%  or USG%  (usage rate)
    - ts%   or TS%   (true shooting percentage)
    - bpm   or BPM   (box plus/minus)
    """
    df = df.copy()
    
    # -----------------------------------------------------------------------
    # Find the possession-based columns
    # (column names vary — we search case-insensitively)
    # -----------------------------------------------------------------------
    col_map = {}
    target_stats = {
        "stl": ["stl%", "stl_pct", "stl_per", "steal_rate", "stl_rate"],
        "tov": ["tov%", "tov_pct", "to_per", "turnover_rate", "to%", "to_rate"],
        "orb": ["orb%", "orb_pct", "orb_per", "off_reb_rate", "oreb%"],
        "drb": ["drb%", "drb_pct", "drb_per", "def_reb_rate", "dreb%"],
        "ast": ["ast%", "ast_pct", "ast_per", "assist_rate"],
        "usg": ["usg%", "usg_pct", "usg", "usage_rate", "usage%"],
        "ts":  ["ts%", "ts_pct", "ts_per", "true_shooting"],
    }
    
    available_cols_lower = {c.lower().strip(): c for c in df.columns}
    
    for stat_key, possible_names in target_stats.items():
        for name in possible_names:
            if name.lower() in available_cols_lower:
                col_map[stat_key] = available_cols_lower[name.lower()]
                break
    
    print(f"\n  Found college stat columns: {col_map}")
    
    # -----------------------------------------------------------------------
    # Convert found columns to numeric
    # -----------------------------------------------------------------------
    for stat_key, col_name in col_map.items():
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    
    # -----------------------------------------------------------------------
    # Create the Possession Creation Index (PCI)
    # 
    # YOUR signature metric. This encodes your theory:
    #   - High steal rate = creating possessions via defense
    #   - High offensive rebound rate = creating second chances
    #   - LOW turnover rate = preserving possessions
    #   - High assist rate = converting possessions into quality shots
    #
    # We normalise each component to z-scores (mean=0, std=1)
    # so they're on the same scale, then combine.
    #
    # The weights below are a starting point. You can experiment with
    # different weights as you develop your intuition about what matters.
    # -----------------------------------------------------------------------
    
    components = {}
    weights = {
        "stl": 1.0,    # Steal rate (positive: more steals = more possessions)
        "orb": 0.8,    # Off rebound rate (positive: more second chances)
        "tov": -1.0,   # Turnover rate (NEGATIVE: high TO% is bad)
        "ast": 0.6,    # Assist rate (positive: converting possessions)
    }
    
    for stat_key, weight in weights.items():
        if stat_key in col_map:
            col = col_map[stat_key]
            # Z-score normalisation
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                z_score = (df[col] - mean) / std
                components[stat_key] = z_score * weight
    
    if components:
        # Sum the weighted z-scores
        df["possession_creation_index"] = sum(components.values())
        print(f"  ✅ Created Possession Creation Index from: {list(components.keys())}")
    else:
        print("  ⚠️  Could not create PCI — no possession stat columns found")
        print("  You may need to manually map column names in create_possession_features()")

    # -----------------------------------------------------------------------
    # Create a threshold-based scoring profile (0-6).
    # One point is awarded for each benchmark met; the elite flag requires
    # all six benchmarks. Barttorvik stores FT% as a decimal (0.75 = 75%).
    # -----------------------------------------------------------------------
    scoring_columns = {
        "ts_above_55": "TS_per",
        "usage_above_20": "usg",
        "ft_at_least_75": "FT_per",
        "ftr_above_25": "ftr",
        "bpm_above_1": "bpm_college" if "bpm_college" in df.columns else "bpm",
        "three_per_100_above_5": "3p/100?",
    }
    missing_scoring_columns = [
        column for column in scoring_columns.values() if column not in df.columns
    ]
    if missing_scoring_columns:
        print(f"  ⚠️ Could not create scoring profile; missing: {missing_scoring_columns}")
    else:
        for column in set(scoring_columns.values()):
            df[column] = pd.to_numeric(df[column], errors="coerce")

        criteria = pd.DataFrame({
            "ts_above_55": df["TS_per"] > 55,
            "usage_above_20": df["usg"] > 20,
            "ft_at_least_75": df["FT_per"] >= 0.75,
            "ftr_above_25": df["ftr"] > 25,
            "bpm_above_1": df[scoring_columns["bpm_above_1"]] > 1,
            "three_per_100_above_5": df["3p/100?"] > 5,
        }, index=df.index)
        complete = df[list(dict.fromkeys(scoring_columns.values()))].notna().all(axis=1)
        df["scoring_profile_score"] = criteria.sum(axis=1).where(complete)
        df["elite_scoring_profile"] = criteria.all(axis=1).where(complete)
        for criterion in criteria.columns:
            df[criterion] = criteria[criterion].where(complete)

        elite_count = int(df["elite_scoring_profile"].fillna(False).sum())
        print(f"  ✅ Created 0-6 scoring profile; {elite_count} players meet all six thresholds")
    
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🎓 NBA Draft Surplus Value — College Stats Collection")
    print("-" * 50)
    
    # Load the draft data from Step 1
    if not os.path.exists("data/nba_draft_clean.csv"):
        print("❌ Error: data/nba_draft_clean.csv not found")
        print("   Run 01_collect_draft_data.py first!")
        exit(1)
    
    draft_df = pd.read_csv("data/nba_draft_clean.csv")
    print(f"📂 Loaded draft data: {len(draft_df)} players")
    
    # -----------------------------------------------------------------------
    # Collect college stats
    # First try automatic scraping; fall back to manual CSVs
    # -----------------------------------------------------------------------
    print("\nAttempting to fetch college stats from Barttorvik...")
    
    # We need college stats for the seasons corresponding to each draft
    # Draft year 2020 → college season 2019-20 → Barttorvik year = 2020
    draft_years = sorted(
        int(year) for year in draft_df["Draft_Year"].unique()
        if BARTTORVIK_START_YEAR <= int(year) <= 2023
    )
    
    college_dfs = []
    missing_years = []
    for year in draft_years:
        cache_path = f"data/barttorvik_cache/{year}.csv"
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            college_dfs.append(df)
            print(f"  Using saved {year} season ({len(df)} players)")
            continue

        print(f"  Fetching {year}...")
        try:
            df = fetch_barttorvik_season(year)
        except NetworkUnavailableError as e:
            print(f"  ❌ Network/DNS error: {e}")
            raise SystemExit(1)
        except BarttorvikBlockedError as e:
            print(f"  ❌ {e}")
            missing_years.append(year)
            time.sleep(WAIT_SECONDS)
            continue
        if df is not None:
            df.to_csv(cache_path, index=False)
            college_dfs.append(df)
            print(f"  ✅ Got and saved {len(df)} players")
        else:
            missing_years.append(year)
        time.sleep(WAIT_SECONDS)

    if missing_years:
        print(f"\n⚠️ Missing seasons: {', '.join(map(str, missing_years))}")
        print("Successful seasons were cached. Run again later to retry only missing seasons.")
        raise SystemExit(1)
    
    if college_dfs:
        college_df = pd.concat(college_dfs, ignore_index=True)
        print(f"\n✅ Collected college stats: {len(college_df)} player-seasons")
    else:
        print("\n⚠️  Automatic fetch failed. Checking for manual CSVs...")
        college_df = load_manual_barttorvik()
        
        if college_df is None:
            print("\n❌ No college data available.")
            print("   To continue, manually download season data from barttorvik.com")
            print("   and save CSVs to data/barttorvik/season_YYYY.csv")
            print("\n   Alternatively, you can proceed to Step 3 using only")
            print("   Basketball Reference stats (PPG, RPG, etc.) from the draft pages.")
            exit(0)
    
    # Save raw college stats
    college_df.to_csv("data/college_stats.csv", index=False)
    print(f"💾 Saved: data/college_stats.csv")
    
    # -----------------------------------------------------------------------
    # Match college stats to drafted players
    # -----------------------------------------------------------------------
    print("\nMatching players between draft data and college stats...")
    merged_df = match_players(draft_df, college_df)
    
    # -----------------------------------------------------------------------
    # Create possession-based features
    # -----------------------------------------------------------------------
    print("\nCreating possession-based features...")
    merged_df = create_possession_features(merged_df)
    merged_df = create_granular_prospect_index(merged_df)
    merged_df = add_primary_position(merged_df)
    
    # Save the merged dataset
    merged_df.to_csv("data/draft_with_college.csv", index=False)
    print(f"\n💾 Saved: data/draft_with_college.csv ({len(merged_df)} rows)")
    
    # -----------------------------------------------------------------------
    # Quick preview: PCI vs surplus value
    # -----------------------------------------------------------------------
    if "possession_creation_index" in merged_df.columns:
        corr = merged_df["possession_creation_index"].corr(
            merged_df["surplus_value"]
        )
        print(f"\n📊 Possession Creation Index ↔ Surplus Value correlation: {corr:.3f}")
        
        if abs(corr) > 0.1:
            print("   👀 There might be something here! Run Step 3 to explore further.")
        else:
            print("   🤔 Weak initial signal. But don't worry yet —")
            print("   The model may find non-linear patterns that correlation misses.")

    if "granular_prospect_index" in merged_df.columns:
        granular_corr = merged_df["granular_prospect_index"].corr(
            merged_df["surplus_value"]
        )
        print(
            "\n📊 Granular Prospect Index ↔ Surplus Value "
            f"in-sample correlation: {granular_corr:.3f}"
        )
        print("   ℹ️  This is a fitted descriptive correlation, not a held-out score.")
    
    print("\n✅ Done! Next step: 03_grouped_nested_cv.py")
