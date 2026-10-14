"""
NBA Draft Surplus Value Project — Step 1: Data Collection
=========================================================

This script pulls draft data from Basketball Reference for drafts 2009-2023.
Each draft page contains both the draft position AND career stats, so we get
everything we need from one source.

Setup (run once in terminal):
    pip install pandas requests beautifulsoup4 lxml matplotlib

Usage:
    python 01_collect_draft_data.py

Output:
    - data/nba_draft_raw.csv       (raw scraped data)
    - data/nba_draft_clean.csv     (cleaned with surplus value calculated)
    - plots/draft_value_curve.png  (your anchor visual)

IMPORTANT: Basketball Reference rate-limits requests. This script waits
3 seconds between each page to be respectful. The full scrape takes ~1 min.

Author: [Your name]
Project: NBA Draft Surplus Value Model
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import re
from io import StringIO

from position_utils import add_primary_position

# ---------------------------------------------------------------------------
# CONFIG — Change these if you want different draft range
# ---------------------------------------------------------------------------
START_YEAR = 2009
END_YEAR = 2023
WAIT_SECONDS = 10  # Delay between requests (don't lower this — bbref will block you)
LAST_COMPLETE_NBA_SEASON = 2026  # 2025-26
PEAK_WINDOW_YEARS = 3

# Create output directories
os.makedirs("data", exist_ok=True)
os.makedirs("data/draft_years", exist_ok=True)
os.makedirs("data/nba_seasons", exist_ok=True)
os.makedirs("plots", exist_ok=True)


class BasketballReferenceBlockedError(RuntimeError):
    """Raised when Basketball Reference refuses automated access."""


class NetworkUnavailableError(RuntimeError):
    """Raised when the host cannot be reached or resolved."""


def normalize_table_headers(df):
    """Flatten Basketball Reference's grouped, two-row table header."""
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        header_pairs = list(df.columns)
    elif "Pk" not in df.columns and not df.empty and "Pk" in df.iloc[0].astype(str).values:
        # Repair cache files created by the older one-header parser.
        header_pairs = list(zip(df.columns, df.iloc[0].astype(str)))
        df = df.iloc[1:].reset_index(drop=True)
    else:
        return df

    flattened = []
    for group, name in header_pairs:
        group = str(group).strip()
        group = re.sub(r"\.\d+$", "", group)
        name = str(name).strip()
        if (
            group.startswith("Unnamed")
            or group.startswith("Round ")
            or group == name
            or group == "Draft_Year"
        ):
            flattened.append("Draft_Year" if group == "Draft_Year" else name)
        else:
            flattened.append(f"{group}_{name}")

    df.columns = flattened
    return df


def scrape_draft_page(year):
    """
    Scrape one draft year from Basketball Reference.
    
    The draft page table includes columns like:
    - Pk (pick number)
    - Tm (drafting team)
    - Player
    - College
    - Yrs (years played)
    - G (games)
    - MP (minutes played)
    - PTS, TRB, AST (per game stats)
    - WS (career win shares)
    - WS/48 (win shares per 48 minutes)
    - BPM (box plus/minus)
    - VORP (value over replacement player)
    
    Returns a DataFrame for that draft class, or None if the request fails.
    """
    url = f"https://www.basketball-reference.com/draft/NBA_{year}.html"
    
    # Send the same basic headers as a normal desktop browser.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/134.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 403:
            raise BasketballReferenceBlockedError(
                f"Basketball Reference returned 403 for {year}."
            )
        response.raise_for_status()
    except BasketballReferenceBlockedError:
        raise
    except requests.ConnectionError as e:
        raise NetworkUnavailableError(
            f"Could not connect to {response.url if 'response' in locals() else url}: {e}"
        ) from e
    except requests.RequestException as e:
        print(f"  ❌ Failed to fetch {year}: {e}")
        return None
    
    # Parse the HTML and find the stats table
    soup = BeautifulSoup(response.content, "lxml")
    
    # Basketball Reference uses id="stats" for the main draft table
    table = soup.find("table", {"id": "stats"})
    if table is None:
        print(f"  ❌ No stats table found for {year}")
        return None
    
    # pandas can read HTML tables directly
    # We wrap the table HTML back into a string for pandas
    df = pd.read_html(StringIO(str(table)), header=[0, 1])[0]
    df = normalize_table_headers(df)
    
    # The table has sub-header rows we need to remove
    # These rows have "Pk" in the pick column (they're repeated headers)
    if "Pk" in df.columns:
        df = df[df["Pk"] != "Pk"]
    elif "Pick" in df.columns:
        df = df[df["Pick"] != "Pick"]
    
    # Add the draft year
    df["Draft_Year"] = year
    
    return df


def collect_all_drafts():
    """
    Loop through all draft years and combine into one DataFrame.
    """
    all_drafts = []
    missing_years = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        cache_path = f"data/draft_years/{year}.csv"

        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            df = normalize_table_headers(df)
            # Rewrite old malformed caches in the corrected format.
            df.to_csv(cache_path, index=False)
            all_drafts.append(df)
            print(f"Using saved {year} draft ({len(df)} picks)")
            continue

        print(f"Scraping {year} draft...")
        try:
            df = scrape_draft_page(year)
        except NetworkUnavailableError as e:
            print(f"  ❌ Network/DNS error: {e}")
            print("  Check your internet connection and DNS, then run the script again.")
            raise SystemExit(1)
        except BasketballReferenceBlockedError as e:
            print(f"  ❌ {e}")
            print("  Skipping this year; it will be retried on the next run.")
            missing_years.append(year)
            time.sleep(WAIT_SECONDS)
            continue
        
        if df is not None:
            df.to_csv(cache_path, index=False)
            all_drafts.append(df)
            print(f"  ✅ Got and saved {len(df)} picks")
        else:
            missing_years.append(year)
        
        # Be respectful — wait between requests
        time.sleep(WAIT_SECONDS)

    if missing_years:
        years = ", ".join(str(year) for year in missing_years)
        print(f"\n⚠️ Missing draft years: {years}")
        print("Successful years are saved in data/draft_years/.")
        print("Run the script again later; only missing years will be requested.")
        raise SystemExit(1)
    
    combined = pd.concat(all_drafts, ignore_index=True)
    return combined


def collect_nba_season_stats():
    """Collect league-wide season BPM/VORP tables needed for peak windows."""
    seasons = []
    missing_seasons = []
    for season in range(START_YEAR + 1, LAST_COMPLETE_NBA_SEASON + 1):
        cache_path = f"data/nba_seasons/{season}.csv"
        if os.path.exists(cache_path):
            cached = pd.read_csv(cache_path)
            cached["nba_season"] = season
            seasons.append(cached)
            print(f"Using saved NBA {season} advanced stats")
            continue

        url = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/134.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        response = requests.get(url, headers=headers, timeout=20)
        if response.status_code == 403:
            print(f"❌ Basketball Reference returned 403 for NBA season {season}")
            print("Skipping it for now; this season will be retried next run.")
            missing_seasons.append(season)
            time.sleep(WAIT_SECONDS)
            continue
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text), attrs={"id": "advanced"})
        if not tables:
            raise ValueError(f"No advanced table found for NBA season {season}")
        df = tables[0]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col[-1]) for col in df.columns]
        df = df[df["Player"].astype(str) != "Player"].copy()
        df["nba_season"] = season
        df.to_csv(cache_path, index=False)
        seasons.append(df)
        print(f"✅ Saved NBA {season} advanced stats")
        time.sleep(WAIT_SECONDS)

    if missing_seasons:
        print(f"\n⚠️ Missing NBA seasons: {', '.join(map(str, missing_seasons))}")
        print("Successful seasons are saved in data/nba_seasons/.")
        print("Run the script again later; only missing seasons will be requested.")
        raise SystemExit(1)

    return pd.concat(seasons, ignore_index=True)


def add_peak_three_year_value(draft_df, season_df):
    """Add the best same-window three-season VORP and BPM averages."""
    season_df = season_df.copy()
    season_df["match_name"] = (
        season_df["Player"].astype(str).str.lower().str.strip()
        .str.replace(r"[^a-z\s]", "", regex=True)
    )
    season_df["BPM"] = pd.to_numeric(season_df["BPM"], errors="coerce")
    season_df["VORP"] = pd.to_numeric(season_df["VORP"], errors="coerce")
    season_df["minutes_numeric"] = pd.to_numeric(
        season_df.get("MP"), errors="coerce"
    )
    season_df["games_numeric"] = pd.to_numeric(
        season_df.get("G"), errors="coerce"
    )
    season_df["nba_player_id"] = (
        season_df["Player-additional"].astype("string")
        if "Player-additional" in season_df.columns
        else season_df["match_name"]
    )
    if "Tm" in season_df.columns and "Team" in season_df.columns:
        team_values = season_df["Tm"].fillna(season_df["Team"])
    elif "Tm" in season_df.columns:
        team_values = season_df["Tm"]
    else:
        team_values = season_df["Team"]
    # Preserve every source row, but prefer the combined traded-player row
    # (2TM, 3TM, etc.; TOT is the legacy label) for peak calculations/plots.
    season_df["tot_priority"] = (
        team_values.astype(str).str.fullmatch(r"(?:TOT|\d+TM)")
    ).astype(int)
    season_df = season_df[season_df["games_numeric"].gt(0)].copy()
    season_df = (
        season_df.sort_values(
            [
                "match_name", "nba_player_id", "nba_season", "tot_priority",
                "minutes_numeric",
            ],
            ascending=[True, True, True, False, False],
        )
        .drop_duplicates(["match_name", "nba_player_id", "nba_season"])
    )
    season_df["career_debut"] = season_df.groupby(
        ["match_name", "nba_player_id"]
    )["nba_season"].transform("min")
    season_df["career_minutes"] = season_df.groupby(
        ["match_name", "nba_player_id"]
    )["minutes_numeric"].transform("sum")

    peak_rows = []
    for (match_name, nba_player_id), player_seasons in season_df.groupby(
        ["match_name", "nba_player_id"]
    ):
        player_seasons = player_seasons.sort_values("nba_season")
        records = player_seasons[["nba_season", "VORP", "BPM"]].dropna(subset=["VORP", "BPM"])
        candidates = []
        if len(records) <= PEAK_WINDOW_YEARS:
            if not records.empty:
                candidates = [records]
        else:
            for i in range(len(records) - PEAK_WINDOW_YEARS + 1):
                window = records.iloc[i:i + PEAK_WINDOW_YEARS]
                if int(window.iloc[-1]["nba_season"] - window.iloc[0]["nba_season"]) == 2:
                    candidates.append(window)
        if not candidates:
            continue
        best = max(candidates, key=lambda w: (w["VORP"].mean(), w["BPM"].mean()))
        peak_rows.append({
            "match_name": match_name,
            "nba_player_id": nba_player_id,
            "nba_career_debut": int(player_seasons["career_debut"].iloc[0]),
            "nba_career_minutes": float(player_seasons["career_minutes"].iloc[0]),
            "peak_3yr_vorp": best["VORP"].mean(),
            "peak_3yr_bpm": best["BPM"].mean(),
            "peak_start_season": int(best.iloc[0]["nba_season"]),
            "peak_end_season": int(best.iloc[-1]["nba_season"]),
            "nba_seasons_available": len(records),
        })

    peaks = pd.DataFrame(peak_rows)
    result = draft_df.copy()
    result["match_name"] = (
        result["player"].astype(str).str.lower().str.strip()
        .str.replace(r"[^a-z\s]", "", regex=True)
    )
    result["_draft_row_id"] = range(len(result))
    candidates = result[
        ["_draft_row_id", "match_name", "Draft_Year"]
    ].merge(peaks, on="match_name", how="left")
    candidates = candidates[
        candidates["nba_career_debut"] > candidates["Draft_Year"]
    ].copy()
    # Earliest eligible debut is the least ambiguous match.  Career minutes
    # break rare same-name/same-debut ties such as the two Tony Mitchells.
    matched_peaks = (
        candidates.sort_values(
            ["_draft_row_id", "nba_career_debut", "nba_career_minutes"],
            ascending=[True, True, False],
            na_position="last",
        )
        .drop_duplicates("_draft_row_id")
        .drop(columns=["match_name", "Draft_Year"])
    )
    result = result.merge(matched_peaks, on="_draft_row_id", how="left")
    result["peak_3yr_vorp"] = result["peak_3yr_vorp"].fillna(0)
    result["peak_3yr_bpm"] = result["peak_3yr_bpm"].fillna(0)
    expected = result.groupby("pick")["peak_3yr_vorp"].mean().rename("expected_peak_3yr_vorp_for_slot")
    result = result.merge(expected, left_on="pick", right_index=True)
    result["surplus_value"] = result["peak_3yr_vorp"] - result["expected_peak_3yr_vorp_for_slot"]
    return result.drop(
        columns=["match_name", "_draft_row_id", "nba_career_minutes"]
    )


def clean_draft_data(df):
    """
    Clean the raw scraped data:
    1. Standardise column names
    2. Convert numeric columns from strings to numbers
    3. Handle missing values (undrafted/overseas players who never played)
    4. Calculate surplus value
    """
    # Make a copy so we don't modify the original
    df = df.copy()
    
    # -----------------------------------------------------------------------
    # Standardise column names
    # Basketball Reference column names can vary slightly across years.
    # We'll rename to consistent names.
    # -----------------------------------------------------------------------
    rename_map = {}
    available_column_names = {str(col).lower().strip() for col in df.columns}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ["pk", "pick"] or (
            col_lower == "rk" and not ({"pk", "pick"} & available_column_names)
        ):
            rename_map[col] = "pick"
        elif col_lower == "tm":
            rename_map[col] = "team"
        elif col_lower == "player":
            rename_map[col] = "player"
        elif col_lower in ["college", "college/univ"]:
            rename_map[col] = "college"
        elif col_lower == "yrs":
            rename_map[col] = "years_played"
        elif col_lower in ["g", "totals_g"]:
            rename_map[col] = "games"
        elif col_lower in ["mp", "totals_mp"]:
            rename_map[col] = "minutes_played"
        elif col_lower in ["pts", "per game_pts"]:
            rename_map[col] = "ppg"
        elif col_lower in ["trb", "per game_trb"]:
            rename_map[col] = "rpg"
        elif col_lower in ["ast", "per game_ast"]:
            rename_map[col] = "apg"
        elif col_lower in ["ws", "advanced_ws"]:
            rename_map[col] = "win_shares"
        elif col_lower in ["ws/48", "advanced_ws/48"]:
            rename_map[col] = "ws_per_48"
        elif col_lower in ["bpm", "advanced_bpm"]:
            rename_map[col] = "bpm"
        elif col_lower in ["vorp", "advanced_vorp"]:
            rename_map[col] = "vorp"
    
    df = df.rename(columns=rename_map)
    
    # -----------------------------------------------------------------------
    # Convert numeric columns
    # -----------------------------------------------------------------------
    numeric_cols = [
        "pick", "years_played", "games", "minutes_played",
        "ppg", "rpg", "apg", "win_shares", "ws_per_48", "bpm", "vorp"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # -----------------------------------------------------------------------
    # Filter: only keep actual picks (remove rows with no pick number)
    # -----------------------------------------------------------------------
    df = df.dropna(subset=["pick"])
    df["pick"] = df["pick"].astype(int)
    
    # -----------------------------------------------------------------------
    # Filter: only first and second round (picks 1-60)
    # Some years have fewer picks, but 60 is the modern standard
    # -----------------------------------------------------------------------
    df = df[df["pick"] <= 60]
    
    # -----------------------------------------------------------------------
    # Fill missing stats with 0 (players who never played in the NBA)
    # -----------------------------------------------------------------------
    for col in numeric_cols:
        if col in df.columns and col != "pick":
            df[col] = df[col].fillna(0)
    
    # Preserve career aggregates for reference. Peak-window values are added
    # later from season-level advanced tables and become the modelling target.
    df = df.rename(columns={"vorp": "career_vorp", "bpm": "career_bpm"})
    
    return df


def plot_draft_value_curve(df):
    """
    Create the anchor visual: expected VORP by draft pick,
    with individual players scattered around the curve.
    
    This is the plot that will appear in every presentation.
    """
    import matplotlib.pyplot as plt
    
    # Calculate average VORP per pick
    avg_by_pick = df.groupby("pick")["peak_3yr_vorp"].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Scatter all players (light dots)
    ax.scatter(df["pick"], df["peak_3yr_vorp"], alpha=0.15, s=15, color="#6366f1", 
               label="Individual players")
    
    # Average curve (bold line)
    ax.plot(avg_by_pick["pick"], avg_by_pick["peak_3yr_vorp"], color="#f97316", 
            linewidth=2.5, label="Average VORP by pick")
    
    # Label some famous steals (high surplus value)
    top_steals = df.nlargest(8, "surplus_value")
    for _, row in top_steals.iterrows():
        if pd.notna(row.get("player")):
            ax.annotate(
                f"{row['player']} (#{int(row['pick'])})",
                xy=(row["pick"], row["peak_3yr_vorp"]),
                fontsize=7,
                alpha=0.8,
                textcoords="offset points",
                xytext=(5, 5),
            )
    
    ax.set_xlabel("Draft Pick", fontsize=12)
    ax.set_ylabel("Best 3-Year Average VORP", fontsize=12)
    ax.set_title("NBA Draft Value Curve: Peak 3-Year VORP by Draft Position (2009-2023)", 
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 62)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("plots/draft_value_curve.png", dpi=150)
    plt.close()
    print("📊 Saved: plots/draft_value_curve.png")


def print_top_steals(df, n=15):
    """Show the biggest surplus value players — the 'steals'."""
    cols = ["Draft_Year", "pick", "player", "college", "peak_3yr_vorp",
            "peak_3yr_bpm", "peak_start_season", "peak_end_season",
            "expected_peak_3yr_vorp_for_slot", "surplus_value"]
    available_cols = [c for c in cols if c in df.columns]
    
    print("\n" + "=" * 70)
    print(f"TOP {n} DRAFT STEALS (highest surplus value)")
    print("=" * 70)
    top = df.nlargest(n, "surplus_value")[available_cols]
    print(top.to_string(index=False))
    
    print("\n" + "=" * 70)
    print(f"TOP {n} DRAFT BUSTS (lowest surplus value)")
    print("=" * 70)
    bottom = df.nsmallest(n, "surplus_value")[available_cols]
    print(bottom.to_string(index=False))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🏀 NBA Draft Surplus Value — Data Collection")
    print(f"   Drafts: {START_YEAR} to {END_YEAR}")
    print(f"   Delay between requests: {WAIT_SECONDS}s")
    print("-" * 50)
    
    # Step 1: Scrape
    raw_df = collect_all_drafts()
    raw_df.to_csv("data/nba_draft_raw.csv", index=False)
    print(f"\n💾 Saved raw data: data/nba_draft_raw.csv ({len(raw_df)} rows)")
    
    # Step 2: Clean career aggregates
    clean_df = clean_draft_data(raw_df)

    # Step 3: Add best same-window three-season VORP/BPM and surplus value
    print("\nCollecting completed NBA season advanced stats...")
    season_df = collect_nba_season_stats()
    clean_df = add_peak_three_year_value(clean_df, season_df)
    clean_df = add_primary_position(
        clean_df, allow_statistical_fallback=False
    )
    clean_df.to_csv("data/nba_draft_clean.csv", index=False)
    print(f"💾 Saved clean data: data/nba_draft_clean.csv ({len(clean_df)} rows)")
    
    # Step 4: Plot the draft value curve
    plot_draft_value_curve(clean_df)
    
    # Step 5: Show the biggest steals and busts
    print_top_steals(clean_df)
    
    print("\n✅ Done! Next step: run 02_collect_college_stats.py")
