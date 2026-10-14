"""Composite index (bpm + adjoe + porpag + usg + age_at_draft) vs peak 3-yr VORP."""
import sys
from pathlib import Path
_DEPS = Path(__file__).resolve().parent / ".deps"
if _DEPS.exists():
    sys.path.insert(0, str(_DEPS))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = Path("data/draft_with_college.csv")
OUT = Path("plots/composite_index_vs_peak_vorp.png")
FEATS = ["bpm", "adjoe", "porpag", "usg", "age_at_draft"]

# ---- palette (single-series scatter: one accessible hue + a warm trend line) --
INK, MUTED = "#1f2933", "#6b7280"
DOT, DOT_EDGE = "#2f6fb0", "#ffffff"      # blue points
TOP, TOP_EDGE = "#c1462f", "#ffffff"      # highlighted top-20
TREND = "#c1462f"
GRID = "#e5e7eb"

df = pd.read_csv(DATA, low_memory=False)


def ht(t):
    try:
        f, i = str(t).split("-")
        return int(f) * 12 + int(i)
    except (ValueError, AttributeError):
        return np.nan


df["height_in"] = df["ht"].map(ht)
dob = pd.to_datetime(df["dob"], errors="coerce")
dd = pd.to_datetime(df["Draft_Year"].astype("Int64").astype(str) + "-06-25", errors="coerce")
df["age_at_draft"] = (dd - dob).dt.days / 365.25

y_all = pd.to_numeric(df["peak_3yr_vorp"], errors="coerce")
has = df["bpm"].notna()
sub = df.loc[has].copy()
y = y_all[has]


def z(s):
    x = pd.to_numeric(s, errors="coerce")
    return (x - x.mean()) / x.std()


# sign-aligned z-score mean so higher composite = more peak VORP
parts = []
for f in FEATS:
    zi = z(sub[f])
    m = zi.notna() & y.notna()
    sign = np.sign(np.corrcoef(zi[m], y[m])[0, 1]) or 1.0
    parts.append(sign * zi)
comp = pd.concat(parts, axis=1).mean(axis=1)
comp = comp.fillna(comp.median())

m = comp.notna() & y.notna()
r = np.corrcoef(comp[m], y[m])[0, 1]

sub = sub.assign(comp=comp.values, y=y.values)
top20 = sub.sort_values("y", ascending=False).head(20)
top_idx = set(top20.index)

# ---- figure ------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(13, 9))
ax.set_axisbelow(True)
ax.grid(True, color=GRID, linewidth=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

rest = sub[~sub.index.isin(top_idx)]
ax.scatter(rest["comp"], rest["y"], s=34, c=DOT, edgecolors=DOT_EDGE,
           linewidths=0.5, alpha=0.55, zorder=2)
ax.scatter(top20["comp"], top20["y"], s=90, c=TOP, edgecolors=TOP_EDGE,
           linewidths=1.1, zorder=4)

# OLS trend line
b1, b0 = np.polyfit(sub["comp"], sub["y"], 1)
xs = np.linspace(sub["comp"].min(), sub["comp"].max(), 100)
ax.plot(xs, b0 + b1 * xs, color=TREND, linewidth=2, zorder=3)

# label the top 20 — stack labels down the right, leader lines to each point
tv = top20.sort_values("y", ascending=False).reset_index()
x_lab = sub["comp"].max() + 0.15
y_hi, y_lo = sub["y"].max(), 3.3
ys = np.linspace(y_hi, y_lo, len(tv))
for k, row in tv.iterrows():
    ax.annotate(
        f"{row['player']}", (row["comp"], row["y"]), xytext=(x_lab, ys[k]),
        textcoords="data", fontsize=8.3, color=INK, va="center", ha="left",
        arrowprops=dict(arrowstyle="-", color="#c9ccd1", linewidth=0.6,
                        connectionstyle="arc3,rad=0.0"), zorder=5)

ax.set_xlim(sub["comp"].min() - 0.15, x_lab + 1.9)
ax.set_xlabel("College composite index  (z-mean of bpm, adjoe, porpag, usg, youth)", fontsize=12)
ax.set_ylabel("NBA peak 3-yr VORP", fontsize=12)
ax.set_title("A 5-stat college composite vs NBA peak value",
             fontsize=16, fontweight="bold", loc="left", pad=26)
ax.text(0, 1.02, f"Pearson r = {r:.2f}  ·  n = {int(m.sum())} drafted players with college stats  ·  "
        "red = top-20 peak VORP", transform=ax.transAxes, fontsize=10.5, color=MUTED)

handles = [Line2D([], [], marker="o", color="none", markerfacecolor=TOP,
                  markeredgecolor=TOP_EDGE, markersize=10, label="Top-20 peak VORP"),
           Line2D([], [], marker="o", color="none", markerfacecolor=DOT,
                  markeredgecolor=DOT_EDGE, markersize=7, label="Other drafted players"),
           Line2D([], [], color=TREND, linewidth=2, label="OLS trend")]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9.5)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(OUT, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"r={r:.3f}  saved {OUT}")
