"""Equal-weight composite vs regression-weighted index, side by side, vs peak VORP."""
import sys
from pathlib import Path
_DEPS = Path(__file__).resolve().parent / ".deps"
if _DEPS.exists():
    sys.path.insert(0, str(_DEPS))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.impute import SimpleImputer

DATA = Path("data/draft_with_college.csv")
OUT = Path("plots/composite_equal_vs_weighted.png")
FEATS = ["bpm", "adjoe", "porpag", "usg", "age_at_draft"]

INK, MUTED = "#1f2933", "#6b7280"
DOT, DOT_EDGE = "#2f6fb0", "#ffffff"
TOP, TOP_EDGE = "#c1462f", "#ffffff"
TREND, GRID = "#c1462f", "#e5e7eb"

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
y = y_all[has].reset_index(drop=True)


def z(s):
    x = pd.to_numeric(s, errors="coerce")
    return (x - x.mean()) / x.std()


# --- index A: equal-weight sign-aligned z-mean --------------------------------
parts = []
for f in FEATS:
    zi = z(sub[f]).reset_index(drop=True)
    m = zi.notna() & y.notna()
    sign = np.sign(np.corrcoef(zi[m], y[m])[0, 1]) or 1.0
    parts.append(sign * zi)
comp_eq = pd.concat(parts, axis=1).mean(axis=1)
comp_eq = comp_eq.fillna(comp_eq.median())

# --- index B: regression-weighted, out-of-fold (honest, not overfit) ----------
X = sub[FEATS].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=FEATS)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
comp_wt = pd.Series(cross_val_predict(LinearRegression(), X, y, cv=cv))
weights = LinearRegression().fit(X, y).coef_

r_eq = np.corrcoef(comp_eq, y)[0, 1]
r_wt = np.corrcoef(comp_wt, y)[0, 1]

yv = y.values
top_order = np.argsort(-yv)[:20]
top_mask = np.zeros(len(yv), bool)
top_mask[top_order] = True
names = sub["player"].reset_index(drop=True)

plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(17, 8.5), sharey=True)

panels = [
    (axes[0], comp_eq.values, r_eq, "Equal-weight composite",
     "plain z-mean — every stat weighted the same"),
    (axes[1], comp_wt.values, r_wt, "Regression-weighted index",
     "optimal weights, cross-validated (out-of-fold)"),
]
label_a = {"James Harden", "Stephen Curry", "Anthony Davis", "Draymond Green", "Isaiah Thomas"}

for ax, xv, r, title, sub_t in panels:
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.scatter(xv[~top_mask], yv[~top_mask], s=30, c=DOT, edgecolors=DOT_EDGE,
               linewidths=0.5, alpha=0.5, zorder=2)
    ax.scatter(xv[top_mask], yv[top_mask], s=85, c=TOP, edgecolors=TOP_EDGE,
               linewidths=1.1, zorder=4)
    b1, b0 = np.polyfit(xv, yv, 1)
    xs = np.linspace(xv.min(), xv.max(), 100)
    ax.plot(xs, b0 + b1 * xs, color=TREND, linewidth=2, zorder=3)
    for i in top_order:
        if names[i] in label_a:
            ax.annotate(names[i], (xv[i], yv[i]), xytext=(6, 6),
                        textcoords="offset points", fontsize=8.5, color=INK, zorder=5)
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=20)
    ax.text(0, 1.015, f"{sub_t}    ·    Pearson r = {r:.2f}",
            transform=ax.transAxes, fontsize=10.5, color=MUTED)
    ax.set_xlabel("College index  (higher = stronger profile)", fontsize=11.5)

axes[0].set_ylabel("NBA peak 3-yr VORP", fontsize=12.5)
handles = [Line2D([], [], marker="o", color="none", markerfacecolor=TOP,
                  markeredgecolor=TOP_EDGE, markersize=10, label="Top-20 peak VORP"),
           Line2D([], [], marker="o", color="none", markerfacecolor=DOT,
                  markeredgecolor=DOT_EDGE, markersize=7, label="Other drafted players"),
           Line2D([], [], color=TREND, linewidth=2, label="OLS trend")]
axes[1].legend(handles=handles, loc="lower right", frameon=False, fontsize=9.5)

fig.suptitle("Optimal weighting tightens the cloud — but not past r = 0.5",
             fontsize=17, fontweight="bold", x=0.09, ha="left", y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.97])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
plt.close(fig)

print(f"equal r={r_eq:.3f}   weighted (CV) r={r_wt:.3f}")
print("fitted weights (per-unit):")
for f, w in zip(FEATS, weights):
    print(f"  {f:14s} {w:+.4f}")
print(f"saved {OUT}")
