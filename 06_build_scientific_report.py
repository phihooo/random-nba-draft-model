"""Build the publication-ready scientific report and its report-only figures.

The script reads only derived analysis outputs. It does not recollect data or
refit any model, which keeps report production separate from estimation.

Outputs
-------
output/pdf/NBA_Draft_Possession_Position_Report.pdf
reports/figures/*.png
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
PDF_DIR = ROOT / "output" / "pdf"
PDF_PATH = PDF_DIR / "NBA_Draft_Possession_Position_Report.pdf"

REPORT_DATE = "4 August 2026"
POSITIONS = ["PG", "SG", "SF", "PF", "C"]

NAVY = "#17324D"
BLUE = "#2E74B5"
DARK_BLUE = "#1F4D78"
SKY = "#4E9FAD"
ORANGE = "#E07A5F"
GOLD = "#B8862B"
INK = "#1E2630"
MUTED = "#65717D"
LIGHT = "#F4F6F9"
GRID = "#D9E0E7"
RED = "#A33A3A"
GREEN = "#247A5A"

POSITION_COLORS = {
    "PG": "#2F67E4",
    "SG": "#7A38E8",
    "SF": "#D62878",
    "PF": "#F05A0B",
    "C": "#18A24A",
    "Unassigned": "#929AA3",
}

REQUIRED_INPUTS = [
    DATA / "draft_with_college.csv",
    DATA / "position_summary.csv",
    DATA / "position_feature_correlations.csv",
    DATA / "grouped_nested_cv_model_metrics.csv",
    DATA / "grouped_nested_cv_comparisons.csv",
    DATA / "grouped_nested_cv_position_metrics.csv",
    DATA / "tsne_cluster_metrics.csv",
    PLOTS / "draft_value_curve.png",
    PLOTS / "by_position" / "pci_vs_surplus_value_by_position.png",
    PLOTS / "tsne" / "possession.png",
    PLOTS / "tsne" / "by_position" / "SF.png",
    PLOTS / "by_position" / "surplus_value_correlations_by_position_partial.png",
]


def validate_inputs() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing report inputs: " + ", ".join(missing))


def register_fonts() -> None:
    """Register Arial, the explicit scientific-report font override."""
    font_dir = Path("/System/Library/Fonts/Supplemental")
    font_files = {
        "Arial": font_dir / "Arial.ttf",
        "Arial-Bold": font_dir / "Arial Bold.ttf",
        "Arial-Italic": font_dir / "Arial Italic.ttf",
        "Arial-BoldItalic": font_dir / "Arial Bold Italic.ttf",
    }
    if all(path.exists() for path in font_files.values()):
        for name, path in font_files.items():
            pdfmetrics.registerFont(TTFont(name, str(path)))
        pdfmetrics.registerFontFamily(
            "Arial",
            normal="Arial",
            bold="Arial-Bold",
            italic="Arial-Italic",
            boldItalic="Arial-BoldItalic",
        )


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#66717C",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.65,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "master": pd.read_csv(DATA / "draft_with_college.csv", low_memory=False),
        "summary": pd.read_csv(DATA / "position_summary.csv"),
        "correlations": pd.read_csv(DATA / "position_feature_correlations.csv"),
        "model_metrics": pd.read_csv(DATA / "grouped_nested_cv_model_metrics.csv"),
        "comparisons": pd.read_csv(DATA / "grouped_nested_cv_comparisons.csv"),
        "position_metrics": pd.read_csv(DATA / "grouped_nested_cv_position_metrics.csv"),
        "tsne_metrics": pd.read_csv(DATA / "tsne_cluster_metrics.csv"),
    }


def save_figure(fig: plt.Figure, filename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def make_workflow_figure() -> Path:
    fig, ax = plt.subplots(figsize=(12, 3.35))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.35)
    ax.axis("off")
    boxes = [
        (0.15, "Cohort", "896 draft picks\n2009-2023"),
        (2.55, "College match", "682 BartTorvik\nprofiles"),
        (4.95, "Role control", "PG / SG / SF / PF / C\nplus provenance"),
        (7.35, "Confirmatory test", "Held-out draft years\nfirst-3yr VORP"),
        (9.75, "Exploration", "Correlations and\nsame-position t-SNE"),
    ]
    for x, title, detail in boxes:
        rect = plt.Rectangle((x, 0.85), 2.05, 1.7, facecolor=LIGHT, edgecolor=BLUE,
                             linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + 1.025, 2.08, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=NAVY)
        ax.text(x + 1.025, 1.42, detail, ha="center", va="center",
                fontsize=9, color=INK, linespacing=1.3)
    for start in [2.2, 4.6, 7.0, 9.4]:
        ax.annotate("", xy=(start + 0.28, 1.7), xytext=(start, 1.7),
                    arrowprops={"arrowstyle": "-|>", "color": GOLD, "lw": 1.8})
    ax.text(6, 3.02, "Scientific workflow and claim hierarchy", ha="center",
            va="center", fontsize=14, fontweight="bold", color=NAVY)
    ax.text(6, 0.35,
            "Prediction is evaluated without using post-draft position; position is descriptive only.",
            ha="center", va="center", fontsize=9, color=MUTED, style="italic")
    return save_figure(fig, "scientific_workflow.png")


def make_cohort_figure(tables: dict[str, pd.DataFrame]) -> Path:
    summary = tables["summary"].copy()
    order = POSITIONS + ["Unassigned"]
    summary = summary.set_index("primary_position").reindex(order)
    matched = summary["college_matched_players"].fillna(0).to_numpy(dtype=float)
    total = summary["players"].fillna(0).to_numpy(dtype=float)
    unmatched = total - matched

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    x = np.arange(len(order))
    colors_matched = [POSITION_COLORS[p] for p in order]
    colors_unmatched = [POSITION_COLORS[p] for p in order]
    ax.bar(x, matched, color=colors_matched, width=0.7, label="College matched")
    ax.bar(x, unmatched, bottom=matched, color=colors_unmatched, alpha=0.26,
           hatch="///", edgecolor=colors_unmatched, width=0.7,
           label="No matched college profile")
    for i, (m, t) in enumerate(zip(matched, total)):
        ax.text(i, t + 5, f"{int(t)}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=INK)
        if m > 0:
            ax.text(i, m / 2, f"{int(m)}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
    ax.set_xticks(x, order)
    ax.set_ylabel("Drafted players")
    fig.suptitle("Cohort coverage by primary position", y=0.985,
                 fontsize=14, fontweight="bold", color=NAVY)
    ax.set_title(
            "896 total; 682 college matched; all matched players assigned a position",
            fontsize=9.5, color=MUTED, fontweight="normal", pad=9,
    )
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(total) * 1.18)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return save_figure(fig, "cohort_position_coverage.png")


def make_pci_forest(tables: dict[str, pd.DataFrame]) -> Path:
    c = tables["correlations"]
    rows = c[(c["target"] == "surplus_value") &
             (c["feature"] == "possession_creation_index")].copy()
    order = ["ALL", "WITHIN_POSITION", "PG", "SG", "SF", "PF", "C"]
    rows["primary_position"] = pd.Categorical(rows["primary_position"], order, ordered=True)
    rows = rows.sort_values("primary_position")
    labels = ["Pooled", "Within-position", "PG", "SG", "SF", "PF", "C"]
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    r = rows["pearson_r"].to_numpy()
    low = rows["pearson_ci_low"].to_numpy()
    high = rows["pearson_ci_high"].to_numpy()
    partial = rows["partial_r_controlling_pick"].to_numpy()
    ax.errorbar(r, y + 0.10, xerr=np.vstack([r - low, high - r]), fmt="o",
                color=BLUE, ecolor=BLUE, capsize=4, linewidth=1.4,
                label="Pearson r with Fisher 95% CI")
    ax.scatter(partial, y - 0.10, marker="D", s=38, color=ORANGE,
               label="Partial r controlling quadratic draft pick", zorder=3)
    ax.axvline(0, color="#5B6570", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Correlation with exploratory draft surplus value")
    fig.suptitle("Possession Creation Index: small positive descriptive associations",
                 y=0.985, fontsize=14, fontweight="bold", color=NAVY)
    ax.set_title(
            "A filled star indicates Benjamini-Hochberg q < 0.05 for that estimate",
            fontsize=9.2, color=MUTED, fontweight="normal", pad=9,
    )
    for idx, (_, row) in enumerate(rows.iterrows()):
        y_i = y[idx]
        if row["pearson_q_value"] < 0.05:
            ax.text(row["pearson_r"] + 0.013, y_i + 0.10, "*", color=BLUE,
                    va="center", fontweight="bold")
        if row["partial_q_value"] < 0.05:
            ax.text(row["partial_r_controlling_pick"] + 0.013, y_i - 0.10, "*",
                    color=ORANGE, va="center", fontweight="bold")
    ax.set_xlim(-0.18, 0.35)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=2, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=[0, 0.09, 1, 0.92])
    return save_figure(fig, "pci_surplus_forest.png")


def make_cv_forest(tables: dict[str, pd.DataFrame]) -> Path:
    comparisons = tables["comparisons"]
    rows = comparisons[(comparisons["target"] == "first_3yr_vorp") &
                       (comparisons["comparison"] == "adds_beyond_pick")].copy()
    label_map = {
        "pick_plus_fixed_pci": "Fixed PCI",
        "pick_plus_possession": "Possession family",
        "pick_plus_scoring": "Scoring family",
        "pick_plus_impact_offense": "Impact / offense",
        "pick_plus_defense_size": "Defense / size",
        "pick_plus_context_age": "Context / age",
        "pick_plus_all": "All college features",
    }
    order = list(label_map)
    rows["candidate_model"] = pd.Categorical(rows["candidate_model"], order, ordered=True)
    rows = rows.sort_values("candidate_model")
    y = np.arange(len(rows))[::-1]
    mean = rows["mean_year_delta_mae"].to_numpy()
    low = rows["year_delta_mae_ci_low"].to_numpy()
    high = rows["year_delta_mae_ci_high"].to_numpy()

    fig, ax = plt.subplots(figsize=(10.2, 5.9))
    point_colors = [ORANGE if "possession" in str(m) or "fixed_pci" in str(m)
                    else BLUE for m in rows["candidate_model"].astype(str)]
    for i in range(len(rows)):
        ax.errorbar(mean[i], y[i], xerr=[[mean[i] - low[i]], [high[i] - mean[i]]],
                    fmt="o", color=point_colors[i], ecolor=point_colors[i],
                    capsize=4, linewidth=1.6, markersize=6)
        q = rows.iloc[i]["q_value"]
        ax.text(high[i] + 0.0012, y[i], f"q={q:.3f}", va="center", fontsize=8.5,
                color=MUTED)
    ax.axvline(0, color="#4F5963", linewidth=1.2)
    ax.set_yticks(y, [label_map[str(v)] for v in rows["candidate_model"]])
    ax.set_xlabel("Mean held-out-year MAE improvement versus draft-pick baseline")
    fig.suptitle("No college-stat addition reliably improves first-3yr VORP MAE",
                 y=0.985, fontsize=14, fontweight="bold", color=NAVY)
    ax.set_title(
            "Positive values favor the college-stat model; intervals resample complete draft years",
            fontsize=9.2, color=MUTED, fontweight="normal", pad=9,
    )
    ax.set_xlim(min(low) - 0.003, max(high) + 0.008)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return save_figure(fig, "cv_mae_gain_forest.png")


def make_position_oof_figure(tables: dict[str, pd.DataFrame]) -> Path:
    p = tables["position_metrics"]
    p = p[p["target"] == "first_3yr_vorp"].copy()
    base = p[p["model"] == "pick_only"].set_index("primary_position")
    poss = p[p["model"] == "pick_plus_possession"].set_index("primary_position")
    full = p[p["model"] == "pick_plus_all"].set_index("primary_position")

    mae_poss = (base.loc[POSITIONS, "mae"] - poss.loc[POSITIONS, "mae"]).to_numpy()
    mae_full = (base.loc[POSITIONS, "mae"] - full.loc[POSITIONS, "mae"]).to_numpy()
    r_poss = (poss.loc[POSITIONS, "pearson_r"] - base.loc[POSITIONS, "pearson_r"]).to_numpy()
    r_full = (full.loc[POSITIONS, "pearson_r"] - base.loc[POSITIONS, "pearson_r"]).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))
    x = np.arange(len(POSITIONS))
    width = 0.34
    axes[0].bar(x - width / 2, mae_poss, width, color=ORANGE, label="Possession family")
    axes[0].bar(x + width / 2, mae_full, width, color=BLUE, label="All college features")
    axes[0].axhline(0, color="#4F5963", linewidth=1)
    axes[0].set_xticks(x, POSITIONS)
    axes[0].set_ylabel("MAE improvement versus pick-only")
    axes[0].set_title("Prediction error")

    axes[1].bar(x - width / 2, r_poss, width, color=ORANGE)
    axes[1].bar(x + width / 2, r_full, width, color=BLUE)
    axes[1].axhline(0, color="#4F5963", linewidth=1)
    axes[1].set_xticks(x, POSITIONS)
    axes[1].set_ylabel("Change in held-out Pearson r")
    axes[1].set_title("Ranking agreement")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Position audit of pooled, leakage-safe predictions", y=1.02,
                 fontsize=14, fontweight="bold", color=NAVY)
    fig.text(0.5, -0.01,
             "Descriptive subgroup audit only: these are not five independently fitted position models.",
             ha="center", fontsize=9, color=MUTED, style="italic")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.96), frameon=False, ncol=2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.91])
    return save_figure(fig, "position_oof_audit.png")


def make_tsne_quality_figure(tables: dict[str, pd.DataFrame]) -> Path:
    t = tables["tsne_metrics"]
    t = t[t["analysis_scope"] == "position_only"].copy()
    panels = ["possession", "scoring", "impact_offense", "defense_size",
              "context_age", "all_features"]
    panel_labels = ["Possession", "Scoring", "Impact / offense", "Defense / size",
                    "Context / age", "All features"]
    sil = t.pivot(index="primary_position", columns="panel", values="silhouette_score").loc[POSITIONS, panels]
    trust = t.pivot(index="primary_position", columns="panel", values="tsne_trustworthiness").loc[POSITIONS, panels]

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.8), gridspec_kw={"hspace": 0.38})
    for ax, values, title, vmin, vmax, cmap in [
        (axes[0], sil, "Cluster separation in standardized source space (silhouette)",
         0.1, 0.6, "YlOrRd"),
        (axes[1], trust, "Local-neighborhood preservation in 2D (t-SNE trustworthiness)",
         0.84, 1.0, "YlGnBu"),
    ]:
        im = ax.imshow(values.to_numpy(), aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(np.arange(len(panels)), panel_labels)
        ax.set_yticks(np.arange(len(POSITIONS)), POSITIONS)
        ax.set_title(title, fontsize=11.5)
        ax.grid(False)
        for i in range(len(POSITIONS)):
            for j in range(len(panels)):
                value = values.iloc[i, j]
                ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                        fontsize=8.5, color="white" if value > (vmin + vmax) / 2 else INK,
                        fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.suptitle("What the position-specific t-SNE diagnostics support", y=0.99,
                 fontsize=14, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_figure(fig, "tsne_quality_heatmap.png")


def make_selected_partial_heatmap(tables: dict[str, pd.DataFrame]) -> Path:
    c = tables["correlations"]
    features = [
        "granular_prospect_index", "possession_creation_index", "stl_per",
        "ORB_per", "DRB_per", "bpm", "gbpm", "TS_per",
        "scoring_profile_score",
    ]
    labels = ["GPI*", "PCI", "STL%", "ORB%", "DRB%", "BPM", "GBPM", "TS%",
              "Scoring profile"]
    rows = c[(c["target"] == "surplus_value") &
             c["primary_position"].isin(POSITIONS) &
             c["feature"].isin(features)].copy()
    matrix = rows.pivot(index="feature", columns="primary_position",
                        values="partial_r_controlling_pick").reindex(features).reindex(columns=POSITIONS)
    q = rows.pivot(index="feature", columns="primary_position",
                   values="partial_q_value").reindex(features).reindex(columns=POSITIONS)

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    im = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(np.arange(len(POSITIONS)), POSITIONS)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Selected draft-pick-adjusted surplus correlations by position")
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            star = "*" if q.iloc[i, j] < 0.05 else ""
            ax.text(j, i, f"{value:+.2f}{star}", ha="center", va="center",
                    fontsize=9, color="white" if abs(value) > 0.32 else INK,
                    fontweight="bold" if star else "normal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Partial r controlling quadratic draft pick")
    ax.text(0.5, -0.12,
            "* Benjamini-Hochberg q < 0.05 within the position-target family. GPI is in-sample descriptive.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    return save_figure(fig, "selected_partial_correlations.png")


def make_report_figures(tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    set_plot_style()
    paths = {
        "workflow": make_workflow_figure(),
        "cohort": make_cohort_figure(tables),
        "pci_forest": make_pci_forest(tables),
        "cv_forest": make_cv_forest(tables),
        "position_oof": make_position_oof_figure(tables),
        "tsne_quality": make_tsne_quality_figure(tables),
        "selected_partial": make_selected_partial_heatmap(tables),
    }
    paths.update({
        "draft_curve": PLOTS / "draft_value_curve.png",
        "pci_scatter": PLOTS / "by_position" / "pci_vs_surplus_value_by_position.png",
        "tsne_possession": PLOTS / "tsne" / "possession.png",
        "tsne_sf": PLOTS / "tsne" / "by_position" / "SF.png",
        "full_partial": PLOTS / "by_position" / "surplus_value_correlations_by_position_partial.png",
    })
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReportDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            if style_name in {"Heading1", "Heading2"}:
                level = 0 if style_name == "Heading1" else 1
                text = flowable.getPlainText()
                key = f"toc-{level}-{self.page}-{abs(hash(text))}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=level, closed=False)
                if style_name == "Heading1":
                    self.notify("TOCEntry", (0, text, self.page, key))


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}
    styles["Body"] = ParagraphStyle(
        "Body", parent=base["BodyText"], fontName="Arial", fontSize=10.5,
        leading=14.0, textColor=HexColor(INK), alignment=TA_LEFT,
        spaceBefore=0, spaceAfter=8,
    )
    styles["Small"] = ParagraphStyle(
        "Small", parent=styles["Body"], fontSize=8.5, leading=11,
        textColor=HexColor(MUTED), spaceAfter=5,
    )
    styles["Caption"] = ParagraphStyle(
        "Caption", parent=styles["Small"], fontSize=8.5, leading=11,
        textColor=HexColor("#4E5863"), alignment=TA_LEFT,
        spaceBefore=5, spaceAfter=10,
    )
    styles["Heading1"] = ParagraphStyle(
        "Heading1", parent=base["Heading1"], fontName="Arial-Bold",
        fontSize=16, leading=19, textColor=HexColor(BLUE),
        spaceBefore=18, spaceAfter=10, keepWithNext=True,
    )
    styles["Heading2"] = ParagraphStyle(
        "Heading2", parent=base["Heading2"], fontName="Arial-Bold",
        fontSize=13, leading=16, textColor=HexColor(BLUE),
        spaceBefore=12, spaceAfter=6, keepWithNext=True,
    )
    styles["Heading3"] = ParagraphStyle(
        "Heading3", parent=base["Heading3"], fontName="Arial-Bold",
        fontSize=11.5, leading=14, textColor=HexColor(DARK_BLUE),
        spaceBefore=8, spaceAfter=4, keepWithNext=True,
    )
    styles["Title"] = ParagraphStyle(
        "ReportTitle", parent=base["Title"], fontName="Arial-Bold",
        fontSize=28, leading=32, textColor=HexColor(NAVY), alignment=TA_CENTER,
        spaceBefore=0, spaceAfter=10,
    )
    styles["Subtitle"] = ParagraphStyle(
        "Subtitle", parent=styles["Body"], fontName="Arial",
        fontSize=14, leading=18, textColor=HexColor(DARK_BLUE),
        alignment=TA_CENTER, spaceAfter=18,
    )
    styles["Kicker"] = ParagraphStyle(
        "Kicker", parent=styles["Small"], fontName="Arial-Bold",
        fontSize=9.5, leading=12, textColor=HexColor(GOLD),
        alignment=TA_CENTER, spaceAfter=14, uppercase=True,
    )
    styles["CoverMeta"] = ParagraphStyle(
        "CoverMeta", parent=styles["Small"], fontSize=9.5, leading=13,
        alignment=TA_CENTER, textColor=HexColor(MUTED), spaceAfter=4,
    )
    styles["Callout"] = ParagraphStyle(
        "Callout", parent=styles["Body"], fontSize=10.2, leading=13.6,
        textColor=HexColor(NAVY), spaceAfter=0,
    )
    styles["TableCell"] = ParagraphStyle(
        "TableCell", parent=styles["Body"], fontSize=8.4, leading=10.4,
        spaceAfter=0,
    )
    styles["TableHeader"] = ParagraphStyle(
        "TableHeader", parent=styles["TableCell"], fontName="Arial-Bold",
        textColor=colors.white, alignment=TA_CENTER,
    )
    styles["Reference"] = ParagraphStyle(
        "Reference", parent=styles["Small"], leftIndent=14, firstLineIndent=-14,
        fontSize=8.3, leading=10.8, spaceAfter=5,
    )
    styles["Code"] = ParagraphStyle(
        "Code", parent=styles["Small"], fontName="Courier", fontSize=7.8,
        leading=10, textColor=HexColor(INK), leftIndent=8,
    )
    styles["TOCHeading"] = ParagraphStyle(
        "TOCHeading", parent=styles["Heading1"], alignment=TA_LEFT,
        spaceBefore=0,
    )
    return styles


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullet_list(items: list[str], styles: dict[str, ParagraphStyle]) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, styles["Body"]), leftIndent=18) for item in items],
        bulletType="bullet", start="circle", leftIndent=23, bulletFontName="Arial",
        bulletFontSize=7, bulletOffsetY=2, spaceAfter=8,
    )


def callout(text: str, styles: dict[str, ParagraphStyle], accent: str = BLUE) -> Table:
    table = Table([[p(text, styles["Callout"])]], colWidths=[6.5 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor(LIGHT)),
        ("BOX", (0, 0), (-1, -1), 0.6, HexColor(GRID)),
        ("LINEBEFORE", (0, 0), (0, -1), 4, HexColor(accent)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return table


def data_table(rows: list[list[object]], widths: list[float],
               styles: dict[str, ParagraphStyle], header: bool = True) -> Table:
    formatted = []
    for r_idx, row in enumerate(rows):
        row_style = styles["TableHeader"] if header and r_idx == 0 else styles["TableCell"]
        formatted.append([
            value if hasattr(value, "wrapOn") else p(escape(str(value)), row_style)
            for value in row
        ])
    table = Table(formatted, colWidths=[w * inch for w in widths],
                  repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(NAVY)) if header else
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, HexColor(GRID)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        for idx in range(1, len(rows)):
            if idx % 2 == 0:
                commands.append(("BACKGROUND", (0, idx), (-1, idx), HexColor("#F8FAFC")))
    table.setStyle(TableStyle(commands))
    return table


def image_flowable(path: Path, max_width: float = 6.5, max_height: float = 6.8) -> Image:
    with PILImage.open(path) as im:
        width_px, height_px = im.size
    ratio = min((max_width * inch) / width_px, (max_height * inch) / height_px)
    return Image(str(path), width=width_px * ratio, height=height_px * ratio)


def figure(path: Path, caption: str, styles: dict[str, ParagraphStyle],
           max_width: float = 6.5, max_height: float = 6.8) -> KeepTogether:
    image = image_flowable(path, max_width=max_width, max_height=max_height)
    return KeepTogether([
        image,
        p(caption, styles["Caption"]),
    ])


def fmt_r(value: float) -> str:
    return f"{value:.3f}".replace("-0.", "-.").replace("0.", ".")


def draw_header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, LETTER[0], LETTER[1], fill=1, stroke=0)
    canvas.setStrokeColor(HexColor(GRID))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, LETTER[1] - 0.58 * inch,
                LETTER[0] - doc.rightMargin, LETTER[1] - 0.58 * inch)
    canvas.setFont("Arial", 7.8)
    canvas.setFillColor(HexColor(MUTED))
    canvas.drawString(doc.leftMargin, LETTER[1] - 0.48 * inch,
                      "NBA Draft Possession-Economics Study")
    canvas.drawRightString(LETTER[0] - doc.rightMargin, 0.43 * inch,
                           f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def draw_cover(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, LETTER[0], LETTER[1], fill=1, stroke=0)
    canvas.setFillColor(HexColor(NAVY))
    canvas.rect(0, LETTER[1] - 0.18 * inch, LETTER[0], 0.18 * inch, fill=1, stroke=0)
    canvas.setTitle("NBA Draft Possession-Economics and Position Study")
    canvas.setAuthor("NBA Draft Surplus Value Project")
    canvas.setSubject("Position-stratified correlations, grouped nested validation, and t-SNE")
    canvas.restoreState()


def build_pdf(tables: dict[str, pd.DataFrame], figures: dict[str, Path]) -> Path:
    register_fonts()
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    styles = build_styles()

    doc = ReportDocTemplate(
        str(PDF_PATH), pagesize=LETTER,
        leftMargin=1.0 * inch, rightMargin=1.0 * inch,
        topMargin=0.78 * inch, bottomMargin=0.72 * inch,
        title="Do College Possession-Economics Indicators Add Signal Beyond Draft Position?",
        author="NBA Draft Surplus Value Project",
    )
    story = []

    # Cover: editorial_cover pattern, resolved with an Arial font override.
    story.extend([
        Spacer(1, 0.72 * inch),
        p("SCIENTIFIC REPORT", styles["Kicker"]),
        p("Do College Possession-Economics Indicators Add Signal Beyond Draft Position?",
          styles["Title"]),
        p("A position-stratified study of NBA draft classes, 2009-2023",
          styles["Subtitle"]),
        HRFlowable(width="66%", thickness=1.2, color=HexColor(GOLD),
                   spaceBefore=4, spaceAfter=18, hAlign="CENTER"),
        p("Grouped nested validation and exploratory t-SNE maps of player archetypes",
          styles["CoverMeta"]),
        p(f"Analysis date: {REPORT_DATE} | Reproducible local data and code",
          styles["CoverMeta"]),
        Spacer(1, 0.34 * inch),
        callout(
            "<b>Primary conclusion.</b> Possession-related college indicators contain a modest, "
            "position-dependent descriptive signal. They have not yet shown a reliable, practically "
            "large improvement beyond the information already summarized by draft position.",
            styles, accent=GOLD,
        ),
        Spacer(1, 0.32 * inch),
        data_table([
            ["Cohort", "Descriptive PCI result", "Confirmatory result"],
            ["896 drafted; 682 college matched",
             "Pooled surplus r=.185",
             "Possession MAE gain=.0035; q=.569"],
        ], [2.05, 2.05, 2.40], styles),
        Spacer(1, 0.26 * inch),
        p("This report distinguishes association, prediction, and visualization. It does not make "
          "causal claims or treat t-SNE clusters as talent tiers.", styles["CoverMeta"]),
        PageBreak(),
    ])

    # Contents.
    story.append(p("Contents", styles["TOCHeading"]))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC1", fontName="Arial-Bold", fontSize=10.5, leading=15,
                       leftIndent=0, firstLineIndent=0, textColor=HexColor(NAVY),
                       spaceBefore=5),
        ParagraphStyle("TOC2", fontName="Arial", fontSize=9, leading=13,
                       leftIndent=18, firstLineIndent=0, textColor=HexColor(MUTED)),
    ]
    story.extend([toc, PageBreak()])

    # Executive summary.
    story.append(p("Executive abstract", styles["Heading1"]))
    story.append(p(
        "This retrospective study tests whether possession-related college statistics add "
        "information beyond NBA draft position. The cohort contains 896 players drafted from "
        "2009 through 2023; 682 were matched to final pre-draft-season BartTorvik profiles. "
        "The fixed Possession Creation Index (PCI) was weakly associated with exploratory draft "
        "surplus (r=.185, 95% CI .112 to .257). The strongest position-specific point estimate "
        "was among small forwards (r=.245). In grouped nested evaluation of first-three-year "
        "VORP, adding the broader possession family raised held-out correlation from .351 to .417 "
        "but reduced MAE by only .0035 VORP. The draft-year interval crossed zero (-.0065 to "
        ".0129; q=.569). Broad impact/offense measures supplied the clearest, though still small, "
        "unique early-career contribution. Position audits suggested that college information "
        "helped most for shooting guards, small forwards, and power forwards. t-SNE preserved "
        "local same-position neighborhoods but showed weak all-feature cluster separation. The "
        "evidence supports a modest role-dependent possession-economics signal, not a strong "
        "universal draft-market inefficiency.",
        styles["Body"],
    ))
    story.append(callout(
        "<b>How to read the evidence.</b> Correlations and surplus-value graphs are exploratory. "
        "The leave-one-draft-year-out first-three-year VORP analysis is the main predictive test. "
        "t-SNE is an exploratory similarity display.", styles,
    ))
    story.append(Spacer(1, 10))
    story.append(data_table([
        ["Finding", "Evidence", "Interpretation"],
        ["Small descriptive possession signal",
         "PCI: pooled r=.185; within-position r=.173",
         "Positive, but only about 3-4% bivariate variance."],
        ["Role-specific texture",
         "PCI r: PG .139; SG .186; SF .245; PF .199; C .075",
         "Point estimates differ; no formal interaction test was run."],
        ["No reliable incremental prediction",
         "Possession MAE gain=.0035; 95% year CI crosses 0; q=.569",
         "Holds out entire draft classes and controls preprocessing leakage."],
        ["Broader impact is more useful",
         "Impact/offense unique ablation benefit q=.029",
         "The detectable effect is small and fails the five-year sensitivity test."],
        ["Useful similarity maps, weak natural tiers",
         "Position possession silhouettes .236-.259; high local trust",
         "Use for comparables, not star/bust ranking."],
    ], [1.55, 2.35, 2.60], styles))
    story.extend([Spacer(1, 8), figure(
        figures["workflow"],
        "<b>Figure 1.</b> Analysis workflow. The confirmatory branch predicts a common-horizon "
        "NBA outcome while the position and t-SNE branches remain descriptive and exploratory.",
        styles, max_height=2.25,
    )])

    # Research framing.
    story.append(p("1. Research question and scientific method", styles["Heading1"]))
    story.append(p("1.1 Observation", styles["Heading2"]))
    story.append(p(
        "Prospect discussion often emphasizes scoring volume and shot creation. A competing "
        "basketball hypothesis is that gaining extra opportunities, avoiding lost opportunities, "
        "and converting possessions productively may be underweighted. Draft position already "
        "summarizes much of the league's information, so the stringent question is not whether "
        "college possession statistics correlate with NBA value. It is whether they add useful "
        "signal after draft position is known.", styles["Body"],
    ))
    story.append(p("1.2 Question", styles["Heading2"]))
    story.append(callout(
        "Do possession-economics indicators from a player's final pre-draft college season "
        "predict NBA value or draft surplus, and does that relationship vary across PG, SG, SF, "
        "PF, and C roles?", styles, accent=GOLD,
    ))
    story.append(p("1.3 Hypotheses and decision rules", styles["Heading2"]))
    story.append(data_table([
        ["Hypothesis", "Pre-specified test in this report", "Outcome"],
        ["H1: PCI is positively associated with surplus within role.",
         "Position-specific Pearson and pick-adjusted partial r with BH q-values.",
         "Partially supported descriptively."],
        ["H2: PCI or the possession family improves prediction beyond pick.",
         "Grouped nested CV; positive year-level MAE gain; corrected q < .05.",
         "Not supported."],
        ["H3: possession effects differ by position.",
         "Compare position estimates and uncertainty; formal interaction needed for confirmation.",
         "Suggestive pattern only."],
        ["H4: same-position prospects form coherent feature archetypes.",
         "Source-space silhouette plus t-SNE trustworthiness.",
         "Local comparables supported; crisp tiers not supported."],
    ], [1.65, 3.05, 1.80], styles))
    story.append(p(
        "The project was not preregistered. H1-H4 organize the completed analyses rather than "
        "constituting a prospective preregistration. Consequently, confirmatory language is "
        "reserved for the leakage-safe cross-validation design and its corrected tests.",
        styles["Small"],
    ))

    # Data and definitions.
    story.append(p("2. Data and cohort construction", styles["Heading1"]))
    story.append(p("2.1 Sources and coverage", styles["Heading2"]))
    story.append(p(
        "Basketball Reference supplies draft history, NBA player identifiers, season positions, "
        "and VORP. BartTorvik supplies final pre-draft-season college box-score and advanced "
        "statistics. The frozen local cohort covers the 2009-2023 draft classes. The full draft "
        "table has 896 unique draft-year and pick records; 682 players (76.1%) match to college "
        "data and constitute the main analytic sample.", styles["Body"],
    ))
    story.append(figure(
        figures["cohort"],
        "<b>Figure 2.</b> Cohort coverage by role. Solid segments are the 682 college-matched "
        "players used in correlation, prediction, and t-SNE analyses. Hatched segments lack a "
        "matched final college profile; the 53 unassigned players are outside the analytic sample.",
        styles, max_height=3.65,
    ))
    story.append(data_table([
        ["Position", "Full cohort", "College matched", "Direct NBA-entry label", "College-model fallback"],
        ["PG", "148", "126", "119", "7"],
        ["SG", "217", "188", "181", "7"],
        ["SF", "160", "119", "113", "6"],
        ["PF", "196", "163", "151", "12"],
        ["C", "122", "86", "83", "3"],
        ["Unassigned", "53", "0", "0", "0"],
        ["Total", "896", "682", "647", "35"],
    ], [1.15, 1.05, 1.20, 1.60, 1.50], styles))
    story.append(p("2.2 Identity and position controls", styles["Heading2"]))
    story.append(p(
        "NBA season joins retain Basketball Reference player IDs, prefer combined traded-team "
        "rows, and require the NBA career to begin after the draft year. These rules prevent "
        "same-name collisions, including the two Marcus Thornton and two Justin Jackson records. "
        "The preferred position is the first recorded NBA position in a season with games played. "
        "Thirty-five college-matched players without an NBA appearance receive a regularized "
        "five-position estimate from pre-NBA height, BartTorvik role, and college statistics. "
        "Those probabilities rank ambiguity but are not calibrated.", styles["Body"],
    ))
    story.append(callout(
        "<b>Leakage warning.</b> First-NBA-season position is post-draft information. It is used "
        "only to describe historical role differences and audit predictions. It is never passed "
        "into the leakage-safe predictive model.", styles, accent=RED,
    ))

    story.append(p("3. Operational definitions", styles["Heading1"]))
    story.append(p("3.1 What 'possession' means here", styles["Heading2"]))
    story.append(callout(
        "<b>Possession does not mean pace, touches, or time of possession.</b> It denotes pre-NBA "
        "indicators associated with gaining opportunities, retaining them, or converting them "
        "productively. Steals and offensive rebounds gain opportunities; turnovers surrender "
        "them; assists proxy playmaking and shot creation but do not literally create an extra "
        "team possession.", styles, accent=ORANGE,
    ))
    story.append(p("Fixed Possession Creation Index (PCI)", styles["Heading3"]))
    story.append(p(
        "PCI = 1.0 x z(STL%) + 0.8 x z(ORB%) - 1.0 x z(TO%) + 0.6 x z(AST%). "
        "The weights are theory-driven. In descriptive files the z-scores use the analytic sample; "
        "in cross-validation, medians, means, standard deviations, and PCI construction are fitted "
        "only inside the training fold.", styles["Body"],
    ))
    story.append(p("Broader possession family", styles["Heading3"]))
    story.append(p(
        "The model and t-SNE possession family contains ORB%, AST%, TO%, STL%, assist-to-turnover "
        "ratio, offensive rebounds, assists, and steals. PCI is therefore a compact theory score, "
        "while the possession family lets ridge regression learn how much each related measure "
        "contributes in training data.", styles["Body"],
    ))
    story.append(p("3.2 Outcomes", styles["Heading2"]))
    story.append(data_table([
        ["Outcome", "Definition", "Role in inference"],
        ["First-3yr VORP", "Mean VORP in first three post-draft seasons; missing seasons entered as zero.",
         "Primary common-horizon predictive outcome; n=682 across 15 drafts."],
        ["Best-3yr VORP in first five", "Best three-season average within first five post-draft seasons.",
         "Sensitivity outcome; n=596, 2009-2021 drafts."],
        ["Peak 3yr VORP", "Best three consecutive NBA seasons across locally observed career.",
         "Exploratory; older cohorts have a longer opportunity window."],
        ["Draft surplus", "Peak 3yr VORP minus full-sample average peak VORP for the exact draft slot.",
         "Exploratory descriptive target; not used in grouped prediction."],
    ], [1.45, 2.75, 2.30], styles))
    story.append(figure(
        figures["draft_curve"],
        "<b>Figure 3.</b> Observed peak three-year VORP by exact draft position. The noisy slot mean "
        "establishes the exploratory surplus-value baseline; it is not a causal or smoothed value "
        "curve and is not used as the cross-validation target.",
        styles, max_height=3.25,
    ))

    # Methods.
    story.append(p("4. Statistical methods", styles["Heading1"]))
    story.append(p("4.1 Descriptive correlations", styles["Heading2"]))
    story.append(p(
        "For each target and each of PG, SG, SF, PF, C, pooled, and within-position-centered scopes, "
        "the analysis reports Pearson r, a Fisher-transformed 95% confidence interval, Spearman r, "
        "and partial r after residualizing both feature and outcome against a quadratic draft-pick "
        "basis. Benjamini-Hochberg false-discovery correction is applied separately within each "
        "position-target set across the available feature comparisons. A result is labeled "
        "detectable when q &lt; .05. Separate significance in one position but not another is not, by "
        "itself, evidence of a statistically significant between-position difference.",
        styles["Body"],
    ))
    story.append(p("4.2 Grouped nested prediction", styles["Heading2"]))
    story.append(bullet_list([
        "Outer loop: leave one complete draft class out for testing.",
        "Inner loop: five-fold GroupKFold over remaining draft years selects ridge alpha from 0.01 to 1000.",
        "Fold-local processing: draft-pick spline, median imputation, missing indicators, scaling, and fixed PCI standardization are learned on training rows only.",
        "Primary benchmark: a nonlinear draft-pick-only model. Candidate models add fixed PCI, one basketball feature family, or all college features.",
        "Inference: draft-year MAE differences, a 20,000-repeat year-cluster bootstrap interval, a one-sided exact sign-flip p-value, and BH correction within each comparison family.",
    ], styles))
    story.append(p(
        "Raw VORP is predicted directly. Stored surplus value is not used because its expected-pick "
        "curve was calculated from the full sample. The subgroup position table evaluates the same "
        "pooled out-of-fold predictions within each role; it does not fit five independent role "
        "models or provide subgroup hypothesis tests.", styles["Body"],
    ))
    story.append(p("4.3 t-SNE, clustering, and nearest neighbors", styles["Heading2"]))
    story.append(p(
        "For each feature family, missing values are median-imputed with missing indicators and all "
        "dimensions are standardized. K-means selects k from 2 through 8 by the highest silhouette "
        "score in standardized source-feature space, with 25 initializations. Nearest neighbors are "
        "also calculated in that source space. t-SNE then displays two dimensions using PCA "
        "initialization, automatic learning rate, seed 42, and perplexity 30 subject to the sample-"
        "size cap. Trustworthiness measures local-neighborhood preservation. Every position-only map "
        "is refit within a single role, preventing basic guard-center size differences from "
        "dominating the display.", styles["Body"],
    ))
    story.append(callout(
        "<b>Interpretation boundary.</b> t-SNE axes, rotations, and global distances have no direct "
        "basketball meaning. Cluster colors describe feature-space groups, not NBA success. High "
        "trustworthiness supports local visual comparison; it does not validate K-means or prove "
        "predictive value.", styles, accent=RED,
    ))

    # Results.
    story.append(p("5. Results", styles["Heading1"]))
    story.append(p("5.1 Descriptive possession associations", styles["Heading2"]))
    story.append(p(
        "Across all 682 matched players, PCI correlates .185 with exploratory draft surplus "
        "(95% CI .112 to .257; q&lt;.001). Centering both variables within position leaves r=.173 "
        "(.099 to .245; q&lt;.001), so the small pooled association is not explained only by mean "
        "differences between positions. The bivariate squared correlations are roughly 3-4%, "
        "which is evidence of a tendency rather than strong individual-player discrimination.",
        styles["Body"],
    ))
    story.append(figure(
        figures["pci_scatter"],
        "<b>Figure 4.</b> PCI and exploratory draft surplus within each primary position. Each line is "
        "a separate within-role fit. Only SF clears the Pearson FDR threshold; scatter remains wide "
        "at every role.", styles, max_height=2.15,
    ))
    story.append(figure(
        figures["pci_forest"],
        "<b>Figure 5.</b> PCI correlations with draft surplus. Pearson intervals quantify sampling "
        "uncertainty; orange diamonds show draft-pick-adjusted partial correlations. SF is detectable "
        "for both estimates; PF becomes detectable after pick adjustment. This does not constitute a "
        "formal test that SF or PF differs from another position.", styles, max_height=3.85,
    ))
    story.append(data_table([
        ["Position", "n", "PCI r (95% CI)", "Pearson q", "Pick-adjusted partial r", "Partial q"],
        ["PG", "126", ".139 (-.037, .306)", ".290", ".106", ".603"],
        ["SG", "188", ".186 (.044, .321)", ".091", ".188", ".084"],
        ["SF", "119", ".245 (.068, .407)", ".037", ".252", ".027"],
        ["PF", "163", ".199 (.046, .342)", ".086", ".223", ".031"],
        ["C", "86", ".075 (-.139, .283)", ".728", ".081", ".682"],
    ], [0.72, 0.52, 1.65, 0.95, 1.65, 1.01], styles))

    story.append(p("5.2 Position-specific texture beyond PCI", styles["Heading2"]))
    story.append(figure(
        figures["selected_partial"],
        "<b>Figure 6.</b> Selected feature-surplus partial correlations after controlling quadratic "
        "draft pick. GPI is included for context but was fitted on this cohort and is not held-out "
        "evidence. The compact panel replaces the denser full heatmap in the main narrative.",
        styles, max_height=4.25,
    ))
    story.append(data_table([
        ["Role", "Most defensible descriptive pattern", "What not to infer"],
        ["PG", "General impact and efficiency (GPI, GBPM, TS%, BPM) exceed PCI.",
         "No proof that playmaking is unimportant."],
        ["SG", "GPI and BPM lead; broad college information improves held-out ranking.",
         "No isolated scoring score is reliable."],
        ["SF", "Defensive rebounding, offensive rebounding, and PCI form the clearest possession-related pattern.",
         "No causal role mechanism is established."],
        ["PF", "Two-way impact and PCI retain modest pick-adjusted relationships.",
         "Effect size remains small and subgroup prediction is descriptive."],
        ["C", "Aggregate PCI is weak; steal rate is the more promising disruption measure.",
         "Mobility was not measured, so it should not be claimed."],
    ], [0.55, 3.45, 2.50], styles))
    story.append(p(
        "The Granular Prospect Index is positive at every position (r=.260 to .466), but its weights "
        "were fitted on the same 2009-2023 cohort and outcome. Those correlations are useful for "
        "description only. The threshold-based scoring profile is weak: pooled surplus r=.066, "
        "within-position r=.043, and no position clears FDR.", styles["Body"],
    ))

    story.append(p("5.3 Main predictive test: unseen draft classes", styles["Heading2"]))
    story.append(data_table([
        ["Model", "MAE", "RMSE", "Held-out r", "Pooled MAE gain vs pick", "Adjusted q"],
        ["Draft pick only", ".3824", ".5942", ".351", "-", "-"],
        ["Pick + fixed PCI", ".3802", ".5807", ".407", ".0022", ".569"],
        ["Pick + possession", ".3789", ".5778", ".417", ".0035", ".569"],
        ["Pick + scoring", ".3874", ".5927", ".358", "-.0049", ".925"],
        ["Pick + impact/offense", ".3742", ".5679", ".449", ".0083", ".512"],
        ["Pick + defense/size", ".3809", ".5811", ".402", ".0016", ".569"],
        ["Pick + context/age", ".3861", ".5944", ".350", "-.0036", ".925"],
        ["Pick + all college", ".3749", ".5689", ".449", ".0076", ".512"],
    ], [1.50, 0.70, 0.72, 0.88, 1.45, 0.95], styles))
    story.append(figure(
        figures["cv_forest"],
        "<b>Figure 7.</b> Draft-year-level MAE changes for first-three-year VORP. Every interval "
        "crosses zero and no addition survives FDR correction. The possession family improves 10 "
        "of 15 held-out years, but its mean gain is less than 1% of baseline MAE.",
        styles, max_height=4.1,
    ))
    story.append(p(
        "The central possession result is therefore a change in ranking agreement more than a "
        "meaningful reduction in typical error. Fixed PCI raises held-out r from .351 to .407, and "
        "the broader possession family raises it to .417. Yet the possession model's pooled MAE "
        "gain is .0035, with a year-clustered 95% interval of -.0065 to .0129 and q=.569. The data "
        "remain compatible with a small benefit, no effect, or a small cost in a new draft class.",
        styles["Body"],
    ))
    story.append(p("5.4 Unique contribution and sensitivity", styles["Heading2"]))
    story.append(p(
        "In leave-one-family-out ablation of the full model, restoring impact/offense provides a "
        "small detectable benefit: mean year-level MAE gain .0057 (95% CI .0020 to .0094; q=.029). "
        "Restoring possession is suggestive but does not meet the corrected threshold: gain .0043 "
        "(-.0009 to .0085; q=.092). On the best-three-year-within-first-five sensitivity outcome, "
        "no addition or ablation survives correction. The full model is slightly worse than pick "
        "alone on MAE, so the impact finding is outcome-specific rather than universal.",
        styles["Body"],
    ))
    story.append(p("5.5 Position audit of held-out predictions", styles["Heading2"]))
    story.append(figure(
        figures["position_oof"],
        "<b>Figure 8.</b> Descriptive role audit of the pooled out-of-fold models. Possession lowers "
        "MAE most for SF and PF and worsens it for PG; the all-feature model improves SG, SF, and PF "
        "but not PG or C. These subgroup differences have not been inferentially tested.",
        styles, max_height=3.4,
    ))
    story.append(p(
        "For the all-feature model, held-out r changes from .191 to .341 for SG, .124 to .330 for "
        "SF, and .409 to .577 for PF. It changes from .530 to .519 for PG and from .400 to .344 for "
        "C. The possession family specifically lowers MAE from .3862 to .3542 for SF and from .3518 "
        "to .3360 for PF, while it is nearly flat for SG and C and worse for PG. This pattern is a "
        "useful diagnostic for future interaction models, not evidence that five separate models "
        "would generalize.", styles["Body"],
    ))

    story.append(p("5.6 What t-SNE reveals", styles["Heading2"]))
    story.append(figure(
        figures["tsne_possession"],
        "<b>Figure 9.</b> Pooled possession-family t-SNE (n=682). K-means selects two source-space "
        "clusters (silhouette=.328), and the 2D display has trustworthiness=.977. The labeled "
        "players are cluster representatives, not best or worst prospects.",
        styles, max_height=4.85,
    ))
    story.append(p(
        "The map shows smooth neighborhoods of prospects with similar possession profiles. It does "
        "not reveal a clean boundary between NBA stars and misses because NBA outcomes are absent "
        "from both clustering and t-SNE. Position-only possession silhouettes are .236 to .259, "
        "which is modest. All-feature silhouettes are weaker (.138 to .190), indicating that the "
        "full college profile does not break into crisp natural prospect classes within role.",
        styles["Body"],
    ))
    story.append(figure(
        figures["tsne_quality"],
        "<b>Figure 10.</b> Source-space cluster separation and t-SNE local preservation by position. "
        "Impact/offense and context/age separate more clearly than possession or all features. "
        "Context/age clusters likely reflect experience, minutes, and age rather than latent talent.",
        styles, max_height=4.35,
    ))
    story.append(figure(
        figures["tsne_sf"],
        "<b>Figure 11.</b> Small-forward gallery refitted entirely within SF. It illustrates why "
        "one prospect can have different nearest statistical neighborhoods in possession, scoring, "
        "impact, defense/size, context/age, and all-feature spaces. The axes cannot be compared across "
        "panels.", styles, max_height=4.0,
    ))

    # Discussion.
    story.append(p("6. Discussion", styles["Heading1"]))
    story.append(p("6.1 What the evidence adds", styles["Heading2"]))
    story.append(bullet_list([
        "Position splitting reduces role confusion. The pooled PCI signal remains after within-position centering, but its strength is not uniform in the point estimates.",
        "Possession is more informative as a family of behaviors than as a single universal score. SF and PF deserve the clearest follow-up; PG and C require different feature definitions or interactions.",
        "Draft position is a formidable baseline because it already embeds scouting, measurements, medical information, and team beliefs. Small gains beyond it can be real while still being operationally modest.",
        "General impact/offense measures carry the strongest small early-career signal in the current validation. This is not the same as saying scoring volume is the answer; the threshold scoring profile performs poorly.",
        "t-SNE adds a retrieval tool: analysts can find same-position statistical comparables and inspect which family makes them similar. It does not add a prospect ranking.",
    ], styles))
    story.append(p("6.2 Updated answer to the original hypothesis", styles["Heading2"]))
    story.append(callout(
        "The evidence does not justify the claim that the NBA draft market systematically "
        "undervalues possession creators at every position. A narrower statement is supported: "
        "possession-related college indicators show a modest association with later surplus, and "
        "the most promising descriptive patterns occur among wings and power forwards. Their "
        "incremental predictive value beyond draft position remains uncertain.",
        styles, accent=GOLD,
    ))
    story.append(p("6.3 Practical use", styles["Heading2"]))
    story.append(data_table([
        ["Appropriate use", "Inappropriate use"],
        ["Flag SF/PF prospects whose possession profiles are unusual for their position.",
         "Automatically move a prospect up the draft board because PCI is high."],
        ["Retrieve same-position comparables from standardized source-feature neighbors.",
         "Treat t-SNE distance as a calibrated similarity score or talent gap."],
        ["Use PCI as one component in a broader, prospectively validated model.",
         "Present GPI's in-sample correlation as out-of-sample proof."],
        ["Audit model error and ranking by role after pooled validation.",
         "Use first-NBA-season position as a pre-draft predictor."],
    ], [3.25, 3.25], styles))

    # Limitations.
    story.append(p("7. Shortfalls and threats to validity", styles["Heading1"]))
    story.append(p("7.1 Data and outcome limitations", styles["Heading2"]))
    story.append(bullet_list([
        "The study is observational and cannot show that possession traits cause NBA success or that teams ignored them.",
        "Only drafted, college-matched players enter the main analysis. International players and unmatched prospects create selection bias and limit transportability.",
        "VORP is box-score based and opportunity dependent. It reflects health, role, coaching, team context, and minutes as well as ability.",
        "Zero-filled early seasons combine lack of ability with injury, stashing, roster access, and development choices.",
        "Exploratory peak VORP gives older cohorts a longer search window, and exact-slot surplus uses an expected curve estimated from the same complete sample.",
    ], styles))
    story.append(p("7.2 Position and model limitations", styles["Heading2"]))
    story.append(bullet_list([
        "Historical entry position is post-draft and collapses hybrid roles and role changes into one label.",
        "The 35 college-model position probabilities are not calibrated; centers have only 86 matched observations.",
        "PCI weights are theory-driven and have not been tuned or validated on an external cohort.",
        "GPI is fitted on this cohort and cannot serve as held-out evidence.",
        "The outer loop holds out draft classes but can train on seasons later than the test class. Rolling-origin validation would more closely simulate real-time forecasting.",
        "Position differences are descriptive; formal feature-by-position interaction tests and prospectively fitted role models remain future work.",
    ], styles))
    story.append(p("7.3 Multiplicity and visualization limitations", styles["Heading2"]))
    story.append(bullet_list([
        "BH correction is local to each target and analysis family; the wider project remains exploratory and was not preregistered.",
        "t-SNE can change with scaling, imputation, seed, perplexity, and sample composition. No multi-seed or multi-perplexity stability analysis is reported.",
        "K-means favors roughly spherical source-space groups, and choosing k by the same sample's silhouette can overstate apparent separation.",
        "High t-SNE trustworthiness validates only local neighborhood retention. It does not validate clusters, outcomes, or causality.",
    ], styles))

    # Conclusion and future work.
    story.append(p("8. Conclusions and next experiments", styles["Heading1"]))
    story.append(p(
        "Possession-economics indicators pass the weakest test: they have a positive descriptive "
        "relationship with later draft surplus in the pooled and within-position data. The signal "
        "is small and appears most promising for small forwards and power forwards. They do not yet "
        "pass the stronger test of producing a reliable, practically large improvement beyond draft "
        "position in unseen draft classes. Broader impact/offense variables show the clearest small "
        "unique early-career contribution, while threshold scoring profiles add little. t-SNE "
        "reveals local statistical archetypes and useful same-position comparables, not stable talent "
        "tiers.", styles["Body"],
    ))
    story.append(callout(
        "<b>Final answer.</b> Possession-related college indicators appear to contain a modest, "
        "position-dependent signal, but they have not yet demonstrated a reliable, practically "
        "large improvement beyond information already summarized by draft position.",
        styles, accent=GREEN,
    ))
    story.append(CondPageBreak(4.0 * inch))
    story.append(p("Priority follow-up studies", styles["Heading2"]))
    story.append(bullet_list([
        "Freeze PCI and every analytic choice, then evaluate prospectively on later draft classes without refitting the formula.",
        "Use rolling-origin validation so every test class occurs strictly after all training classes.",
        "Add pre-draft measurements, age curves, injuries, role, league strength, and authoritative pre-draft positions inside each training fold.",
        "Fit penalized feature-by-position interactions and test the interaction terms directly instead of comparing separate significance labels.",
        "Run t-SNE and K-means stability checks across seeds, perplexities, resamples, and alternative clustering methods.",
        "Evaluate decision value: prospect rank lift, top-k hit rate, calibration, and scouting utility, not only global MAE and r.",
    ], styles))

    # Reproducibility.
    story.append(p("9. Reproducibility", styles["Heading1"]))
    story.append(p("9.1 Execution order", styles["Heading2"]))
    commands = [
        "python3 01_collect_draft_data.py",
        "python3 02_collect_college_stats.py",
        "python3 03_grouped_nested_cv.py",
        "python3 05_position_analysis.py",
        "python3 04_tsne_similarity_clusters.py --seed 42",
        "python3 plot.py",
        "python3 new.py",
        "python3 06_build_scientific_report.py",
    ]
    command_table = Table([[p(escape(command), styles["Code"])] for command in commands],
                          colWidths=[6.5 * inch])
    command_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor("#F7F9FB")),
        ("BOX", (0, 0), (-1, -1), 0.5, HexColor(GRID)),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, HexColor(GRID)),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(command_table)
    story.append(p("9.2 Frozen output checksums", styles["Heading2"]))
    checksum_files = [
        DATA / "draft_with_college.csv",
        DATA / "position_feature_correlations.csv",
        DATA / "grouped_nested_cv_model_metrics.csv",
        DATA / "grouped_nested_cv_comparisons.csv",
        DATA / "grouped_nested_cv_position_metrics.csv",
        DATA / "tsne_cluster_metrics.csv",
    ]
    checksum_rows = [["File", "SHA-256 (first 16 hex)"]]
    checksum_rows.extend([
        [str(path.relative_to(ROOT)), sha256(path)[:16]] for path in checksum_files
    ])
    story.append(data_table(checksum_rows, [4.65, 1.85], styles))
    story.append(p(
        "Random seed is 42. t-SNE defaults are perplexity 30 with a sample-size cap, PCA "
        "initialization, and 1,000 iterations. K-means searches k=2 through 8 with n_init=25. "
        "The predictive alpha grid is log-spaced from 0.01 through 1000. This exported folder does "
        "not contain Git repository metadata, so a commit hash cannot be reported.", styles["Small"],
    ))
    story.append(p("9.3 Validation checks completed", styles["Heading2"]))
    story.append(bullet_list([
        "896 unique draft-year and pick records in the full master table.",
        "682 college-matched analytic rows, all assigned exactly one primary position.",
        "808 direct position labels, 35 college-model fallbacks, and 53 explicitly unassigned full-cohort rows.",
        "No cross-position neighbors in position-only t-SNE output.",
        "All position-specific t-SNE perplexities are below the relevant sample size.",
        "Fold-local target construction, draft-pick baseline fitting, imputation, scaling, and PCI standardization in grouped nested CV.",
    ], styles))

    # Supplement.
    story.append(PageBreak())
    story.append(p("Supplement A. Complete selected correlation display", styles["Heading1"]))
    story.append(p(
        "The following full-resolution heatmap displays every available pre-NBA feature's partial "
        "correlation with exploratory draft surplus after controlling quadratic draft pick. Stars "
        "mark q&lt;.05 within the relevant position-target set. It is supplied for transparency rather "
        "than as a shortlist for model selection.", styles["Body"],
    ))
    story.append(figure(
        figures["full_partial"],
        "<b>Figure S1.</b> Full position-by-feature partial-correlation matrix. Read row patterns "
        "alongside sample sizes and uncertainty in data/position_feature_correlations.csv.",
        styles, max_width=4.65, max_height=6.75,
    ))
    story.append(p("Supplement B. Data dictionary", styles["Heading1"]))
    story.append(data_table([
        ["Family", "Variables", "Interpretive role"],
        ["Possession", "ORB%, AST%, TO%, STL%, AST/TO, OREB, AST, STL",
         "Gain, preserve, and convert opportunities."],
        ["Scoring", "Usage, eFG%, TS%, FTM, FTA, FT%, 2P%, 3P%, FTr, points",
         "Volume, efficiency, and shot-result profile."],
        ["Impact/offense", "ORtg, PORPAG, adj. offensive efficiency, BPM, OBPM, GBPM, OGBPM",
         "Broad college productivity and impact."],
        ["Defense/size", "DRB%, BLK%, defensive ratings, DPORPAG, stops, DBPM, DGBPM, rebounds, blocks, height",
         "Defensive activity, rebounding, rim protection, and size."],
        ["Context/age", "Games, minutes share, height where modeled, age at draft",
         "Experience, opportunity, physical context, and development timing."],
        ["Market", "Draft pick",
         "Consensus baseline and adjustment variable."],
    ], [1.15, 3.20, 2.15], styles))
    story.append(p("Supplement C. References", styles["Heading1"]))
    references = [
        "1. Basketball Reference. NBA draft and player-season data. Local frozen extracts for draft classes 2009-2023 and NBA seasons through the available 2026 file.",
        "2. BartTorvik. College basketball advanced player statistics. Local frozen extracts for seasons 2009-2023.",
        "3. van der Maaten L, Hinton G. Visualizing Data using t-SNE. Journal of Machine Learning Research. 2008;9:2579-2605.",
        "4. Rousseeuw PJ. Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics. 1987;20:53-65.",
        "5. Davies DL, Bouldin DW. A Cluster Separation Measure. IEEE Transactions on Pattern Analysis and Machine Intelligence. 1979;PAMI-1(2):224-227.",
        "6. Benjamini Y, Hochberg Y. Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. Journal of the Royal Statistical Society, Series B. 1995;57(1):289-300.",
        "7. Hoerl AE, Kennard RW. Ridge Regression: Biased Estimation for Nonorthogonal Problems. Technometrics. 1970;12(1):55-67.",
        "8. Pedregosa F, et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research. 2011;12:2825-2830.",
    ]
    for reference in references:
        story.append(p(escape(reference), styles["Reference"]))
    story.append(Spacer(1, 12))
    story.append(p(
        "Report source files and complete machine-readable results are retained in the project "
        "folder. The scientific conclusions should be updated if any upstream data or analysis "
        "table changes.", styles["Small"],
    ))

    doc.multiBuild(story, onFirstPage=draw_cover, onLaterPages=draw_header_footer)
    return PDF_PATH


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/nba-draft-report-mpl")
    validate_inputs()
    tables = load_tables()
    figures = make_report_figures(tables)
    pdf_path = build_pdf(tables, figures)
    print(f"Saved {pdf_path}")
    for path in figures.values():
        if path.is_relative_to(FIGURE_DIR):
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
