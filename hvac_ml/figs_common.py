"""
figs_common.py - shared style, colours and drawing helpers for the report figures.
"""
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from hvac_ml import config

log = logging.getLogger(__name__)

BLUE, ORANGE, GREEN, PURPLE, GREY, NAVY, RED, GOLD = "#2E86AB", "#E07A1F", "#3A9D5D", "#8E44AD", "#8D99AE", "#1F4E79", "#C0392B", "#D4A017"
LIGHT = "#F4F6F8"
REPORT_FIG_DIR = config.FIG_DIR / "report"

# who owns which lifecycle stage (from the group's agreed split of the report)
OWNERS = {
    "Kriti": {"full": "Kriti Vajpayee", "enroll": "25931008", "stages": [1, 2], "color": PURPLE},
    "Dhruv": {"full": "Dhruv Aggarwal", "enroll": "26911004", "stages": [3, 4], "color": BLUE},
    "Ankit": {"full": "Ankit Kumar", "enroll": "26911002", "stages": [5, 6], "color": GREEN},
    "Animesh": {"full": "Animesh Basak", "enroll": "26911001", "stages": [7, 8], "color": ORANGE},
    "Saurav": {"full": "Saurav Kumar Boro", "enroll": "26535026", "stages": [9, 10], "color": RED},
}
STAGES = ["Problem definition", "Data collection", "Data cleaning and preprocessing", "Exploratory data analysis", "Feature engineering",
          "Model selection", "Model training", "Model evaluation and tuning", "Model deployment", "Monitoring and maintenance"]


def owner_of(stage):
    for key, o in OWNERS.items():
        if stage in o["stages"]:
            return key
    raise KeyError(stage)


plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"]})


def load_facts():
    return json.loads((config.RESULTS_DIR / "facts.json").read_text(encoding="utf-8"))


def load_summary():
    return pd.read_csv(config.RESULTS_DIR / "summary_mean_std.csv")


def val(summary, pillar, split, model, col):
    return float(summary[(summary["pillar"] == pillar) & (summary["split"] == split) & (summary["model"] == model)].iloc[0][f"{col}_mean"])


def sd(summary, pillar, split, model, col):
    return float(summary[(summary["pillar"] == pillar) & (summary["split"] == split) & (summary["model"] == model)].iloc[0][f"{col}_std"])


def save(fig, name, tight=True):
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(REPORT_FIG_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("  saved report figure %s", name)


# diagram helpers
def canvas(w, h, title=None, xlim=(0, 100), ylim=(0, 60)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY, pad=8)
    return fig, ax


def box(ax, x, y, w, h, text="", color=BLUE, fill="white", size=9, bold=False, text_color="#222222", lw=1.8, align="center", radius=1.0, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill, edgecolor=color, linewidth=lw, alpha=alpha))
    if text:
        if align == "center":
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=text_color, fontweight="bold" if bold else "normal", linespacing=1.3)
        else:
            ax.text(x + 1.0, y + h - 1.0, text, ha="left", va="top", fontsize=size, color=text_color, fontweight="bold" if bold else "normal", linespacing=1.35)


def header_box(ax, x, y, w, h, title, body, color, size=8.5, title_size=9.5, header_h=3.4):
    """Box with a coloured title strip and a text body."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", facecolor="white", edgecolor=color, linewidth=1.8))
    ax.add_patch(FancyBboxPatch((x, y + h - header_h), w, header_h, boxstyle="round,pad=0,rounding_size=0.8", facecolor=color, edgecolor=color))
    ax.text(x + w / 2, y + h - header_h / 2, title, ha="center", va="center", fontsize=title_size, color="white", fontweight="bold")
    ax.text(x + 1.0, y + h - header_h - 0.8, body, ha="left", va="top", fontsize=size, color="#222222", linespacing=1.35)


def arrow(ax, p0, p1, color=NAVY, lw=1.8, style="-|>", rad=0.0, mutation=14):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=color, lw=lw, mutation_scale=mutation, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


def table_figure(df, title=None, col_widths=None, font=9, header_color=NAVY, figsize=None, row_height=0.5, zebra=True, first_col_left=True, wrap=None):
    """Render a DataFrame as a clean picture of a table."""
    n_rows, n_cols = df.shape
    figsize = figsize or (min(14, 2.0 + 2.0 * n_cols), 0.9 + row_height * (n_rows + 1))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    cell_text = [[str(v) for v in row] for row in df.itertuples(index=False)]
    tbl = ax.table(cellText=cell_text, colLabels=list(df.columns), loc="center", cellLoc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font)
    tbl.scale(1, 1.0 + row_height * 1.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F4F6F8" if (zebra and r % 2 == 0) else "white")
            if c == 0 and first_col_left:
                cell.set_text_props(ha="left")
                cell._loc = "left"
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY, pad=6)
    return fig
