"""
generate_all_visualizations.py

Draws every figure in this folder. Charts read the saved results in
hvac_gml_project/artifacts/ (telemetry JSON and CSV files) and the season table
in hvac_gml_project/config.py, so the numbers match what was actually run.
The diagrams (02 to 05) describe the code in hvac_gml_project/.

Run:  python visualization/generate_all_visualizations.py
Needs: numpy, pandas, matplotlib
"""
import json
import os
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(OUT_DIR)
ART = os.path.join(ROOT_DIR, "hvac_gml_project", "artifacts")
DPI = 200
NAVY = "#1F4E79"
BLUE, GREEN, RED, ORANGE, PURPLE, GREY = "#2980B9", "#27AE60", "#C0392B", "#D35400", "#8E44AD", "#7F8C8D"

# ---------------------------------------------------------------- real data
with open(os.path.join(ART, "fine_tune_telemetry.json"), encoding="utf-8") as f:
    TEL = json.load(f)
PRE, POST = TEL["pre_fine_tune_metrics"], TEL["post_fine_tune_metrics"]
HIST = pd.DataFrame(TEL["training_history"])
FIN = pd.read_csv(os.path.join(ART, "fine_tuned_financial_metrics.csv"))
BASE = pd.read_csv(os.path.join(ART, "baseline_model_comparison.csv"), index_col=0)

# Season table copied out of config.py (importing config would need torch).
with open(os.path.join(ROOT_DIR, "hvac_gml_project", "config.py"), encoding="utf-8") as f:
    _cfg = f.read()
REGIME = {int(m): (n, float(c), float(h)) for m, n, _, c, h in
          re.findall(r'(\d+): \("([^"]+)", (\d), ([\d.]+), ([\d.]+)\)', _cfg)}
assert len(REGIME) == 12, "could not read the 12 months from config.py"


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(name, "created.")


def box(ax, x, y, w, h, title, lines=(), color=BLUE, fill="white", title_size=10.5, size=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.2",
                                facecolor=fill, edgecolor=color, linewidth=2))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=color)
    if lines:
        ax.text(x + 1.8, y + h - 6.5, "\n".join(lines), ha="left", va="top", fontsize=size,
                color="#222222", linespacing=1.45)


def arrow(ax, p0, p1, color=NAVY, lw=2.2, rad=0.0):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(
        arrowstyle="-|>", color=color, lw=lw, mutation_scale=16, shrinkA=2, shrinkB=2,
        connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY, pad=10)
    return fig, ax


# ================================================================ 01 lifecycle (copied)
src = os.path.join(ROOT_DIR, "docs", "ml_lifecycle", "ml_lifecycle.png")
if os.path.exists(src):
    shutil.copyfile(src, os.path.join(OUT_DIR, "01_ml_lifecycle_10_stages.png"))
    print("01_ml_lifecycle_10_stages.png created.")

# ================================================================ 02 graph schema
fig, ax = canvas(12, 7.8, "Graph schema: 4 node types, 3 relations (each also stored in reverse)")
month = "sin(month), cos(month)"
nodes = {
    "Location": (3, 58, BLUE, ["is_metropolitan", "base temperature / 45", "grid stability",
                               "population density", month, "cooling multiplier",
                               "heating multiplier"]),
    "Resident": (3, 4, ORANGE, ["is_metropolitan", "app active (0/1)", "dynamic engagement",
                                "tenure / 48 months", "income tier", month, "season regime / 2"]),
    "Device": (38, 29, GREEN, ["is_cooling_hvac", "unit price / 60000", "service margin / 8000",
                               "power draw / 3.5", "sensor failure probability", month,
                               "seasonal stress"]),
    "Part": (73, 29, PURPLE, ["for_cooling_hvac", "unit cost / 10000", "holding cost / 300",
                              "current stock / 150", month, "cooling multiplier",
                              "heating multiplier"]),
}
for name, (x, y, col, feats) in nodes.items():
    box(ax, x, y, 25, 34, f"{name} node  (8 features)", ["- " + f for f in feats], color=col, title_size=10)
arrow(ax, (10, 42), (10, 58), BLUE)
ax.text(11.5, 50, "located_in\n(reverse: has_resident)", fontsize=8.5, color=BLUE, va="center", fontweight="bold")
arrow(ax, (27, 24), (40, 30), GREEN)
ax.text(30, 21, "owns  (reverse: owned_by)", fontsize=8.5, color=GREEN, fontweight="bold")
arrow(ax, (62, 48), (73, 48), PURPLE)
ax.text(67.5, 52, "requires_part\n(reverse: used_in)", fontsize=8.5, color=PURPLE, ha="center", fontweight="bold")
ax.text(50, -1, "Month features (sin, cos, multipliers) are added to every node, so one model handles all 12 months.",
        ha="center", fontsize=9, color="#444444")
save(fig, "02_heterogeneous_graph_schema.png")

# ================================================================ 03 model architecture
fig, ax = canvas(13, 7.6, "MultiTaskHVACHeteroGNN (hetero_gnn_model.py)")
box(ax, 1, 24, 14, 58, "Inputs", ["Location x 8", "Resident x 8", "Device x 8", "Part x 8", "",
                                  "one graph per", "month (12)"], color=GREY)
box(ax, 19, 24, 16, 58, "Input layer", ["One Linear(8 -> 128)", "per node type,", "then LeakyReLU"], color=BLUE)
box(ax, 39, 24, 24, 58, "3 x HeteroConv", ["SAGEConv on all 6 relations", "(mean aggregation)", "",
                                           "after each layer:", "- LeakyReLU", "- residual connection",
                                           "- dropout 0.2", "", "hidden size 128"], color=GREEN)
for i, (name, node, col) in enumerate([("demand", "location", BLUE), ("power", "device", GREEN),
                                       ("failure", "device", RED), ("sleeper", "resident", ORANGE),
                                       ("inventory", "part", PURPLE)]):
    y = 82 - i * 13
    box(ax, 67, y - 10.5, 31, 10.5, f"head_{name}  ({node})", [f"Linear(128 -> 64), ReLU, Linear(64 -> 1)"],
        color=col, title_size=9.5, size=8)
    arrow(ax, (63, 53), (67, y - 5.5), "#888888", lw=1.4)
arrow(ax, (15, 53), (19, 53))
arrow(ax, (35, 53), (39, 53))
ax.text(33, 11, "Training loss has 4 terms: demand, maintenance (power + failure), churn, inventory.\n"
        "Fine-tuning weights them with learned uncertainty terms (Kendall et al., 2018).\n"
        "Cross-sell links: predict_cross_sell_links() takes a dot product of resident and device embeddings.",
        ha="center", va="center", fontsize=9, color="#333333", linespacing=1.5)
save(fig, "03_hetero_gnn_model_architecture.png")

# ================================================================ 04 pipeline
red_loss = (1 - POST["total_dossier_loss"] / PRE["total_dossier_loss"]) * 100
fig, ax = canvas(14, 7.6, "SmartHVAC pipeline, from data to dashboard")
steps = [
    (2, 58, BLUE, "1. Data", ["UCI AI4I 2020 (failures)", "UCI Online Retail II (customers)", "Kaggle HVAC power use",
                              "CSV tables: locations, residents,", "  devices, parts and 3 edge lists"]),
    (35, 58, GREEN, "2. Graph building", ["12 monthly graphs (PyG HeteroData)", "8 features per node",
                                          "sin/cos of the month and the", "  cooling / heating multipliers",
                                          "edges stored in both directions"]),
    (68, 58, PURPLE, "3. Training", ["120 epochs, FP16 on a Tesla V100", "4 loss terms, summed with fixed weights",
                                     "  If the loss turns NaN or Inf:", "  halve the learning rate and restart"]),
    (68, 6, ORANGE, "4. Fine-tuning", ["60 epochs, learned loss weights", "Huber loss for regression heads",
                                       "pos_weight 2.5 (failure), 1.8 (churn)", "cosine learning rate, grad clip 1.0",
                                       f"took {HIST['elapsed_sec'].iloc[-1]:.1f} s"]),
    (35, 6, GREEN, "5. Evaluation", [f"Total loss {PRE['total_dossier_loss']:.1f} -> {POST['total_dossier_loss']:.1f} (-{red_loss:.1f}%)",
                                     f"Failure F1 {PRE['failure_f1']:.2f} -> {POST['failure_f1']:.2f}",
                                     f"Churn F1 {PRE['sleeper_f1']:.2f} -> {POST['sleeper_f1']:.2f}",
                                     "XGBoost scores higher on most", "  metrics (figure 13)"]),
    (2, 6, RED, "6. Dashboard", ["Streamlit app: 4 task tabs,", "graph explorer, revenue and cost",
                                 "Revenue and cost are simulated", "  from assumed prices",
                                 "Neo4j export is optional"]),
]
for x, y, col, title, lines in steps:
    box(ax, x, y, 30, 34, title, [l if l.startswith("  ") else "- " + l for l in lines], color=col, size=9.2)
arrow(ax, (32, 75), (35, 75))
arrow(ax, (65, 75), (68, 75))
arrow(ax, (83, 58), (83, 40))
arrow(ax, (68, 23), (65, 23))
arrow(ax, (35, 23), (32, 23))
save(fig, "04_end_to_end_system_pipeline_flowchart.png")

# ================================================================ 05 divergence handling
fig, ax = canvas(13, 6.8, "What happens when training goes wrong (train_and_monitor.py and fine_tune_v100.py)")
box(ax, 2, 62, 44, 28, "Pre-training: each epoch", ["Run the 12 monthly graphs through the model",
                                                     "Check the loss for NaN or Inf"], color=BLUE)
box(ax, 2, 6, 20, 40, "Loss is fine", ["Add to the epoch average", "", "If it is the best so far,", "save the checkpoint"],
    color=GREEN)
box(ax, 26, 6, 20, 40, "Loss is NaN / Inf", ["Halve the learning rate", "", "Restart training from", "the beginning",
                                             "", "Give up with an error", "after the retry limit"], color=RED)
arrow(ax, (14, 62), (12, 46), GREEN)
arrow(ax, (34, 62), (36, 46), RED)
box(ax, 54, 62, 44, 28, "Fine-tuning: extra safeguards", ["Learning rate follows a cosine curve",
                                                         "(2.5e-4 down to 1e-5)", "Gradient norm clipped to 1.0"], color=ORANGE)
box(ax, 54, 6, 44, 40, "Checkpoints", ["The best total loss so far is tracked", "and its weights are saved.", "",
                                       "Note: fine-tuning has no NaN/Inf check;", "only the pre-training script retries."],
    color=PURPLE)
arrow(ax, (76, 62), (76, 46), PURPLE)
save(fig, "05_training_divergence_handling.png")

# ================================================================ 06 fine-tuning loss (real)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
a1.plot(HIST["epoch"], HIST["dossier_equivalent_loss"], "o-", color=GREEN, lw=2, label="total loss (logged epochs)")
a1.axhline(PRE["total_dossier_loss"], color=RED, ls="--", lw=1.4, label=f"before fine-tuning: {PRE['total_dossier_loss']:.2f}")
a1.axhline(POST["total_dossier_loss"], color=PURPLE, ls=":", lw=1.4, label=f"best after fine-tuning: {POST['total_dossier_loss']:.2f}")
a1.set(xlabel="Fine-tuning epoch", ylabel="Total loss (all tasks added)", title="(a) Total loss during fine-tuning")
a1.legend(fontsize=8.5)
a1.grid(alpha=0.3)
a2.plot(HIST["epoch"], HIST["failure_f1"], "o-", color=RED, lw=2, label="failure F1")
a2.plot(HIST["epoch"], HIST["sleeper_f1"], "s-", color=ORANGE, lw=2, label="churn F1")
a2.set(xlabel="Fine-tuning epoch", ylabel="F1 score", ylim=(0, 1), title="(b) F1 scores during fine-tuning")
a2.legend(fontsize=8.5)
a2.grid(alpha=0.3)
fig.suptitle("Fine-tuning progress on the Tesla V100 (values logged at epochs 1, 5, 10, ... 60)", fontweight="bold", color=NAVY)
fig.text(0.5, -0.03, "The loss history of the first 120-epoch training run was overwritten by the fine-tuning log, "
         "so only its final value (the red line) is available.", ha="center", fontsize=8.5, color="#555555")
save(fig, "06_pretrain_vs_finetune_loss_trajectory.png")

# ================================================================ 07 before / after
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
reg = [("Demand MSE", "demand_mse"), ("Inventory MSE", "inventory_mse"), ("Power MSE", "power_mse")]
x = np.arange(len(reg))
pre_v = [PRE[k] for _, k in reg]
post_v = [POST[k] for _, k in reg]
a1.bar(x - 0.2, pre_v, 0.4, color=GREY, label="before")
a1.bar(x + 0.2, post_v, 0.4, color=GREEN, label="after")
for i, (p, q) in enumerate(zip(pre_v, post_v)):
    a1.text(i + 0.2, q * 1.15, f"{q:.3g}\n({(q - p) / p * 100:+.1f}%)", ha="center", fontsize=8)
a1.set_yscale("log")
a1.set(xticks=x, xticklabels=[n for n, _ in reg], ylabel="Mean squared error (log scale)",
       title="(a) Regression error, lower is better")
a1.legend()
cls = [("Failure F1", "failure_f1"), ("Churn F1", "sleeper_f1")]
x = np.arange(len(cls))
pre_v = [PRE[k] for _, k in cls]
post_v = [POST[k] for _, k in cls]
a2.bar(x - 0.2, pre_v, 0.4, color=GREY, label="before")
a2.bar(x + 0.2, post_v, 0.4, color=GREEN, label="after")
for i, (p, q) in enumerate(zip(pre_v, post_v)):
    a2.text(i + 0.2, q + 0.02, f"{q:.3f}\n({q - p:+.3f})", ha="center", fontsize=8.5)
a2.set(xticks=x, xticklabels=[n for n, _ in cls], ylim=(0, 1), ylabel="F1 score",
       title="(b) Classification F1, higher is better")
a2.legend(loc="upper left")
fig.suptitle("Our model before and after fine-tuning", fontweight="bold", color=NAVY)
save(fig, "07_pillar_performance_comparison_before_after.png")

# ================================================================ 08 revenue and cost (CSV)
fig, ax = plt.subplots(figsize=(11.5, 4.8))
m = FIN["month"].values
ax.bar(m - 0.2, FIN["revenue_inr"] / 1e5, 0.4, color="#00A6CF", label="Revenue")
ax.bar(m + 0.2, FIN["expenditure_inr"] / 1e5, 0.4, color="#2A9D8F", label="Operating cost")
ax.set(xlabel="Month", ylabel="INR lakh", xticks=m)
ax2 = ax.twinx()
ax2.plot(m, FIN["cost_efficiency_ratio"], "o--", color=RED, lw=2, label="Revenue / cost ratio")
ax2.set_ylabel("Revenue / cost ratio", color=RED)
ax2.set_ylim(0, FIN["cost_efficiency_ratio"].max() * 1.25)
ax2.spines["right"].set_visible(True)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right", ncol=3, fontsize=9)
tot_r, tot_c = FIN["revenue_inr"].sum(), FIN["expenditure_inr"].sum()
ax.set_title(f"Simulated monthly revenue and cost (12-month total: revenue Rs {tot_r / 1e7:.2f} crore, "
             f"cost Rs {tot_c / 1e5:.1f} lakh)", fontsize=11, fontweight="bold", color=NAVY)
ax.grid(axis="y", alpha=0.3)
save(fig, "08_12month_financial_ledger.png")

# ================================================================ 09 seasons and demand
fig, ax = plt.subplots(figsize=(11.5, 4.8))
ax.plot(FIN["month"], FIN["metro_demand_units"], "s-", color=NAVY, lw=2.2, label="Metro demand")
ax.plot(FIN["month"], FIN["village_demand_units"], "^-", color=GREEN, lw=2.2, label="Village demand")
ax.set(xlabel="Month", ylabel="Predicted service demand (units)", xticks=range(1, 13))
ax.set_xticklabels([f"{i}\n{REGIME[i][0].replace('-', chr(10), 1)}" for i in range(1, 13)], fontsize=7.5)
ax2 = ax.twinx()
ax2.plot(range(1, 13), [REGIME[i][1] for i in range(1, 13)], "o--", color=BLUE, label="Cooling multiplier")
ax2.plot(range(1, 13), [REGIME[i][2] for i in range(1, 13)], "o--", color=ORANGE, label="Heating multiplier")
ax2.set(ylabel="Season multiplier (from config.py)", ylim=(0, 1.2))
ax2.spines["right"].set_visible(True)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=4, fontsize=9)
ax.grid(alpha=0.3)
save(fig, "09_seasonal_cooling_heating_regimes.png")

# ================================================================ 10 classification detail
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
cls = [("Failure", "failure_bce", "failure_flagged"), ("Churn", "sleeper_bce", "sleeper_flagged")]
x = np.arange(2)
a1.bar(x - 0.2, [PRE[b] for _, b, _ in cls], 0.4, color=GREY, label="before")
a1.bar(x + 0.2, [POST[b] for _, b, _ in cls], 0.4, color=GREEN, label="after")
for i, (_, b, _) in enumerate(cls):
    a1.text(i + 0.2, POST[b] + 0.01, f"{POST[b]:.3f}", ha="center", fontsize=9)
    a1.text(i - 0.2, PRE[b] + 0.01, f"{PRE[b]:.3f}", ha="center", fontsize=9)
a1.set(xticks=x, xticklabels=[n for n, _, _ in cls], ylabel="Binary cross-entropy",
       title="(a) Classification loss, lower is better", ylim=(0, 0.9))
a1.legend()
a2.bar(x - 0.2, [PRE[c] for _, _, c in cls], 0.4, color=GREY, label="before")
a2.bar(x + 0.2, [POST[c] for _, _, c in cls], 0.4, color=GREEN, label="after")
for i, (_, _, c) in enumerate(cls):
    a2.text(i - 0.2, PRE[c] + 8, str(PRE[c]), ha="center", fontsize=9)
    a2.text(i + 0.2, POST[c] + 8, str(POST[c]), ha="center", fontsize=9)
a2.set(xticks=x, xticklabels=["Devices flagged as failing", "Customers flagged as inactive"],
       ylabel="Number flagged (12 months combined)", title="(b) How many each model flags")
a2.legend()
fig.suptitle("Failure and churn classifiers: detail behind the F1 scores", fontweight="bold", color=NAVY)
save(fig, "10_failure_and_churn_classification_detail.png")

# ================================================================ 11 learned task weights
names = ["Demand", "Maintenance\n(power + failure)", "Churn", "Inventory"]
s = np.array(TEL["learned_uncertainty_log_vars"])
w = 0.5 * np.exp(-s)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
cols = [BLUE, GREEN, ORANGE, PURPLE]
a1.bar(names, s, color=cols)
for i, v in enumerate(s):
    a1.text(i, v + (0.08 if v >= 0 else -0.25), f"{v:.2f}", ha="center", fontsize=9)
a1.axhline(0, color="black", lw=0.8)
a1.set(ylabel="Learned log-variance s", title="(a) Learned task log-variances (final values)")
a2.bar(names, w, color=cols)
for i, v in enumerate(w):
    a2.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
a2.set(ylabel="Weight on the task loss = 0.5 x exp(-s)", title="(b) Resulting weight of each task")
for a in (a1, a2):
    a.tick_params(axis="x", labelsize=8.5)
fig.suptitle("Learned uncertainty weights after fine-tuning (4 loss terms)", fontweight="bold", color=NAVY)
save(fig, "11_kendall_uncertainty_task_weights.png")

# ================================================================ 12 monthly savings (CSV)
fig, ax = plt.subplots(figsize=(11.5, 4.8))
m = FIN["month"].values
bars = ax.bar(m, FIN["cost_efficiency_gain_pct"], 0.6, color=GREEN)
mean_gain = FIN["cost_efficiency_gain_pct"].mean()
ax.axhline(mean_gain, color=ORANGE, ls="--", lw=2, label=f"average: {mean_gain:.2f}%")
for b, d, r in zip(bars, FIN["high_risk_devices_flagged"], FIN["sleepers_reactivated"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
            f"{b.get_height():.1f}%\n{d} dev", ha="center", fontsize=7.5)
ax.set(xlabel="Month", ylabel="Cost saved vs fixing after failure (%)", xticks=m,
       ylim=(0, FIN["cost_efficiency_gain_pct"].max() * 1.3))
ax.set_title("Planned maintenance vs reactive repair: simulated monthly saving (label shows devices flagged)",
             fontsize=11, fontweight="bold", color=NAVY)
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
save(fig, "12_economic_proactive_savings_breakdown.png")

# ================================================================ 13 baseline comparison
order = ["XGBoost (Tabular)", "Vanilla GCN (Homogeneous)", "Vanilla GAT (Homogeneous)", "Hetero-GNN (Ours)"]
short = ["XGBoost", "GCN", "GAT", "Hetero-GNN\n(ours)"]
bc = [GREY, GREY, GREY, GREEN]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
tl = [BASE.loc[n, "total_dossier_loss"] for n in order]
a1.bar(short, tl, color=bc)
for i, v in enumerate(tl):
    a1.text(i, v * 1.1, f"{v:.1f}", ha="center", fontsize=9)
a1.set_yscale("log")
a1.set(ylabel="Total loss (log scale)", title="(a) Total loss over all tasks, lower is better")
x = np.arange(len(order))
a2.bar(x - 0.2, [BASE.loc[n, "failure_f1"] for n in order], 0.4, color=RED, label="failure F1")
a2.bar(x + 0.2, [BASE.loc[n, "sleeper_f1"] for n in order], 0.4, color=ORANGE, label="churn F1")
for i, n in enumerate(order):
    a2.text(i - 0.2, BASE.loc[n, "failure_f1"] + 0.015, f"{BASE.loc[n, 'failure_f1']:.2f}", ha="center", fontsize=8.5)
    a2.text(i + 0.2, BASE.loc[n, "sleeper_f1"] + 0.015, f"{BASE.loc[n, 'sleeper_f1']:.2f}", ha="center", fontsize=8.5)
a2.set(xticks=x, xticklabels=short, ylim=(0, 1), ylabel="F1 score", title="(b) Classification F1, higher is better")
a2.legend(loc="upper right")
fig.suptitle("Baseline comparison on the training data (no held-out set): XGBoost fits best",
             fontweight="bold", color=NAVY)
save(fig, "13_baseline_comparison.png")

# ================================================================ 14 held-out test scores
HO_MAIN = os.path.join(ART, "heldout_evaluation_months3-6-9-12_nodes.csv")
HO_FORECAST = os.path.join(ART, "heldout_evaluation_forecast.csv")
if os.path.exists(HO_MAIN) and os.path.exists(HO_FORECAST):
    d_main, d_fc = pd.read_csv(HO_MAIN), pd.read_csv(HO_FORECAST)
    d_main, d_fc = d_main[d_main["data"] == "test"], d_fc[d_fc["data"] == "test"]
    groups = [("Months\n3,6,9,12", d_main[d_main["split"] == "months"]),
              ("20% of\nnodes", d_main[d_main["split"] == "nodes"]),
              ("Forecast\n10-12", d_fc)]
    model_styles = [("Hetero-GNN", GREEN), ("XGBoost (own features)", GREY), ("XGBoost (+ neighbour features)", BLUE)]
    panels = [("demand_mse", "Demand MSE (log, lower is better)", True), ("inventory_mse", "Inventory MSE (log, lower is better)", True),
              ("power_mse", "Power MSE (log, lower is better)", True), ("failure_f1", "Failure F1 (higher is better)", False),
              ("churn_f1", "Churn F1 (higher is better)", False)]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.6))
    for ax, (col, title, use_log) in zip(axes, panels):
        for gi, (_, df) in enumerate(groups):
            ref = df[df["model"].str.startswith("Reference")][col].iloc[0]
            for mi, (name, color) in enumerate(model_styles):
                ax.bar(gi + (mi - 1) * 0.27, df[df["model"] == name][col].iloc[0], 0.26, color=color,
                       label=name if gi == 0 else None)
            ax.hlines(ref, gi - 0.42, gi + 0.42, colors="black", linestyles="--", lw=1.5,
                      label="Reference (predict mean / flag all)" if gi == 0 else None)
        if use_log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1)
        ax.set_xticks(range(3))
        ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
        ax.set_title(title, fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Held-out test scores (single run, seed 42): no model wins everywhere", fontweight="bold", color=NAVY)
    fig.subplots_adjust(bottom=0.28, top=0.86, wspace=0.28)
    save(fig, "14_heldout_test_scores.png")
else:
    print("14_heldout_test_scores.png skipped (held-out CSV files not found in artifacts/)")
