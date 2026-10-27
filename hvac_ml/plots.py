"""
plots.py - every figure for the report and slides (PNG files in outputs/figures/).

Colours: blue = Ridge, orange = XGBoost, grey = reference. Lighter bars use the basic features,
darker bars the engineered features.
"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from hvac_ml import config

log = logging.getLogger(__name__)
RIDGE, XGB, GREY, RED = "#2E86AB", "#E07A1F", "#8D99AE", "#C0392B"
STAT, FOREST = "#3A9D5D", "#8E44AD"   # statistical baseline, Random Forest baseline
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

FAMILY_ORDER = ["Linear regression (engineered features)", "Logistic regression (engineered features)", "Random Forest (engineered features)",
                "Ridge (basic features)", "Ridge (engineered features)", "XGBoost (basic features)", "XGBoost (engineered features)"]


def models_in(summary):
    present = set(summary["model"])
    return [m for m in FAMILY_ORDER if m in present]


def _style(model):
    if model.startswith(("Linear", "Logistic")):
        return STAT, 1.0
    if model.startswith("Random Forest"):
        return FOREST, 1.0
    color = RIDGE if model.startswith("Ridge") else XGB if model.startswith("XGBoost") else GREY
    return color, (0.45 if "basic" in model else 1.0)


def _short(model):
    return (model.replace("Logistic regression", "Logistic").replace("Linear regression", "Linear").replace("Random Forest", "Random\nForest")
            .replace(" (engineered features)", "\n(engineered)").replace(" (basic features)", "\n(basic)"))


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("  saved figure %s", name)


# data exploration
def eda_demand(hvac):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode, color in (("Cooling", "#2E86AB"), ("Heating", "#C0392B"), ("Idle", GREY)):
        d = hvac[hvac["mode"] == mode]
        axes[0].scatter(d["ambient_c"], d["power_kw"], s=5, alpha=0.4, color=color, label=mode)
    axes[0].axvline(config.HEATWAVE_TEMP_C, color="black", ls="--", lw=1)
    axes[0].text(config.HEATWAVE_TEMP_C + 0.3, 0.2, "heatwave test\nregion", fontsize=8)
    axes[0].set(xlabel="Outdoor temperature (°C)", ylabel="HVAC power (kW)", title="Power rises with the gap to the indoor setpoint")
    axes[0].legend(markerscale=3)
    monthly = hvac.groupby("month")["power_kw"].mean()
    axes[1].bar(monthly.index, monthly.values, color=RIDGE)
    axes[1].set(xlabel="Month", ylabel="Mean HVAC power (kW)", title="Demand peaks in summer", xticks=range(1, 13))
    _save(fig, "01_hvac_power_vs_temperature.png")


def eda_failure(ai4i):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    power = ai4i["torque_nm"] * ai4i["rpm"] * 2 * np.pi / 60
    for ax, (name, series) in zip(axes, (("Torque (Nm)", ai4i["torque_nm"]), ("Tool wear (min)", ai4i["tool_wear_min"]), ("Mechanical power (W)", power))):
        bins = pd.qcut(series, 10, duplicates="drop")
        rate = ai4i.groupby(bins, observed=True)["failure"].mean() * 100
        ax.bar(range(len(rate)), rate.values, color=XGB)
        ax.axhline(ai4i["failure"].mean() * 100, color="black", ls="--", lw=1, label=f"average {ai4i['failure'].mean() * 100:.1f}%")
        ax.set(xlabel=f"{name}, in 10 equal-size groups (low to high)", ylabel="Failure rate (%)", xticks=range(len(rate)))
        ax.legend()
    fig.suptitle("Failures are rare (3.4%) and cluster at extreme torque, wear and power", fontweight="bold")
    _save(fig, "02_ai4i_failure_drivers.png")


def eda_churn(snapshots):
    cut = config.TEST_CUTOFF
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    rates = pd.Series({c[:7]: s["lapsed"].mean() * 100 for c, s in snapshots.items()})
    axes[0].bar(rates.index, rates.values, color=RIDGE)
    axes[0].set(xlabel="Cutoff date (month)", ylabel="Customers who lapse in the next 90 days (%)", title="Share of active customers who go quiet")
    test = snapshots[cut]
    bins = pd.cut(test["recency_days"], [-1, 15, 30, 45, 60, 75, 90])
    r = test.groupby(bins, observed=True)["lapsed"].mean() * 100
    axes[1].bar([str(i) for i in r.index], r.values, color=XGB)
    axes[1].set(xlabel="Days since last purchase", ylabel="Lapse rate (%)", title=f"Recency is informative but weak (cutoff {cut})")
    axes[1].tick_params(axis="x", rotation=30)
    _save(fig, "03_churn_base_rates.png")


# Pillar 1
def p1_comparison(summary):
    splits = ["random", "unseen_months", "forecast", "heatwave"]
    labels = ["Random 20%\n(easy)", "Unseen months\n3,6,9,12", "Forecast\nmonths 10-12", "Heatwave\n(>= 40 °C)"]
    models = models_in(summary)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    width = 0.84 / len(models)
    for i, model in enumerate(models):
        d = summary[summary["model"] == model].set_index("split").loc[splits]
        color, alpha = _style(model)
        ax.bar(np.arange(4) + (i - (len(models) - 1) / 2) * width, d["mae_mean"], width * 0.92, yerr=d["mae_std"], color=color, alpha=alpha, label=model, capsize=2)
    ref = summary[summary["model"] == "Reference: average by hour of day"].set_index("split").loc[splits]
    ax.hlines(ref["mae_mean"], np.arange(4) - 0.45, np.arange(4) + 0.45, colors="black", linestyles="--", lw=1.3, label="Reference: average by hour of day")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set(ylabel="Mean absolute error (kW), lower is better", title="Pillar 1: demand forecast error on data the models never saw")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    _save(fig, "04_p1_model_comparison.png")

def p1_heatwave(kept):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, split, title in zip(axes, ["forecast", "heatwave"], ["Forecast months 10-12", f"Heatwave: ambient >= {config.HEATWAVE_TEMP_C:.0f} °C (never seen in training)"]):
        df, ridge_pred = kept[(split, "Ridge (engineered features)")]
        _, xgb_pred = kept[(split, "XGBoost (engineered features)")]
        ax.scatter(df["ambient_c"], df["power_kw"], s=8, color=GREY, alpha=0.6, label="Actual")
        ax.scatter(df["ambient_c"], ridge_pred, s=8, color=RIDGE, alpha=0.6, label="Ridge (engineered)")
        ax.scatter(df["ambient_c"], xgb_pred, s=8, color=XGB, alpha=0.6, label="XGBoost (engineered)")
        ax.set(xlabel="Outdoor temperature (°C)", title=title)
    axes[0].set_ylabel("HVAC power (kW)")
    axes[0].legend(markerscale=2)
    _save(fig, "05_p1_heatwave_extrapolation.png")


# Pillar 2
def p2_results(summary, kept):
    models = models_in(summary)
    rule = "Reference: most worn tool first"
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for model in models + [rule]:
        y, scores, _ = kept[model]
        p, r, _ = precision_recall_curve(y, scores)
        color, alpha = _style(model)
        axes[0].plot(r, p, color="black" if model == rule else color, alpha=max(alpha, 0.6), ls="-" if "engineered" in model else ":", lw=1.8, label=model)
    prevalence = float(np.mean(kept[models[0]][0]))
    axes[0].axhline(prevalence, color="black", ls="--", lw=1, label=f"Reference: random ({prevalence:.1%})")
    axes[0].set(xlabel="Recall (share of failures found)", ylabel="Precision (share of alerts that are real)", title="Precision-recall curves (seed 42)", xlim=(0, 1), ylim=(0, 1.02))
    axes[0].legend(fontsize=7)
    names = models + [rule]
    d = summary.set_index("model").loc[names]
    axes[1].bar(range(len(names)), d["pr_auc_mean"], yerr=d["pr_auc_std"], color=[_style(m)[0] for m in names], capsize=3)
    for i, m in enumerate(names):
        axes[1].patches[i].set_alpha(_style(m)[1])
        axes[1].text(i, d["pr_auc_mean"].iloc[i] + d["pr_auc_std"].iloc[i] + 0.015, f"{d['pr_auc_mean'].iloc[i]:.2f}", ha="center", fontsize=8.5)
    axes[1].set(xticks=range(len(names)), xticklabels=[_short(m).replace("Reference: most worn tool first", "Rule:\nmost worn\ntool") for m in names],
                ylabel="PR-AUC (higher is better)", title="Pillar 2: failure prediction, mean of 5 seeds", ylim=(0, 1.05))
    axes[1].tick_params(axis="x", labelsize=7.5)
    axes[1].axhline(prevalence, color="black", ls="--", lw=1)
    _save(fig, "06_p2_failure_results.png")

def importance_figure(final, feature_names, name, title, xgb_key, ridge_key):
    """Side-by-side XGBoost importance and absolute standardised Ridge coefficients."""
    xgb_model = final[xgb_key][0] if isinstance(final[xgb_key], tuple) else final[xgb_key]
    ridge_model = final[ridge_key][0] if isinstance(final[ridge_key], tuple) else final[ridge_key]
    imp = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values()
    coef = ridge_model.named_steps["model"].coef_
    coef = pd.Series(np.abs(np.ravel(coef)), index=feature_names).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.8, 0.32 * len(feature_names))))
    axes[0].barh(imp.index, imp.values, color=XGB)
    axes[0].set(xlabel="Importance (share of total gain)", title="XGBoost: which inputs it relies on")
    axes[1].barh(coef.index, coef.values, color=RIDGE)
    axes[1].set(xlabel="|standardised coefficient|", title="Ridge: which inputs get weight")
    fig.suptitle(title, fontweight="bold")
    _save(fig, name)


# Pillar 3
def p3_results(summary, kept):
    models = models_in(summary)
    rule = "Reference: recency only"
    names = models + [rule]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for model in names:
        y, scores, _ = kept[model]
        order = np.argsort(-scores, kind="stable")
        gains = np.concatenate([[0], np.cumsum(np.asarray(y)[order]) / max(1, np.sum(y))])
        color, alpha = _style(model)
        axes[0].plot(np.linspace(0, 1, len(gains)), gains, color="black" if model == rule else color, alpha=max(alpha, 0.6),
                     ls="-" if "engineered" in model else ":" if "basic" in model else "-.", lw=1.8, label=model)
    axes[0].plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="Reference: random targeting")
    axes[0].set(xlabel="Share of customers contacted (highest risk first)", ylabel="Share of lapsing customers reached",
                title="Gains curve on unseen customers (seed 42)")
    axes[0].legend(fontsize=7, loc="lower right")
    d = summary.set_index("model").loc[names]
    axes[1].bar(range(len(names)), d["roc_auc_mean"], yerr=d["roc_auc_std"], color=[_style(m)[0] for m in names], capsize=3)
    for i, m in enumerate(names):
        axes[1].patches[i].set_alpha(_style(m)[1])
        axes[1].text(i, d["roc_auc_mean"].iloc[i] + d["roc_auc_std"].iloc[i] + 0.008, f"{d['roc_auc_mean'].iloc[i]:.2f}", ha="center", fontsize=8.5)
    axes[1].axhline(0.5, color="black", ls="--", lw=1)
    axes[1].set(xticks=range(len(names)), xticklabels=[_short(m).replace("Reference: recency only", "Rule:\nrecency\nonly") for m in names],
                ylabel="ROC-AUC (0.5 = guessing)", title="Pillar 3: lapse prediction, mean of 5 seeds", ylim=(0.4, 0.9))
    axes[1].tick_params(axis="x", labelsize=7.5)
    _save(fig, "08_p3_churn_results.png")

def xgb_vs_ridge(summary_all):
    """One picture of where XGBoost helps and where it does not: XGBoost minus Ridge on the engineered features,
    expressed as a relative change in the main metric (positive = XGBoost better)."""
    rows = []
    for pillar, split, metric, lower_better, label in [
        ("demand", "random", "mae", True, "Demand, random split"), ("demand", "unseen_months", "mae", True, "Demand, unseen months"),
        ("demand", "forecast", "mae", True, "Demand, forecast"), ("demand", "heatwave", "mae", True, "Demand, heatwave"),
        ("failure", "stratified_20pct", "pr_auc", False, "Failure (PR-AUC)"), ("churn", "unseen_customers_later_period", "pr_auc", False, "Churn (PR-AUC)"),
    ]:
        d = summary_all[(summary_all["pillar"] == pillar) & (summary_all["split"] == split)].set_index("model")
        r, x = d.loc["Ridge (engineered features)", f"{metric}_mean"], d.loc["XGBoost (engineered features)", f"{metric}_mean"]
        rows.append((label, (r - x) / r * 100 if lower_better else (x - r) / r * 100))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels, vals = zip(*rows)
    ax.barh(labels, vals, color=[XGB if v >= 0 else RIDGE for v in vals])
    ax.axvline(0, color="black", lw=1)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + (2 if v >= 0 else -2), i, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.set(xlabel="XGBoost vs Ridge (both with engineered features): % better  (orange: XGBoost better, blue: Ridge better)",
           title="Where XGBoost helps and where it does not")
    ax.set_xlim(min(vals) - 40, max(vals) + 40)
    _save(fig, "10_xgboost_vs_ridge.png")
