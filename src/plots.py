"""
plots.py - every figure used in the report and the PPT.

All figures are saved as PNG in outputs/figures/ with a number prefix so they
are easy to refer to (Figure 1, Figure 2, ...). See docs/04_RESULTS_GUIDE.md for
what each figure means and how to explain it.
"""
import logging

import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import learning_curve

from src import config

log = logging.getLogger(__name__)
P = config.PRETTY
HEAT, COOL, GREY = "#d1495b", "#2e86ab", "#8d99ae"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 10})


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    log.info("  saved figure %s", name)


# ============================================================ EDA (data only)
def eda_figures(df):
    # Fig 1 - how the two targets are distributed
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, t, c in zip(axes, config.TARGETS, (HEAT, COOL)):
        ax.hist(df[t], bins=30, color=c, alpha=0.85, edgecolor="white")
        ax.axvline(df[t].mean(), color="black", ls="--", lw=1, label=f"mean = {df[t].mean():.1f}")
        ax.set(xlabel=P[t], ylabel="Number of buildings", title=f"Distribution of {t.replace('_', ' ')}")
        ax.legend()
    _save(fig, "01_target_distributions.png")

    # Fig 2 - correlation heat-map
    corr = df[config.FEATURES + config.TARGETS].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    names = [P[c].split(" (")[0] for c in corr.columns]
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Correlation between building features and HVAC loads")
    _save(fig, "02_correlation_heatmap.png")

    # Fig 3 - heating vs cooling (are they related?) coloured by height
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for h, c, lab in [(3.5, COOL, "1 storey (3.5 m)"), (7.0, HEAT, "2 storeys (7 m)")]:
        d = df[df.overall_height == h]
        ax.scatter(d.heating_load, d.cooling_load, s=14, alpha=0.6, color=c, label=lab)
    ax.set(xlabel=P["heating_load"], ylabel=P["cooling_load"],
           title="Heating vs cooling load (colour = building height)")
    ax.legend()
    _save(fig, "03_heating_vs_cooling.png")

    # Fig 4 - effect of glazing (windows) and of height on loads
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, feat in zip(axes, ["glazing_area", "overall_height"]):
        levels = sorted(df[feat].unique())
        w = 0.38
        for k, (t, c) in enumerate(zip(config.TARGETS, (HEAT, COOL))):
            data = [df.loc[df[feat] == v, t] for v in levels]
            pos = np.arange(len(levels)) + (k - 0.5) * w
            bp = ax.boxplot(data, positions=pos, widths=w * 0.9, patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set(facecolor=c, alpha=0.75)
            for med in bp["medians"]:
                med.set(color="black")
        ax.set_xticks(range(len(levels)), [str(v) for v in levels])
        ax.set(xlabel=P[feat], ylabel="Load (kWh/m²)", title=f"Loads by {P[feat].split(' (')[0].lower()}")
        ax.legend([plt.Rectangle((0, 0), 1, 1, fc=HEAT), plt.Rectangle((0, 0), 1, 1, fc=COOL)],
                  ["Heating", "Cooling"], loc="upper left")
    _save(fig, "04_loads_by_glazing_and_height.png")


# ============================================================ model results
def model_comparison_figure(comparison):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, t, c in zip(axes, config.TARGETS, (HEAT, COOL)):
        d = comparison[comparison.target == t].sort_values("val_RMSE", ascending=False)
        colors = [c if i == len(d) - 1 else GREY for i in range(len(d))]
        ax.barh(d.model, d.val_RMSE, color=colors)
        for y, v in enumerate(d.val_RMSE):
            ax.text(v, y, f" {v:.2f}", va="center", fontsize=8.5)
        ax.set(xlabel="Validation RMSE (kWh/m²) - lower is better",
               title=f"Model comparison: {t.replace('_', ' ')}")
        ax.set_xlim(0, d.val_RMSE.max() * 1.18)
    _save(fig, "05_model_comparison.png")


def predicted_vs_actual_figure(test, art):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, t, c in zip(axes, config.TARGETS, (HEAT, COOL)):
        y, p, m = test[t].values, art[t]["y_pred"], art[t]["margin"]
        lo, hi = min(y.min(), p.min()) - 2, max(y.max(), p.max()) + 2
        ax.fill_between([lo, hi], [lo - m, hi - m], [lo + m, hi + m], color=c, alpha=0.12,
                        label=f"90% interval (±{m:.2f})")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect prediction")
        ax.scatter(y, p, s=16, color=c, alpha=0.8, label="test buildings")
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=f"Actual {P[t]}", ylabel=f"Predicted {P[t]}",
               title=f"{art[t]['name']} on unseen test data")
        ax.legend(loc="upper left", fontsize=8.5)
    _save(fig, "06_predicted_vs_actual.png")


def residuals_figure(test, art):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, t, c in zip(axes, config.TARGETS, (HEAT, COOL)):
        res = test[t].values - art[t]["y_pred"]
        ax.hist(res, bins=25, color=c, alpha=0.85, edgecolor="white")
        ax.axvline(0, color="black", lw=1)
        ax.set(xlabel="Error = actual - predicted (kWh/m²)", ylabel="Number of test buildings",
               title=f"Prediction errors: {t.replace('_', ' ')} (mean {res.mean():+.2f})")
    _save(fig, "07_residuals.png")


def feature_importance_figure(art):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, t, c in zip(axes, config.TARGETS, (HEAT, COOL)):
        imp = art[t]["importance"]
        ax.barh([P[f].split(" (")[0] for f in imp.index], imp.values, color=c)
        ax.set(xlabel="Drop in R² when feature is shuffled", title=f"What drives {t.replace('_', ' ')}?")
    _save(fig, "08_feature_importance.png")


def confusion_matrix_figure(clf_summary):
    cm, labels = clf_summary["confusion_matrix"], config.DEMAND_CLASSES
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set(xlabel="Predicted demand class", ylabel="True demand class",
           title=f"{clf_summary['best_model']}\naccuracy {clf_summary['test_accuracy']:.1%} on test")
    _save(fig, "09_confusion_matrix.png")


def learning_curve_figure(model, X, y, name):
    sizes, tr, va = learning_curve(model, X, y, cv=5, scoring="neg_root_mean_squared_error",
                                   train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for scores, c, lab in [(-tr, HEAT, "training error"), (-va, COOL, "cross-validation error")]:
        ax.plot(sizes, scores.mean(1), "o-", color=c, label=lab)
        ax.fill_between(sizes, scores.mean(1) - scores.std(1), scores.mean(1) + scores.std(1),
                        color=c, alpha=0.15)
    ax.set(xlabel="Number of training buildings", ylabel="RMSE (kWh/m²)",
           title=f"Learning curve - {name} (heating load)")
    ax.legend()
    _save(fig, "10_learning_curve.png")


def partial_dependence_figure(model, X, name):
    feats = ["relative_compactness", "overall_height", "glazing_area", "glazing_distribution"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    # newer scikit-learn refuses integer columns here, so convert to float first
    PartialDependenceDisplay.from_estimator(model, X.astype(float), feats, ax=axes, line_kw={"color": HEAT})
    for ax, f in zip(axes, feats):
        ax.set_xlabel(P[f])
    axes[0].set_ylabel("Predicted heating load")
    fig.suptitle(f"What-if analysis: average effect of each feature ({name})", y=1.02)
    _save(fig, "11_partial_dependence.png")
