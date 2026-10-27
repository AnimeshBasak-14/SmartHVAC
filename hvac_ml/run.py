"""
run.py - runs the whole project: load data, run the three pillars, save tables, figures and final models.

    python run_all.py            # full run (5 seeds)      about 3 minutes
    python run_all.py --quick    # 1 seed, to check that everything works
"""
import argparse
import logging
import sys
from datetime import datetime

import joblib
import pandas as pd

from hvac_ml import config, data, pillar1_demand as p1, pillar2_failure as p2, pillar3_churn as p3, plots
from hvac_ml.common import summarise

log = logging.getLogger("run")

# simplest first: used to break ties
COMPLEXITY = ["Reference", "Linear regression (engineered features)", "Logistic regression (engineered features)",
              "Ridge (basic features)", "Ridge (engineered features)", "Random Forest (engineered features)",
              "XGBoost (basic features)", "XGBoost (engineered features)"]
PILLAR_METRICS = {
    "demand": (["mae", "rmse", "r2", "peak_mae"], "mae", True),
    "failure": (["roc_auc", "pr_auc", "f1", "precision", "recall", "top10_capture", "prevalence"], "pr_auc", False),
    "churn": (["roc_auc", "pr_auc", "f1", "precision", "recall", "top10_capture", "top10_lift", "prevalence"], "pr_auc", False),
}


def setup_logging():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(config.OUTPUT_DIR / "run_log.txt", mode="w", encoding="utf-8")])
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def pick(summary, metric, lower_is_better):
    """Model with the best mean; any model within one standard deviation of it counts as a tie and the simplest tied model wins."""
    s = summary[~summary["model"].str.startswith("Reference")].copy()
    s["score"] = s[f"{metric}_mean"] * (1 if lower_is_better else -1)
    best = s.sort_values("score").iloc[0]
    std = best[f"{metric}_std"]
    std = 0.0 if pd.isna(std) else std            # one seed (--quick) has no spread: only exact ties count
    tied = s[(s["score"] - best["score"]) <= std]
    rank = lambda m: COMPLEXITY.index(m) if m in COMPLEXITY else len(COMPLEXITY)
    return sorted(tied["model"], key=rank)[0], list(tied["model"])


def fmt(metric, mean, std):
    return f"{mean:.3f} ± {std:.3f}" if abs(mean) < 100 else f"{mean:.0f} ± {std:.0f}"


def df_to_md(df):
    """Plain markdown table for a DataFrame (avoids needing the tabulate package)."""
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(":---" for _ in df.columns) + " |"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def markdown_table(summary, metric_cols, names):
    lines = ["| Split | Model | " + " | ".join(names) + " |", "| :--- | :--- | " + " | ".join(":---:" for _ in names) + " |"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['split']} | {r['model']} | " + " | ".join(fmt(c, r[f'{c}_mean'], r[f'{c}_std']) for c in metric_cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one seed only")
    args = ap.parse_args()
    if args.quick:                                  # a quick check must never overwrite the real results
        config.SEEDS = config.SEEDS[:1]
        config.OUTPUT_DIR = config.OUTPUT_DIR / "quick_check"
        config.FIG_DIR, config.RESULTS_DIR, config.MODEL_DIR = (config.OUTPUT_DIR / d for d in ("figures", "results", "models"))
    setup_logging()
    keep_seed = config.SEEDS[0]
    for d in (config.FIG_DIR, config.RESULTS_DIR, config.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    log.info("Seeds: %s", config.SEEDS)

    log.info("Loading data")
    hvac, ai4i, invoices = data.load_hvac(), data.load_ai4i(), data.load_invoices()
    snapshots = p3.build_snapshots(invoices)

    log.info("Figures: data exploration")
    plots.eda_demand(hvac)
    plots.eda_failure(ai4i)
    plots.eda_churn(snapshots)

    log.info("Pillar 1: demand forecasting")
    r1, k1 = p1.run(hvac, keep_predictions_for_seed=keep_seed)
    log.info("Pillar 2: failure prediction")
    r2, k2 = p2.run(ai4i, keep_predictions_for_seed=keep_seed)
    log.info("Pillar 3: churn prediction")
    r3, k3 = p3.run(snapshots, keep_predictions_for_seed=keep_seed)

    runs = {"demand": r1, "failure": r2, "churn": r3}
    pd.concat(runs.values(), ignore_index=True).to_csv(config.RESULTS_DIR / "all_runs_by_seed.csv", index=False)
    summaries = {p: summarise(r, PILLAR_METRICS[p][0]) for p, r in runs.items()}
    summary_all = pd.concat(summaries.values(), ignore_index=True)
    summary_all.to_csv(config.RESULTS_DIR / "summary_mean_std.csv", index=False)

    log.info("Figures: results")
    plots.p1_comparison(summaries["demand"])
    plots.p1_heatwave(k1)
    plots.p2_results(summaries["failure"], k2)
    plots.p3_results(summaries["churn"], k3)
    plots.xgb_vs_ridge(summary_all)

    log.info("Final models on all data")
    final1, final2, final3 = p1.fit_final(hvac), p2.fit_final(ai4i), p3.fit_final(snapshots)
    f1_names = list(p1.build_features(hvac, True).columns)
    f2_names = list(p2.build_features(ai4i, True).columns)
    plots.importance_figure(final2, f2_names, "07_p2_feature_importance.png", "Pillar 2: what drives failure predictions",
                            "XGBoost (engineered features)", "Ridge (engineered features)")
    plots.importance_figure(final3, p3.ENGINEERED, "09_p3_feature_importance.png", "Pillar 3: what drives lapse predictions",
                            "XGBoost (engineered features)", "Ridge (engineered features)")

    # recommendation per pillar and split, with the tie rule
    recs = []
    for pillar, (cols, metric, lower) in PILLAR_METRICS.items():
        for split, grp in summaries[pillar].groupby("split", sort=False):
            choice, tied = pick(grp, metric, lower)
            recs.append({"pillar": pillar, "split": split, "metric": metric, "pick": choice, "tied_models": "; ".join(tied)})
    recs = pd.DataFrame(recs)
    recs.to_csv(config.RESULTS_DIR / "model_choice.csv", index=False)

    joblib.dump({"demand": final1, "failure": final2, "churn": final3,
                 "features": {"demand": f1_names, "failure": f2_names, "churn": p3.ENGINEERED},
                 "recommended": recs, "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M")}, config.MODEL_DIR / "hvac_models.joblib")

    lines = ["# Results summary (auto-generated by run_all.py)", "", f"Seeds: {config.SEEDS}. Every cell is mean ± standard deviation over seeds, on data never used for training.", ""]
    lines += ["## Pillar 1: demand forecasting (kW, lower is better)", "", markdown_table(summaries["demand"], ["mae", "rmse", "r2", "peak_mae"], ["MAE", "RMSE", "R²", "Peak-hour MAE"]), ""]
    lines += ["## Pillar 2: failure prediction (higher is better)", "", markdown_table(summaries["failure"], ["roc_auc", "pr_auc", "f1", "precision", "recall", "top10_capture", "prevalence"],
                                                                               ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Failures in top 10%", "Base rate"]), ""]
    lines += ["## Pillar 3: churn prediction (higher is better)", "", markdown_table(summaries["churn"], ["roc_auc", "pr_auc", "f1", "precision", "recall", "top10_capture", "top10_lift", "prevalence"],
                                                                               ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Lapsers in top 10%", "Top-10% lift", "Base rate"]), ""]
    lines += ["## Model choice (best mean; ties within one standard deviation go to the simpler model)", "", df_to_md(recs), ""]
    (config.RESULTS_DIR / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Done. See %s", config.RESULTS_DIR / "RESULTS_SUMMARY.md")


if __name__ == "__main__":
    main()
