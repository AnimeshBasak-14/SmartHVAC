"""
figs_models.py - figures for lifecycle stages 6 to 8 (model selection, training, evaluation) that need fitted models.
Everything uses seed 42 unless the figure says it averages over seeds.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hvac_ml import common, config, data, pillar1_demand as p1, pillar2_failure as p2, pillar3_churn as p3
from hvac_ml.figs_common import BLUE, GOLD, GREEN, GREY, LIGHT, NAVY, ORANGE, PURPLE, RED, load_facts, load_summary, save, sd, val

R, X, RF, LG, LN = "Ridge (engineered features)", "XGBoost (engineered features)", "Random Forest (engineered features)", "Logistic regression (engineered features)", "Linear regression (engineered features)"
SP3 = "unseen_customers_later_period"
_CACHE = {}


def seed42():
    """Run each pillar once with seed 42 and keep the test predictions (cached for all figures)."""
    if "run" not in _CACHE:
        old = config.SEEDS
        config.SEEDS = [42]
        try:
            h, a = data.load_hvac(), data.load_ai4i()
            snaps = p3.build_snapshots(data.load_invoices())
            _, k1 = p1.run(h, 42)
            _, k2 = p2.run(a, 42)
            _, k3 = p3.run(snaps, 42)
        finally:
            config.SEEDS = old
        _CACHE["run"] = (h, a, snaps, k1, k2, k3)
    return _CACHE["run"]


# stage 6
def performance_matrix():
    s = load_summary()
    cols = [("demand", "random", "mae", "Demand\nrandom", True), ("demand", "unseen_months", "mae", "Demand\nunseen months", True), ("demand", "forecast", "mae", "Demand\nforecast", True),
            ("demand", "heatwave", "mae", "Demand\nheatwave", True), ("failure", "stratified_20pct", "pr_auc", "Failure\nPR-AUC", False), ("churn", SP3, "roc_auc", "Churn\nROC-AUC", False)]
    rows = [("Linear / logistic\n(engineered)", {"demand": LN, "failure": LG, "churn": LG}), ("Random Forest\n(engineered)", {"demand": RF, "failure": RF, "churn": RF}),
            ("Ridge (basic)", {k: "Ridge (basic features)" for k in ("demand", "failure", "churn")}), ("Ridge (engineered)", {k: R for k in ("demand", "failure", "churn")}),
            ("XGBoost (basic)", {k: "XGBoost (basic features)" for k in ("demand", "failure", "churn")}), ("XGBoost (engineered)", {k: X for k in ("demand", "failure", "churn")})]
    values = np.array([[val(s, p, sp, m[p], c) for p, sp, c, _, _ in cols] for _, m in rows])
    score = np.zeros_like(values)
    for j, (_, _, _, _, lower) in enumerate(cols):
        v = values[:, j]
        score[:, j] = (v.max() - v) / (v.max() - v.min()) if lower else (v - v.min()) / (v.max() - v.min())
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.imshow(score, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto"); ax.grid(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}" if cols[j][2] == "mae" else f"{values[i, j]:.2f}", ha="center", va="center", fontsize=10, fontweight="bold" if score[i, j] > 0.97 else "normal")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[3] for c in cols], fontsize=9.5); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.xaxis.tick_top()
    fig.suptitle("Every model on every test (green = best in its column, red = worst; bold = within 3% of the best)", fontweight="bold", color=NAVY, y=0.99, fontsize=11)
    save(fig, "s6_03_performance_matrix.png")


def selection_rule():
    s = load_summary()
    order = ["Reference", LG, "Ridge (basic features)", R, RF, "XGBoost (basic features)", X]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, (p, sp, m, title, lower) in zip(axes, (("failure", "stratified_20pct", "pr_auc", "Pillar 2, failure (PR-AUC)", False), ("churn", SP3, "pr_auc", "Pillar 3, churn (PR-AUC)", False))):
        d = s[(s["pillar"] == p) & (s["split"] == sp) & (~s["model"].str.startswith("Reference"))].copy()
        d["rank"] = d["model"].map(lambda x: order.index(x) if x in order else 99)
        d = d.sort_values(f"{m}_mean")
        best = d.iloc[-1]
        lo = best[f"{m}_mean"] - best[f"{m}_std"]
        tied = d[d[f"{m}_mean"] >= lo]
        pick = tied.sort_values("rank").iloc[0]["model"]
        ax.axvspan(lo, best[f"{m}_mean"] + 0.5, color="#E8F5E9", alpha=0.9)
        for i, (_, r) in enumerate(d.iterrows()):
            is_tied = r[f"{m}_mean"] >= lo
            ax.errorbar(r[f"{m}_mean"], i, xerr=r[f"{m}_std"], fmt="o", color=GREEN if r["model"] == pick else (ORANGE if is_tied else GREY), ms=9 if r["model"] == pick else 7, capsize=3, lw=1.6)
            ax.text(r[f"{m}_mean"] + r[f"{m}_std"] + 0.012, i, f"{r[f'{m}_mean']:.3f}", va="center", fontsize=8.5)
        ax.set_yticks(range(len(d))); ax.set_yticklabels([x.replace(" (engineered features)", "\n(engineered)").replace(" (basic features)", "\n(basic)") for x in d["model"]], fontsize=8.5)
        ax.axvline(lo, color=GREEN, ls="--", lw=1)
        ax.set_xlim(d[f"{m}_mean"].min() - 0.12, 1.02 if p == "failure" else 0.72)
        ax.set_title(f"{title}: tied with the best = {len(tied)}, chosen = {pick.replace(' (engineered features)', '')}", fontsize=10)
        ax.set_xlabel("mean over 5 seeds, bars = one standard deviation; green band = within one std of the best")
    fig.suptitle("How the tie rule picks a model: the simplest model inside the green band wins", fontweight="bold", color=NAVY, y=1.02)
    save(fig, "s6_04_selection_rule.png")


# stage 7
def _p1_xy(h, engineered):
    return p1.build_features(h, engineered), h["power_kw"]


def learning_curves():
    h, a, snaps, *_ = seed42()
    fracs = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    # pillar 1: forecast split
    test = p1.make_split(h, "forecast", 42)
    Xf, y = _p1_xy(h, True)
    tr_idx = np.flatnonzero(~test)
    for name, mk, col in (("Ridge", common.ridge_regressor, BLUE), ("XGBoost", lambda: common.xgb_regressor(42), ORANGE)):
        m, s_ = [], []
        for f in fracs:
            vals = []
            for rep in range(3):
                idx = rng.choice(tr_idx, max(30, int(f * len(tr_idx))), replace=False)
                model = mk().fit(Xf.iloc[idx], y.iloc[idx])
                vals.append(common.regression_metrics(y[test], model.predict(Xf[test]))["mae"])
            m.append(np.mean(vals)); s_.append(np.std(vals))
        m, s_ = np.array(m), np.array(s_)
        axes[0].plot(np.array(fracs) * len(tr_idx), m, "o-", color=col, label=name); axes[0].fill_between(np.array(fracs) * len(tr_idx), m - s_, m + s_, color=col, alpha=0.2)
    axes[0].set(xlabel="training hours", ylabel="test error, MAE (kW), lower is better", title="Pillar 1: forecast months", xscale="log"); axes[0].legend()
    # pillar 2
    tr, te = train_test_split(np.arange(len(a)), test_size=config.TEST_SIZE, stratify=a["failure"], random_state=42)
    Xa = p2.build_features(a, True); ya = a["failure"]
    for name, mk, col in (("Ridge", common.ridge_classifier, BLUE), ("XGBoost", lambda: common.xgb_classifier(42), ORANGE)):
        m, s_ = [], []
        for f in fracs:
            vals = []
            for rep in range(3):
                idx = rng.choice(tr, max(200, int(f * len(tr))), replace=False)
                while ya.iloc[idx].sum() < 5:
                    idx = rng.choice(tr, max(200, int(f * len(tr))), replace=False)
                model = mk().fit(Xa.iloc[idx], ya.iloc[idx])
                vals.append(average_precision_score(ya.iloc[te], model.predict_proba(Xa.iloc[te])[:, 1]))
            m.append(np.mean(vals)); s_.append(np.std(vals))
        m, s_ = np.array(m), np.array(s_)
        axes[1].plot(np.array(fracs) * len(tr), m, "o-", color=col, label=name); axes[1].fill_between(np.array(fracs) * len(tr), m - s_, m + s_, color=col, alpha=0.2)
    axes[1].set(xlabel="training machines", ylabel="test PR-AUC, higher is better", title="Pillar 2: failure", xscale="log"); axes[1].legend()
    # pillar 3
    all_train = pd.concat([snaps[c] for c in config.TRAIN_CUTOFFS], ignore_index=True)
    train, test3 = p3.split_customers(all_train, snaps[config.TEST_CUTOFF], 42)
    for name, mk, col in (("Ridge", common.ridge_classifier, BLUE), ("XGBoost", lambda: common.xgb_classifier(42), ORANGE)):
        m, s_ = [], []
        for f in fracs:
            vals = []
            for rep in range(3):
                idx = rng.choice(len(train), max(100, int(f * len(train))), replace=False)
                model = mk().fit(train.iloc[idx][p3.ENGINEERED], train.iloc[idx]["lapsed"])
                vals.append(roc_auc_score(test3["lapsed"], model.predict_proba(test3[p3.ENGINEERED])[:, 1]))
            m.append(np.mean(vals)); s_.append(np.std(vals))
        m, s_ = np.array(m), np.array(s_)
        axes[2].plot(np.array(fracs) * len(train), m, "o-", color=col, label=name); axes[2].fill_between(np.array(fracs) * len(train), m - s_, m + s_, color=col, alpha=0.2)
    axes[2].set(xlabel="training rows (customer snapshots)", ylabel="test ROC-AUC, higher is better", title="Pillar 3: churn", xscale="log"); axes[2].legend()
    for ax in axes:
        ax.minorticks_off()
    fig.suptitle("Learning curves: test score against the amount of training data (mean of 3 draws, band = one std)", fontweight="bold", color=NAVY, y=1.03)
    save(fig, "s7_02_learning_curves.png")


def sensitivity():
    h, a, snaps, *_ = seed42()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    Xf, y = _p1_xy(h, True)
    tests = {"forecast": p1.make_split(h, "forecast", 42), "heatwave": p1.make_split(h, "heatwave", 42)}
    alphas = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    for sp, col in (("forecast", BLUE), ("heatwave", RED)):
        t = tests[sp]
        axes[0].plot(alphas, [common.regression_metrics(y[t], make_pipeline(StandardScaler(), Ridge(alpha=al)).fit(Xf[~t], y[~t]).predict(Xf[t]))["mae"] for al in alphas], "o-", color=col, label=sp)
    axes[0].set(xscale="log", xlabel="Ridge penalty (alpha)", ylabel="test MAE (kW)", title="Demand: Ridge penalty"); axes[0].legend(fontsize=8.5)
    ns = [25, 50, 100, 200, 400, 800]
    for sp, col in (("forecast", BLUE), ("heatwave", RED)):
        t = tests[sp]
        axes[1].plot(ns, [common.regression_metrics(y[t], xgb.XGBRegressor(**{**config.XGB_REG, "n_estimators": n}, random_state=42, n_jobs=-1).fit(Xf[~t], y[~t]).predict(Xf[t]))["mae"] for n in ns], "o-", color=col, label=sp)
    axes[1].set(xscale="log", xlabel="XGBoost trees", ylabel="test MAE (kW)", title="Demand: number of trees"); axes[1].legend(fontsize=8.5)
    tr, te = train_test_split(np.arange(len(a)), test_size=config.TEST_SIZE, stratify=a["failure"], random_state=42)
    Xa, ya = p2.build_features(a, True), a["failure"]
    depths = [1, 2, 3, 4, 6, 8]
    axes[2].plot(depths, [average_precision_score(ya.iloc[te], xgb.XGBClassifier(**{**config.XGB_CLF, "max_depth": d}, random_state=42, n_jobs=-1).fit(Xa.iloc[tr], ya.iloc[tr]).predict_proba(Xa.iloc[te])[:, 1]) for d in depths], "o-", color=ORANGE)
    axes[2].axvline(config.XGB_CLF["max_depth"], color="black", ls="--", lw=1); axes[2].text(config.XGB_CLF["max_depth"] + 0.1, 0.80, "used", fontsize=8.5)
    axes[2].set(xlabel="XGBoost tree depth", ylabel="test PR-AUC", title="Failure: tree depth")
    all_train = pd.concat([snaps[c] for c in config.TRAIN_CUTOFFS], ignore_index=True)
    train, test3 = p3.split_customers(all_train, snaps[config.TEST_CUTOFF], 42)
    Cs = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    axes[3].plot(Cs, [roc_auc_score(test3["lapsed"], make_pipeline(common.SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=c, max_iter=3000)).fit(train[p3.ENGINEERED], train["lapsed"])
                                    .predict_proba(test3[p3.ENGINEERED])[:, 1]) for c in Cs], "o-", color=GREEN)
    axes[3].set(xscale="log", xlabel="logistic regression penalty strength C", ylabel="test ROC-AUC", title="Churn: Ridge-type penalty")
    for ax in axes:
        ax.minorticks_off()
    fig.suptitle("Sensitivity of the main settings: the results do not hinge on a lucky setting", fontweight="bold", color=NAVY, y=1.03)
    save(fig, "s7_03_sensitivity.png")


def fit_times():
    f = load_facts()["timing"]["fit_seconds"]
    names = [("Linear / logistic", {"demand": LN, "failure": LG, "churn": LG}, GREEN), ("Ridge (engineered)", {k: R for k in ("demand", "failure", "churn")}, BLUE),
             ("Random Forest", {k: RF for k in ("demand", "failure", "churn")}, PURPLE), ("XGBoost (engineered)", {k: X for k in ("demand", "failure", "churn")}, ORANGE)]
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    pill = ["demand", "failure", "churn"]
    for i, (lab, m, col) in enumerate(names):
        v = [f[f"{p}|{m[p]}"] for p in pill]
        ax.bar(np.arange(3) + (i - 1.5) * 0.2, v, 0.19, color=col, label=lab)
        for j, x in enumerate(v):
            ax.text(j + (i - 1.5) * 0.2, x * 1.12, f"{x:.2f}", ha="center", fontsize=7.8)
    ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels(["Pillar 1 (7,008 hours)", "Pillar 2 (8,000 machines)", "Pillar 3 (about 8,500 rows)"])
    ax.set(ylabel="seconds to fit once (log scale)", title="Training time on a laptop CPU (one fit, seed 42)", ylim=(0.01, 6)); ax.legend(fontsize=9, ncol=4, loc="upper center")
    save(fig, "s7_04_fit_times.png")


# stage 8
def p1_residuals():
    h, a, snaps, k1, *_ = seed42()
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
    for r, (sp, title) in enumerate((("forecast", "Forecast months 10-12"), ("heatwave", "Heatwave, 40 °C and above"))):
        df, pr = k1[(sp, R)]
        _, px = k1[(sp, X)]
        for name, pred, col in (("Ridge", pr, BLUE), ("XGBoost", px, ORANGE)):
            res = df["power_kw"].to_numpy() - pred
            axes[r, 0].hist(res, bins=60, alpha=0.6, color=col, label=f"{name}: mean {res.mean():+.3f}, std {res.std():.3f}")
            axes[r, 1].scatter(df["ambient_c"], res, s=5, alpha=0.4, color=col, label=name)
        axes[r, 0].axvline(0, color="black", lw=1); axes[r, 0].set(xlabel="actual minus predicted (kW)", ylabel="hours", title=f"{title}: error distribution"); axes[r, 0].legend(fontsize=8.5)
        axes[r, 1].axhline(0, color="black", lw=1); axes[r, 1].set(xlabel="outdoor temperature (°C)", ylabel="actual minus predicted (kW)", title=f"{title}: error by temperature")
    save(fig, "s8_01_p1_residuals.png")


def p1_error_by_time():
    h, a, snaps, k1, *_ = seed42()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    df, _ = k1[("unseen_months", R)]
    for name, key, col in (("Ridge", R, BLUE), ("XGBoost", X, ORANGE)):
        d, pred = k1[("unseen_months", key)]
        e = pd.DataFrame({"hour": d["hour"], "err": (d["power_kw"] - pred).abs()}).groupby("hour")["err"].mean()
        axes[0].plot(e.index, e.values, "o-", color=col, label=name, ms=4)
    axes[0].set(xlabel="hour of day", ylabel="MAE (kW)", title="Error by hour of day (unseen months 3, 6, 9, 12)"); axes[0].legend()
    for name, key, col in (("Ridge", R, BLUE), ("XGBoost", X, ORANGE)):
        parts = []
        for sp in ("unseen_months", "forecast"):
            d, pred = k1[(sp, key)]
            parts.append(pd.DataFrame({"month": d["month"], "err": (d["power_kw"] - pred).abs(), "sp": sp}))
        e = pd.concat(parts).drop_duplicates("month", keep="first") if False else pd.concat(parts)
        g = e[e["sp"] == "unseen_months"].groupby("month")["err"].mean()
        gf = e[e["sp"] == "forecast"].groupby("month")["err"].mean()
        axes[1].plot(g.index, g.values, "o", color=col, ms=8, label=f"{name}, unseen-months test")
        axes[1].plot(gf.index, gf.values, "s", color=col, ms=8, mfc="white", mew=2, label=f"{name}, forecast test")
    axes[1].set(xlabel="month", ylabel="MAE (kW)", title="Error by month in the two month-based tests", xticks=range(1, 13)); axes[1].legend(fontsize=8)
    save(fig, "s8_02_p1_error_by_time.png")


def p2_roc_threshold():
    h, a, snaps, k1, k2, k3 = seed42()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, col, lab in ((X, ORANGE, "XGBoost"), (RF, PURPLE, "Random Forest"), (R, BLUE, "Ridge (logistic)"), (LG, GREEN, "Logistic regression"), ("Reference: most worn tool first", GREY, "rule: most worn tool")):
        y, s_, _ = k2[name]
        fpr, tpr, _ = roc_curve(y, s_)
        axes[0].plot(fpr, tpr, color=col, lw=2, label=f"{lab} (AUC {roc_auc_score(y, s_):.3f})")
    axes[0].plot([0, 1], [0, 1], color="black", ls="--", lw=1, label="random guess")
    axes[0].set(xlabel="false positive rate (healthy machines flagged)", ylabel="true positive rate (failures found)", title="ROC curves (seed 42)"); axes[0].legend(fontsize=8.5, loc="lower right")
    y, s_, thr = k2[X]
    p, r, t = precision_recall_curve(y, s_)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-9)
    axes[1].plot(t, p[:-1], color=BLUE, lw=2, label="precision"); axes[1].plot(t, r[:-1], color=ORANGE, lw=2, label="recall"); axes[1].plot(t, f1, color="black", lw=2, ls="--", label="F1")
    axes[1].axvline(thr, color=RED, lw=1.5); axes[1].text(thr + 0.01, 0.15, f"threshold used\n(from training data): {thr:.2f}", color=RED, fontsize=9)
    axes[1].set(xlabel="alert threshold on the failure risk score", ylabel="score", title="XGBoost: what the threshold trades off", xlim=(0, 1), ylim=(0, 1.02)); axes[1].legend(loc="lower left", fontsize=9)
    save(fig, "s8_03_p2_roc_threshold.png")


def confusion_matrices():
    h, a, snaps, k1, k2, k3 = seed42()
    items = [("Pillar 2: XGBoost", k2[X], ["no failure", "failure"]), ("Pillar 2: Ridge (logistic)", k2[R], ["no failure", "failure"]),
             ("Pillar 3: Ridge (logistic)", k3[R], ["stays active", "lapses"]), ("Pillar 3: XGBoost", k3[X], ["stays active", "lapses"])]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, (title, (y, s_, thr), lab) in zip(axes, items):
        cm = confusion_matrix(y, (s_ >= thr).astype(int), labels=[0, 1])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{norm[i, j] * 100:.0f}%", ha="center", va="center", fontsize=11, color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["predicted\n" + lab[0], "predicted\n" + lab[1]], fontsize=8.5); ax.set_yticks([0, 1]); ax.set_yticklabels(["actual\n" + lab[0], "actual\n" + lab[1]], fontsize=8.5)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Confusion matrices at each model's own alert threshold (percentages are of each actual row; seed 42, unseen data)", fontweight="bold", color=NAVY, y=1.03, fontsize=10.5)
    save(fig, "s8_04_confusion_matrices.png")


def calibration():
    h, a, snaps, k1, k2, k3 = seed42()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (k, title, bins, lim) in zip(axes, ((k2, "Pillar 2: failure", 6, 0.4), (k3, "Pillar 3: churn", 8, 1.02))):
        for name, col, lab in ((X, ORANGE, "XGBoost"), (R, BLUE, "Ridge (logistic)")):
            y, s_, _ = k[name]
            frac, mean = calibration_curve(y, s_, n_bins=bins, strategy="quantile")
            ax.plot(mean, frac, "o-", color=col, lw=2, label=lab)
        ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1, label="perfectly calibrated")
        ax.set(xlabel="predicted probability (average in the group)", ylabel="share that really happened", title=title, xlim=(0, lim), ylim=(0, lim)); ax.legend(fontsize=9)
    fig.suptitle("Do the probabilities mean what they say? (groups of equal size, unseen data)", fontweight="bold", color=NAVY, y=1.02)
    save(fig, "s8_05_calibration.png")


def seed_variability():
    runs = pd.read_csv(config.RESULTS_DIR / "all_runs_by_seed.csv")
    panels = [("demand", "random", "mae", "Demand, random split: MAE (kW)"), ("failure", "stratified_20pct", "pr_auc", "Failure: PR-AUC"), ("churn", SP3, "roc_auc", "Churn: ROC-AUC"),
              ("churn", SP3, "top10_capture", "Churn: lapsers in the top 10%")]
    order = [LN, LG, RF, "Ridge (basic features)", R, "XGBoost (basic features)", X]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    rng = np.random.default_rng(1)
    for ax, (p, sp, m, title) in zip(axes, panels):
        d = runs[(runs["pillar"] == p) & (runs["split"] == sp)]
        present = [o for o in order if o in set(d["model"])]
        for i, name in enumerate(present):
            v = d[d["model"] == name][m].to_numpy()
            col = GREEN if name.startswith(("Linear", "Logistic")) else PURPLE if name.startswith("Random") else BLUE if name.startswith("Ridge") else ORANGE
            ax.scatter(np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), v, color=col, s=36, alpha=0.8)
            ax.hlines(v.mean(), i - 0.28, i + 0.28, color="black", lw=2)
        ax.set_xticks(range(len(present))); ax.set_xticklabels([n.replace(" (engineered features)", "\n(eng.)").replace(" (basic features)", "\n(basic)").replace("Logistic regression", "Logistic").replace("Linear regression", "Linear").replace("Random Forest", "Rand.\nForest") for n in present], fontsize=7.4)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Spread over the five seeds: each dot is one seed, the black line is the mean", fontweight="bold", color=NAVY, y=1.03)
    save(fig, "s8_06_seed_variability.png")


def failure_top_k():
    h, a, snaps, k1, k2, k3 = seed42()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    needed = {}
    for name, col, lab in ((X, ORANGE, "XGBoost"), (RF, PURPLE, "Random Forest"), (R, BLUE, "Ridge (logistic)"), ("Reference: most worn tool first", GREY, "rule: most worn tool")):
        y, s_, _ = k2[name]
        order = np.argsort(-s_, kind="stable")
        gains = np.concatenate([[0], np.cumsum(y[order]) / y.sum()])
        x = np.arange(len(gains)) / len(y) * 100
        axes[0].plot(x, gains * 100, color=col, lw=2, label=lab)
        needed[lab] = x[np.argmax(gains >= 0.8)]
    axes[0].plot([0, 100], [0, 100], color="black", ls="--", lw=1, label="random inspection")
    axes[0].axvline(10, color=RED, ls=":", lw=1.3); axes[0].text(11, 8, "top 10%", color=RED, fontsize=9)
    axes[0].set(xlim=(0, 40), ylim=(0, 102), xlabel="machines inspected (% of all machines, riskiest first)", ylabel="failures found (%)", title="Failures found as the inspection list gets longer"); axes[0].legend(fontsize=8.5, loc="lower right")
    labs = list(needed)
    axes[1].barh(labs[::-1], [needed[l] for l in labs[::-1]], color=[GREY, BLUE, PURPLE, ORANGE])
    for i, l in enumerate(labs[::-1]):
        axes[1].text(needed[l] + 0.6, i, f"{needed[l]:.0f}% of machines", va="center", fontsize=9)
    axes[1].set(xlabel="share of machines to inspect to find 80% of the failures (%)", title="Inspection effort needed to find 80% of failures", xlim=(0, 100))
    save(fig, "s8_07_failure_top_k.png")


ALL_STAGE6_8 = [performance_matrix, selection_rule, learning_curves, sensitivity, fit_times, p1_residuals, p1_error_by_time, p2_roc_threshold, confusion_matrices, calibration, seed_variability, failure_top_k]
