"""
figs_ops.py - figures for lifecycle stages 9 and 10 (deployment outputs and monitoring on real data).
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hvac_ml import common, config, data, monitoring, pillar1_demand as p1, pillar2_failure as p2, pillar3_churn as p3
from hvac_ml.figs_common import BLUE, GREEN, GREY, ORANGE, RED, save, table_figure

R, X = "Ridge (engineered features)", "XGBoost (engineered features)"
_C = {}


def _load():
    if not _C:
        _C["b"] = joblib.load(config.MODEL_DIR / "hvac_models.joblib")
        _C["h"], _C["a"] = data.load_hvac(), data.load_ai4i()
        _C["snaps"] = p3.build_snapshots(data.load_invoices())
    return _C["b"], _C["h"], _C["a"], _C["snaps"]


def _day(month, tmin, tmax, indoor=22.5, hum=55):
    hours = np.arange(24)
    amb = tmin + (tmax - tmin) * (0.5 - 0.5 * np.cos(2 * np.pi * (hours - 5) / 24))
    mode = np.where(amb > 27, "Cooling", np.where(amb < 19, "Heating", "Idle"))
    return pd.DataFrame({"month": month, "hour": hours, "ambient_c": amb, "indoor_c": indoor, "humidity": hum, "mode": mode})


def planner_output():
    b, h, *_ = _load()
    lo, hi = h["ambient_c"].min(), h["ambient_c"].max()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (title, day) in zip(axes, (("A normal July day (24 to 38 °C)", _day(7, 24, 38)), ("A heatwave day (30 to 48 °C)", _day(7, 30, 48)))):
        Xd = p1.build_features(day, True)
        px, pr = b["demand"][X].predict(Xd), b["demand"][R].predict(Xd)
        outside = ((day["ambient_c"] > config.HEATWAVE_TEMP_C) | (day["ambient_c"] < lo) | (day["ambient_c"] > hi)).to_numpy()
        plan = np.where(outside, pr, px)
        ax.plot(day["hour"], px, color=ORANGE, lw=1.6, ls="--", label="XGBoost only")
        ax.plot(day["hour"], pr, color=BLUE, lw=1.6, ls="--", label="Ridge only")
        ax.plot(day["hour"], plan, color="black", lw=2.6, label="planned demand (what the app shows)")
        if outside.any():
            hrs = day["hour"][outside]
            ax.axvspan(hrs.min() - 0.5, hrs.max() + 0.5, color="#FDECEA", zorder=0)
            ax.text(hrs.mean(), plan.min() + 0.05, "safeguard on:\nRidge used", ha="center", color=RED, fontsize=9)
        ax.set(xlabel="hour of day", title=f"{title}: peak {plan.max():.2f} kW, energy {plan.sum():.1f} kWh")
        ax.legend(fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("predicted power (kW)")
    save(fig, "s9_02_planner_output.png")


def failure_radar_table():
    b, h, a, _ = _load()
    model, thr = b["failure"][X]
    s = a.sample(500, random_state=1).copy()
    s["risk"] = model.predict_proba(p2.build_features(s, True))[:, 1]
    top = s.sort_values("risk", ascending=False).head(12)
    df = pd.DataFrame({"Product ID": top["Product ID"], "Type": top["Type"], "Torque (Nm)": top["torque_nm"].round(1), "Speed (rpm)": top["rpm"],
                       "Tool wear (min)": top["tool_wear_min"], "Risk (%)": [f"{v * 100:.1f}" for v in top["risk"]], "Alert": np.where(top["risk"] >= thr, "inspect", "watch"),
                       "Actually failed": np.where(top["failure"] == 1, "yes", "no")})
    df.to_csv(config.RESULTS_DIR / "failure_radar_example.csv", index=False)
    fig = table_figure(df, title=f"Failure Radar: the 12 riskiest machines in a sample of 500 (alert threshold {thr:.0%})", font=10, figsize=(11, 5.2), row_height=0.45)
    save(fig, "s9_03_failure_radar_table.png")


def retention_table():
    b, h, a, snaps = _load()
    snap = snaps[config.TEST_CUTOFF].copy()
    model, thr = b["churn"][R]
    snap["risk"] = model.predict_proba(snap[p3.ENGINEERED])[:, 1]
    top = snap.sort_values("risk", ascending=False).head(12)
    df = pd.DataFrame({"Customer": top["customer"].astype(int), "Risk (%)": [f"{v * 100:.1f}" for v in top["risk"]], "Days idle": top["recency_days"].astype(int),
                       "Orders": top["frequency"].astype(int), "Spend": top["monetary"].round(0).astype(int), "Orders (90 d)": top["orders_last_90d"].astype(int),
                       "Actually lapsed": np.where(top["lapsed"] == 1, "yes", "no")})
    df.to_csv(config.RESULTS_DIR / "retention_list_example.csv", index=False)
    fig = table_figure(df, title="Retention List: the 12 active customers most likely to go quiet (illustration; the saved model saw these customers)", font=10, figsize=(11, 5.2), row_height=0.45)
    save(fig, "s9_04_retention_table.png")


def input_drift():
    b, h, a, snaps = _load()
    h = h.copy()
    h["gap"] = h["ambient_c"] - h["indoor_c"]
    cols1 = ["ambient_c", "indoor_c", "humidity", "hour", "gap"]
    d1 = monitoring.drift_table(h[h["month"] <= 9], h[h["month"] >= 10], cols1)
    d1h = monitoring.drift_table(h[h["ambient_c"] < config.HEATWAVE_TEMP_C], h[h["ambient_c"] >= config.HEATWAVE_TEMP_C], cols1)
    ref3 = pd.concat([snaps[c] for c in config.TRAIN_CUTOFFS], ignore_index=True)
    cols3 = [c for c in ["recency_days", "frequency", "monetary", "orders_last_90d", "tenure_days"] if c in ref3.columns]
    d3 = monitoring.drift_table(ref3, snaps[config.TEST_CUTOFF], cols3)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, d, title in ((axes[0], d1, "Demand: months 10-12 against months 1-9"), (axes[1], d1h, "Demand: hours at 40 °C and above\nagainst all cooler hours"),
                         (axes[2], d3, "Churn: test cutoff against the training cutoffs")):
        col = [{"ok": GREEN, "warning": ORANGE, "alert": RED}[s] for s in d["status"]]
        ax.barh(d["feature"][::-1], d["psi"][::-1], color=col[::-1])
        for i, v in enumerate(d["psi"][::-1]):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
        ax.axvline(config.PSI_WARN, color=ORANGE, ls="--", lw=1)
        ax.axvline(config.PSI_ALERT, color=RED, ls="--", lw=1)
        ax.set(xlabel="population stability index (PSI); dashed lines at 0.10 and 0.25", title=title, xlim=(0, max(0.5, d["psi"].max() * 1.2)))
    save(fig, "s10_02_input_drift.png")
    pd.concat([d1.assign(check="demand months 10-12 vs 1-9"), d1h.assign(check="demand 40C+ vs cooler"),
               d3.assign(check="churn test vs train cutoffs")]).to_csv(config.RESULTS_DIR / "drift_checks.csv", index=False)


def performance_over_time():
    b, h, a, snaps = _load()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    Xf, y = p1.build_features(h, True), h["power_kw"]
    ref = 0.065
    for name, mk, col in (("XGBoost", lambda: common.xgb_regressor(42), ORANGE), ("Ridge", common.ridge_regressor, BLUE)):
        maes = []
        for m in range(4, 13):
            tr, te = (h["month"] < m).to_numpy(), (h["month"] == m).to_numpy()
            mod = mk().fit(Xf[tr], y[tr])
            maes.append(common.regression_metrics(y[te], mod.predict(Xf[te]))["mae"])
        axes[0].plot(range(4, 13), maes, "o-", color=col, label=name)
    axes[0].axhline(ref * config.ERROR_WARN_RATIO, color=ORANGE, ls="--", lw=1)
    axes[0].axhline(ref * config.ERROR_ALERT_RATIO, color=RED, ls="--", lw=1)
    axes[0].text(4.05, ref * config.ERROR_WARN_RATIO + 0.004, "warning: 1.5 x test error", color=ORANGE, fontsize=8.5)
    axes[0].text(4.05, ref * config.ERROR_ALERT_RATIO + 0.004, "alert: 2 x test error", color=RED, fontsize=8.5)
    axes[0].set(xlabel="month being predicted (model trained only on earlier months)", ylabel="MAE (kW)", title="Demand: error as the data moves forward", xticks=range(4, 13))
    axes[0].legend(loc="upper right")
    cuts = config.TRAIN_CUTOFFS + [config.TEST_CUTOFF]
    pred, act, lab = [], [], []
    for i in range(1, len(cuts)):
        tr = pd.concat([snaps[c] for c in cuts[:i]], ignore_index=True)
        te = snaps[cuts[i]]
        mod = common.ridge_classifier().fit(tr[p3.ENGINEERED], tr["lapsed"])
        pred.append(mod.predict_proba(te[p3.ENGINEERED])[:, 1].mean())
        act.append(te["lapsed"].mean())
        lab.append(str(cuts[i])[:7])
    xs = np.arange(len(lab))
    axes[1].bar(xs - 0.2, pred, 0.38, color=BLUE, label="predicted lapse rate")
    axes[1].bar(xs + 0.2, act, 0.38, color=GREY, label="actual lapse rate")
    for i in range(len(lab)):
        axes[1].text(xs[i] - 0.2, pred[i] + 0.01, f"{pred[i]:.2f}", ha="center", fontsize=8.5)
        axes[1].text(xs[i] + 0.2, act[i] + 0.01, f"{act[i]:.2f}", ha="center", fontsize=8.5)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(lab)
    axes[1].set(xlabel="cutoff month (model trained only on earlier cutoffs)", ylabel="share of active customers who lapse", title="Churn: predicted against actual lapse rate", ylim=(0, 0.75))
    axes[1].legend(loc="upper right", fontsize=9)
    save(fig, "s10_03_performance_over_time.png")


ALL = [planner_output, failure_radar_table, retention_table, input_drift, performance_over_time]
