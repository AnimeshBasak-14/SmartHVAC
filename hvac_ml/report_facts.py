"""
report_facts.py - computes every number quoted in the report, slides and lifecycle documents and saves them to
outputs/results/facts.json, so the text never depends on numbers typed by hand.

    python -m hvac_ml.report_facts
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from hvac_ml import common, config, data, pillar1_demand as p1, pillar2_failure as p2, pillar3_churn as p3

RAW_RETAIL_LINES = 1_067_371          # lines in the original Online Retail II workbook (counted when the invoice table was built)
KEPT_RETAIL_LINES = 824_293           # lines left after dropping rows without a customer id and rows with a price of zero or less


def q(series, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    return {f"p{int(x * 100)}": float(series.quantile(x)) for x in qs}


def describe(series):
    return {"min": float(series.min()), "mean": float(series.mean()), "max": float(series.max()), "std": float(series.std())}


def hvac_facts(h):
    h = h.copy()
    h["gap"] = h["ambient_c"] - h["indoor_c"]
    h["abs_gap"] = h["gap"].abs()
    out = {"rows": len(h), "start": str(h["timestamp"].min()), "end": str(h["timestamp"].max()),
           "power": describe(h["power_kw"]), "ambient": describe(h["ambient_c"]), "indoor": describe(h["indoor_c"]), "humidity": describe(h["humidity"]),
           "mode_counts": h["mode"].value_counts().to_dict(), "mode_share": (h["mode"].value_counts(normalize=True)).round(4).to_dict()}
    out["power_by_mode"] = {m: describe(g["power_kw"]) for m, g in h.groupby("mode")}
    out["ambient_by_mode"] = {m: describe(g["ambient_c"]) for m, g in h.groupby("mode")}
    out["hours_at_or_above_40"] = int((h["ambient_c"] >= config.HEATWAVE_TEMP_C).sum())
    out["months_with_40_plus"] = sorted(h.loc[h["ambient_c"] >= config.HEATWAVE_TEMP_C, "month"].unique().tolist())
    out["corr"] = {"power_vs_abs_gap": float(h["power_kw"].corr(h["abs_gap"])), "power_vs_ambient": float(h["power_kw"].corr(h["ambient_c"])),
                   "power_vs_hour": float(h["power_kw"].corr(h["hour"])), "power_vs_humidity": float(h["power_kw"].corr(h["humidity"])),
                   "power_vs_indoor": float(h["power_kw"].corr(h["indoor_c"]))}
    out["monthly_mean_power"] = h.groupby("month")["power_kw"].mean().round(3).to_dict()
    out["monthly_mean_ambient"] = h.groupby("month")["ambient_c"].mean().round(2).to_dict()
    out["hourly_mean_power"] = h.groupby("hour")["power_kw"].mean().round(3).to_dict()
    daily = h.groupby(h["timestamp"].dt.date)["power_kw"].agg(["max", "sum"])
    out["daily_peak_kw"] = describe(daily["max"])
    out["daily_energy_kwh"] = describe(daily["sum"])
    out["power_floor_rows"] = int((h["power_kw"] <= 0.1).sum())
    out["mode_rule"] = {"cooling_min_ambient": float(h.loc[h["mode"] == "Cooling", "ambient_c"].min()),
                        "heating_max_ambient": float(h.loc[h["mode"] == "Heating", "ambient_c"].max())}
    return out


def ai4i_facts(a):
    out = {"rows": len(a), "failures": int(a["failure"].sum()), "failure_rate": float(a["failure"].mean()),
           "type_counts": a["Type"].value_counts().to_dict(),
           "failure_rate_by_type": a.groupby("Type")["failure"].mean().round(4).to_dict()}
    modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    out["failure_modes"] = {m: int(a[m].sum()) for m in modes}
    out["failures_with_no_mode_flag"] = int(((a["failure"] == 1) & (a[modes].sum(axis=1) == 0)).sum())
    out["rows_with_more_than_one_mode"] = int((a[modes].sum(axis=1) > 1).sum())
    num = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]
    out["numeric"] = {c: describe(a[c]) for c in num}
    out["corr_with_failure"] = {c: float(a[c].corr(a["failure"])) for c in num}
    power = a["torque_nm"] * a["rpm"] * 2 * np.pi / 60
    gap = a["process_temp_k"] - a["air_temp_k"]
    out["power_w"] = describe(power)
    out["temp_gap_k"] = describe(gap)
    out["corr_with_failure"]["power_w"] = float(power.corr(a["failure"]))
    out["corr_with_failure"]["temp_gap_k"] = float(gap.corr(a["failure"]))
    out["corr_with_failure"]["wear_x_torque"] = float((a["tool_wear_min"] * a["torque_nm"]).corr(a["failure"]))
    for name, s in (("torque_nm", a["torque_nm"]), ("tool_wear_min", a["tool_wear_min"]), ("power_w", power)):
        top = s >= s.quantile(0.9)
        out[f"failure_rate_top_decile_{name}"] = float(a.loc[top, "failure"].mean())
    low_rpm = a["rpm"] < a["rpm"].quantile(0.1)
    out["failure_rate_lowest_rpm_decile"] = float(a.loc[low_rpm, "failure"].mean())
    return out


def retail_facts(inv, snaps):
    sales = inv[~inv["is_return"]]
    out = {"raw_lines": RAW_RETAIL_LINES, "kept_lines": KEPT_RETAIL_LINES, "dropped_lines": RAW_RETAIL_LINES - KEPT_RETAIL_LINES,
           "dropped_share": (RAW_RETAIL_LINES - KEPT_RETAIL_LINES) / RAW_RETAIL_LINES,
           "invoices": len(inv), "purchase_invoices": len(sales), "return_invoices": int(inv["is_return"].sum()),
           "return_share_of_invoices": float(inv["is_return"].mean()), "customers": int(inv["customer"].nunique()),
           "start": str(inv["date"].min()), "end": str(inv["date"].max()),
           "uk_share_of_invoices": float((inv["country"] == "United Kingdom").mean()), "countries": int(inv["country"].nunique()),
           "top_countries": inv["country"].value_counts().head(5).to_dict()}
    per_cust = sales.groupby("customer").agg(freq=("invoice", "nunique"), spend=("value", "sum"), last=("date", "max"), first=("date", "min"))
    out["invoices_per_customer"] = {**describe(per_cust["freq"]), **q(per_cust["freq"])}
    out["spend_per_customer"] = {**describe(per_cust["spend"]), **q(per_cust["spend"])}
    out["single_purchase_customers_share"] = float((per_cust["freq"] == 1).mean())
    out["invoice_value"] = {**describe(sales["value"]), **q(sales["value"])}
    out["monthly_invoices"] = sales.groupby(sales["date"].dt.to_period("M").astype(str)).size().to_dict()
    out["monthly_active_customers"] = sales.groupby(sales["date"].dt.to_period("M").astype(str))["customer"].nunique().to_dict()
    out["snapshots"] = {c[:10]: {"active_customers": int(len(s)), "lapse_rate": float(s["lapsed"].mean())} for c, s in snaps.items()}
    out["total_train_rows"] = int(sum(len(snaps[c]) for c in config.TRAIN_CUTOFFS))
    test = snaps[config.TEST_CUTOFF]
    out["test_cutoff_rows"] = int(len(test))
    corr = {c: float(test[c].corr(test["lapsed"])) for c in p3.ENGINEERED}
    out["corr_with_lapse_test_cutoff"] = corr
    return out


def timing_facts(h, a, snaps):
    """Seconds to fit each model once on its training split, and milliseconds for one prediction (seed 42)."""
    seed = 42
    rows = {}

    def timeit(fn, repeat=1):
        t = time.perf_counter()
        for _ in range(repeat):
            fn()
        return (time.perf_counter() - t) / repeat

    # pillar 1
    test = p1.make_split(h, "random", seed)
    for name, (eng, model) in p1.model_zoo(seed).items():
        X = p1.build_features(h, eng)
        rows[f"demand|{name}"] = {"fit_s": timeit(lambda: model.fit(X[~test], h.loc[~test, "power_kw"]))}
    # pillar 2
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(a)), test_size=config.TEST_SIZE, stratify=a["failure"], random_state=seed)
    for name, (eng, factory) in p2.model_zoo(seed).items():
        X = p2.build_features(a, eng)
        rows[f"failure|{name}"] = {"fit_s": timeit(lambda: factory().fit(X.iloc[tr], a["failure"].iloc[tr]))}
    # pillar 3
    all_train = pd.concat([snaps[c] for c in config.TRAIN_CUTOFFS], ignore_index=True)
    train, _ = p3.split_customers(all_train, snaps[config.TEST_CUTOFF], seed)
    for name, (cols, factory) in p3.model_zoo(seed).items():
        rows[f"churn|{name}"] = {"fit_s": timeit(lambda: factory().fit(train[cols], train["lapsed"]))}
    # prediction latency of the saved models
    bundle = joblib.load(config.MODEL_DIR / "hvac_models.joblib")
    X1 = p1.build_features(h.iloc[:1], True)
    X2 = p2.build_features(a.iloc[:1], True)
    lat = {}
    for name, m in bundle["demand"].items():
        lat[f"demand|{name}"] = timeit(lambda: m.predict(X1), 50) * 1000
    for name, (m, _) in bundle["failure"].items():
        lat[f"failure|{name}"] = timeit(lambda: m.predict_proba(X2), 50) * 1000
    for name, (m, _) in bundle["churn"].items():
        X3 = train[p3.ENGINEERED].iloc[:1]
        lat[f"churn|{name}"] = timeit(lambda: m.predict_proba(X3), 50) * 1000
    size_mb = (config.MODEL_DIR / "hvac_models.joblib").stat().st_size / 1e6
    return {"fit_seconds": {k: round(v["fit_s"], 3) for k, v in rows.items()}, "predict_ms_single_row": {k: round(v, 2) for k, v in lat.items()},
            "model_file_mb": round(size_mb, 2)}


def main():
    h, a, inv = data.load_hvac(), data.load_ai4i(), data.load_invoices()
    snaps = p3.build_snapshots(inv)
    facts = {"hvac": hvac_facts(h), "ai4i": ai4i_facts(a), "retail": retail_facts(inv, snaps), "timing": timing_facts(h, a, snaps)}
    facts["seeds"] = config.SEEDS
    out = config.RESULTS_DIR / "facts.json"
    out.write_text(json.dumps(facts, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print("saved", out)
    return facts


if __name__ == "__main__":
    main()
