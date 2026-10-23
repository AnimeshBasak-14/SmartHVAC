"""
figs_data.py - figures for lifecycle stages 1 to 5 (problem, data, cleaning, exploration, features).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hvac_ml import config, data, pillar1_demand as p1, pillar3_churn as p3
from hvac_ml.figs_common import BLUE, GOLD, GREEN, GREY, NAVY, ORANGE, PURPLE, RED, load_facts, load_summary, save, sd, val

R, X = "Ridge (engineered features)", "XGBoost (engineered features)"
MODE_COLORS = {"Cooling": BLUE, "Heating": RED, "Idle": GREY}


def _rate_bars(ax, x, y, bins, xlabel, color=ORANGE, labels=None, min_count=1):
    """Bar chart of the mean of y (a 0/1 target) inside bins of x, with the row count printed under each bar."""
    cats = pd.cut(x, bins) if not hasattr(bins, "categories") else bins
    g = pd.DataFrame({"y": y.to_numpy(), "c": cats.to_numpy() if hasattr(cats, "to_numpy") else cats}).groupby("c", observed=True)["y"].agg(["mean", "count"])
    g = g[g["count"] >= min_count]
    ax.bar(range(len(g)), g["mean"] * 100, color=color)
    ax.axhline(y.mean() * 100, color="black", ls="--", lw=1)
    for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
        ax.text(i, m * 100 + 0.8, f"{m * 100:.0f}%", ha="center", fontsize=8)
        ax.text(i, 0.6, f"n={n}", ha="center", fontsize=6.8, color="white", rotation=90, va="bottom")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(labels or [str(c) for c in g.index], rotation=35, ha="right", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("lapse rate (%)")


# ============================================================================ stage 1
def rule_of_thumb():
    s = load_summary()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    splits = [("random", "random"), ("unseen_months", "unseen\nmonths"), ("forecast", "forecast\n10-12"), ("heatwave", "heatwave")]
    ax = axes[0]
    ref = [val(s, "demand", sp, "Reference: average by hour of day", "mae") for sp, _ in splits]
    xg = [val(s, "demand", sp, X, "mae") for sp, _ in splits]
    rg = [val(s, "demand", sp, R, "mae") for sp, _ in splits]
    xs = np.arange(4)
    ax.bar(xs - 0.27, ref, 0.26, color=GREY, label="rule: average by hour")
    ax.bar(xs, rg, 0.26, color=BLUE, label="Ridge")
    ax.bar(xs + 0.27, xg, 0.26, color=ORANGE, label="XGBoost")
    ax.set_xticks(xs); ax.set_xticklabels([l for _, l in splits]); ax.set_ylabel("average error (kW), lower is better")
    ax.set_title("Pillar 1: demand forecast error", fontsize=10.5); ax.legend(fontsize=8)
    ax = axes[1]
    names = ["random\nguess", "rule: most\nworn tool", "logistic\nregression", "Random\nForest", "XGBoost"]
    vals = [0.10, val(s, "failure", "stratified_20pct", "Reference: most worn tool first", "top10_capture"),
            val(s, "failure", "stratified_20pct", "Logistic regression (engineered features)", "top10_capture"),
            val(s, "failure", "stratified_20pct", "Random Forest (engineered features)", "top10_capture"), val(s, "failure", "stratified_20pct", X, "top10_capture")]
    ax.bar(range(5), np.array(vals) * 100, color=[GREY, GREY, PURPLE, GREEN, ORANGE])
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 1.5, f"{v * 100:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=8.5); ax.set_ylim(0, 108)
    ax.set_ylabel("failures found by inspecting the top 10% (%)"); ax.set_title("Pillar 2: failures in the top 10% of machines", fontsize=10.5)
    ax = axes[2]
    names = ["random\ncontact", "rule: longest\nsince purchase", "logistic\nregression", "Ridge", "XGBoost"]
    sp = "unseen_customers_later_period"
    vals = [0.10, val(s, "churn", sp, "Reference: recency only", "top10_capture"), val(s, "churn", sp, "Logistic regression (engineered features)", "top10_capture"),
            val(s, "churn", sp, R, "top10_capture"), val(s, "churn", sp, X, "top10_capture")]
    ax.bar(range(5), np.array(vals) * 100, color=[GREY, GREY, PURPLE, BLUE, ORANGE])
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 0.5, f"{v * 100:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=8.5); ax.set_ylim(0, 30)
    ax.set_ylabel("lapsing customers reached by calling the top 10% (%)"); ax.set_title("Pillar 3: lapsers in the top 10% of customers", fontsize=10.5)
    fig.suptitle("The bar to beat: what a rule of thumb achieves, against our models (unseen data)", fontweight="bold", color=NAVY, y=1.02)
    save(fig, "s1_03_rule_of_thumb.png")


# ============================================================================ stage 2
def hvac_year_overview():
    h = data.load_hvac()
    d = h.groupby(h["timestamp"].dt.date).agg(power=("power_kw", "mean"), ambient=("ambient_c", "mean"), peak=("power_kw", "max"))
    d.index = pd.to_datetime(d.index)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True)
    axes[0].plot(d.index, d["ambient"], color=RED, lw=1.2)
    for yv, lab in ((19, "heating below 19 °C"), (27, "cooling above 27 °C"), (config.HEATWAVE_TEMP_C, "heatwave test: 40 °C")):
        axes[0].axhline(yv, color="black", ls="--", lw=0.8)
        axes[0].text(d.index[2], yv + 0.6, lab, fontsize=8)
    axes[0].set_ylabel("daily mean outdoor\ntemperature (°C)")
    axes[1].plot(d.index, d["power"], color=BLUE, lw=1.2, label="daily mean power")
    axes[1].plot(d.index, d["peak"], color=ORANGE, lw=0.9, alpha=0.8, label="daily peak power")
    axes[1].set_ylabel("HVAC power (kW)"); axes[1].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Pillar 1 data: one year of hourly readings, shown by day", fontweight="bold", color=NAVY)
    save(fig, "s2_02_hvac_year_overview.png")


def ai4i_composition():
    f = load_facts()["ai4i"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    types = ["L", "M", "H"]
    axes[0].bar(types, [f["type_counts"][t] for t in types], color=[BLUE, GREEN, PURPLE])
    for i, t in enumerate(types):
        axes[0].text(i, f["type_counts"][t] + 100, f"{f['type_counts'][t]:,}", ha="center", fontsize=9)
    axes[0].set(title="Machines by quality type", ylabel="machines")
    axes[1].bar(types, [f["failure_rate_by_type"][t] * 100 for t in types], color=[BLUE, GREEN, PURPLE])
    axes[1].axhline(f["failure_rate"] * 100, color="black", ls="--", lw=1, label=f"all machines {f['failure_rate'] * 100:.2f}%")
    for i, t in enumerate(types):
        axes[1].text(i, f["failure_rate_by_type"][t] * 100 + 0.08, f"{f['failure_rate_by_type'][t] * 100:.1f}%", ha="center", fontsize=9)
    axes[1].set(title="Failure rate by type", ylabel="% that fail"); axes[1].legend(fontsize=8.5)
    modes = list(f["failure_modes"]) + ["no flag"]
    counts = list(f["failure_modes"].values()) + [f["failures_with_no_mode_flag"]]
    axes[2].bar(modes, counts, color=[ORANGE] * 5 + [GREY])
    for i, c in enumerate(counts):
        axes[2].text(i, c + 2, str(c), ha="center", fontsize=9)
    axes[2].set(title="Failure types (a failure can have more than one)", ylabel="machines")
    fig.suptitle(f"Pillar 2 data: {f['rows']:,} machines, {f['failures']} failures", fontweight="bold", color=NAVY, y=1.03)
    save(fig, "s2_03_ai4i_composition.png")


def retail_activity():
    f = load_facts()["retail"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1.6, 1.6, 1]})
    months = list(f["monthly_invoices"])
    x = np.arange(len(months))
    axes[0].bar(x, list(f["monthly_invoices"].values()), color=GREEN)
    axes[0].set_xticks(x[::3]); axes[0].set_xticklabels([m[2:] for m in months][::3], rotation=45, ha="right")
    axes[0].set(title="Purchase invoices per month (Dec 2011 is partial)", ylabel="invoices")
    axes[1].plot(x, list(f["monthly_active_customers"].values()), color=PURPLE, marker="o", ms=3.5)
    axes[1].set_xticks(x[::3]); axes[1].set_xticklabels([m[2:] for m in months][::3], rotation=45, ha="right")
    axes[1].set(title="Customers who bought that month (Dec 2011 is partial)", ylabel="customers")
    c = f["top_countries"]
    axes[2].barh(list(c)[::-1], np.array(list(c.values())[::-1]) / f["invoices"] * 100, color=BLUE)
    axes[2].set(title="Share of invoices by country", xlabel="% of all invoices")
    fig.suptitle(f"Pillar 3 data: {f['invoices']:,} invoices from {f['customers']:,} customers; the UK is {f['uk_share_of_invoices'] * 100:.0f}% of invoices; "
                 f"{f['return_invoices']:,} are returns", fontweight="bold", color=NAVY, y=1.04, fontsize=10.5)
    save(fig, "s2_04_retail_activity.png")


# ============================================================================ stage 3
def retail_cleaning_funnel():
    f = load_facts()["retail"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    labels = ["lines in the\noriginal workbook", "lines kept\n(customer id, price > 0)", "invoices\n(lines grouped)", "  of which\n  purchases", "  of which\n  returns"]
    vals = [f["raw_lines"], f["kept_lines"], f["invoices"], f["purchase_invoices"], f["return_invoices"]]
    cols = [GREY, BLUE, GREEN, GREEN, RED]
    ax.barh(range(5)[::-1], vals, color=cols)
    for i, v in zip(range(5)[::-1], vals):
        ax.text(v + 15000, i, f"{v:,}", va="center", fontsize=9)
    ax.set_yticks(range(5)[::-1]); ax.set_yticklabels(labels, fontsize=8.8); ax.set_xlim(0, 1.35e6)
    ax.set_xlabel("rows"); ax.set_title(f"{f['dropped_share'] * 100:.1f}% of lines had no customer id and were dropped", fontsize=10.5)
    ax = axes[1]
    snaps = f["snapshots"]
    names = list(snaps)
    n = [snaps[k]["active_customers"] for k in names]
    ax.bar(range(len(names)), n, color=[BLUE] * 5 + [ORANGE])
    ax2 = ax.twinx()
    ax2.plot(range(len(names)), [snaps[k]["lapse_rate"] * 100 for k in names], color="black", marker="o", lw=1.4)
    ax2.set_ylabel("lapse rate (%)"); ax2.set_ylim(0, 70); ax2.grid(False)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([k[:7] for k in names], rotation=30, ha="right")
    ax.set_ylabel("active customers (rows in the churn table)"); ax.set_title("Rows in the churn table, one cutoff at a time", fontsize=10.5)
    for i, v in enumerate(n):
        ax.text(i, v + 40, f"{v:,}", ha="center", fontsize=8.5)
    save(fig, "s3_01_retail_cleaning_funnel.png")


def invoice_value_outliers():
    inv = data.load_invoices()
    sales = inv[~inv["is_return"]]
    spend = sales.groupby("customer")["value"].sum().sort_values(ascending=False)
    top1 = spend.head(int(len(spend) * 0.01)).sum() / spend.sum()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    axes[0].hist(np.log10(sales["value"].clip(lower=0.5)), bins=60, color=GREEN)
    top = axes[0].get_ylim()[1]
    for (pct, c), frac in zip(((50, "black"), (95, ORANGE), (99, RED)), (0.95, 0.82, 0.69)):
        v = np.percentile(sales["value"], pct)
        axes[0].axvline(np.log10(v), color=c, ls="--"); axes[0].text(np.log10(v) + 0.04, top * frac, f"p{pct}: {v:,.0f}", color=c, fontsize=8.8, fontweight="bold")
    axes[0].set(xlabel="invoice value (log10 of currency units)", ylabel="invoices", title=f"Invoice values (largest: {sales['value'].max():,.0f})")
    cum = spend.cumsum() / spend.sum()
    axes[1].plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum.values * 100, color=PURPLE, lw=2)
    axes[1].plot([0, 100], [0, 100], color=GREY, ls="--", lw=1)
    axes[1].set(xlabel="customers, biggest spenders first (%)", ylabel="share of total spend (%)", title=f"The top 1% of customers account for {top1 * 100:.0f}% of spend")
    fig.suptitle("Outliers are real big buyers, so we keep them (the models use log versions of spend)", fontweight="bold", color=NAVY, y=1.03, fontsize=10.5)
    save(fig, "s3_04_invoice_value_outliers.png")


# ============================================================================ stage 4
def hvac_heatmap():
    h = data.load_hvac()
    pv = h.pivot_table(index="hour", columns="month", values="power_kw", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    im = ax.imshow(pv.values, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_xticks(range(12)); ax.set_xticklabels(range(1, 13)); ax.set_yticks(range(0, 24, 2)); ax.set_yticklabels(range(0, 24, 2)); ax.grid(False)
    ax.set(xlabel="month", ylabel="hour of day", title="Average HVAC power (kW) by hour and month")
    for i in range(24):
        for j in range(12):
            ax.text(j, i, f"{pv.values[i, j]:.1f}", ha="center", va="center", fontsize=5.6, color="black" if pv.values[i, j] < 3.0 else "white")
    fig.colorbar(im, ax=ax, label="kW")
    save(fig, "s4_01_hvac_heatmap.png")


def hvac_modes():
    h = data.load_hvac()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    order = ["Idle", "Heating", "Cooling"]
    bp = axes[0].boxplot([h.loc[h["mode"] == m, "power_kw"] for m in order], tick_labels=order, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], order):
        patch.set(facecolor=MODE_COLORS[m], alpha=0.8)
    axes[0].set(ylabel="power (kW)", title="Power by HVAC mode")
    for m in ("Heating", "Idle", "Cooling"):
        axes[1].hist(h.loc[h["mode"] == m, "ambient_c"], bins=40, color=MODE_COLORS[m], alpha=0.85, label=m)
    for xv in (19, 27, config.HEATWAVE_TEMP_C):
        axes[1].axvline(xv, color="black", ls="--", lw=1)
    axes[1].set(xlabel="outdoor temperature (°C)", ylabel="hours", title="The mode follows the temperature (19 and 27 °C)"); axes[1].legend(fontsize=8.5)
    g = h.groupby("hour")[["power_kw", "ambient_c"]].mean()
    axes[2].plot(g.index, g["power_kw"], color=BLUE, lw=2, label="power (kW)")
    ax2 = axes[2].twinx(); ax2.plot(g.index, g["ambient_c"], color=RED, lw=1.6, ls="--", label="outdoor temp (°C)"); ax2.grid(False)
    axes[2].set(xlabel="hour of day", ylabel="mean power (kW)", title="Daily rhythm"); ax2.set_ylabel("mean outdoor temperature (°C)")
    save(fig, "s4_02_hvac_modes.png")


def _corr_heatmap(df, title, name, figsize=(8.2, 6.6), fmt="{:.2f}", font=8):
    c = df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=45, ha="right", fontsize=8.5); ax.set_yticks(range(len(c))); ax.set_yticklabels(c.columns, fontsize=8.5)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, fmt.format(c.values[i, j]), ha="center", va="center", fontsize=font, color="white" if abs(c.values[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="correlation"); ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY)
    save(fig, name)


def hvac_corr():
    h = data.load_hvac()
    df = pd.DataFrame({"power_kw": h["power_kw"], "outdoor temp": h["ambient_c"], "indoor temp": h["indoor_c"], "humidity": h["humidity"], "hour": h["hour"], "month": h["month"],
                       "|outdoor - indoor|": (h["ambient_c"] - h["indoor_c"]).abs()})
    _corr_heatmap(df, "Pillar 1: correlations (the absolute gap is the strongest)", "s4_03_hvac_corr.png")


def ai4i_distributions():
    a = data.load_ai4i()
    a = a.assign(power_w=a["torque_nm"] * a["rpm"] * 2 * np.pi / 60)
    cols = [("air_temp_k", "air temperature (K)"), ("process_temp_k", "process temperature (K)"), ("rpm", "rotational speed (rpm)"),
            ("torque_nm", "torque (Nm)"), ("tool_wear_min", "tool wear (min)"), ("power_w", "mechanical power (W)")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (c, lab) in zip(axes.ravel(), cols):
        lo, hi = a[c].quantile(0.001), a[c].quantile(0.999)
        bins = np.linspace(lo, hi, 45)
        ax.hist(a.loc[a["failure"] == 0, c].clip(lo, hi), bins=bins, density=True, color=BLUE, alpha=0.65, label="no failure")
        ax.hist(a.loc[a["failure"] == 1, c].clip(lo, hi), bins=bins, density=True, color=RED, alpha=0.65, label="failure")
        ax.set(xlabel=lab, ylabel="density"); ax.legend(fontsize=8)
    fig.suptitle("Pillar 2: each sensor reading, machines that failed (red) against those that did not (blue)", fontweight="bold", color=NAVY, y=1.0)
    save(fig, "s4_04_ai4i_distributions.png")


def ai4i_corr():
    a = data.load_ai4i()
    df = pd.DataFrame({"failure": a["failure"], "air temp": a["air_temp_k"], "process temp": a["process_temp_k"], "speed": a["rpm"], "torque": a["torque_nm"], "tool wear": a["tool_wear_min"],
                       "temp gap": a["process_temp_k"] - a["air_temp_k"], "power": a["torque_nm"] * a["rpm"] * 2 * np.pi / 60, "wear x torque": a["tool_wear_min"] * a["torque_nm"]})
    _corr_heatmap(df, "Pillar 2: correlations (engineered inputs on the right and bottom)", "s4_05_ai4i_corr.png", figsize=(8.8, 7.2))


def retail_distributions():
    inv = data.load_invoices()
    sales = inv[~inv["is_return"]]
    cust = sales.groupby("customer").agg(freq=("invoice", "nunique"), spend=("value", "sum"), last=("date", "max"))
    snap = p3.customer_snapshot(inv, config.TEST_CUTOFF, inv.drop_duplicates("customer").set_index("customer")["country"])
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4))
    axes[0, 0].hist(cust["freq"].clip(upper=60), bins=60, color=GREEN); axes[0, 0].set(xlabel="invoices per customer (capped at 60 for display)", ylabel="customers",
                                                                                         title=f"{(cust['freq'] == 1).mean() * 100:.0f}% of customers bought once; median {cust['freq'].median():.0f}")
    axes[0, 1].hist(np.log10(cust["spend"].clip(lower=1)), bins=50, color=PURPLE); axes[0, 1].set(xlabel="total spend (log10)", ylabel="customers", title=f"Total spend per customer, median {cust['spend'].median():,.0f}")
    axes[1, 0].hist(snap["recency_days"], bins=45, color=BLUE); axes[1, 0].set(xlabel="days since last purchase (active customers at the test cutoff)", ylabel="customers", title="Recency of the customers in the test table")
    axes[1, 1].hist(snap["avg_gap_days"].dropna().clip(upper=200), bins=50, color=ORANGE); axes[1, 1].set(xlabel="average days between orders (capped at 200)", ylabel="customers",
                                                                                                           title="Order rhythm of the customers in the test table")
    fig.suptitle("Pillar 3: how customers behave", fontweight="bold", color=NAVY)
    save(fig, "s4_06_retail_distributions.png")


def _all_snapshots():
    inv = data.load_invoices()
    snaps = p3.build_snapshots(inv)
    return pd.concat(snaps.values(), ignore_index=True)


def retail_lapse_drivers():
    d = _all_snapshots()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    _rate_bars(axes[0], d["frequency"], d["lapsed"], [0, 1, 2, 4, 8, 16, 1000], "orders so far (frequency)", labels=["1", "2", "3-4", "5-8", "9-16", "17+"], color=GREEN)
    _rate_bars(axes[1], d["orders_last_90d"], d["lapsed"], [-1, 0, 1, 2, 4, 1000], "orders in the last 90 days", labels=["0", "1", "2", "3-4", "5+"], color=PURPLE)
    gap = d["avg_gap_days"].fillna(-5)
    _rate_bars(axes[2], gap, d["lapsed"], [-10, 0, 20, 40, 70, 120, 1000], "average days between orders", labels=["one order", "under 20", "20-40", "40-70", "70-120", "120+"], color=ORANGE)
    fig.suptitle("Pillar 3: who lapses (all six cutoffs combined; dashed line = overall lapse rate)", fontweight="bold", color=NAVY, y=1.03)
    save(fig, "s4_07_retail_lapse_drivers.png")


def retail_corr():
    d = _all_snapshots()
    cols = ["lapsed", "recency_days", "frequency", "monetary", "tenure_days", "avg_order_value", "avg_gap_days", "orders_last_90d", "orders_prev_90d", "order_trend", "return_share", "is_uk"]
    df = d[cols].rename(columns={"lapsed": "lapsed (target)"})
    _corr_heatmap(df, "Pillar 3: correlations between customer features and the target", "s4_08_retail_corr.png", figsize=(9.6, 8), font=7.5)


# ============================================================================ stage 5
def hvac_gap_features():
    h = data.load_hvac()
    gap = h["ambient_c"] - h["indoor_c"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, x, title, xl in ((axes[0], h["ambient_c"], "Raw input: outdoor temperature", "outdoor temperature (°C)"), (axes[1], gap, "Signed gap: outdoor minus indoor", "gap (°C)"),
                             (axes[2], gap.abs(), "Engineered: absolute gap", "|outdoor - indoor| (°C)")):
        for m in ("Cooling", "Heating", "Idle"):
            sel = h["mode"] == m
            ax.scatter(x[sel], h.loc[sel, "power_kw"], s=4, alpha=0.35, color=MODE_COLORS[m], label=m)
        ax.set(xlabel=xl, title=title)
    axes[0].set_ylabel("power (kW)"); axes[0].legend(markerscale=3, fontsize=8.5)
    for m in ("Cooling", "Heating"):
        sel = h["mode"] == m
        co = np.polyfit(gap.abs()[sel], h.loc[sel, "power_kw"], 1)
        xs = np.linspace(gap.abs()[sel].min(), gap.abs()[sel].max(), 50)
        axes[2].plot(xs, np.polyval(co, xs), color="black", lw=1.4)
    axes[2].text(0.03, 0.9, "straight lines fit per mode", transform=axes[2].transAxes, fontsize=8.5)
    fig.suptitle("Why the absolute gap helps: power is a V-shape in temperature but nearly a straight line in the gap", fontweight="bold", color=NAVY, y=1.03)
    save(fig, "s5_02_hvac_gap_features.png")


def ai4i_engineered():
    a = data.load_ai4i()
    a = a.assign(temp_gap=a["process_temp_k"] - a["air_temp_k"], power_w=a["torque_nm"] * a["rpm"] * 2 * np.pi / 60, wear_torque=a["tool_wear_min"] * a["torque_nm"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (c, lab, col) in zip(axes, (("temp_gap", "temperature gap (process - air, K)", BLUE), ("power_w", "mechanical power (W)", ORANGE), ("wear_torque", "wear x torque", PURPLE))):
        bins = pd.qcut(a[c], 10, duplicates="drop")
        r = a.groupby(bins, observed=True)["failure"].mean() * 100
        ax.bar(range(len(r)), r.values, color=col)
        ax.axhline(a["failure"].mean() * 100, color="black", ls="--", lw=1)
        ax.set_xticks(range(len(r))); ax.set_xticklabels([f"{i.mid:.0f}" if abs(i.mid) >= 10 else f"{i.mid:.1f}" for i in r.index], rotation=45, ha="right", fontsize=8)
        ax.set(xlabel=f"{lab}, ten equal-size groups (group midpoint)", ylabel="failure rate (%)")
    fig.suptitle("Pillar 2: the engineered inputs separate failures better than the raw sensors (dashed = 3.4% average)", fontweight="bold", color=NAVY, y=1.03)
    save(fig, "s5_03_ai4i_engineered.png")


def churn_engineered():
    d = _all_snapshots()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    _rate_bars(axes[0], d["order_trend"], d["lapsed"], [-100, -3, -1, 0, 1, 3, 100], "orders in last 90 days minus the 90 days before", labels=["-4 or less", "-3 to -2", "-1", "0", "1-2", "3+"], color=GREEN)
    _rate_bars(axes[1], d["return_share"], d["lapsed"], [-0.01, 0.0001, 0.1, 0.25, 1.0], "share of the customer's invoices that were returns", labels=["none", "up to 10%", "10-25%", "over 25%"], color=RED)
    _rate_bars(axes[2], d["tenure_days"], d["lapsed"], [-1, 90, 180, 365, 540, 800], "days since the first purchase (tenure)", labels=["under 3 months", "3-6 months", "6-12 months", "12-18 months", "over 18 months"], color=PURPLE)
    fig.suptitle("Pillar 3: engineered customer features and the lapse rate", fontweight="bold", color=NAVY, y=1.03)
    save(fig, "s5_04_churn_engineered.png")


def basic_vs_engineered():
    s = load_summary()
    panels = [("demand", "forecast", "mae", "Demand, forecast months\n(MAE kW, lower is better)"), ("demand", "heatwave", "mae", "Demand, heatwave\n(MAE kW, lower is better)"),
              ("failure", "stratified_20pct", "pr_auc", "Failure\n(PR-AUC, higher is better)"), ("churn", "unseen_customers_later_period", "roc_auc", "Churn\n(ROC-AUC, higher is better)")]
    models = [("Ridge (basic features)", BLUE, 0.45), (R, BLUE, 1.0), ("XGBoost (basic features)", ORANGE, 0.45), (X, ORANGE, 1.0)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    for ax, (p, sp, m, title) in zip(axes, panels):
        for i, (name, c, a) in enumerate(models):
            v, e = val(s, p, sp, name, m), sd(s, p, sp, name, m)
            ax.bar(i, v, color=c, alpha=a, yerr=e if not np.isnan(e) else 0, capsize=3)
            ax.text(i, v + (e if not np.isnan(e) else 0) + (ax.get_ylim()[1] * 0.01), f"{v:.2f}" if m != "mae" else f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(range(4)); ax.set_xticklabels(["Ridge\nbasic", "Ridge\nengin.", "XGB\nbasic", "XGB\nengin."], fontsize=8.5); ax.set_title(title, fontsize=10)
    fig.suptitle("What the engineered features add (light bars: basic inputs, dark bars: engineered inputs)", fontweight="bold", color=NAVY, y=1.03)
    save(fig, "s5_05_basic_vs_engineered.png")


ALL = [rule_of_thumb, hvac_year_overview, ai4i_composition, retail_activity, retail_cleaning_funnel, invoice_value_outliers, hvac_heatmap, hvac_modes, hvac_corr,
       ai4i_distributions, ai4i_corr, retail_distributions, retail_lapse_drivers, retail_corr, hvac_gap_features, ai4i_engineered, churn_engineered, basic_vs_engineered]
