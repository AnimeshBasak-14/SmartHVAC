"""
pillar3_churn.py - Pillar 3: which active customers will go quiet ("sleepers")?

The Online Retail II file only has transactions, so we build the table ourselves:
  * pick a cutoff date
  * features  = what we know about each customer from purchases BEFORE the cutoff
  * target    = 1 if the customer buys nothing in the 90 days AFTER the cutoff (they become a sleeper: no purchase for 90+ days)
  * only customers who bought in the 90 days before the cutoff are included (they are "active now")
Using the "Recency > 90 days" flag as the target and Recency as an input would be circular, so we avoid it.

Split (no information crosses it):
  * customers are split 80/20
  * training rows : the 5 earlier cutoffs (2010-06 to 2011-06), training customers only
  * test rows     : the cutoff 2011-09-09 (the Sep-Dec holiday season), test customers only
Two feature sets: "basic" (recency, frequency, monetary) and "engineered" (adds tenure, order value,
gaps between orders, recent trend, returns, country).
"""
import numpy as np
import pandas as pd

from hvac_ml import config
from hvac_ml.common import (best_f1_threshold, classification_metrics, forest_classifier, logistic_classifier, oof_scores,
                            ridge_classifier, xgb_classifier)

SPLIT = "unseen_customers_later_period"
BASIC = ["recency_days", "frequency", "monetary"]
ENGINEERED = BASIC + ["tenure_days", "avg_order_value", "avg_gap_days", "avg_items", "avg_products", "orders_last_90d",
                      "orders_prev_90d", "order_trend", "return_share", "is_uk", "log_monetary", "log_frequency", "log_avg_order_value"]


def customer_snapshot(inv, cutoff, countries):
    """One row per customer who was active in the 90 days before `cutoff`, with features and the 'lapsed' target.
    `inv` has one row per invoice (see data.to_invoices)."""
    cutoff = pd.Timestamp(cutoff)
    horizon = pd.Timedelta(days=config.CHURN_HORIZON_DAYS)
    past = inv[inv["date"] < cutoff]                       # only what we knew at the cutoff
    sales = past[~past["is_return"]]
    g = sales.groupby("customer")
    feat = pd.DataFrame({"last": g["date"].max(), "first": g["date"].min(), "frequency": g.size(), "monetary": g["value"].sum(),
                         "avg_items": g["items"].mean(), "avg_products": g["products"].mean()})
    feat["recency_days"] = (cutoff - feat["last"]).dt.days
    feat["tenure_days"] = (cutoff - feat["first"]).dt.days
    feat["avg_order_value"] = feat["monetary"] / feat["frequency"]
    feat["avg_gap_days"] = np.where(feat["frequency"] > 1, feat["tenure_days"] / (feat["frequency"] - 1).clip(lower=1), np.nan)
    last90 = sales[sales["date"] >= cutoff - pd.Timedelta(days=90)].groupby("customer").size()
    prev90 = sales[(sales["date"] >= cutoff - pd.Timedelta(days=180)) & (sales["date"] < cutoff - pd.Timedelta(days=90))].groupby("customer").size()
    feat["orders_last_90d"] = last90.reindex(feat.index).fillna(0)
    feat["orders_prev_90d"] = prev90.reindex(feat.index).fillna(0)
    feat["order_trend"] = feat["orders_last_90d"] - feat["orders_prev_90d"]
    returns = past[past["is_return"]].groupby("customer").size().reindex(feat.index).fillna(0)
    feat["return_share"] = returns / (returns + feat["frequency"])
    feat["is_uk"] = (countries.reindex(feat.index) == "United Kingdom").astype(float)
    feat["log_monetary"] = np.log1p(feat["monetary"])
    feat["log_frequency"] = np.log1p(feat["frequency"])
    feat["log_avg_order_value"] = np.log1p(feat["avg_order_value"])

    feat = feat[feat["recency_days"] <= config.ACTIVE_WINDOW_DAYS].copy()      # active right now
    future = inv[(inv["date"] >= cutoff) & (inv["date"] < cutoff + horizon) & (~inv["is_return"])]
    feat["lapsed"] = (~feat.index.isin(set(future["customer"]))).astype(int)
    feat["cutoff"] = cutoff
    return feat.drop(columns=["last", "first"]).reset_index().rename(columns={"index": "customer"})


def build_snapshots(inv):
    countries = inv.drop_duplicates("customer").set_index("customer")["country"]
    return {c: customer_snapshot(inv, c, countries) for c in config.TRAIN_CUTOFFS + [config.TEST_CUTOFF]}


def model_zoo(seed):
    return {
        "Logistic regression (engineered features)": (ENGINEERED, logistic_classifier),   # statistical baseline
        "Random Forest (engineered features)": (ENGINEERED, lambda: forest_classifier(seed)),   # classic ML baseline
        "Ridge (basic features)": (BASIC, ridge_classifier),
        "Ridge (engineered features)": (ENGINEERED, ridge_classifier),
        "XGBoost (basic features)": (BASIC, lambda: xgb_classifier(seed)),
        "XGBoost (engineered features)": (ENGINEERED, lambda: xgb_classifier(seed)),
    }


def split_customers(all_train, test_all, seed):
    """Train rows (earlier cutoffs, training customers) and test rows (later cutoff, unseen customers)."""
    customers = np.array(sorted(set(all_train["customer"]) | set(test_all["customer"])))
    rng = np.random.default_rng(seed)
    test_customers = set(rng.permutation(customers)[: int(round(config.TEST_SIZE * len(customers)))])
    train = all_train[~all_train["customer"].isin(test_customers)].reset_index(drop=True)
    test = test_all[test_all["customer"].isin(test_customers)].reset_index(drop=True)
    return train, test


def run(snapshots, keep_predictions_for_seed=None):
    rows, kept = [], {}
    all_train = pd.concat([snapshots[c] for c in config.TRAIN_CUTOFFS], ignore_index=True)
    test_all = snapshots[config.TEST_CUTOFF]
    for seed in config.SEEDS:
        train, test = split_customers(all_train, test_all, seed)
        ytr, yte = train["lapsed"], test["lapsed"]
        # reference: the classic rule "the longer since the last purchase, the more likely to lapse"
        thr = best_f1_threshold(ytr, train["recency_days"].to_numpy(float))
        rows.append({"pillar": "churn", "split": SPLIT, "model": "Reference: recency only", "seed": seed, "threshold": thr,
                     **classification_metrics(yte, test["recency_days"].to_numpy(float), thr), "n_train": len(train), "n_test": len(test)})
        for name, (cols, factory) in model_zoo(seed).items():
            thr = best_f1_threshold(ytr, oof_scores(factory, train[cols], ytr, groups=train["customer"], seed=seed))
            model = factory().fit(train[cols], ytr)
            scores = model.predict_proba(test[cols])[:, 1]
            rows.append({"pillar": "churn", "split": SPLIT, "model": name, "seed": seed, "threshold": thr,
                         **classification_metrics(yte, scores, thr), "n_train": len(train), "n_test": len(test)})
            if seed == keep_predictions_for_seed:
                kept[name] = (yte.to_numpy(), scores, thr)
        if seed == keep_predictions_for_seed:
            kept["Reference: recency only"] = (yte.to_numpy(), test["recency_days"].to_numpy(float), None)
    return pd.DataFrame(rows), kept


def fit_final(snapshots, seed=config.SEEDS[0]):
    """Final models on every snapshot we have (all customers, all cutoffs), with the F1 threshold from out-of-fold scores."""
    data = pd.concat(snapshots.values(), ignore_index=True)
    y = data["lapsed"]
    final = {}
    for name, (cols, factory) in model_zoo(seed).items():
        if name in ("Ridge (engineered features)", "XGBoost (engineered features)"):
            thr = best_f1_threshold(y, oof_scores(factory, data[cols], y, groups=data["customer"], seed=seed))
            final[name] = (factory().fit(data[cols], y), thr)
    return final
