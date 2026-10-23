"""
pillar2_failure.py - Pillar 2: predict equipment failure from sensor readings.

Question: which machines should a technician inspect first? Only about 3.4% of the 10,000 AI4I records are
failures, so accuracy is meaningless (predicting "no failure" is 96.6% accurate). We report ROC-AUC, PR-AUC
(average precision), F1 at a threshold chosen on the TRAINING data only, and the share of all failures that
fall in the 10% of machines with the highest risk score.

Split: stratified 80/20, repeated with 5 seeds.
Two feature sets: "basic" (raw sensors and machine type) and "engineered" (adds the temperature gap, mechanical
power = torque x speed, and wear x torque, which are the physical quantities behind the failure modes).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hvac_ml import config
from hvac_ml.common import (best_f1_threshold, classification_metrics, forest_classifier, logistic_classifier, oof_scores,
                            ridge_classifier, xgb_classifier)

SPLIT = "stratified_20pct"


def build_features(df, engineered):
    X = pd.DataFrame({"air_temp_k": df["air_temp_k"], "process_temp_k": df["process_temp_k"], "rpm": df["rpm"],
                      "torque_nm": df["torque_nm"], "tool_wear_min": df["tool_wear_min"]})
    for t in ["L", "M", "H"]:
        X[f"type_{t}"] = (df["Type"] == t).astype(float)
    if engineered:
        X["temp_gap_k"] = df["process_temp_k"] - df["air_temp_k"]
        X["power_w"] = df["torque_nm"] * df["rpm"] * 2 * np.pi / 60
        X["wear_x_torque"] = df["tool_wear_min"] * df["torque_nm"]
    return X


def model_zoo(seed):
    return {
        "Logistic regression (engineered features)": (True, logistic_classifier),    # statistical baseline
        "Random Forest (engineered features)": (True, lambda: forest_classifier(seed)),   # classic ML baseline
        "Ridge (basic features)": (False, ridge_classifier),
        "Ridge (engineered features)": (True, ridge_classifier),
        "XGBoost (basic features)": (False, lambda: xgb_classifier(seed)),
        "XGBoost (engineered features)": (True, lambda: xgb_classifier(seed)),
    }


def run(df, keep_predictions_for_seed=None):
    rows, kept = [], {}
    y = df["failure"]
    for seed in config.SEEDS:
        train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=config.TEST_SIZE, stratify=y, random_state=seed)
        # rule of thumb a workshop could use today: look at the most worn tools first
        wear_thr = best_f1_threshold(y.iloc[train_idx], df["tool_wear_min"].iloc[train_idx].to_numpy(float))
        wear_scores = df["tool_wear_min"].iloc[test_idx].to_numpy(float)
        rows.append({"pillar": "failure", "split": SPLIT, "model": "Reference: most worn tool first", "seed": seed, "threshold": wear_thr,
                     **classification_metrics(y.iloc[test_idx], wear_scores, wear_thr), "n_train": len(train_idx), "n_test": len(test_idx)})
        if seed == keep_predictions_for_seed:
            kept["Reference: most worn tool first"] = (y.iloc[test_idx].to_numpy(), wear_scores, wear_thr)
        for name, (engineered, factory) in model_zoo(seed).items():
            X = build_features(df, engineered)
            Xtr, ytr, Xte, yte = X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx]
            threshold = best_f1_threshold(ytr, oof_scores(factory, Xtr, ytr, seed=seed))
            model = factory().fit(Xtr, ytr)
            scores = model.predict_proba(Xte)[:, 1]
            rows.append({"pillar": "failure", "split": SPLIT, "model": name, "seed": seed, "threshold": threshold,
                         **classification_metrics(yte, scores, threshold), "n_train": len(train_idx), "n_test": len(test_idx)})
            if seed == keep_predictions_for_seed:
                kept[name] = (yte.to_numpy(), scores, threshold)
    return pd.DataFrame(rows), kept


def fit_final(df, seed=config.SEEDS[0]):
    """Final models on all 10,000 records, with the F1 threshold from out-of-fold scores."""
    final = {}
    y = df["failure"]
    X = build_features(df, True)
    for name, (engineered, factory) in model_zoo(seed).items():
        if name in ("Ridge (engineered features)", "XGBoost (engineered features)"):
            threshold = best_f1_threshold(y, oof_scores(factory, X, y, seed=seed))
            final[name] = (factory().fit(X, y), threshold)
    return final
