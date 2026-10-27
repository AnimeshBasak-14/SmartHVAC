"""
common.py - the two model families and the metrics used by all three pillars.

Statistical baseline : ordinary linear regression (numbers) and plain logistic regression (yes/no), no penalty.
Classic ML baseline  : Random Forest.
Ridge  : L2-penalised linear model. For numbers it is Ridge regression; for yes/no targets it is logistic
         regression with an L2 (ridge-type) penalty. The penalty strength is chosen by cross-validation
         on the training data only.
XGBoost: gradient-boosted decision trees with fixed settings from config.py.
"""
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hvac_ml import config

# scikit-learn 1.9 prints harmless FutureWarnings about LogisticRegressionCV settings that change in 1.10
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model._logistic")
warnings.filterwarnings("ignore", message="The fitted attributes of LogisticRegressionCV")


# models
def ridge_regressor():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", RidgeCV(alphas=config.RIDGE_ALPHAS))])


def xgb_regressor(seed):
    return xgb.XGBRegressor(**config.XGB_REG, random_state=seed, n_jobs=-1)


def ridge_classifier():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", LogisticRegressionCV(Cs=config.LOGISTIC_C, cv=5, scoring="average_precision", max_iter=3000))])


def xgb_classifier(seed):
    return xgb.XGBClassifier(**config.XGB_CLF, random_state=seed, n_jobs=-1, eval_metric="logloss")


def linear_regressor():
    """Statistical baseline for numbers: ordinary least squares."""
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LinearRegression())])


def logistic_classifier():
    """Statistical baseline for yes/no: logistic regression with a negligible penalty (C = 1e6 is almost no penalty)."""
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", LogisticRegression(C=1e6, max_iter=5000))])


def forest_regressor(seed):
    """Classic ML baseline for numbers."""
    return RandomForestRegressor(**config.RF, random_state=seed, n_jobs=-1)


def forest_classifier(seed):
    """Classic ML baseline for yes/no."""
    return RandomForestClassifier(**config.RF, random_state=seed, n_jobs=-1)


# regression metrics
def regression_metrics(y_true, y_pred, peak_quantile=0.9):
    """MAE, RMSE, R2, and the MAE on the busiest 10% of test hours (the hours that overload the grid)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    peak = y_true >= np.quantile(y_true, peak_quantile)
    return {"mae": mean_absolute_error(y_true, y_pred), "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan"),
            "peak_mae": mean_absolute_error(y_true[peak], y_pred[peak])}


# classification metrics
def oof_scores(model_factory, X, y, groups=None, n_splits=5, seed=0):
    """Out-of-fold predicted probabilities on the TRAINING data (used only to pick a decision threshold)."""
    if groups is None:
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)
    else:
        splitter = GroupKFold(n_splits).split(X, y, groups)
    scores = np.zeros(len(y))
    for tr, va in splitter:
        model = model_factory().fit(X.iloc[tr], y.iloc[tr])
        scores[va] = model.predict_proba(X.iloc[va])[:, 1]
    return scores


def best_f1_threshold(y, scores):
    """Threshold that maximises F1 on (training) scores."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
    f1s = [f1_score(y, scores >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(f1s))])


def top_share(y, scores, fraction):
    """Share of all positives found among the `fraction` highest-scored rows (gains / capture rate)."""
    y = np.asarray(y)
    n = max(1, int(round(fraction * len(y))))
    order = np.argsort(-np.asarray(scores), kind="stable")
    return float(y[order[:n]].sum() / max(1, y.sum()))


def classification_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true)
    pred = scores >= threshold
    top10 = top_share(y_true, scores, 0.10)
    return {"roc_auc": roc_auc_score(y_true, scores), "pr_auc": average_precision_score(y_true, scores),
            "f1": f1_score(y_true, pred, zero_division=0), "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0), "flagged_share": float(pred.mean()),
            "top10_capture": top10, "top10_lift": top10 / 0.10, "prevalence": float(y_true.mean())}


# summary
def summarise(results, metric_cols, keys=("pillar", "split", "model")):
    """Mean and standard deviation over seeds for every (pillar, split, model)."""
    grouped = results.groupby(list(keys), sort=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    return grouped.reset_index()
