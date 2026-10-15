"""
train.py - train, validate, tune and test all models.

For EACH target (heating_load, cooling_load):
  1. Compare every regression model:  5-fold cross-validation on TRAIN + score on VALIDATION
  2. Pick the model with the lowest validation RMSE
  3. Fine-tune it with GridSearchCV (small grid)
  4. Retrain on TRAIN + VALIDATION, then score ONCE on the hidden TEST set
  5. Build a 90% prediction interval (split-conformal idea, using cross-validated errors)
  6. Measure feature importance on the TEST set (permutation importance)

Then the CLASSIFICATION task (Low / Medium / High energy demand):
  - compare classifiers the same way (macro F1 on validation)
  - compare with "regress-then-classify" (use the predicted loads to assign the class)
"""
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_validate

from src import config
from src.data import demand_thresholds, make_demand_labels
from src.models import PARAM_GRIDS, classification_models, regression_models

log = logging.getLogger(__name__)
X_COLS, CV = config.FEATURES, KFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def conformal_margin(model, X, y, coverage=config.INTERVAL_COVERAGE):
    """Half-width of the prediction interval: the `coverage` quantile of cross-validated
    absolute errors (with the standard (n+1) finite-sample correction)."""
    errors = np.abs(y - cross_val_predict(clone(model), X, y, cv=CV))
    level = min(1.0, np.ceil((len(errors) + 1) * coverage) / len(errors))
    return float(np.quantile(errors, level))


def train_regressors(train, val, test, quick=False):
    """Returns (comparison_df, test_df, artefacts) for both targets."""
    rows, test_rows, art = [], [], {}
    train_val = pd.concat([train, val])

    for target in config.TARGETS:
        log.info("=" * 60)
        log.info("TARGET: %s", target)
        Xtr, ytr, Xva, yva = train[X_COLS], train[target], val[X_COLS], val[target]

        # ---- 1. compare models -------------------------------------------------
        fitted = {}
        for name, model in regression_models(quick).items():
            t0 = time.time()
            cv = cross_validate(clone(model), Xtr, ytr, cv=CV,
                                scoring=("r2", "neg_root_mean_squared_error"))
            model.fit(Xtr, ytr)
            fitted[name] = model
            m = regression_metrics(yva, model.predict(Xva))
            rows.append({"target": target, "model": name,
                         "cv_R2_mean": cv["test_r2"].mean(), "cv_R2_std": cv["test_r2"].std(),
                         "cv_RMSE_mean": -cv["test_neg_root_mean_squared_error"].mean(),
                         "val_MAE": m["MAE"], "val_RMSE": m["RMSE"], "val_R2": m["R2"],
                         "train_seconds": round(time.time() - t0, 2)})
            log.info("  %-27s CV R2=%.4f  |  val RMSE=%.3f  val R2=%.4f", name,
                     cv["test_r2"].mean(), m["RMSE"], m["R2"])

        target_rows = [r for r in rows if r["target"] == target]
        best_name = min(target_rows, key=lambda r: r["val_RMSE"])["model"]
        best_model, best_rmse = fitted[best_name], min(r["val_RMSE"] for r in target_rows)
        log.info("  -> best on validation: %s (RMSE %.3f)", best_name, best_rmse)

        # ---- 2. hyper-parameter tuning -----------------------------------------
        best_params = "default"
        if best_name in PARAM_GRIDS and not quick:
            grid = GridSearchCV(clone(best_model), PARAM_GRIDS[best_name], cv=CV,
                                scoring="neg_root_mean_squared_error", n_jobs=-1)
            grid.fit(Xtr, ytr)
            tuned_rmse = regression_metrics(yva, grid.best_estimator_.predict(Xva))["RMSE"]
            log.info("  tuned params %s -> val RMSE %.3f", grid.best_params_, tuned_rmse)
            if tuned_rmse < best_rmse:
                best_model, best_params = grid.best_estimator_, grid.best_params_
            else:
                log.info("  tuning did not help on validation; keeping default settings")

        # ---- 3. final model: retrain on train+val, test ONCE -------------------
        final = clone(best_model).fit(train_val[X_COLS], train_val[target])
        y_pred = final.predict(test[X_COLS])
        tm = regression_metrics(test[target], y_pred)
        margin = conformal_margin(best_model, train_val[X_COLS], train_val[target])
        covered = np.mean(np.abs(test[target] - y_pred) <= margin)
        log.info("  TEST: MAE=%.3f RMSE=%.3f R2=%.4f | 90%% interval ±%.2f covers %.1f%% of test",
                 tm["MAE"], tm["RMSE"], tm["R2"], margin, 100 * covered)

        # baseline: always predict the training mean (shows how much ML helps)
        base = regression_metrics(test[target], np.full(len(test), train_val[target].mean()))
        test_rows.append({"target": target, "model": best_name, "params": str(best_params),
                          "test_MAE": tm["MAE"], "test_RMSE": tm["RMSE"], "test_R2": tm["R2"],
                          "baseline_RMSE": base["RMSE"], "interval_margin": margin,
                          "interval_coverage_test": covered})

        # ---- 4. which features matter? -----------------------------------------
        imp = permutation_importance(final, test[X_COLS], test[target], n_repeats=20,
                                     random_state=config.RANDOM_STATE, n_jobs=-1)
        art[target] = {"model": final, "name": best_name, "margin": margin, "y_pred": y_pred,
                       "importance": pd.Series(imp.importances_mean, index=X_COLS).sort_values()}

    return pd.DataFrame(rows), pd.DataFrame(test_rows), art


def train_classifiers(train, val, test, reg_art, quick=False):
    """Low / Medium / High demand classification + regress-then-classify comparison."""
    log.info("=" * 60)
    log.info("CLASSIFICATION: Low / Medium / High energy demand")
    thresholds = demand_thresholds(train[config.TARGETS].sum(axis=1))
    log.info("  class thresholds (total load, from TRAIN only): %.2f, %.2f", *thresholds)
    labels = {name: make_demand_labels(df[config.TARGETS].sum(axis=1), thresholds)
              for name, df in [("train", train), ("val", val), ("test", test)]}

    rows, fitted = [], {}
    for name, model in classification_models(quick).items():
        cv = cross_validate(clone(model), train[X_COLS], labels["train"], cv=CV, scoring="f1_macro")
        model.fit(train[X_COLS], labels["train"])
        fitted[name] = model
        pred = model.predict(val[X_COLS])
        rows.append({"model": name, "cv_F1_mean": cv["test_score"].mean(),
                     "val_accuracy": accuracy_score(labels["val"], pred),
                     "val_F1_macro": f1_score(labels["val"], pred, average="macro")})
        log.info("  %-22s CV F1=%.4f | val acc=%.4f F1=%.4f", name, rows[-1]["cv_F1_mean"],
                 rows[-1]["val_accuracy"], rows[-1]["val_F1_macro"])

    best_name = max(rows, key=lambda r: r["val_F1_macro"])["model"]
    y_tv = np.concatenate([labels["train"], labels["val"]])
    final = clone(fitted[best_name]).fit(pd.concat([train, val])[X_COLS], y_tv)
    pred = final.predict(test[X_COLS])

    # Alternative: use the two regression models, add the loads, then apply the thresholds
    total_pred = reg_art["heating_load"]["y_pred"] + reg_art["cooling_load"]["y_pred"]
    pred_via_reg = make_demand_labels(total_pred, thresholds)

    summary = {
        "best_model": best_name,
        "thresholds": thresholds,
        "test_accuracy": accuracy_score(labels["test"], pred),
        "test_F1_macro": f1_score(labels["test"], pred, average="macro"),
        "regress_then_classify_accuracy": accuracy_score(labels["test"], pred_via_reg),
        "regress_then_classify_F1_macro": f1_score(labels["test"], pred_via_reg, average="macro"),
        "confusion_matrix": confusion_matrix(labels["test"], pred, labels=config.DEMAND_CLASSES),
        "report": classification_report(labels["test"], pred, labels=config.DEMAND_CLASSES, digits=3),
    }
    log.info("  -> best: %s | TEST acc=%.4f F1=%.4f | regress-then-classify acc=%.4f",
             best_name, summary["test_accuracy"], summary["test_F1_macro"],
             summary["regress_then_classify_accuracy"])
    return pd.DataFrame(rows), summary, final


def build_bundle(reg_art, clf_summary, classifier):
    """Everything the GUI / predict.py needs, in one object saved with joblib."""
    return {
        "regressors": {t: reg_art[t]["model"] for t in config.TARGETS},
        "regressor_names": {t: reg_art[t]["name"] for t in config.TARGETS},
        "margins": {t: reg_art[t]["margin"] for t in config.TARGETS},
        "classifier": classifier,
        "classifier_name": clf_summary["best_model"],
        "thresholds": clf_summary["thresholds"],
        "features": X_COLS,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sklearn_version": sklearn.__version__,
    }
