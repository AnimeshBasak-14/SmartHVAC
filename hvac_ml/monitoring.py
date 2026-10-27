"""
monitoring.py - checks to run after the models are deployed (lifecycle stage 10).

Three kinds of check:
  1. Input drift      : have the inputs moved away from what the model was trained on?
                        (Population Stability Index and the Kolmogorov-Smirnov test, one value per input column)
  2. Performance      : is the error still as small as in testing? (needs the true outcome, so it runs after the fact)
  3. Rate calibration : does the predicted share of positives match the share seen later? (churn, failure)

Each check returns a plain table and a status ("ok", "warning", "alert") using the thresholds in config.py.
"""
import numpy as np
import pandas as pd
from scipy import stats

from hvac_ml import config


def psi(reference, new, bins=10):
    """Population Stability Index. Bin edges are the quantiles of the reference sample.
    Common reading: under 0.10 stable, 0.10 to 0.25 moderate shift, above 0.25 large shift."""
    reference, new = np.asarray(reference, float), np.asarray(new, float)
    reference, new = reference[~np.isnan(reference)], new[~np.isnan(new)]
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                                   # almost constant column
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    new_share = np.histogram(new, edges)[0] / len(new)
    ref_share, new_share = np.clip(ref_share, 1e-4, None), np.clip(new_share, 1e-4, None)   # avoid log(0)
    return float(np.sum((new_share - ref_share) * np.log(new_share / ref_share)))


def status_from_psi(value):
    if value >= config.PSI_ALERT:
        return "alert"
    if value >= config.PSI_WARN:
        return "warning"
    return "ok"


def drift_table(reference_df, new_df, columns):
    """One row per input column: PSI, KS statistic and p-value, shift of the mean (in reference standard deviations), status."""
    rows = []
    for col in columns:
        ref, new = reference_df[col].to_numpy(float), new_df[col].to_numpy(float)
        ks = stats.ks_2samp(ref[~np.isnan(ref)], new[~np.isnan(new)])
        sd = np.nanstd(ref) or 1.0
        value = psi(ref, new)
        rows.append({"feature": col, "psi": value, "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
                     "mean_shift_sd": float((np.nanmean(new) - np.nanmean(ref)) / sd), "status": status_from_psi(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def error_by_period(period, y_true, y_pred):
    """Mean absolute error for every value of `period` (for example the month)."""
    df = pd.DataFrame({"period": np.asarray(period), "abs_err": np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))})
    return df.groupby("period")["abs_err"].agg(["mean", "count"]).rename(columns={"mean": "mae"}).reset_index()


def performance_status(current_mae, reference_mae):
    """Compare the error of a recent period with the error measured in testing."""
    ratio = current_mae / reference_mae if reference_mae > 0 else np.inf
    if ratio >= config.ERROR_ALERT_RATIO:
        return "alert", ratio
    if ratio >= config.ERROR_WARN_RATIO:
        return "warning", ratio
    return "ok", ratio


def rate_check(y_true, scores, threshold):
    """Compare the share predicted positive at `threshold` and the average predicted probability with the share that really was positive."""
    y_true, scores = np.asarray(y_true, float), np.asarray(scores, float)
    return {"actual_rate": float(y_true.mean()), "mean_predicted_probability": float(scores.mean()),
            "share_flagged": float((scores >= threshold).mean()),
            "calibration_gap": float(scores.mean() - y_true.mean())}
