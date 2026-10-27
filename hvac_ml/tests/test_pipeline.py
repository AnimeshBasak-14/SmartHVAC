"""
Automatic checks. Run from the project folder:   python -m pytest -q

The most important ones guard against look-ahead leakage in the churn task and against train/test overlap.
"""
import numpy as np
import pandas as pd
import pytest

from hvac_ml import config, data, pillar1_demand as p1, pillar2_failure as p2, pillar3_churn as p3
from hvac_ml import monitoring
from hvac_ml.common import best_f1_threshold, classification_metrics, regression_metrics, top_share


@pytest.fixture(scope="module")
def hvac():
    return data.load_hvac()


@pytest.fixture(scope="module")
def ai4i():
    return data.load_ai4i()


@pytest.fixture(scope="module")
def invoices():
    return data.load_invoices()


# data
def test_datasets_load_with_expected_size(hvac, ai4i, invoices):
    assert hvac.shape[0] == 8760 and ai4i.shape[0] == 10000
    assert {"customer", "invoice", "date", "value", "is_return", "country"} <= set(invoices.columns)
    assert abs(ai4i["failure"].mean() - 0.0339) < 0.001
    assert invoices["customer"].nunique() == 5939


# Pillar 1 splits
@pytest.mark.parametrize("split", p1.SPLITS)
def test_demand_splits_are_not_empty_and_do_not_overlap(hvac, split):
    test = p1.make_split(hvac, split, 42)
    assert 0 < test.sum() < len(hvac)
    if split == "heatwave":
        assert (hvac.loc[test, "ambient_c"] >= config.HEATWAVE_TEMP_C).all()
        assert (hvac.loc[~test, "ambient_c"] < config.HEATWAVE_TEMP_C).all()
    if split == "unseen_months":
        assert set(hvac.loc[test, "month"]) == set(config.HELD_OUT_MONTHS)
        assert not set(hvac.loc[~test, "month"]) & set(config.HELD_OUT_MONTHS)
    if split == "forecast":
        assert hvac.loc[~test, "month"].max() < min(config.FORECAST_MONTHS)


def test_demand_features_have_no_missing_values(hvac):
    for engineered in (False, True):
        X = p1.build_features(hvac, engineered)
        assert not X.isna().any().any() and len(X) == len(hvac)


# Pillar 2
def test_failure_features(ai4i):
    X = p2.build_features(ai4i, True)
    assert not X.isna().any().any()
    assert {"temp_gap_k", "power_w", "wear_x_torque"} <= set(X.columns)
    assert "failure" not in X.columns and not any(c in X.columns for c in ["TWF", "HDF", "PWF", "OSF", "RNF"])  # no target leakage


# Pillar 3: the leakage checks
def test_churn_features_use_only_the_past(invoices):
    """Deleting every invoice after the cutoff must not change a single feature."""
    cutoff = pd.Timestamp(config.TEST_CUTOFF)
    countries = invoices.drop_duplicates("customer").set_index("customer")["country"]
    full = p3.customer_snapshot(invoices, cutoff, countries).set_index("customer")
    past_only = p3.customer_snapshot(invoices[invoices["date"] < cutoff], cutoff, countries).set_index("customer")
    common = full.index.intersection(past_only.index)
    assert len(common) == len(full) > 500
    pd.testing.assert_frame_equal(full.loc[common, p3.ENGINEERED], past_only.loc[common, p3.ENGINEERED])


def test_churn_label_comes_from_the_future(invoices):
    """A customer is 'lapsed' exactly when they have no purchase in the 90 days after the cutoff."""
    cutoff = pd.Timestamp(config.TEST_CUTOFF)
    countries = invoices.drop_duplicates("customer").set_index("customer")["country"]
    snap = p3.customer_snapshot(invoices, cutoff, countries)
    window = invoices[(invoices["date"] >= cutoff) & (invoices["date"] < cutoff + pd.Timedelta(days=90)) & (~invoices["is_return"])]
    buyers = set(window["customer"])
    assert (snap["lapsed"] == (~snap["customer"].isin(buyers)).astype(int)).all()
    assert (snap["recency_days"] <= config.ACTIVE_WINDOW_DAYS).all()      # only customers who are active now


def test_train_cutoffs_end_before_test_cutoff():
    last_train_window_end = pd.Timestamp(config.TRAIN_CUTOFFS[-1]) + pd.Timedelta(days=config.CHURN_HORIZON_DAYS)
    assert last_train_window_end <= pd.Timestamp(config.TEST_CUTOFF)


@pytest.mark.parametrize("seed", [42, 43])
def test_churn_test_customers_never_appear_in_training(invoices, seed):
    snaps = p3.build_snapshots(invoices)
    all_train = pd.concat([snaps[c] for c in config.TRAIN_CUTOFFS], ignore_index=True)
    train, test = p3.split_customers(all_train, snaps[config.TEST_CUTOFF], seed)
    assert len(train) > 1000 and len(test) > 100
    assert not set(train["customer"]) & set(test["customer"])
    assert (test["cutoff"] == pd.Timestamp(config.TEST_CUTOFF)).all()
    assert (train["cutoff"] < pd.Timestamp(config.TEST_CUTOFF)).all()


# metrics
def test_metrics_behave_sensibly():
    y = np.array([0] * 90 + [1] * 10)
    perfect = y.astype(float)
    assert top_share(y, perfect, 0.10) == 1.0                      # all positives in the top 10%
    assert abs(top_share(y, np.random.default_rng(0).random(100), 0.10) - 0.10) < 0.35
    thr = best_f1_threshold(y, perfect)
    m = classification_metrics(y, perfect, thr)
    assert m["f1"] == 1.0 and m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    r = regression_metrics([1, 2, 3, 4], [1, 2, 3, 4])
    assert r["mae"] == 0 and r["r2"] == 1.0


# a short end-to-end run
def test_quick_end_to_end_demand_and_failure(hvac, ai4i, monkeypatch):
    monkeypatch.setattr(config, "SEEDS", [42])
    r1, _ = p1.run(hvac)
    r2, _ = p2.run(ai4i)
    ridge = r1[(r1["split"] == "heatwave") & (r1["model"] == "Ridge (engineered features)")]["mae"].iloc[0]
    xgb = r1[(r1["split"] == "heatwave") & (r1["model"] == "XGBoost (engineered features)")]["mae"].iloc[0]
    assert ridge < xgb                                              # the heatwave finding: trees cannot extrapolate
    best = r2[r2["model"] == "XGBoost (engineered features)"]["pr_auc"].iloc[0]
    rule = r2[r2["model"] == "Reference: most worn tool first"]["pr_auc"].iloc[0]
    assert best > 3 * rule                                          # the model clearly beats the rule of thumb


# monitoring
def test_psi_is_small_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    a, b, shifted = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000)
    assert monitoring.psi(a, b) < 0.05
    assert monitoring.psi(a, shifted) > config.PSI_ALERT
    assert monitoring.status_from_psi(0.30) == "alert" and monitoring.status_from_psi(0.15) == "warning" and monitoring.status_from_psi(0.01) == "ok"


def test_drift_table_flags_the_column_that_moved():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"stable": rng.normal(size=2000), "moved": rng.normal(size=2000)})
    new = pd.DataFrame({"stable": rng.normal(size=2000), "moved": rng.normal(2.0, 1, size=2000)})
    table = monitoring.drift_table(ref, new, ["stable", "moved"]).set_index("feature")
    assert table.loc["stable", "status"] == "ok" and table.loc["moved", "status"] == "alert"


def test_performance_status_uses_the_ratio_to_the_tested_error():
    assert monitoring.performance_status(0.07, 0.07)[0] == "ok"
    assert monitoring.performance_status(0.11, 0.07)[0] == "warning"
    assert monitoring.performance_status(0.20, 0.07)[0] == "alert"
