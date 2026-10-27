"""
pillar1_demand.py - Pillar 1: forecast hourly HVAC power demand (kW) from weather and calendar.

Question: can we predict how much power a building's HVAC will draw, so a service company can plan
technicians, spare units and grid load before a heatwave or a cold snap?

Four test situations (the training data never contains the test rows):
  random         : random 20% of hours (easiest, optimistic)
  unseen_months  : months 3, 6, 9 and 12 are held out
  forecast       : train on months 1-9, predict months 10-12
  heatwave       : train below 40 C ambient, predict the hours at or above 40 C
Two feature sets: "basic" (raw columns) and "engineered" (adds the indoor/outdoor temperature gap, which
is what drives HVAC power).
"""
import numpy as np
import pandas as pd

from hvac_ml import config
from hvac_ml.common import forest_regressor, linear_regressor, regression_metrics, ridge_regressor, xgb_regressor

MODES = ["Cooling", "Heating", "Idle"]
SPLITS = ["random", "unseen_months", "forecast", "heatwave"]


def build_features(df, engineered):
    X = pd.DataFrame({
        "ambient_c": df["ambient_c"], "indoor_c": df["indoor_c"], "humidity": df["humidity"],
        "hour_sin": np.sin(2 * np.pi * df["hour"] / 24), "hour_cos": np.cos(2 * np.pi * df["hour"] / 24),
        "month_sin": np.sin(2 * np.pi * (df["month"] - 1) / 12), "month_cos": np.cos(2 * np.pi * (df["month"] - 1) / 12),
    })
    for m in MODES:
        X[f"mode_{m}"] = (df["mode"] == m).astype(float)
    if engineered:
        gap = df["ambient_c"] - df["indoor_c"]
        X["abs_gap"] = gap.abs()
        X["abs_gap_cooling"] = gap.abs() * X["mode_Cooling"]
        X["abs_gap_heating"] = gap.abs() * X["mode_Heating"]
    return X


def make_split(df, name, seed):
    """Boolean mask of the TEST rows for a named split."""
    if name == "random":
        rng = np.random.default_rng(seed)
        mask = np.zeros(len(df), dtype=bool)
        mask[rng.permutation(len(df))[: int(round(config.TEST_SIZE * len(df)))]] = True
        return mask
    if name == "unseen_months":
        return df["month"].isin(config.HELD_OUT_MONTHS).to_numpy()
    if name == "forecast":
        return df["month"].isin(config.FORECAST_MONTHS).to_numpy()
    if name == "heatwave":
        return (df["ambient_c"] >= config.HEATWAVE_TEMP_C).to_numpy()
    raise ValueError(name)


def model_zoo(seed):
    """name -> (feature set is engineered?, model)"""
    return {
        "Linear regression (engineered features)": (True, linear_regressor()),       # statistical baseline
        "Random Forest (engineered features)": (True, forest_regressor(seed)),       # classic ML baseline
        "Ridge (basic features)": (False, ridge_regressor()),
        "Ridge (engineered features)": (True, ridge_regressor()),
        "XGBoost (basic features)": (False, xgb_regressor(seed)),
        "XGBoost (engineered features)": (True, xgb_regressor(seed)),
    }


def run(df, keep_predictions_for_seed=None):
    """Returns (results DataFrame with one row per split/model/seed, dict of kept test predictions)."""
    rows, kept = [], {}
    y = df["power_kw"]
    for seed in config.SEEDS:
        for split in SPLITS:
            test = make_split(df, split, seed)
            train = ~test
            # reference models that learn nothing about the weather
            ref_mean = np.full(test.sum(), y[train].mean())
            hour_mean = df.loc[test, "hour"].map(y[train].groupby(df.loc[train, "hour"]).mean()).to_numpy()
            for name, pred in (("Reference: training mean", ref_mean), ("Reference: average by hour of day", hour_mean)):
                rows.append({"pillar": "demand", "split": split, "model": name, "seed": seed, **regression_metrics(y[test], pred)})
            for name, (engineered, model) in model_zoo(seed).items():
                X = build_features(df, engineered)
                model.fit(X[train], y[train])
                pred = model.predict(X[test])
                rows.append({"pillar": "demand", "split": split, "model": name, "seed": seed, **regression_metrics(y[test], pred),
                             "n_train": int(train.sum()), "n_test": int(test.sum())})
                if seed == keep_predictions_for_seed:
                    kept[(split, name)] = (df.loc[test].copy(), pred)
    return pd.DataFrame(rows), kept


def fit_final(df, seed=config.SEEDS[0]):
    """Both final models trained on ALL hours, with the feature builder, for the dashboard."""
    final = {}
    for name, (engineered, model) in model_zoo(seed).items():
        if name in ("Ridge (engineered features)", "XGBoost (engineered features)"):
            final[name] = model.fit(build_features(df, True), df["power_kw"])
    return final
