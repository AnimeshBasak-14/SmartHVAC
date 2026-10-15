"""
recommend.py - turns model predictions into business advice for an HVAC
service provider (the "Advisor" part of the project).

  predict_building()   -> heating & cooling load + 90% range + demand class
  annual_energy_cost() -> kWh per year and ₹ per year for the whole building
  service_plan()       -> which equipment tier & maintenance plan to offer
  design_optimizer()   -> tries every window / orientation option and ranks them
"""
from itertools import product

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data import make_demand_labels

# The 12 building shapes in the dataset. Features X1-X5 always come together
# (they describe one shape), so the GUI lets the user pick a shape, not each number.
BUILDING_SHAPES = [
    # compactness, surface, wall, roof, height
    (0.98, 514.5, 294.0, 110.25, 7.0), (0.90, 563.5, 318.5, 122.50, 7.0),
    (0.86, 588.0, 294.0, 147.00, 7.0), (0.82, 612.5, 318.5, 147.00, 7.0),
    (0.79, 637.0, 343.0, 147.00, 7.0), (0.76, 661.5, 416.5, 122.50, 7.0),
    (0.74, 686.0, 245.0, 220.50, 3.5), (0.71, 710.5, 269.5, 220.50, 3.5),
    (0.69, 735.0, 294.0, 220.50, 3.5), (0.66, 759.5, 318.5, 220.50, 3.5),
    (0.64, 784.0, 343.0, 220.50, 3.5), (0.62, 808.5, 367.5, 220.50, 3.5),
]
SHAPE_KEYS = ["relative_compactness", "surface_area", "wall_area", "roof_area", "overall_height"]


def shape_label(i):
    s = BUILDING_SHAPES[i]
    storeys = "2-storey" if s[4] == 7.0 else "1-storey"
    return f"Shape {i + 1}: {storeys}, compactness {s[0]}, surface {s[1]} m²"


def building_from_shape(i, orientation=2, glazing_area=0.25, glazing_distribution=1):
    b = dict(zip(SHAPE_KEYS, BUILDING_SHAPES[i]))
    b.update(orientation=orientation, glazing_area=glazing_area, glazing_distribution=glazing_distribution)
    return b


def load_bundle(path=config.MODEL_BUNDLE):
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            pass
    # Cloud fallback: automatically train bundle if missing or unpickling fails
    from src.data import load_data, split_data
    from src.train import build_bundle, train_classifiers, train_regressors

    df = load_data()
    train, val, test = split_data(df)
    _, _, art = train_regressors(train, val, test, quick=True)
    _, clf, classifier = train_classifiers(train, val, test, art, quick=True)
    bundle = build_bundle(art, clf, classifier)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(bundle, path)
    except Exception:
        pass
    return bundle


def _frame(buildings):
    return pd.DataFrame(buildings if isinstance(buildings, list) else [buildings])[config.FEATURES]


def predict_building(bundle, building):
    X = _frame(building)
    out = {}
    for t in config.TARGETS:
        p = float(bundle["regressors"][t].predict(X)[0])
        m = bundle["margins"][t]
        out[t] = p
        out[f"{t}_low"], out[f"{t}_high"] = max(0.0, p - m), p + m
    out["total_load"] = out["heating_load"] + out["cooling_load"]
    out["demand_class"] = str(make_demand_labels([out["total_load"]], bundle["thresholds"])[0])
    clf = bundle["classifier"]
    out["classifier_class"] = str(clf.predict(X)[0])
    out["classifier_confidence"] = float(np.max(clf.predict_proba(X)[0]))
    return out


def floor_area(building):
    """Total floor area (m²): roof area x number of storeys."""
    return building["roof_area"] * (2 if building["overall_height"] == 7.0 else 1)


def annual_energy_cost(building, pred, price=config.ELECTRICITY_PRICE_INR):
    """ASSUMPTION: loads are annual energy per m² (kWh/m² per year) as in Tsanas & Xifara (2012)."""
    area = floor_area(building)
    kwh = pred["total_load"] * area
    return {"floor_area_m2": area, "annual_kwh": kwh, "annual_cost_inr": kwh * price}


def service_plan(demand_class):
    """Business rules: what a service provider should offer for each demand class."""
    plans = {
        "Low": {
            "equipment": "Standard-capacity split / packaged unit (smaller unit is enough)",
            "maintenance": "Annual check-up plan (1 visit / year)",
            "advice": "Building is already efficient. Do NOT over-size the unit - it wastes money and "
                      "short-cycles. Offer a basic maintenance contract.",
        },
        "Medium": {
            "equipment": "Mid-capacity inverter unit with good energy-star rating",
            "maintenance": "Bi-annual plan (pre-summer + pre-winter visits)",
            "advice": "Normal demand. Recommend an inverter system and a smart thermostat; check the "
                      "design optimizer for cheap window/orientation improvements.",
        },
        "High": {
            "equipment": "High-capacity inverter / heat-pump system (consider zoning for 2 storeys)",
            "maintenance": "Quarterly priority plan (4 visits / year, filter + coil cleaning)",
            "advice": "High demand = high running cost and more wear. Upsell an energy audit and "
                      "retrofit (glazing / shading) before installing - it can lower the unit size needed.",
        },
    }
    return plans[demand_class]


def design_optimizer(bundle, building, vary=("glazing_area", "glazing_distribution", "orientation"),
                     allow_no_windows=False):
    """Try every combination of the 'changeable' features, predict loads, rank by total load.
    Returns a DataFrame sorted from best (lowest load) to worst, with % saving vs. the current design.
    By default a building must keep some windows (daylight / building code), so 'no windows' is excluded."""
    options = {"glazing_area": [0.10, 0.25, 0.40],
               "glazing_distribution": [1, 2, 3, 4, 5],
               "orientation": [2, 3, 4, 5]}
    if allow_no_windows:
        options["glazing_area"].insert(0, 0.0)
        options["glazing_distribution"].insert(0, 0)
    combos = []
    for values in product(*[options[k] for k in vary]):
        b = dict(building, **dict(zip(vary, values)))
        # "no windows" (area 0) only makes sense with distribution 0, and vice-versa
        if (b["glazing_area"] == 0) != (b["glazing_distribution"] == 0):
            continue
        combos.append(b)
    X = _frame(combos)
    res = X[list(vary)].copy()
    res["heating_load"] = bundle["regressors"]["heating_load"].predict(X)
    res["cooling_load"] = bundle["regressors"]["cooling_load"].predict(X)
    res["total_load"] = res.heating_load + res.cooling_load
    current = predict_building(bundle, building)["total_load"]
    res["saving_vs_current_%"] = 100 * (current - res.total_load) / current
    if "orientation" in res:
        res["orientation"] = res["orientation"].map(config.ORIENTATION_NAMES)
    if "glazing_distribution" in res:
        res["glazing_distribution"] = res["glazing_distribution"].map(config.GLAZING_DIST_NAMES)
    return res.sort_values("total_load").reset_index(drop=True)
