"""
Quick automatic checks.  Run:  python -m pytest -q
They train small models (fast) so they take ~20-40 seconds.
"""
import numpy as np
import pytest

from src import config
from src.data import demand_thresholds, load_data, make_demand_labels, split_data
from src.recommend import (BUILDING_SHAPES, annual_energy_cost, building_from_shape, design_optimizer,
                           predict_building)
from src.train import build_bundle, train_classifiers, train_regressors


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def bundle(data):
    train, val, test = split_data(data)
    _, _, art = train_regressors(train, val, test, quick=True)
    _, clf, classifier = train_classifiers(train, val, test, art, quick=True)
    return build_bundle(art, clf, classifier)


def test_data_shape(data):
    assert data.shape == (768, 10)
    assert list(data.columns) == config.FEATURES + config.TARGETS
    assert data.isna().sum().sum() == 0


def test_shapes_match_dataset(data):
    in_data = {tuple(r) for r in data[["relative_compactness", "surface_area", "wall_area",
                                       "roof_area", "overall_height"]].itertuples(index=False)}
    assert in_data == set(BUILDING_SHAPES)


def test_split_has_no_overlap(data):
    train, val, test = split_data(data)
    assert len(train) + len(val) + len(test) == 768
    assert not (set(train.index) & set(test.index)) and not (set(val.index) & set(test.index))


def test_demand_labels_balanced(data):
    total = data[config.TARGETS].sum(axis=1)
    labels = make_demand_labels(total, demand_thresholds(total))
    counts = {c: np.sum(labels == c) for c in config.DEMAND_CLASSES}
    assert all(200 < n < 320 for n in counts.values()), counts


def test_prediction_and_advice(bundle):
    b = building_from_shape(0, orientation=2, glazing_area=0.25, glazing_distribution=1)
    r = predict_building(bundle, b)
    assert 0 < r["heating_load"] < 60 and 0 < r["cooling_load"] < 60
    assert r["heating_load_low"] <= r["heating_load"] <= r["heating_load_high"]
    assert r["demand_class"] in config.DEMAND_CLASSES
    assert annual_energy_cost(b, r)["floor_area_m2"] == pytest.approx(220.5)
    opt = design_optimizer(bundle, b)
    assert len(opt) == 3 * 5 * 4 and opt.total_load.is_monotonic_increasing
