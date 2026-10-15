"""
data.py - load, check, label and split the dataset.

Steps:
  1. load_data()          -> read data/energy_efficiency.csv (downloads it if missing)
  2. make_demand_labels() -> turn total load into Low / Medium / High demand classes
  3. split_data()         -> train (70%) / validation (15%) / test (15%)
"""
import io
import logging
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

log = logging.getLogger(__name__)


def download_data():
    """Download the original UCI zip and save it as a clean CSV (needs `openpyxl`)."""
    log.info("Downloading dataset from %s", config.DATA_URL)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(config.DATA_URL, timeout=60) as resp:
        content = resp.read()
    (config.RAW_DIR / "energy_efficiency.zip").write_bytes(content)
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        xlsx_name = next(n for n in z.namelist() if n.endswith(".xlsx"))
        df = pd.read_excel(z.open(xlsx_name))
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.to_csv(config.DATA_CSV, index=False)
    log.info("Saved %s", config.DATA_CSV)


def load_data():
    """Return the dataset as a DataFrame with readable column names."""
    if not config.DATA_CSV.exists():
        download_data()
    df = pd.read_csv(config.DATA_CSV)
    df = df.dropna(how="all").dropna(axis=1, how="all")   # remove empty Excel rows/cols
    df = df.rename(columns=config.COLUMN_NAMES)
    df = df[config.FEATURES + config.TARGETS]

    # basic data-quality checks (fail loudly if the file is broken)
    assert len(df) == 768, f"Expected 768 buildings, found {len(df)}"
    assert df.isna().sum().sum() == 0, "Dataset has missing values"
    df[["orientation", "glazing_distribution"]] = df[["orientation", "glazing_distribution"]].astype(int)
    log.info("Loaded %d buildings x %d features, %d targets", len(df), len(config.FEATURES), len(config.TARGETS))
    return df


def demand_thresholds(total_load):
    """Tertile cut-points of total load (heating + cooling). Computed on TRAINING data only."""
    return [float(np.quantile(total_load, 1 / 3)), float(np.quantile(total_load, 2 / 3))]


def make_demand_labels(total_load, thresholds):
    """Map total load values to 'Low' / 'Medium' / 'High'."""
    idx = np.digitize(np.asarray(total_load), thresholds)   # 0, 1 or 2
    return np.array(config.DEMAND_CLASSES)[idx]


def split_data(df):
    """Split into train / validation / test DataFrames (70 / 15 / 15)."""
    train_val, test = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    val_fraction = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(train_val, test_size=val_fraction, random_state=config.RANDOM_STATE)
    log.info("Split -> train: %d, validation: %d, test: %d", len(train), len(val), len(test))
    return train, val, test
