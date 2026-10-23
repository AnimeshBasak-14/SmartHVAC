"""
config.py - all settings for the project in one place.

Change a split, a seed or a file location HERE, not inside the other files.
"""
from pathlib import Path

# ---------------------------------------------------------------- folders
PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent
DATA_DIR = PKG_DIR / "data"
OUTPUT_DIR = PKG_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
MODEL_DIR = OUTPUT_DIR / "models"

# ---------------------------------------------------------------- data files and sources
AI4I_CSV = DATA_DIR / "ai4i2020.csv"
HVAC_CSV = DATA_DIR / "hvac_power_consumption.csv"
RETAIL_XLSX = DATA_DIR / "online_retail_II.xlsx"
RETAIL_INVOICES = DATA_DIR / "retail_invoices.csv.gz"   # one row per invoice (about 1 MB); built once from the xlsx

SOURCES = {
    "ai4i": "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
    "retail": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
    "hvac": "https://www.kaggle.com/datasets/felixpradoh/hvac-power-consumption",
}
DOWNLOAD_URLS = {
    "ai4i": "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
    "retail": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
}

# ---------------------------------------------------------------- experiment settings
SEEDS = [42, 43, 44, 45, 46]     # every experiment is repeated with these seeds; we report mean and std
TEST_SIZE = 0.20                 # share of rows (AI4I) or customers (retail) kept for testing

# Pillar 1 - demand: which months / temperatures are held out
HELD_OUT_MONTHS = [3, 6, 9, 12]  # "unseen months" split
FORECAST_MONTHS = [10, 11, 12]   # "forecast" split: train on months 1-9
HEATWAVE_TEMP_C = 40.0           # "heatwave" split: train below this ambient temperature, test at or above it

# Pillar 3 - churn: a customer who bought in the last 90 days "lapses" if they buy nothing in the next 90 days
CHURN_HORIZON_DAYS = 90
ACTIVE_WINDOW_DAYS = 90
TRAIN_CUTOFFS = ["2010-06-01", "2010-09-01", "2010-12-01", "2011-03-01", "2011-06-01"]
TEST_CUTOFF = "2011-09-09"       # window ends 2011-12-08, the last full day in the data

# ---------------------------------------------------------------- model settings (fixed, not tuned on test data)
XGB_REG = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
XGB_CLF = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
RF = dict(n_estimators=300, min_samples_leaf=5)                  # Random Forest (classic ML baseline)
RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]   # chosen by cross-validation on the training data
LOGISTIC_C = [0.01, 0.1, 1.0, 10.0, 100.0]                    # same idea for the L2 (ridge-type) classifier
