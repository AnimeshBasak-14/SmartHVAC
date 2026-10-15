"""
config.py - ALL settings for the project live here.

If you want to change something (split sizes, random seed, electricity price,
file locations) change it HERE, not inside the other files.
"""
from pathlib import Path

# ---------------------------------------------------------------- folders
ROOT = Path(__file__).resolve().parents[1]          # the Project/ folder
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DATA_CSV = DATA_DIR / "energy_efficiency.csv"       # clean copy we ship with the code

OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"                    # all plots (PNG) for report / PPT
RESULTS_DIR = OUTPUT_DIR / "results"                # all tables (CSV / MD / TXT)
MODEL_DIR = OUTPUT_DIR / "models"                   # trained models (.joblib)
MODEL_BUNDLE = MODEL_DIR / "hvac_models.joblib"
LOG_FILE = OUTPUT_DIR / "run_log.txt"

# ---------------------------------------------------------------- data source
# UCI Machine Learning Repository - Energy Efficiency dataset (Tsanas & Xifara, 2012)
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/242/energy+efficiency"
DATA_URL = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"

# Original column codes -> readable names
COLUMN_NAMES = {
    "X1": "relative_compactness",   # how "cube-like" the building is (higher = more compact)
    "X2": "surface_area",           # m^2, total outer surface
    "X3": "wall_area",              # m^2
    "X4": "roof_area",              # m^2
    "X5": "overall_height",         # m (3.5 = 1 storey, 7.0 = 2 storeys)
    "X6": "orientation",            # 2=North, 3=East, 4=South, 5=West
    "X7": "glazing_area",           # window area as fraction of floor area (0, 0.10, 0.25, 0.40)
    "X8": "glazing_distribution",   # 0=no windows, 1=uniform, 2=North, 3=East, 4=South, 5=West
    "Y1": "heating_load",           # kWh/m^2  (target 1)
    "Y2": "cooling_load",           # kWh/m^2  (target 2)
}
FEATURES = [COLUMN_NAMES[f"X{i}"] for i in range(1, 9)]
TARGETS = ["heating_load", "cooling_load"]

# Friendly labels used in plots and in the GUI
PRETTY = {
    "relative_compactness": "Relative compactness",
    "surface_area": "Surface area (m²)",
    "wall_area": "Wall area (m²)",
    "roof_area": "Roof area (m²)",
    "overall_height": "Overall height (m)",
    "orientation": "Orientation",
    "glazing_area": "Glazing (window) area",
    "glazing_distribution": "Glazing distribution",
    "heating_load": "Heating load (kWh/m²)",
    "cooling_load": "Cooling load (kWh/m²)",
}
ORIENTATION_NAMES = {2: "North", 3: "East", 4: "South", 5: "West"}
GLAZING_DIST_NAMES = {0: "No windows", 1: "Uniform", 2: "North", 3: "East", 4: "South", 5: "West"}

# ---------------------------------------------------------------- experiment
RANDOM_STATE = 42       # same seed = same results every run (reproducible)
TEST_SIZE = 0.15        # 15% of buildings kept hidden until the very end
VAL_SIZE = 0.15         # 15% used to compare / choose models
CV_FOLDS = 5            # 5-fold cross-validation on the training set
INTERVAL_COVERAGE = 0.90  # prediction ranges should contain the true value ~90% of the time

# Demand classes (classification task). Thresholds = tertiles of the TRAINING
# total load (heating + cooling), so each class has ~1/3 of buildings.
DEMAND_CLASSES = ["Low", "Medium", "High"]

# ---------------------------------------------------------------- business
ELECTRICITY_PRICE_INR = 8.0   # ₹ per kWh, used for the annual cost estimate (editable in the GUI)
