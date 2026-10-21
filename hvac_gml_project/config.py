"""
config.py
Central Configuration for HVAC & Smart Appliance Heterogeneous GML Pipeline
Hardware Target: NVIDIA Tesla V100 GPU (CUDA + FP16 AMP) + Neo4j Graph DB
"""
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # Hardware & V100 GPU Settings
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    USE_AMP_FP16 = True if torch.cuda.is_available() else False  # V100 Tensor Cores
    NUM_WORKERS = 4
    PIN_MEMORY = True if torch.cuda.is_available() else False
    SEED = 42

    # Neo4j Graph Database Connection
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
    PUSH_TO_NEO4J = True

    # Dataset Paths (Auto-falls back to realistic synthetic generator if CSVs are missing)
    DATA_DIR = os.path.join(BASE_DIR, "data")
    HVAC_POWER_CSV = os.path.join(DATA_DIR, "hvac_power_consumption.csv")
    HVAC_ANOMALY_CSV = os.path.join(DATA_DIR, "hvac_anomaly_detection.csv")
    IITK_APPLIANCE_CSV = os.path.join(DATA_DIR, "iitk_seasonal_appliances.csv")
    FLIPKART_AC_CSV = os.path.join(DATA_DIR, "flipkart_air_conditioners.csv")
    RETAIL_CHURN_CSV = os.path.join(DATA_DIR, "customer_transactions_olist.csv")

    # Graph & Temporal Hyperparameters (Month 1..12 is a core dynamic feature)
    MONTHS = list(range(1, 13))
    HIDDEN_CHANNELS = 128
    OUT_CHANNELS = 64
    NUM_GNN_LAYERS = 3
    DROPOUT = 0.2
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 120

    # Multi-Task Loss Weights
    W_DEMAND = 1.0       # Model 1: Spatio-Temporal Demand Regression
    W_MAINTENANCE = 1.5  # Model 2: Power & Failure Prediction
    W_CHURN = 1.2        # Model 3: Resident Sleeper / Churn Classification
    W_INVENTORY = 1.0    # Model 4: Spare-Parts Inventory Forecasting

    # Artifacts & Monitoring
    ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
    CHECKPOINT_PATH = os.path.join(ARTIFACT_DIR, "best_hetero_gnn_v100.pt")
    TELEMETRY_JSON = os.path.join(ARTIFACT_DIR, "training_telemetry.json")
    FINANCIAL_REPORT_CSV = os.path.join(ARTIFACT_DIR, "monthly_financial_metrics.csv")

    # Monthly Seasonal Regime Mapping (India Climate Context)
    # (Name, Regime_ID, Cooling_Multiplier, Heating_Multiplier)
    MONTH_REGIME = {
        1: ("Winter", 0, 0.25, 0.90),
        2: ("Winter-Transition", 0, 0.40, 0.65),
        3: ("Spring-Warming", 1, 0.70, 0.30),
        4: ("Peak-Summer", 1, 0.95, 0.15),
        5: ("Peak-Summer", 1, 1.00, 0.10),
        6: ("Late-Summer-Monsoon", 1, 0.88, 0.15),
        7: ("Monsoon-Humid", 2, 0.65, 0.25),
        8: ("Monsoon-Humid", 2, 0.60, 0.30),
        9: ("Post-Monsoon", 2, 0.55, 0.35),
        10: ("Autumn-Transition", 2, 0.40, 0.50),
        11: ("Early-Winter", 0, 0.25, 0.80),
        12: ("Peak-Winter", 0, 0.15, 0.95),
    }

os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.ARTIFACT_DIR, exist_ok=True)

