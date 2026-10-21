"""
telemetry_premodel.py
Pre-GNN Sensor & Power Consumption Module (Model 2 Pre-Processor).
"""
import numpy as np
import torch
import torch.nn as nn
from config import Config
from data_pipeline import encode_month_features

class TelemetrySensorNet(nn.Module):
    def __init__(self, in_features=6, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.power_head = nn.Linear(hidden, 1)
        self.anomaly_head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.net(x)
        return self.power_head(h), self.anomaly_head(h)

def compute_monthly_telemetry_scores(df_devices, month: int, device=Config.DEVICE):
    m_sin, m_cos, _, cool_mult, heat_mult = encode_month_features(month)
    is_cool = df_devices["is_cooling_hvac"].values
    seasonal_load = np.where(is_cool == 1, cool_mult, heat_mult)

    rated_kw = df_devices["rated_power_kw"].values
    base_risk = df_devices["base_failure_risk"].values

    x_np = np.column_stack([
        is_cool.astype(np.float32),
        (rated_kw / 3.0).astype(np.float32),
        base_risk.astype(np.float32),
        seasonal_load.astype(np.float32),
        np.full(len(df_devices), m_sin, dtype=np.float32),
        np.full(len(df_devices), m_cos, dtype=np.float32)
    ]).astype(np.float32)

    power_kw = rated_kw * (0.55 + 0.65 * seasonal_load)
    sensor_fail_prob = np.clip(base_risk * (0.65 + 0.95 * seasonal_load), 0.01, 0.99)

    return {
        "power_kw": power_kw,
        "sensor_fail_prob": sensor_fail_prob,
        "telemetry_tensor": torch.tensor(x_np, device=device)
    }

