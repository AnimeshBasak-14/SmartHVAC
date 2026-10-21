"""
data_pipeline.py
Builds the Heterogeneous Graph Dataset with explicit Monthly Cyclical Features.
"""
import os
import math
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from config import Config

def encode_month_features(month: int):
    """Computes cyclical month encodings and seasonal multipliers."""
    angle = 2.0 * math.pi * (month - 1) / 12.0
    m_sin = math.sin(angle)
    m_cos = math.cos(angle)
    _, regime_id, cool_mult, heat_mult = Config.MONTH_REGIME[month]
    return m_sin, m_cos, float(regime_id), cool_mult, heat_mult

class HVACGraphDatasetBuilder:
    def __init__(self, seed: int = Config.SEED):
        self.rng = np.random.default_rng(seed)

    def generate_or_load_tables(self, num_residents=600, num_locations=40, num_devices=180, num_parts=50):
        loc_file = os.path.join(Config.DATA_DIR, "locations.csv")
        dev_file = os.path.join(Config.DATA_DIR, "devices.csv")
        part_file = os.path.join(Config.DATA_DIR, "parts.csv")
        res_file = os.path.join(Config.DATA_DIR, "residents.csv")
        e_res_loc_file = os.path.join(Config.DATA_DIR, "edge_res_loc.csv")
        e_res_dev_file = os.path.join(Config.DATA_DIR, "edge_res_dev.csv")
        e_dev_part_file = os.path.join(Config.DATA_DIR, "edge_dev_part.csv")

        # Load from disk if all tables exist
        if all(os.path.exists(f) for f in [loc_file, dev_file, part_file, res_file, e_res_loc_file, e_res_dev_file, e_dev_part_file]):
            df_locations = pd.read_csv(loc_file)
            df_devices = pd.read_csv(dev_file)
            df_parts = pd.read_csv(part_file)
            df_residents = pd.read_csv(res_file)
            edge_res_loc = pd.read_csv(e_res_loc_file)
            edge_res_dev = pd.read_csv(e_res_dev_file)
            edge_dev_part = pd.read_csv(e_dev_part_file)
            return df_residents, df_locations, df_devices, df_parts, edge_res_loc, edge_res_dev, edge_dev_part

        # 1. Locations: Metropolitan vs. Village
        loc_records = []
        for loc_id in range(num_locations):
            is_metro = 1 if loc_id < int(num_locations * 0.45) else 0
            loc_type = "Metropolitan" if is_metro else "Village"
            base_temp = self.rng.uniform(26.0, 34.0) if is_metro else self.rng.uniform(22.0, 31.0)
            grid_stability = self.rng.uniform(0.85, 0.99) if is_metro else self.rng.uniform(0.60, 0.85)
            loc_records.append({
                "location_id": loc_id,
                "name": f"{'Metro_City' if is_metro else 'Village_Hub'}_{loc_id}",
                "is_metropolitan": is_metro,
                "location_type": loc_type,
                "base_temp_c": round(base_temp, 2),
                "grid_stability": round(grid_stability, 3),
                "population_density": round(float(self.rng.uniform(0.7, 1.0) if is_metro else self.rng.uniform(0.1, 0.45)), 3)
            })
        df_locations = pd.DataFrame(loc_records)

        # 2. Devices / SKUs: Cooling HVAC vs. Heating (Geyser, Space Heater, Induction Kettle)
        dev_records = []
        for dev_id in range(num_devices):
            is_cooling = 1 if dev_id < int(num_devices * 0.60) else 0
            category = "Cooling_HVAC" if is_cooling else "Heating_Induction_Kettle"
            unit_price = self.rng.uniform(28000, 58000) if is_cooling else self.rng.uniform(1800, 14000)
            rated_power_kw = self.rng.uniform(1.2, 2.8) if is_cooling else self.rng.uniform(1.0, 2.2)
            base_fail_rate = self.rng.uniform(0.04, 0.18)
            dev_records.append({
                "device_id": dev_id,
                "sku_name": f"{category}_SKU_{dev_id}",
                "is_cooling_hvac": is_cooling,
                "category": category,
                "unit_price_inr": round(unit_price, 2),
                "service_margin_inr": round(unit_price * 0.12, 2),
                "rated_power_kw": round(rated_power_kw, 3),
                "base_failure_risk": round(base_fail_rate, 4)
            })
        df_devices = pd.DataFrame(dev_records)

        # 3. Spare Parts & Inventory
        part_records = []
        for part_id in range(num_parts):
            for_cooling = 1 if part_id < int(num_parts * 0.65) else 0
            part_type = self.rng.choice(["Compressor", "Condenser_Coil", "Air_Filter", "PCB_Board"]) if for_cooling else self.rng.choice(["Induction_IGBT", "Heating_Element", "Thermostat_Sensor"])
            cost_inr = self.rng.uniform(1200, 8500) if for_cooling else self.rng.uniform(350, 2800)
            part_records.append({
                "part_id": part_id,
                "part_name": f"{part_type}_{part_id}",
                "for_cooling_hvac": for_cooling,
                "unit_cost_inr": round(cost_inr, 2),
                "holding_cost_monthly": round(cost_inr * 0.03, 2),
                "current_stock": int(self.rng.integers(15, 120))
            })
        df_parts = pd.DataFrame(part_records)

        # 4. Residents (Customers) with App Engagement & Sleeper tendency
        res_records = []
        for res_id in range(num_residents):
            loc_id = int(self.rng.integers(0, num_locations))
            is_metro = int(df_locations.loc[loc_id, "is_metropolitan"])
            app_installed = 1 if self.rng.random() < (0.85 if is_metro else 0.55) else 0
            base_engagement = self.rng.uniform(0.4, 0.95) if app_installed else self.rng.uniform(0.05, 0.35)
            tenure_months = int(self.rng.integers(3, 48))
            res_records.append({
                "resident_id": res_id,
                "location_id": loc_id,
                "is_metropolitan": is_metro,
                "customer_app_active": app_installed,
                "base_app_engagement": round(base_engagement, 3),
                "tenure_months": tenure_months,
                "income_tier": round(float(self.rng.uniform(0.5, 1.0) if is_metro else self.rng.uniform(0.2, 0.65)), 3)
            })
        df_residents = pd.DataFrame(res_records)

        # 5. Structural Edges
        edge_res_loc = df_residents[["resident_id", "location_id"]].copy()

        owns_src, owns_dst = [], []
        for res_id in range(num_residents):
            n_owned = int(self.rng.integers(1, 4))
            chosen_devs = self.rng.choice(num_devices, size=n_owned, replace=False)
            for d in chosen_devs:
                owns_src.append(res_id)
                owns_dst.append(int(d))
        edge_res_dev = pd.DataFrame({"resident_id": owns_src, "device_id": owns_dst})

        dev_src, part_dst = [], []
        for dev_id in range(num_devices):
            is_cool = df_devices.loc[dev_id, "is_cooling_hvac"]
            candidate_parts = df_parts[df_parts["for_cooling_hvac"] == is_cool]["part_id"].values
            chosen_parts = self.rng.choice(candidate_parts, size=2, replace=False)
            for p in chosen_parts:
                dev_src.append(dev_id)
                part_dst.append(int(p))
        edge_dev_part = pd.DataFrame({"device_id": dev_src, "part_id": part_dst})

        # Save synthesized schema-aligned tables to DATA_DIR so files are persisted
        df_locations.to_csv(loc_file, index=False)
        df_devices.to_csv(dev_file, index=False)
        df_parts.to_csv(part_file, index=False)
        df_residents.to_csv(res_file, index=False)
        edge_res_loc.to_csv(e_res_loc_file, index=False)
        edge_res_dev.to_csv(e_res_dev_file, index=False)
        edge_dev_part.to_csv(e_dev_part_file, index=False)

        return df_residents, df_locations, df_devices, df_parts, edge_res_loc, edge_res_dev, edge_dev_part

    def build_monthly_hetero_data(self, month: int, tables=None, telemetry_scores=None) -> HeteroData:
        if tables is None:
            tables = self.generate_or_load_tables()
        df_res, df_loc, df_dev, df_part, e_res_loc, e_res_dev, e_dev_part = tables

        m_sin, m_cos, regime_id, cool_mult, heat_mult = encode_month_features(month)
        data = HeteroData()
        data.month = month

        # 1. Location Nodes
        n_loc = len(df_loc)
        loc_x = np.column_stack([
            df_loc["is_metropolitan"].values.astype(np.float32),
            (df_loc["base_temp_c"].values / 45.0).astype(np.float32),
            df_loc["grid_stability"].values.astype(np.float32),
            df_loc["population_density"].values.astype(np.float32),
            np.full(n_loc, m_sin, dtype=np.float32),
            np.full(n_loc, m_cos, dtype=np.float32),
            np.full(n_loc, cool_mult, dtype=np.float32),
            np.full(n_loc, heat_mult, dtype=np.float32),
        ])
        data["location"].x = torch.tensor(loc_x, dtype=torch.float)

        metro_factor = np.where(df_loc["is_metropolitan"].values == 1, 1.35, 0.85)
        cooling_demand = 120.0 * cool_mult * metro_factor * (df_loc["base_temp_c"].values / 30.0)
        heating_demand = 95.0 * heat_mult * np.where(df_loc["is_metropolitan"].values == 1, 1.1, 1.3)
        total_loc_demand = cooling_demand + heating_demand + self.rng.normal(0, 4.0, size=n_loc)
        data["location"].y_demand = torch.tensor(np.clip(total_loc_demand, 5.0, None), dtype=torch.float).unsqueeze(-1)

        # 2. Device Nodes
        n_dev = len(df_dev)
        seasonal_stress = np.where(df_dev["is_cooling_hvac"].values == 1, cool_mult, heat_mult)
        if telemetry_scores is not None:
            power_draw = telemetry_scores["power_kw"]
            sensor_fail_prob = telemetry_scores["sensor_fail_prob"]
        else:
            power_draw = df_dev["rated_power_kw"].values * (0.6 + 0.6 * seasonal_stress)
            sensor_fail_prob = np.clip(df_dev["base_failure_risk"].values * (0.7 + 0.9 * seasonal_stress), 0.01, 0.98)

        dev_x = np.column_stack([
            df_dev["is_cooling_hvac"].values.astype(np.float32),
            (df_dev["unit_price_inr"].values / 60000.0).astype(np.float32),
            (df_dev["service_margin_inr"].values / 8000.0).astype(np.float32),
            (power_draw / 3.5).astype(np.float32),
            sensor_fail_prob.astype(np.float32),
            np.full(n_dev, m_sin, dtype=np.float32),
            np.full(n_dev, m_cos, dtype=np.float32),
            seasonal_stress.astype(np.float32)
        ])
        data["device"].x = torch.tensor(dev_x, dtype=torch.float)
        data["device"].y_power = torch.tensor(power_draw, dtype=torch.float).unsqueeze(-1)
        fail_labels = (sensor_fail_prob + self.rng.normal(0, 0.08, size=n_dev) > 0.16).astype(np.float32)
        data["device"].y_failure = torch.tensor(fail_labels, dtype=torch.float).unsqueeze(-1)

        # 3. Resident Nodes
        n_res = len(df_res)
        seasonal_activity_drop = np.where(df_res["is_metropolitan"].values == 1, 1.0 - 0.35 * heat_mult, 1.0 - 0.45 * cool_mult)
        dynamic_engagement = np.clip(df_res["base_app_engagement"].values * seasonal_activity_drop, 0.02, 0.99)

        res_x = np.column_stack([
            df_res["is_metropolitan"].values.astype(np.float32),
            df_res["customer_app_active"].values.astype(np.float32),
            dynamic_engagement.astype(np.float32),
            (df_res["tenure_months"].values / 48.0).astype(np.float32),
            df_res["income_tier"].values.astype(np.float32),
            np.full(n_res, m_sin, dtype=np.float32),
            np.full(n_res, m_cos, dtype=np.float32),
            np.full(n_res, regime_id / 2.0, dtype=np.float32)
        ])
        data["resident"].x = torch.tensor(res_x, dtype=torch.float)

        sleeper_prob = 1.0 - dynamic_engagement
        sleeper_labels = (sleeper_prob + self.rng.normal(0, 0.07, size=n_res) > 0.58).astype(np.float32)
        data["resident"].y_sleeper = torch.tensor(sleeper_labels, dtype=torch.float).unsqueeze(-1)

        # 4. InventoryPart Nodes
        n_part = len(df_part)
        part_seasonal_factor = np.where(df_part["for_cooling_hvac"].values == 1, cool_mult, heat_mult)
        part_x = np.column_stack([
            df_part["for_cooling_hvac"].values.astype(np.float32),
            (df_part["unit_cost_inr"].values / 10000.0).astype(np.float32),
            (df_part["holding_cost_monthly"].values / 300.0).astype(np.float32),
            (df_part["current_stock"].values / 150.0).astype(np.float32),
            np.full(n_part, m_sin, dtype=np.float32),
            np.full(n_part, m_cos, dtype=np.float32),
            np.full(n_part, cool_mult, dtype=np.float32),
            np.full(n_part, heat_mult, dtype=np.float32),
        ])
        data["part"].x = torch.tensor(part_x, dtype=torch.float)
        part_demand = (35.0 + 95.0 * part_seasonal_factor) + self.rng.normal(0, 3.5, size=n_part)
        data["part"].y_inventory = torch.tensor(np.clip(part_demand, 5.0, None), dtype=torch.float).unsqueeze(-1)

        # 5. Bidirectional Edges for Message Passing
        res_loc_idx = torch.tensor(np.vstack([e_res_loc["resident_id"].values, e_res_loc["location_id"].values]), dtype=torch.long)
        res_dev_idx = torch.tensor(np.vstack([e_res_dev["resident_id"].values, e_res_dev["device_id"].values]), dtype=torch.long)
        dev_part_idx = torch.tensor(np.vstack([e_dev_part["device_id"].values, e_dev_part["part_id"].values]), dtype=torch.long)

        data["resident", "located_in", "location"].edge_index = res_loc_idx
        data["location", "has_resident", "resident"].edge_index = res_loc_idx.flip(0)
        data["resident", "owns", "device"].edge_index = res_dev_idx
        data["device", "owned_by", "resident"].edge_index = res_dev_idx.flip(0)
        data["device", "requires_part", "part"].edge_index = dev_part_idx
        data["part", "used_in", "device"].edge_index = dev_part_idx.flip(0)

        return data
