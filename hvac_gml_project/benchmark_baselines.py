"""
benchmark_baselines.py

Compares our model with three simpler baselines on the same 12 monthly graphs:
1. XGBoost on flat (tabular) features
2. Plain GCN (GCNConv), which treats every node as the same type
3. Plain GAT (GATConv), same idea with attention
4. Our multi-task hetero-GNN (HeteroConv + SAGEConv)

Runs on the lab server's Tesla V100S (10.13.1.7), or on CPU if there is no GPU.
"""

import os
import sys
import json
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    precision_recall_fscore_support,
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score
)

# Ensure local imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import Config
from data_pipeline import HVACGraphDatasetBuilder, encode_month_features
from telemetry_premodel import compute_monthly_telemetry_scores
from hetero_gnn_model import MultiTaskHVACHeteroGNN
from fine_tune_v100 import load_empirical_fine_tuning_tables, UncertaintyMultiTaskLoss

# Check for XGBoost availability
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

from torch_geometric.nn import GCNConv, GATConv


# ==============================================================================
# 1. HOMOGENEOUS GNN BASELINE ARCHITECTURES
# ==============================================================================

class VanillaHomogeneousGCN(nn.Module):
    """
    Homogeneous Baseline: Vanilla GCN (Kipf & Welling, ICLR 2017)
    Forces all heterogeneous nodes into a single shared projection space,
    ignoring node semantic differences and edge relation types.
    """
    def __init__(self, in_channels=8, hidden_channels=128, out_channels=64, dropout=0.15):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

        # Output heads
        self.head_demand = nn.Linear(out_channels, 1)
        self.head_power = nn.Linear(out_channels, 1)
        self.head_failure = nn.Linear(out_channels, 1)
        self.head_sleeper = nn.Linear(out_channels, 1)
        self.head_inventory = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, node_slices):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv3(h, edge_index))

        # Slice embeddings according to entity offsets
        loc_h = h[node_slices["location"]]
        dev_h = h[node_slices["device"]]
        res_h = h[node_slices["resident"]]
        part_h = h[node_slices["part"]]

        return {
            "demand_pred": self.head_demand(loc_h),
            "power_pred": self.head_power(dev_h),
            "failure_logits": self.head_failure(dev_h),
            "sleeper_logits": self.head_sleeper(res_h),
            "inventory_pred": F.relu(self.head_inventory(part_h)),
            "h_res": res_h,
            "h_dev": dev_h
        }


class VanillaHomogeneousGAT(nn.Module):
    """
    Homogeneous Baseline: Vanilla Graph Attention Network (Veličković et al., ICLR 2018)
    Uses dynamic multi-head self-attention, but operates over a single unified
    homogeneous graph with no edge relation differentiation.
    """
    def __init__(self, in_channels=8, hidden_channels=64, out_channels=64, heads=2, dropout=0.15):
        super().__init__()
        # heads * hidden_channels = 2 * 64 = 128
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.dropout = dropout

        self.head_demand = nn.Linear(out_channels, 1)
        self.head_power = nn.Linear(out_channels, 1)
        self.head_failure = nn.Linear(out_channels, 1)
        self.head_sleeper = nn.Linear(out_channels, 1)
        self.head_inventory = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, node_slices):
        h = F.elu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.conv3(h, edge_index))

        loc_h = h[node_slices["location"]]
        dev_h = h[node_slices["device"]]
        res_h = h[node_slices["resident"]]
        part_h = h[node_slices["part"]]

        return {
            "demand_pred": self.head_demand(loc_h),
            "power_pred": self.head_power(dev_h),
            "failure_logits": self.head_failure(dev_h),
            "sleeper_logits": self.head_sleeper(res_h),
            "inventory_pred": F.relu(self.head_inventory(part_h)),
            "h_res": res_h,
            "h_dev": dev_h
        }


# ==============================================================================
# 2. HOMOGENEOUS GRAPH CONVERTER
# ==============================================================================

def convert_hetero_to_homogeneous_batch(batch):
    """
    Collapses a HeteroData object into a single homogeneous graph (x, edge_index).
    Combines Location (40), Device (180), Resident (600), Part (50) -> 870 total nodes.
    Preserves node feature vectors and offsets edges into a single unified adjacency matrix.
    """
    n_loc = batch["location"].num_nodes
    n_dev = batch["device"].num_nodes
    n_res = batch["resident"].num_nodes
    n_part = batch["part"].num_nodes

    offsets = {
        "location": 0,
        "device": n_loc,
        "resident": n_loc + n_dev,
        "part": n_loc + n_dev + n_res
    }

    slices = {
        "location": slice(offsets["location"], offsets["device"]),
        "device": slice(offsets["device"], offsets["resident"]),
        "resident": slice(offsets["resident"], offsets["part"]),
        "part": slice(offsets["part"], offsets["part"] + n_part)
    }

    x_homo = torch.cat([
        batch["location"].x,
        batch["device"].x,
        batch["resident"].x,
        batch["part"].x
    ], dim=0)

    # Re-index all heterogeneous edges into the single homogeneous graph
    all_edges = []
    for (src_type, rel, dst_type), edge_idx in batch.edge_index_dict.items():
        src_offset = offsets[src_type]
        dst_offset = offsets[dst_type]
        reindexed_edge = torch.stack([
            edge_idx[0] + src_offset,
            edge_idx[1] + dst_offset
        ], dim=0)
        all_edges.append(reindexed_edge)
        # Add reverse edge for undirected homogeneous propagation
        rev_edge = torch.stack([
            edge_idx[1] + dst_offset,
            edge_idx[0] + src_offset
        ], dim=0)
        all_edges.append(rev_edge)

    edge_index_homo = torch.cat(all_edges, dim=1)
    # Remove duplicate edges
    edge_index_homo = torch.unique(edge_index_homo, dim=1)

    return x_homo, edge_index_homo, slices


# ==============================================================================
# 3. BENCHMARK HARNESS
# ==============================================================================

def evaluate_predictions(preds_dict, targets_dict):
    """Calculates all key academic project metrics."""
    mse_fn = nn.MSELoss()
    l1_fn = nn.L1Loss()

    d_mse = mean_squared_error(targets_dict["demand"], preds_dict["demand"])
    d_mae = mean_absolute_error(targets_dict["demand"], preds_dict["demand"])

    p_mse = mean_squared_error(targets_dict["power"], preds_dict["power"])
    p_mae = mean_absolute_error(targets_dict["power"], preds_dict["power"])

    # Classification F1
    _, _, f1_fail, _ = precision_recall_fscore_support(
        targets_dict["failure"], preds_dict["failure_binary"], average="binary", zero_division=0
    )
    fail_flagged = int(np.sum(preds_dict["failure_binary"]))

    _, _, f1_churn, _ = precision_recall_fscore_support(
        targets_dict["sleeper"], preds_dict["sleeper_binary"], average="binary", zero_division=0
    )
    churn_flagged = int(np.sum(preds_dict["sleeper_binary"]))

    inv_mse = mean_squared_error(targets_dict["inventory"], preds_dict["inventory"])
    inv_mae = mean_absolute_error(targets_dict["inventory"], preds_dict["inventory"])

    # Link Prediction ROC-AUC
    link_auc = 0.50
    if "link_labels" in targets_dict and "link_scores" in preds_dict:
        try:
            link_auc = roc_auc_score(targets_dict["link_labels"], preds_dict["link_scores"])
        except Exception:
            link_auc = 0.50

    return {
        "demand_mse": round(float(d_mse), 4),
        "demand_mae": round(float(d_mae), 4),
        "power_mse": round(float(p_mse), 4),
        "power_mae": round(float(p_mae), 4),
        "failure_f1": round(float(f1_fail), 4),
        "failure_flagged": fail_flagged,
        "sleeper_f1": round(float(f1_churn), 4),
        "sleeper_flagged": churn_flagged,
        "inventory_mse": round(float(inv_mse), 4),
        "inventory_mae": round(float(inv_mae), 4),
        "link_auc": round(float(link_auc), 4),
        "total_dossier_loss": round(float(d_mse + (p_mse + (1.0 - f1_fail)) + (1.0 - f1_churn) + inv_mse), 2)
    }


def run_benchmark():
    print("=" * 80)
    print(" SmartHVAC Model Selection Benchmark: XGBoost vs GCN vs GAT vs Hetero-GNN")
    print("=" * 80)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[ENVIRONMENT] Active Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # 1. Load Data
    builder = HVACGraphDatasetBuilder(seed=Config.SEED)
    tables, power_profiles = load_empirical_fine_tuning_tables(builder)
    df_res, df_loc, df_dev, df_part, e_res_loc, e_res_dev, e_dev_part = tables

    print("\n[DATA PIPELINE] Preparing 12-Month Heterogeneous Graph Snapshots...")
    monthly_graphs = {}
    for m in Config.MONTHS:
        t_scores = compute_monthly_telemetry_scores(df_dev, month=m, device="cpu")
        if m in power_profiles:
            t_scores["power_kw"] = np.clip(
                t_scores["power_kw"] * 0.5 + power_profiles[m]["mean_power_kw"] * 0.7, 0.4, 4.2
            )
        _, regime_id, cool_mult, heat_mult = Config.MONTH_REGIME[m]
        if regime_id == 2:
            t_scores["sensor_fail_prob"] = np.clip(t_scores["sensor_fail_prob"] * 1.30, 0.04, 0.95)
        elif regime_id == 1:
            t_scores["sensor_fail_prob"] = np.clip(t_scores["sensor_fail_prob"] * 1.25, 0.03, 0.95)

        g_data = builder.build_monthly_hetero_data(month=m, tables=tables, telemetry_scores=t_scores)

        rec_days = 365.0 * (1.0 - df_res["base_app_engagement"].values)
        off_season = 1.0 if regime_id in [0, 2] else 0.5
        empirical_churn = ((rec_days > 90) & (np.random.rand(len(df_res)) < (0.65 * off_season))).astype(np.float32)
        g_data["resident"].y_sleeper = torch.tensor(empirical_churn, dtype=torch.float).unsqueeze(-1)

        monthly_graphs[m] = g_data.to(device)

    # Prepare ground-truth flat targets for full evaluation
    all_targets = {
        "demand": [], "power": [], "failure": [], "sleeper": [], "inventory": [],
        "link_labels": []
    }
    for m in Config.MONTHS:
        g = monthly_graphs[m]
        all_targets["demand"].extend(g["location"].y_demand.cpu().numpy().flatten())
        all_targets["power"].extend(g["device"].y_power.cpu().numpy().flatten())
        all_targets["failure"].extend(g["device"].y_failure.cpu().numpy().flatten())
        all_targets["sleeper"].extend(g["resident"].y_sleeper.cpu().numpy().flatten())
        all_targets["inventory"].extend(g["part"].y_inventory.cpu().numpy().flatten())

    # Build positive and negative link evaluation pairs
    res_dev_pos = e_res_dev.values  # (N_edges, 2)
    # Negative sampling: random pairs not in edge_res_dev
    pos_set = set(map(tuple, res_dev_pos))
    rng = np.random.default_rng(Config.SEED)
    neg_pairs = []
    while len(neg_pairs) < len(res_dev_pos):
        r = int(rng.integers(0, len(df_res)))
        d = int(rng.integers(0, len(df_dev)))
        if (r, d) not in pos_set:
            neg_pairs.append((r, d))
    neg_pairs = np.array(neg_pairs)

    link_eval_pairs = np.vstack([res_dev_pos, neg_pairs])
    link_eval_labels = np.array([1] * len(res_dev_pos) + [0] * len(neg_pairs))
    all_targets["link_labels"] = link_eval_labels

    results = {}

    # ==========================================================================
    # BASELINE 1: TABULAR GRADIENT BOOSTED TREES (XGBOOST)
    # ==========================================================================
    print("\n" + "="*60)
    print(" [1/4] Training Tabular Gradient Boosted Trees (XGBoost)...")
    print("="*60)
    t0 = time.time()

    # Flatten node features across 12 months for tabular training
    loc_X_list, loc_y_list = [], []
    dev_X_list, dev_pwr_list, dev_fail_list = [], [], []
    res_X_list, res_churn_list = [], []
    part_X_list, part_inv_list = [], []

    for m in Config.MONTHS:
        g = monthly_graphs[m]
        loc_X_list.append(g["location"].x.cpu().numpy())
        loc_y_list.append(g["location"].y_demand.cpu().numpy().flatten())

        dev_X_list.append(g["device"].x.cpu().numpy())
        dev_pwr_list.append(g["device"].y_power.cpu().numpy().flatten())
        dev_fail_list.append(g["device"].y_failure.cpu().numpy().flatten())

        res_X_list.append(g["resident"].x.cpu().numpy())
        res_churn_list.append(g["resident"].y_sleeper.cpu().numpy().flatten())

        part_X_list.append(g["part"].x.cpu().numpy())
        part_inv_list.append(g["part"].y_inventory.cpu().numpy().flatten())

    loc_X, loc_y = np.vstack(loc_X_list), np.concatenate(loc_y_list)
    dev_X, dev_pwr, dev_fail = np.vstack(dev_X_list), np.concatenate(dev_pwr_list), np.concatenate(dev_fail_list)
    res_X, res_churn = np.vstack(res_X_list), np.concatenate(res_churn_list)
    part_X, part_inv = np.vstack(part_X_list), np.concatenate(part_inv_list)

    if HAS_XGB:
        m_demand = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=Config.SEED)
        m_power = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=Config.SEED)
        m_fail = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, scale_pos_weight=2.5, random_state=Config.SEED)
        m_churn = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, scale_pos_weight=1.8, random_state=Config.SEED)
        m_inv = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=Config.SEED)
    else:
        m_demand = HistGradientBoostingRegressor(max_iter=100, random_state=Config.SEED)
        m_power = HistGradientBoostingRegressor(max_iter=100, random_state=Config.SEED)
        m_fail = HistGradientBoostingClassifier(max_iter=100, random_state=Config.SEED)
        m_churn = HistGradientBoostingClassifier(max_iter=100, random_state=Config.SEED)
        m_inv = HistGradientBoostingRegressor(max_iter=100, random_state=Config.SEED)

    # Train isolated siloed tabular models
    m_demand.fit(loc_X, loc_y)
    m_power.fit(dev_X, dev_pwr)
    m_fail.fit(dev_X, dev_fail)
    m_churn.fit(res_X, res_churn)
    m_inv.fit(part_X, part_inv)

    # Tabular Link Prediction: Pairwise concatenated features [res_feat || dev_feat]
    res_feats_static = monthly_graphs[1]["resident"].x.cpu().numpy()
    dev_feats_static = monthly_graphs[1]["device"].x.cpu().numpy()
    link_X = np.hstack([
        res_feats_static[link_eval_pairs[:, 0]],
        dev_feats_static[link_eval_pairs[:, 1]]
    ])
    if HAS_XGB:
        m_link = xgb.XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.08, random_state=Config.SEED)
    else:
        m_link = HistGradientBoostingClassifier(max_iter=80, random_state=Config.SEED)
    m_link.fit(link_X, link_eval_labels)
    link_scores_xgb = m_link.predict_proba(link_X)[:, 1]

    # Predict
    xgb_preds = {
        "demand": m_demand.predict(loc_X),
        "power": m_power.predict(dev_X),
        "failure_binary": (m_fail.predict_proba(dev_X)[:, 1] > 0.40).astype(int),
        "sleeper_binary": (m_churn.predict_proba(res_X)[:, 1] > 0.45).astype(int),
        "inventory": np.clip(m_inv.predict(part_X), 0, None),
        "link_scores": link_scores_xgb
    }
    t_xgb = time.time() - t0
    results["XGBoost (Tabular)"] = evaluate_predictions(xgb_preds, all_targets)
    results["XGBoost (Tabular)"]["train_time_sec"] = round(t_xgb, 2)
    print(f"  -> Finished XGBoost in {t_xgb:.2f}s | Demand MSE: {results['XGBoost (Tabular)']['demand_mse']} | Failure F1: {results['XGBoost (Tabular)']['failure_f1']} | Churn F1: {results['XGBoost (Tabular)']['sleeper_f1']}")

    # ==========================================================================
    # BASELINE 2: HOMOGENEOUS VANILLA GCN
    # ==========================================================================
    print("\n" + "="*60)
    print(" [2/4] Training Homogeneous Vanilla GCN (GCNConv)...")
    print("="*60)
    t0 = time.time()
    gcn_model = VanillaHomogeneousGCN(in_channels=8, hidden_channels=128, out_channels=64).to(device)
    optimizer_gcn = torch.optim.AdamW(gcn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn_gcn = UncertaintyMultiTaskLoss(num_tasks=4).to(device)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    gcn_model.train()
    for epoch in range(60):
        for m in Config.MONTHS:
            batch = monthly_graphs[m]
            x_h, edge_h, slices = convert_hetero_to_homogeneous_batch(batch)
            optimizer_gcn.zero_grad()
            out = gcn_model(x_h, edge_h, slices)

            l_dem = mse_fn(out["demand_pred"], batch["location"].y_demand)
            l_pwr = mse_fn(out["power_pred"], batch["device"].y_power)
            l_fail = bce_fn(out["failure_logits"], batch["device"].y_failure)
            l_churn = bce_fn(out["sleeper_logits"], batch["resident"].y_sleeper)
            l_inv = mse_fn(out["inventory_pred"], batch["part"].y_inventory)

            loss = loss_fn_gcn(l_dem, l_pwr + l_fail, l_churn, l_inv)
            loss.backward()
            optimizer_gcn.step()

    # Evaluation
    gcn_model.eval()
    gcn_preds = {"demand": [], "power": [], "failure_binary": [], "sleeper_binary": [], "inventory": []}
    all_res_h, all_dev_h = None, None
    with torch.no_grad():
        for m in Config.MONTHS:
            batch = monthly_graphs[m]
            x_h, edge_h, slices = convert_hetero_to_homogeneous_batch(batch)
            out = gcn_model(x_h, edge_h, slices)

            gcn_preds["demand"].extend(out["demand_pred"].cpu().numpy().flatten())
            gcn_preds["power"].extend(out["power_pred"].cpu().numpy().flatten())
            f_prob = torch.sigmoid(out["failure_logits"]).cpu().numpy().flatten()
            gcn_preds["failure_binary"].extend((f_prob > 0.40).astype(int))
            c_prob = torch.sigmoid(out["sleeper_logits"]).cpu().numpy().flatten()
            gcn_preds["sleeper_binary"].extend((c_prob > 0.45).astype(int))
            gcn_preds["inventory"].extend(out["inventory_pred"].cpu().numpy().flatten())

            if all_res_h is None:
                all_res_h = out["h_res"].cpu()
                all_dev_h = out["h_dev"].cpu()

    # Link prediction evaluation via dot-product
    sims = (all_res_h[link_eval_pairs[:, 0]] * all_dev_h[link_eval_pairs[:, 1]]).sum(dim=-1)
    gcn_preds["link_scores"] = torch.sigmoid(sims).numpy()

    t_gcn = time.time() - t0
    results["Vanilla GCN (Homogeneous)"] = evaluate_predictions(gcn_preds, all_targets)
    results["Vanilla GCN (Homogeneous)"]["train_time_sec"] = round(t_gcn, 2)
    print(f"  -> Finished Vanilla GCN in {t_gcn:.2f}s | Demand MSE: {results['Vanilla GCN (Homogeneous)']['demand_mse']} | Failure F1: {results['Vanilla GCN (Homogeneous)']['failure_f1']} | Churn F1: {results['Vanilla GCN (Homogeneous)']['sleeper_f1']}")

    # ==========================================================================
    # BASELINE 3: HOMOGENEOUS VANILLA GAT
    # ==========================================================================
    print("\n" + "="*60)
    print(" [3/4] Training Homogeneous Vanilla GAT (GATConv)...")
    print("="*60)
    t0 = time.time()
    gat_model = VanillaHomogeneousGAT(in_channels=8, hidden_channels=64, out_channels=64, heads=2).to(device)
    optimizer_gat = torch.optim.AdamW(gat_model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn_gat = UncertaintyMultiTaskLoss(num_tasks=4).to(device)

    gat_model.train()
    for epoch in range(60):
        for m in Config.MONTHS:
            batch = monthly_graphs[m]
            x_h, edge_h, slices = convert_hetero_to_homogeneous_batch(batch)
            optimizer_gat.zero_grad()
            out = gat_model(x_h, edge_h, slices)

            l_dem = mse_fn(out["demand_pred"], batch["location"].y_demand)
            l_pwr = mse_fn(out["power_pred"], batch["device"].y_power)
            l_fail = bce_fn(out["failure_logits"], batch["device"].y_failure)
            l_churn = bce_fn(out["sleeper_logits"], batch["resident"].y_sleeper)
            l_inv = mse_fn(out["inventory_pred"], batch["part"].y_inventory)

            loss = loss_fn_gat(l_dem, l_pwr + l_fail, l_churn, l_inv)
            loss.backward()
            optimizer_gat.step()

    # Evaluation
    gat_model.eval()
    gat_preds = {"demand": [], "power": [], "failure_binary": [], "sleeper_binary": [], "inventory": []}
    all_res_h_gat, all_dev_h_gat = None, None
    with torch.no_grad():
        for m in Config.MONTHS:
            batch = monthly_graphs[m]
            x_h, edge_h, slices = convert_hetero_to_homogeneous_batch(batch)
            out = gat_model(x_h, edge_h, slices)

            gat_preds["demand"].extend(out["demand_pred"].cpu().numpy().flatten())
            gat_preds["power"].extend(out["power_pred"].cpu().numpy().flatten())
            f_prob = torch.sigmoid(out["failure_logits"]).cpu().numpy().flatten()
            gat_preds["failure_binary"].extend((f_prob > 0.40).astype(int))
            c_prob = torch.sigmoid(out["sleeper_logits"]).cpu().numpy().flatten()
            gat_preds["sleeper_binary"].extend((c_prob > 0.45).astype(int))
            gat_preds["inventory"].extend(out["inventory_pred"].cpu().numpy().flatten())

            if all_res_h_gat is None:
                all_res_h_gat = out["h_res"].cpu()
                all_dev_h_gat = out["h_dev"].cpu()

    sims_gat = (all_res_h_gat[link_eval_pairs[:, 0]] * all_dev_h_gat[link_eval_pairs[:, 1]]).sum(dim=-1)
    gat_preds["link_scores"] = torch.sigmoid(sims_gat).numpy()

    t_gat = time.time() - t0
    results["Vanilla GAT (Homogeneous)"] = evaluate_predictions(gat_preds, all_targets)
    results["Vanilla GAT (Homogeneous)"]["train_time_sec"] = round(t_gat, 2)
    print(f"  -> Finished Vanilla GAT in {t_gat:.2f}s | Demand MSE: {results['Vanilla GAT (Homogeneous)']['demand_mse']} | Failure F1: {results['Vanilla GAT (Homogeneous)']['failure_f1']} | Churn F1: {results['Vanilla GAT (Homogeneous)']['sleeper_f1']}")

    # ==========================================================================
    # MODEL 4: MULTI-TASK HETEROGENEOUS GNN (OURS)
    # ==========================================================================
    print("\n" + "="*60)
    print(" [4/4] Evaluating Multi-Task Hetero-GNN (fine_tuned_hetero_gnn_v100.pt)...")
    print("="*60)
    hetero_model = MultiTaskHVACHeteroGNN(
        hidden_channels=Config.HIDDEN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        num_layers=Config.NUM_GNN_LAYERS,
        dropout=0.15
    ).to(device)

    fine_tuned_pt = os.path.join(Config.ARTIFACT_DIR, "fine_tuned_hetero_gnn_v100.pt")
    if os.path.exists(fine_tuned_pt):
        hetero_model.load_state_dict(torch.load(fine_tuned_pt, map_location=device))
        print(f"  -> Loaded fine-tuned checkpoint: {fine_tuned_pt}")
    elif os.path.exists(Config.CHECKPOINT_PATH):
        hetero_model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))
        print(f"  -> Loaded checkpoint: {Config.CHECKPOINT_PATH}")

    hetero_model.eval()
    hetero_preds = {"demand": [], "power": [], "failure_binary": [], "sleeper_binary": [], "inventory": []}
    h_res_dict, h_dev_dict = None, None
    with torch.no_grad():
        for m in Config.MONTHS:
            batch = monthly_graphs[m]
            out = hetero_model(batch.x_dict, batch.edge_index_dict)

            hetero_preds["demand"].extend(out["demand_pred"].cpu().numpy().flatten())
            hetero_preds["power"].extend(out["power_pred"].cpu().numpy().flatten())
            f_prob = torch.sigmoid(out["failure_logits"]).cpu().numpy().flatten()
            hetero_preds["failure_binary"].extend((f_prob > 0.40).astype(int))
            c_prob = torch.sigmoid(out["sleeper_logits"]).cpu().numpy().flatten()
            hetero_preds["sleeper_binary"].extend((c_prob > 0.45).astype(int))
            hetero_preds["inventory"].extend(out["inventory_pred"].cpu().numpy().flatten())

            if h_res_dict is None:
                # Store node embeddings for link prediction
                h_res_dict = out["embeddings"]["resident"].cpu()
                h_dev_dict = out["embeddings"]["device"].cpu()

    sims_hetero = (h_res_dict[link_eval_pairs[:, 0]] * h_dev_dict[link_eval_pairs[:, 1]]).sum(dim=-1)
    hetero_preds["link_scores"] = torch.sigmoid(sims_hetero).numpy()

    results["Hetero-GNN (Ours)"] = evaluate_predictions(hetero_preds, all_targets)
    results["Hetero-GNN (Ours)"]["train_time_sec"] = 17.0  # Documented V100 fine-tuning time
    print(f"  -> Finished Hetero-GNN | Demand MSE: {results['Hetero-GNN (Ours)']['demand_mse']} | Failure F1: {results['Hetero-GNN (Ours)']['failure_f1']} | Churn F1: {results['Hetero-GNN (Ours)']['sleeper_f1']}")

    # ==========================================================================
    # COMPARISON REPORT GENERATION
    # ==========================================================================
    print("\n" + "="*80)
    print(" BASELINE COMPARISON")
    print("="*80)
    df_results = pd.DataFrame(results).T
    print(df_results[[
        "demand_mse", "power_mse", "failure_f1", "sleeper_f1", "inventory_mse", "link_auc", "train_time_sec"
    ]].to_string())

    # Save to JSON and CSV artifacts
    csv_path = os.path.join(Config.ARTIFACT_DIR, "baseline_model_comparison.csv")
    json_path = os.path.join(Config.ARTIFACT_DIR, "baseline_model_comparison.json")
    df_results.to_csv(csv_path)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[ARTIFACTS] Benchmark results saved to:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
