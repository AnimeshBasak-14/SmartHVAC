"""
train_and_monitor.py
Main Training, Evaluation & Telemetry Monitoring Script for NVIDIA Tesla V100 + Neo4j.
"""
import os
import sys
import time
import json
import torch
import torch.nn as nn
import pandas as pd
from config import Config
from data_pipeline import HVACGraphDatasetBuilder
from telemetry_premodel import compute_monthly_telemetry_scores
from hetero_gnn_model import MultiTaskHVACHeteroGNN
from business_financial_engine import BusinessFinancialEngine
from neo4j_connector import Neo4jGraphManager

def train_pipeline(learning_rate=None, max_retries=1):
    lr = learning_rate if learning_rate is not None else Config.LEARNING_RATE
    torch.manual_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"[INIT] Launching HVAC Heterogeneous GML Pipeline on device: {device} | FP16 AMP: {Config.USE_AMP_FP16} | LR: {lr}")

    builder = HVACGraphDatasetBuilder(seed=Config.SEED)
    tables = builder.generate_or_load_tables()
    df_res, df_loc, df_dev, df_part, e_res_loc, e_res_dev, e_dev_part = tables

    neo4j_mgr = Neo4jGraphManager()
    synced = neo4j_mgr.sync_graph_tables(df_res, df_loc, df_dev, df_part, e_res_loc, e_res_dev, e_dev_part, active_month=5)
    print(f"[NEO4J] Graph Database Sync Status: {'SUCCESS' if synced else 'LOCAL_PYG_FALLBACK'}")

    monthly_graphs = {}
    for m in Config.MONTHS:
        t_scores = compute_monthly_telemetry_scores(df_dev, month=m, device="cpu")
        g_data = builder.build_monthly_hetero_data(month=m, tables=tables, telemetry_scores=t_scores)
        monthly_graphs[m] = g_data.to(device)

    model = MultiTaskHVACHeteroGNN(
        hidden_channels=Config.HIDDEN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        num_layers=Config.NUM_GNN_LAYERS,
        dropout=Config.DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP_FP16)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    history = []
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        loss_components = {"demand": 0.0, "maintenance": 0.0, "churn": 0.0, "inventory": 0.0}

        diverged = False
        for m in Config.MONTHS:
            batch = monthly_graphs[m]
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=Config.USE_AMP_FP16):
                out = model(batch.x_dict, batch.edge_index_dict)

                l_demand = mse_loss(out["demand_pred"], batch["location"].y_demand)
                l_power = mse_loss(out["power_pred"], batch["device"].y_power)
                l_fail = bce_loss(out["failure_logits"], batch["device"].y_failure)
                l_maint = l_power + l_fail
                l_churn = bce_loss(out["sleeper_logits"], batch["resident"].y_sleeper)
                l_inv = mse_loss(out["inventory_pred"], batch["part"].y_inventory)

                total_m_loss = (
                    Config.W_DEMAND * l_demand +
                    Config.W_MAINTENANCE * l_maint +
                    Config.W_CHURN * l_churn +
                    Config.W_INVENTORY * l_inv
                )

            # Check for NaN / Inf divergence
            if torch.isnan(total_m_loss) or torch.isinf(total_m_loss):
                diverged = True
                break

            scaler.scale(total_m_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(total_m_loss.item())
            loss_components["demand"] += float(l_demand.item())
            loss_components["maintenance"] += float(l_maint.item())
            loss_components["churn"] += float(l_churn.item())
            loss_components["inventory"] += float(l_inv.item())

        if diverged:
            print(f"[WARNING] Loss diverged (NaN/Inf) at epoch {epoch} with LR={lr}. Retrying with a lower learning rate...")
            if max_retries > 0:
                reduced_lr = lr * 0.5
                print(f"[RETRY] Learning rate halved to {reduced_lr}; restarting training.")
                return train_pipeline(learning_rate=reduced_lr, max_retries=max_retries - 1)
            else:
                raise RuntimeError("Training diverged and max retries exceeded.")

        avg_loss = epoch_loss / 12.0
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)

        if epoch % 10 == 0 or epoch == 1:
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0
            log_entry = {
                "epoch": epoch,
                "avg_total_loss": round(avg_loss, 4),
                "demand_mse": round(loss_components["demand"] / 12.0, 4),
                "maintenance_loss": round(loss_components["maintenance"] / 12.0, 4),
                "sleeper_churn_bce": round(loss_components["churn"] / 12.0, 4),
                "inventory_mse": round(loss_components["inventory"] / 12.0, 4),
                "v100_vram_mb": round(vram_mb, 2),
                "elapsed_sec": round(time.time() - start_time, 2)
            }
            history.append(log_entry)
            print(f"[Epoch {epoch:03d}/{Config.EPOCHS}] Loss: {avg_loss:.4f} | Demand: {log_entry['demand_mse']:.3f} | "
                  f"Maint: {log_entry['maintenance_loss']:.3f} | Churn: {log_entry['sleeper_churn_bce']:.3f} | "
                  f"Inv: {log_entry['inventory_mse']:.3f} | VRAM: {vram_mb:.1f} MB")

    model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))
    model.eval()
    monthly_financials = []
    with torch.no_grad():
        for m in Config.MONTHS:
            batch = monthly_graphs[m]
            out = model(batch.x_dict, batch.edge_index_dict)
            fin_metrics = BusinessFinancialEngine.evaluate_monthly_financials(m, batch, out, tables)
            monthly_financials.append(fin_metrics)

    df_fin = pd.DataFrame(monthly_financials)
    df_fin.to_csv(Config.FINANCIAL_REPORT_CSV, index=False)

    telemetry_payload = {
        "status": "COMPLETED",
        "device": str(device),
        "fp16_amp_enabled": Config.USE_AMP_FP16,
        "best_loss": round(best_loss, 4),
        "annual_revenue_inr": round(float(df_fin["revenue_inr"].sum()), 2),
        "annual_expenditure_inr": round(float(df_fin["expenditure_inr"].sum()), 2),
        "annual_net_profit_inr": round(float(df_fin["net_profit_inr"].sum()), 2),
        "mean_cost_efficiency_ratio": round(float(df_fin["cost_efficiency_ratio"].mean()), 3),
        "mean_cost_efficiency_gain_pct": round(float(df_fin["cost_efficiency_gain_pct"].mean()), 2),
        "training_history": history,
        "cypher_queries": neo4j_mgr.get_cypher_visualization_queries()
    }
    with open(Config.TELEMETRY_JSON, "w") as f:
        json.dump(telemetry_payload, f, indent=2)

    neo4j_mgr.close()
    print(f"[DONE] Saved financial report to {Config.FINANCIAL_REPORT_CSV} and telemetry to {Config.TELEMETRY_JSON}")
    return df_fin, telemetry_payload

if __name__ == "__main__":
    train_pipeline()

