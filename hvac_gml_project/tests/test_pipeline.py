"""
test_pipeline.py
Unit and Integration Tests for HVAC Heterogeneous GML Pipeline.
"""
import os
import sys
import unittest
import torch
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from data_pipeline import encode_month_features, HVACGraphDatasetBuilder
from telemetry_premodel import compute_monthly_telemetry_scores
from hetero_gnn_model import MultiTaskHVACHeteroGNN
from business_financial_engine import BusinessFinancialEngine
from neo4j_connector import Neo4jGraphManager

def test_month_feature_encoding():
    for m in range(1, 13):
        m_sin, m_cos, regime_id, cool_mult, heat_mult = encode_month_features(m)
        assert -1.0 <= m_sin <= 1.0
        assert -1.0 <= m_cos <= 1.0
        assert regime_id in [0.0, 1.0, 2.0]
        assert 0.0 < cool_mult <= 1.0
        assert 0.0 < heat_mult <= 1.0

def test_dataset_builder_and_hetero_data():
    builder = HVACGraphDatasetBuilder(seed=42)
    tables = builder.generate_or_load_tables(num_residents=100, num_locations=10, num_devices=40, num_parts=20)
    df_res, df_loc, df_dev, df_part, e_res_loc, e_res_dev, e_dev_part = tables

    assert len(df_res) == 100
    assert len(df_loc) == 10
    assert len(df_dev) == 40
    assert len(df_part) == 20

    data = builder.build_monthly_hetero_data(month=5, tables=tables)
    assert data["location"].x.shape == (10, 8)
    assert data["device"].x.shape == (40, 8)
    assert data["resident"].x.shape == (100, 8)
    assert data["part"].x.shape == (20, 8)

    assert data["location"].y_demand.shape == (10, 1)
    assert data["device"].y_power.shape == (40, 1)
    assert data["device"].y_failure.shape == (40, 1)
    assert data["resident"].y_sleeper.shape == (100, 1)
    assert data["part"].y_inventory.shape == (20, 1)

def test_hetero_gnn_forward_pass():
    builder = HVACGraphDatasetBuilder(seed=42)
    tables = builder.generate_or_load_tables()
    df_res, df_loc, df_dev, df_part, _, _, _ = tables
    data = builder.build_monthly_hetero_data(month=6, tables=tables)

    model = MultiTaskHVACHeteroGNN(hidden_channels=32, out_channels=16, num_layers=2, dropout=0.1)
    model.eval()

    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)

    assert out["demand_pred"].shape == (len(df_loc), 1)
    assert out["power_pred"].shape == (len(df_dev), 1)
    assert out["failure_logits"].shape == (len(df_dev), 1)
    assert out["sleeper_logits"].shape == (len(df_res), 1)
    assert out["inventory_pred"].shape == (len(df_part), 1)
    assert "embeddings" in out

def test_financial_engine():
    builder = HVACGraphDatasetBuilder(seed=42)
    tables = builder.generate_or_load_tables()
    data = builder.build_monthly_hetero_data(month=5, tables=tables)

    # Validate financial calculation engine with target signals
    mock_outputs = {
        "demand_pred": data["location"].y_demand,
        "failure_logits": data["device"].y_failure,
        "sleeper_logits": data["resident"].y_sleeper,
        "inventory_pred": data["part"].y_inventory
    }
    fin = BusinessFinancialEngine.evaluate_monthly_financials(5, data, mock_outputs, tables)

    assert fin["revenue_inr"] > 0
    assert fin["expenditure_inr"] > 0
    assert fin["cost_efficiency_ratio"] > 1.0
    assert fin["net_profit_inr"] > 0
    assert fin["cost_efficiency_gain_pct"] > 0

def test_neo4j_cypher_queries():
    queries = Neo4jGraphManager.get_cypher_visualization_queries()
    assert "1_full_hetero_subgraph" in queries
    assert "2_sleeper_churn_cross_sell" in queries
    assert "3_high_risk_hvac_parts" in queries

if __name__ == "__main__":
    test_month_feature_encoding()
    print("[PASS] test_month_feature_encoding")
    test_dataset_builder_and_hetero_data()
    print("[PASS] test_dataset_builder_and_hetero_data")
    test_hetero_gnn_forward_pass()
    print("[PASS] test_hetero_gnn_forward_pass")
    test_financial_engine()
    print("[PASS] test_financial_engine")
    test_neo4j_cypher_queries()
    print("[PASS] test_neo4j_cypher_queries")
    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")

