# Code reference: `hvac_gml_project`

This folder holds the whole pipeline. Each file has one job. Everything runs on a Windows
laptop (CPU) and on the lab server (Linux, Tesla V100, optional Neo4j).

```
hvac_gml_project/
├── config.py                      settings, paths, month-to-season table
├── data_pipeline.py               builds the monthly graphs
├── neo4j_connector.py             optional export to Neo4j
├── telemetry_premodel.py          small network for device power and sensor anomalies
├── hetero_gnn_model.py            the multi-task GNN
├── business_financial_engine.py   turns predictions into revenue, cost and profit
├── train_and_monitor.py           first training run
├── fine_tune_v100.py              fine-tuning run
├── benchmark_baselines.py         XGBoost, GCN and GAT baselines
├── dashboard_app.py               Streamlit dashboard
├── requirements.txt
├── run_pipeline.sh / .bat         train, then start the dashboard
└── tests/test_pipeline.py         5 tests
```

## `config.py`
- `Config.DEVICE` is `cuda:0` when a GPU is available, otherwise `cpu`.
- `Config.USE_AMP_FP16` turns on FP16 mixed precision when a GPU is present.
- `Config.MONTH_REGIME` maps each month (1 to 12) to an Indian season (winter, spring, summer,
  monsoon, autumn) with a cooling multiplier and a heating multiplier.
- Paths to `./data` and `./artifacts`.

## `data_pipeline.py`
- `encode_month_features(month)` returns the sin/cos encoding of the month and that month's
  seasonal multipliers.
- `HVACGraphDatasetBuilder.generate_or_load_tables()` reads the CSV tables (locations, devices,
  parts, residents and the three edge tables) from `data/`. If they are missing, it generates them
  with a fixed random seed.
- `HVACGraphDatasetBuilder.build_monthly_hetero_data(month)` adds the month features to all four
  node types and returns a PyTorch Geometric `HeteroData` graph with edges in both directions.

## `neo4j_connector.py`
- `Neo4jGraphManager.sync_graph_tables(...)` writes the tables to Neo4j over Bolt
  (`bolt://localhost:7687`) with Cypher `MERGE` statements.
- If Neo4j isn't running, it logs a warning and the rest of the pipeline carries on without it.
- `get_cypher_visualization_queries()` returns example Cypher queries for Neo4j Browser or Bloom.

## `telemetry_premodel.py`
- `compute_monthly_telemetry_scores(...)` estimates each device's power draw and failure
  probability for a given month (hotter months stress cooling units more).
- `TelemetrySensorNet` is a small network (with BatchNorm) with one output for power and one for anomalies.

## `hetero_gnn_model.py`
- `self.lin_dict`: one linear layer per node type, mapping the 8 input features to 128 dimensions.
- `self.convs`: 3 `HeteroConv` layers with `SAGEConv` on all 6 edge types, plus residual
  connections and dropout.
- Output heads:
  1. `head_demand`: demand per location (regression).
  2. `head_power` and `head_failure`: power draw (regression) and failure risk (logit) per device.
  3. `head_sleeper`: whether a customer is inactive (logit).
  4. `head_inventory`: spare-part demand (regression).
- `predict_cross_sell_links(...)` scores customer-product pairs with a dot product, which we use to suggest
  a product to inactive customers.

## `business_financial_engine.py`
- `CompanyBusinessInputs` holds ticket prices, the AMC fee, inspection and emergency repair costs, and the
  stock-out penalty. The defaults are our assumptions.
- `BusinessFinancialEngine.evaluate_monthly_financials(...)` computes service revenue, AMC revenue,
  cross-sell margin, preventive and emergency repair costs, holding cost, net profit and the
  revenue/cost ratio for one month.

## `train_and_monitor.py`
- Trains on all 12 monthly graphs for 120 epochs, with FP16 mixed precision on the GPU
  (`torch.amp.autocast` and `GradScaler`).
- Logs each task's loss separately.
- If the loss becomes NaN or Inf, it halves the learning rate and starts again.
- Saves `artifacts/best_hetero_gnn_v100.pt`, `artifacts/monthly_financial_metrics.csv` and
  `artifacts/training_telemetry.json`.

## `fine_tune_v100.py`
- Starts from the trained checkpoint and runs 60 more epochs.
- Weights each task's loss with a learned uncertainty term (Kendall et al., 2018) and uses Huber loss
  for regression.
- Uses a cosine learning-rate schedule and gradient clipping.
- Prints and saves the metrics before and after fine-tuning (`artifacts/fine_tune_telemetry.json`).

## `benchmark_baselines.py`
- Trains XGBoost, a plain GCN and a plain GAT on the same data and compares them with our model.
- Results go to `artifacts/baseline_model_comparison.csv` and `.json`.

## `dashboard_app.py`
Streamlit app (the root `app.py` is the same file, set up for Streamlit Cloud). The sidebar picks the
month and graph settings, and an expander lets you change prices and costs.

| Tab | Shows |
| :--- | :--- |
| 1. Demand | Location graph, monthly metro vs village demand, temperature vs demand |
| 2. Maintenance | Device and part graph, power vs failure risk, failure modes, devices to inspect first |
| 3. Inactive customers | Customer graph with suggested products, engagement histogram, monthly inactive count |
| 4. Spare parts | Part and device graph, stock vs forecast, parts to reorder |
| Graph explorer | All four node types together, with filters |
| Revenue and costs | Monthly revenue and costs, this month's breakdown, table for all 12 months |

The charts go through `show_chart()`, which sends plain lists to the browser. Without it, Plotly 6
with an older Streamlit draws empty charts.

## Scripts and tests
- `run_pipeline.sh` (Linux server) and `run_pipeline.bat` (Windows) train the model and then start the dashboard.
- `tests/test_pipeline.py` has 5 tests: month encoding, graph building, a forward pass through the model,
  the financial calculation, and the Cypher queries.
