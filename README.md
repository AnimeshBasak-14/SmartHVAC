# HVAC & Smart Appliance Heterogeneous Graph Machine Learning (GML) Pipeline

## Overview & Architecture
This project is an end-to-end, multi-task Heterogeneous Graph Neural Network (GNN) pipeline designed for HVAC and smart appliance lifecycle management, predictive maintenance, spatio-temporal service demand forecasting, and customer retention. It models the dynamic interactions between **Residents**, **Geographic Locations** (Metropolitan vs. Village), **Appliances/Devices** (Cooling HVAC vs. Heating/Induction Kettles), and **Spare Parts Inventory** conditioned on dynamic monthly climate regimes ($m \in \{1, \dots, 12\}$).

Target hardware: **NVIDIA Tesla V100 GPU** with FP16 Automatic Mixed Precision (AMP) and **Neo4j Graph Database**.

---

## Repository Structure

```
CSC-505/Project/
├── README.md                          # Master documentation and runbook
├── .gitignore                         # Version control ignore rules
├── docs/                              # Extended architectural specifications
└── hvac_gml_project/                  # Modular production codebase
    ├── config.py                      # Central configuration (V100 FP16, paths, seasonal regimes)
    ├── data_pipeline.py               # Month-conditioned HeteroData graph builder (sin/cos temporal encodings)
    ├── neo4j_connector.py             # Bolt Neo4j manager & local PyG fallback
    ├── telemetry_premodel.py          # Pre-GNN sensor & power telemetry module
    ├── hetero_gnn_model.py            # HeteroConv + SAGEConv multi-task GNN & link predictor
    ├── business_financial_engine.py   # Executive financial KPI calculator (Revenue, Expenditure, Profit, Efficiency)
    ├── train_and_monitor.py           # V100 AMP FP16 multi-task training & telemetry logging
    ├── dashboard_app.py               # Interactive Streamlit + Plotly + PyVis command center
    ├── requirements.txt               # Pinned dependencies
    ├── run_pipeline.sh                # Linux execution script
    ├── run_pipeline.bat               # Windows execution script
    ├── data/                          # Persisted schema-aligned relational tables
    │   ├── locations.csv
    │   ├── devices.csv
    │   ├── parts.csv
    │   ├── residents.csv
    │   ├── edge_res_loc.csv
    │   ├── edge_res_dev.csv
    │   └── edge_dev_part.csv
    ├── artifacts/                     # Generated models, logs, and telemetry
    │   ├── best_hetero_gnn_v100.pt    # PyTorch model checkpoint
    │   ├── monthly_financial_metrics.csv # 12-month financial ledger
    │   ├── training_telemetry.json    # Full training telemetry & Cypher queries
    │   └── pyvis_sub_graph.html       # 3D interactive network graph export
    └── tests/
        └── test_pipeline.py           # Automated test suite
```

---

## 4 Multi-Task Loss Heads

1. **`demand_mse` (Model 1: Spatio-Temporal Location Demand Regression)**: Predicts appliance and servicing load across metropolitan and village locations conditioned on cooling and heating multipliers.
2. **`maintenance_loss` (Model 2: Power Regression + Failure Risk Classifier)**: Jointly models operating power draw (kW) and compressor/heating element failure likelihoods.
3. **`sleeper_churn_bce` (Model 3: Resident Sleeper Churn Classifier)**: Flags dormant customers during off-season shifts to trigger cross-selling campaigns.
4. **`inventory_mse` (Model 4: Spare-Parts Inventory Forecasting)**: Forecasts monthly spare parts requirements (compressors, filters, PCBs, induction IGBTs) to minimize stockouts and holding costs.

---

## Live Dashboard Access

The Streamlit Graphical Command Center is active:
- **Lab Server Network URL:** [http://10.13.1.7:8501](http://10.13.1.7:8501)
- **Local Host URL (via SSH Port Forwarding):** [http://localhost:8501](http://localhost:8501)

### SSH Port Forwarding Command
To forward the dashboard from the lab server to your local machine:
```bash
ssh -L 8501:localhost:8501 user@10.13.1.7
```

---

## Running the Pipeline

### On the Tesla V100 Lab Server (`10.13.1.7`):
```bash
cd ~/hvac_gml_project
/home/user/miniconda3/bin/python train_and_monitor.py
/home/user/miniconda3/bin/streamlit run dashboard_app.py --server.port 8501 --server.headless true
```

### On Windows Locally:
```powershell
cd hvac_gml_project
py -3.11 train_and_monitor.py
py -3.11 -m streamlit run dashboard_app.py --server.port 8501
```

---

## Verification & Test Suite
Run the 5-point test suite:
```bash
python tests/test_pipeline.py
```
Validates:
- Cyclical month encodings ($\sin, \cos$) & climate regimes
- Heterogeneous graph snapshot assembly & bidirectional message passing
- Multi-task GNN forward pass dimensions
- Business financial engine calculation validity
- Neo4j Cypher query integrity & local PyG fallback

