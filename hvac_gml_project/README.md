# HVAC & Smart Appliance GML Codebase Reference

## Modular Architecture Overview
The `hvac_gml_project` package contains the complete production-grade implementation of the Heterogeneous Graph Machine Learning pipeline. Each module is decoupled, unit-tested, and optimized for execution across both local environments (Windows/CPU) and the remote high-performance lab server (Linux + NVIDIA Tesla V100 GPU + Neo4j).

---

## Component Breakdown

```
hvac_gml_project/
├── config.py                      # Module 1: Central Environment & Hyperparameters
├── data_pipeline.py               # Module 2: Month-Conditioned Graph Dataset Builder
├── neo4j_connector.py             # Module 3: Neo4j Graph DB Connector & Fallback
├── telemetry_premodel.py          # Module 4: Sensor & Power Pre-Processor
├── hetero_gnn_model.py            # Module 5: Multi-Task Heterogeneous GNN Backbone
├── business_financial_engine.py   # Module 6: Executive Business & Financial Engine
├── train_and_monitor.py           # Module 7: V100 Mixed-Precision Training Orchestrator
├── dashboard_app.py               # Module 8: Interactive Streamlit Command Center
├── requirements.txt               # Module 9: Dependency Pinning
├── run_pipeline.sh / .bat         # Module 10: Platform Launch Scripts
└── tests/test_pipeline.py         # Test Suite: Unit & Integration Verification
```

---

### Module 1: `config.py`
* **Purpose**: Centralized single source of truth for runtime configurations, hyperparameter specifications, and hardware targets.
* **Key Components**:
  * `Config.DEVICE`: Automatically detects CUDA acceleration (`cuda:0` if available, else `cpu`).
  * `Config.USE_AMP_FP16`: Enables Volta Tensor Core mixed-precision training on Tesla V100 GPUs.
  * `Config.MONTH_REGIME`: Comprehensive dictionary mapping months $1 \dots 12$ to Indian climate regimes (Winter, Spring, Summer, Monsoon, Autumn) and setting cooling vs. heating multipliers.
  * Paths: Path-safe resolution for `./data` and `./artifacts`.

---

### Module 2: `data_pipeline.py`
* **Purpose**: Synthesizes or ingests schema-aligned relational CSVs and constructs monthly `HeteroData` PyTorch Geometric objects.
* **Key Functions**:
  * `encode_month_features(month)`: Transforms calendar month into continuous $\sin$ and $\cos$ cyclical encodings and retrieves seasonal multipliers.
  * `HVACGraphDatasetBuilder.generate_or_load_tables()`: Loads persisted CSV tables (`locations`, `devices`, `parts`, `residents`, and edges) from disk, or synthesizes them deterministically if missing.
  * `HVACGraphDatasetBuilder.build_monthly_hetero_data(month)`: Injects cyclical features into all 4 node types and returns a bidirectional `HeteroData` graph snapshot.

---

### Module 3: `neo4j_connector.py`
* **Purpose**: Synchronizes the multi-relational heterogeneous graph into Neo4j via the Bolt protocol (`bolt://localhost:7687`).
* **Key Features**:
  * `Neo4jGraphManager.sync_graph_tables(...)`: Translates pandas entity and edge tables into Cypher `MERGE` transactions.
  * **Zero-Crash Graceful Fallback**: If Neo4j is offline or refused, catches the exception, logs a non-fatal warning, and operates seamlessly in local PyG mode.
  * `get_cypher_visualization_queries()`: Returns ready-to-run Cypher queries for Neo4j Browser and Neo4j Bloom inspection.

---

### Module 4: `telemetry_premodel.py`
* **Purpose**: Pre-GNN sensor telemetry processing module that models real-time appliance power draw (kW) and hardware anomaly indicators conditioned on operating season.
* **Key Functions**:
  * `compute_monthly_telemetry_scores(...)`: Computes device power load and sensor failure probabilities under seasonal thermal stress.
  * `TelemetrySensorNet`: Neural network pre-processor with BatchNorm and non-linear power and anomaly prediction heads.

---

### Module 5: `hetero_gnn_model.py`
* **Purpose**: Deep multi-task heterogeneous graph neural network backbone.
* **Key Architecture**:
  * `self.lin_dict`: Node-specific linear projections transforming $8$-dimensional inputs to $128$-dimensional embeddings.
  * `self.convs`: 3-layer `HeteroConv` utilizing `SAGEConv` over all 6 bidirectional relation types with residual skip-connections and dropout.
  * 4 Specialized Output Heads:
    1. `head_demand`: Location-level demand regression.
    2. `head_power` & `head_failure`: Device-level power draw and failure risk logits.
    3. `head_sleeper`: Customer-level churn/dormancy logits.
    4. `head_inventory`: Warehouse part replenishment forecasting.
  * `predict_cross_sell_links(...)`: Bilateral dot-product link prediction head for sleeper customer re-engagement.

---

### Module 6: `business_financial_engine.py`
* **Purpose**: Translates neural network outputs into audited executive financial statements and operational KPIs.
* **Key Features**:
  * `CompanyBusinessInputs`: Configurable dataclass parameterizing ticket prices, AMC subscription pricing, preventive inspection costs, emergency repair charges, and SLA stockout penalties.
  * `BusinessFinancialEngine.evaluate_monthly_financials(...)`: Calculates on-demand service revenue, AMC subscription ARR, sleeper cross-sell margins, preventive vs. emergency repair costs, inventory holding expenses, net profit, and cost-efficiency ratios.

---

### Module 7: `train_and_monitor.py`
* **Purpose**: Execution engine for training across all 12 monthly graph snapshots on NVIDIA Tesla V100 GPU.
* **Key Features**:
  * Mixed-precision training via `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')`.
  * Multi-task loss tracking across all 4 heads across 120 epochs.
  * **Self-Healing Divergence Protection**: Monitors for `NaN`/`Inf` loss values; automatically halves learning rate and restarts training if loss diverges.
  * Exports `artifacts/best_hetero_gnn_v100.pt`, `artifacts/monthly_financial_metrics.csv`, and `artifacts/training_telemetry.json`.

---

### Module 8: `dashboard_app.py`
* **Purpose**: Interactive Graphical Command Center built with Streamlit, Plotly, and PyVis.
* **Key Tabs**:
  * **Tab 1: 🌐 Interactive Hetero-Graph**: Visualizes the graph topology using Barnes-Hut physics with guaranteed zero-crash node existence validation.
  * **Tab 2: ⚡ Device Failure Telemetry**: Operating power vs. failure risk scatter plot and high-risk SKU alerts.
  * **Tab 3: 👥 Sleeper Retention Engine**: Engagement distributions and link-predicted cross-sell campaigns.
  * **Tab 4: 📦 Spare Parts Forecasting**: Stockout risk identification and replenishment schedules.
  * **Tab 5: 💰 Executive Financial Ledger**: 12-month audited P&L statement, net profit margins, and reactive vs. proactive savings.
  * **Company Business POV Panel**: Expandable real-time controllers to simulate changes in ticket prices, technician wages, and SLA penalties.

---

### Module 9 & 10: Environment & Execution Scripts
* **`requirements.txt`**: Pinned dependencies ensuring seamless deployment on both CUDA servers and Streamlit Cloud.
* **`run_pipeline.sh`**: Production Bash script for Linux V100 server execution.
* **`run_pipeline.bat`**: Native batch script for Windows local execution.
* **`tests/test_pipeline.py`**: Automated 5-point test suite verifying cyclical math, graph builder, forward pass, financial calculations, and Cypher query integrity.

