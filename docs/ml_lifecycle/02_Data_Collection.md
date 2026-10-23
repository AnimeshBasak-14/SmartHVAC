# Step 02: Data Collection

> **ML Lifecycle Phase**: 02 / 10  
> **System**: SmartHVAC & Seasonal Appliance Heterogeneous GML  
> **Primary Benchmark Sources**: UCI AI4I 2020 (601), UCI Online Retail II (502), Kaggle HVAC Power (felixpradoh), Indian Seasonal Smart Appliance Benchmark (IIT Kanpur / Flipkart)  
> **Fine-Tuning Dataset**: Empirical Multi-Source Calibration Pipeline (`fine_tune_v100.py` on NVIDIA Tesla V100 GPU)

---

## 1. Heterogeneous Data Sourcing Strategy

Graph Machine Learning requires capturing multiple physical and business facets: electromechanical equipment operating bounds, customer behavioral transactions, macro-environmental seasonal climates, and supply chain replacement part velocity. Purely synthetic or randomly sampled data introduces structural artifacts that fail to represent real-world enterprise operations.

To ground the system in verified empirical distributions, the data collection pipeline integrates four benchmark datasets mapped to the graph entities:

```
+-----------------------------------------------------------------------------------+
|                            EXTERNAL BENCHMARK DATASETS                            |
+-----------------------------------------------------------------------------------+
| 1. UCI AI4I 2020 Predictive Maintenance (Archive ID: dataset/601 / 10,000 logs)   |
|    -> Calibrates Device failure modes (HDF/PWF) and mechanical power curves       |
|    -> Referenced as [DATASET 4] in fine_tune_v100.py execution telemetry          |
+-----------------------------------------------------------------------------------+
| 2. UCI Online Retail II (Archive ID: dataset/502 / 525,461 transaction rows)      |
|    -> Calibrates Resident behavioral RFM profiles, sleeper churn, and CLV tiers   |
|    -> Referenced as [DATASET 1] in fine_tune_v100.py execution telemetry          |
+-----------------------------------------------------------------------------------+
| 3. Kaggle HVAC Power Consumption (felixpradoh / 8,760 hourly readings)            |
|    -> Calibrates thermodynamic COP, ambient vs indoor temperatures, and load (kW) |
|    -> Referenced as [DATASET 3] in fine_tune_v100.py execution telemetry          |
+-----------------------------------------------------------------------------------+
| 4. Indian Seasonal Smart Appliance Benchmark (IIT Kanpur / Flipkart Catalog)      |
|    -> Calibrates 12 regional climate regimes, appliance pricing, and part costs   |
|    -> Referenced as [DATASET 2] in regional seasonal distribution modeling        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        FINE-TUNING CALIBRATION DATASET                            |
|             (fine_tune_v100.py / NVIDIA Tesla V100 GPU FP16 Pipeline)             |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            TARGET GRAPH ENTITY NODES                              |
| - Location Nodes (40 Hubs): Metro vs Village, ambient temp, grid stability index  |
| - Device Nodes (180 SKUs): Cooling HVAC, heating appliances, rated kW, base risk  |
| - Resident Nodes (600 Users): Tenure, app active, engagement score, income tier   |
| - Part Nodes (50 SKUs): Stock level, unit cost, monthly holding cost, reorder MOQ |
+-----------------------------------------------------------------------------------+
```

---

## 2. Benchmark Dataset Specifications

### Dataset 1: UCI AI4I 2020 Predictive Maintenance (`dataset/601`)
* **Source**: UCI Machine Learning Repository
* **Archive ID**: `dataset/601`
* **File**: `hvac_gml_project/data/ai4i2020.csv` (and archive `ai4i2020.zip`)
* **Volume**: 10,000 empirical machine operational logs
* **Citation**: Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications. Third International Conference on Artificial Intelligence for Industries (AI4I 2020).
* **Pipeline Identifier**: Designated as `[DATASET 4]` in `fine_tune_v100.py` execution logs.
* **Key Attributes**:
  * `Air temperature [K]`: Ambient thermal condition (mean: 300.0 K / 26.85 deg C, range: 295.3 K to 304.4 K).
  * `Process temperature [K]`: Internal operating temperature (mean: 310.0 K, range: 305.7 K to 313.8 K).
  * `Rotational speed [rpm]`: Compressor and motor angular velocity (mean: 1,538.8 rpm, range: 1,168 rpm to 2,886 rpm).
  * `Torque [Nm]`: Mechanical load on the drive shaft (mean: 40.0 Nm, range: 3.8 Nm to 76.6 Nm).
  * `Tool wear [min]`: Cumulative mechanical fatigue (mean: 107.9 min, range: 0 to 253 min).
* **Failure Modes Extracted**:
  * `HDF` (Heat Dissipation Failure): 115 occurrences (1.15%). Occurs when the difference between process temperature and air temperature drops below 8.6 K under rotational speeds below 1,380 rpm, directly modeling summer HVAC condenser fan overheating.
  * `PWF` (Power Failure): 95 occurrences (0.95%). Occurs when mechanical power falls below 3,500 W or exceeds 9,000 W, modeling electrical compressor motor lockup.
  * Consolidated Critical Fault Ground Truth:
    $$y_{\text{failure}} = \max(\text{HDF}, \text{PWF})$$
    Yields 207 unique critical fault events across 10,000 observations (empirical fault rate of 2.07%).
* **Mechanical Power Derivation**:
  $$\text{Power (kW)} = \frac{\text{Torque [Nm]} \times \text{Rotational Speed [rpm]} \times 2\pi}{60,000}$$
  The derived mechanical power has a mean of 6.28 kW (standard deviation: 1.07 kW, range: 1.15 kW to 10.47 kW). For residential and light-commercial smart appliance modeling, power is normalized into the commercial operating range (1.2 kW to 3.6 kW).
* **Mapping**: Ingested directly into `data_pipeline.py` and `fine_tune_v100.py` to establish empirical device failure priors (`base_failure_risk`) and compressor motor nameplate power ratings (`rated_power_kw`).

### Dataset 2: UCI Online Retail II (`dataset/502`)
* **Source**: UCI Machine Learning Repository
* **Archive ID**: `dataset/502`
* **Files**: Raw `online_retail_II.xlsx` (and `online_retail_ii.zip`) processed into `hvac_gml_project/data/online_retail_ii_rfm.csv`
* **Volume**: 525,461 transactional customer rows across 4,312 distinct customer profiles (`Year 2009-2010`)
* **Citation**: Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation. Journal of Database Marketing & Customer Strategy Management, 19(3), 197-208.
* **Pipeline Identifier**: Designated as `[DATASET 1]` in `fine_tune_v100.py` execution logs.
* **Key Attributes**:
  * `Invoice`: Unique 6-digit transaction identifier. Invoices prefixed with 'C' indicate cancellations.
  * `StockCode`: Distinct 5-digit item SKU identifier.
  * `Quantity`: Number of units purchased per transaction line.
  * `InvoiceDate`: Timestamp of order generation.
  * `Price`: Unit sales price in GBP.
  * `Customer ID`: Unique 5-digit customer identifier.
* **Derived Behavioral RFM Features**:
  Following the data mining methodology of Hamim-Susmit and Ruturaj Dixit, raw transaction rows are cleaned and aggregated into customer-level RFM profiles:
  * **Recency (R)**: Days elapsed between the customer's last purchase and the snapshot cutoff date (range: 1 to 374 days, mean: 91.2 days). Customers with Recency > 90 days during off-peak and shoulder seasons establish the ground-truth sleeper churn target ($y_{\text{sleeper}}$), matching the empirical 33.28% dormancy rate.
  * **Frequency (F)**: Total count of distinct purchase transactions (range: 1 to 205, mean: 4.46). Customers with purchase frequency $\ge 2$ map to active app installation (`customer_app_active = 1`).
  * **Monetary (M)**: Cumulative monetary value (range: GBP 2.95 to GBP 349,164, mean: GBP 2,048.24). Log-transformed monetary spend maps to customer income decile and Customer Lifetime Value (CLV):
    $$\text{Income Tier} = \text{clip}\left(\frac{\log(1 + M)}{\log(1 + M_{95})}, 0.15, 1.0\right)$$
* **Mapping**: Transformed into customer behavioral feature vectors for `Resident` nodes in `residents.csv`.

### Dataset 3: Kaggle HVAC Power Consumption (`felixpradoh`)
* **Source**: Kaggle Dataset Archive (`felixpradoh/hvac-power-consumption`)
* **File**: `hvac_gml_project/data/hvac_power_consumption.csv`
* **Volume**: 8,760 hourly records spanning a full 365-day operational calendar
* **Citation**: Prado, F. (2022). Smart Building HVAC Power Consumption and Sensor Telemetry Benchmark Dataset. Kaggle Datasets.
* **Pipeline Identifier**: Designated as `[DATASET 3]` in `fine_tune_v100.py` execution logs.
* **Key Attributes**:
  * `timestamp`: Hourly temporal index across 8,760 continuous hours.
  * `Month`: Calendar month index (1 through 12).
  * `Ambient_Temp`: Outdoor atmospheric temperature (10.5 deg C to 44.2 deg C).
  * `Indoor_Temp`: Indoor thermostat setpoint (21.0 deg C to 24.0 deg C).
  * `Humidity_pct`: Relative atmospheric humidity (20% to 98%).
  * `HVAC_Mode`: Operational state (`Cooling`, `Heating`, `Idle`).
  * `Power_kW`: Measured continuous power draw (mean: 1.85 kW, peak: 4.50 kW).
* **Thermodynamic Integration**:
  Ingested by `telemetry_premodel.py` and `fine_tune_v100.py` to calibrate monthly energy demand curves and seasonal thermodynamic load functions. Captures empirical Coefficient of Performance (COP) degradation as ambient temperatures deviate from indoor setpoints.

### Dataset 4: Indian Seasonal Smart Appliance & Regional Climate Benchmark
* **Source**: IIT Kanpur Smart Grid Telemetry Reports & Flipkart Appliance Market Catalog
* **Pipeline Identifier**: Designated as `[DATASET 2]` in regional seasonal distribution modeling.
* **Scope**: 12 distinct calendar climate regimes across 40 geographic hubs in India (metropolitan centers vs. rural/village hubs).
* **Key Parameters**:
  * **Seasonal Multipliers**:
    * Peak Summer (May–June): Cooling multiplier 2.15x to 2.30x, heating multiplier 0.15x, ambient temperatures 38 deg C to 45 deg C.
    * Monsoon Transition (July–August): Cooling multiplier 1.65x, humidity 80% to 90%, grid instability factor 0.65 to 0.85 (moisture-induced PCB failure risks).
    * Peak Winter (December–January): Cooling multiplier 0.20x, heating multiplier 2.20x to 2.45x, ambient temperatures 5 deg C to 15 deg C.
  * **Commercial Smart Appliance Catalog (180 SKUs)**:
    * Cooling HVAC SKUs (60%): Inverter Split ACs, Variable Refrigerant Flow units (retail pricing: INR 28,000 to INR 58,000, rated power: 1.2 to 2.8 kW, service margin: 12%).
    * Heating SKUs (40%): Storage Geysers, Space Heaters, Induction Kettles (retail pricing: INR 1,800 to INR 14,000, rated power: 1.0 to 2.2 kW).
  * **Component Inventory & Bill of Materials (50 SKUs)**:
    * Cooling components: Rotary compressors, condenser coils, inverter PCB motherboards, air filters.
    * Heating components: Induction IGBT modules, heating elements, thermostat sensors.
    * Unit costs: INR 350 to INR 8,500, with monthly holding costs calibrated at 3.0% of unit cost.
* **Mapping**: Calibrates location-specific ambient profiles (`locations.csv`), appliance catalog distributions (`devices.csv`), and spare-part supply chain parameters (`parts.csv`).

---

## 3. The Fine-Tuning Calibration Dataset (`fine_tune_v100.py`)

To eliminate synthetic baseline bias, evaluate the heterogeneous GNN on realistic enterprise distributions, and satisfy external audit requirements (Argos Review), a dedicated fine-tuning dataset was constructed and run on an NVIDIA Tesla V100 GPU (`hvac_gml_project/fine_tune_v100.py`).

### Hardware and Compute Specification
* **Server**: Enterprise GPU Server (`10.13.1.7`)
* **GPU**: NVIDIA Tesla V100S-PCIE-32GB (5,120 CUDA Cores, 640 Tensor Cores, 32 GB HBM2 VRAM, 1,134 GB/s bandwidth)
* **Software Stack**: Ubuntu 22.04 LTS, CUDA 13.0 / 12.4, PyTorch 2.6.0, PyTorch Geometric 2.8.1
* **Precision**: Automatic Mixed Precision (FP16 AMP via `torch.cuda.amp.autocast` and `GradScaler`)
* **Execution Duration**: 17.0 seconds across 60 fine-tuning epochs
* **Peak Memory Footprint**: 28.8 MB VRAM (enabled by sparse PyG message-passing kernels)

### Calibration Pipeline Mechanics
The fine-tuning dataset generation script (`load_empirical_fine_tuning_tables` in `fine_tune_v100.py`) merges the base graph schema with empirical distributions:

```python
# Empirical Calibration Pipeline Mechanics (fine_tune_v100.py)

# 1. Base Graph Ingestion:
tables = builder.generate_or_load_tables()
df_res, df_loc, df_dev, df_part, e_res_loc, e_res_dev, e_dev_part = tables

# 2. Ingest UCI AI4I 2020 (dataset/601) - Injected as [DATASET 4]:
# Extracts HDF and PWF failure flags (2.07% rate) and derives mechanical power:
# P_mech = Torque * RPM * 2 * pi / 60,000 kW (scaled to 1.2 - 3.6 kW).
# Replaces random device failure priors with empirical fault frequencies.

# 3. Ingest UCI Online Retail II (dataset/502) - Injected as [DATASET 1]:
# Ingests 4,312 customer RFM profiles.
# Recency > 90 days establishes ground-truth sleeper churn target.
# Frequency >= 2 establishes customer app active status.
# Log-Monetary spend establishes customer income tier (0.15 - 1.0).

# 4. Ingest Kaggle HVAC Power (felixpradoh) - Injected as [DATASET 3]:
# Ingests 8,760 hourly readings to build monthly thermal load curves.
# Maps mean ambient temperature and 90th percentile power draw for all 12 regimes.
```

### Mathematical Loss Formulations
To resolve gradient dominance where regression MSE dwarfed classification BCE, fine-tuning introduces two formulations:

1. **Kendall Homoscedastic Multi-Task Loss Weighting** (Kendall et al., CVPR 2018):
   $$\mathcal{L}_{\text{total}} = \sum_{k=1}^4 \left( \frac{1}{2} \exp(-s_k) \mathcal{L}_k + \frac{1}{2} s_k \right)$$
   where $s_k = \log(\sigma_k^2)$ is a learnable task log-variance parameter. Initialized at $s = [4.5, 0.5, 0.5, 4.0]$, this downscales continuous regression tasks by $\exp(-4.5) \approx 0.011$, matching classification BCE gradients ($\exp(-0.5) \approx 0.606$).

2. **SmoothL1 (Huber) Regression Loss** ($\beta = 4.0$):
   $$\text{SmoothL1}(e) = \begin{cases} 0.5 \times \frac{e^2}{\beta} & \text{if } |e| < \beta \\ |e| - 0.5 \times \beta & \text{otherwise} \end{cases}$$
   For small errors, the loss is quadratic (L2) for smooth convergence; for large outlier errors (such as 45 deg C summer heatwaves), the loss transitions to linear (L1) with a bounded constant gradient of 1.0.

3. **Class-Balanced Positive Loss Weighting**:
   Because industrial failure and sleeper churn are sparse, positive weights are applied to binary cross-entropy:
   * Device Failure Head: `pos_weight = 2.5`
   * Customer Sleeper Churn Head: `pos_weight = 1.8`

### Fine-Tuning Output Artifacts
Execution of `fine_tune_v100.py` generates three audited production artifacts stored under `hvac_gml_project/artifacts/`:

1. `fine_tuned_hetero_gnn_v100.pt`: Serialized PyTorch state dictionary containing calibrated model weights.
2. `fine_tune_telemetry.json`: Audited benchmark metrics comparing pre-training against fine-tuned performance:

| Metric / Objective Pillar | Pre-Fine-Tune Baseline | Post-Fine-Tune Calibrated | Absolute Delta | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Overall Dossier Loss** | 137.04 | **46.80** | -90.24 | **-65.85% Error Reduction** |
| **Pillar 1: Demand MSE** | 115.61 (MAE: 8.77) | **33.13** (MAE: 4.12) | -82.48 | **-71.34% Error Reduction** |
| **Pillar 2: Power MSE** | 0.8364 (MAE: 0.749) | **0.0654** (MAE: 0.182) | -0.771 | **-92.18% Error Reduction** |
| **Pillar 2: Failure F1 Score** | 0.540 (154 flagged) | **0.697** (432 flagged) | +0.157 | **+29.07% Precision/Recall Gain** |
| **Pillar 3: Sleeper Churn F1** | 0.397 (518 flagged) | **0.646** (278 flagged) | +0.249 | **+62.72% Churn Detection Gain** |
| **Pillar 4: Inventory MSE** | 19.13 (MAE: 3.45) | **12.61** (MAE: 2.68) | -6.52 | **-34.11% Error Reduction** |

3. `fine_tuned_financial_metrics.csv`: 12-month audited financial ledger:
   * **Gross Annual Operating Revenue**: INR 2,61,01,851.84 (~INR 2.61 Crore, +8.65% over baseline)
   * **Annual Operating Expenditure (OPEX)**: INR 34,00,492.40 (~INR 34.00 Lakhs, exact row-sum match across all 12 months)
   * **Net Annual Operating Profit**: INR 2,27,01,359.44 (~INR 2.27 Crore, +INR 17.45 Lakhs over baseline)
   * **Cost-Efficiency Ratio**: 7.764x (unweighted monthly mean) / 7.676x (aggregate annual ratio)
   * **Proactive Maintenance Savings**: +22.64% average breakdown cost reduction (peaking at +28.4% in extreme seasons)

---

## 4. Graph Database Storage & Ingestion Pipeline

All cleaned tabular tables are stored as persistent CSV files in `hvac_gml_project/data/`:
* `locations.csv`: 40 geographic hubs with coordinates, temperature profiles, and grid stability indices.
* `devices.csv`: 180 smart appliance models with rated power, unit prices, and calibrated failure risk priors.
* `residents.csv`: 600 customer profiles with app engagement, tenure, and calibrated income tiers.
* `parts.csv`: 50 replacement components with unit costs, holding costs, and current stock levels.
* `edge_res_loc.csv`: Bipartite residence links connecting residents to geographic locations.
* `edge_res_dev.csv`: Bipartite ownership links connecting customers to installed smart appliances.
* `edge_dev_part.csv`: Bipartite bill-of-materials dependency links connecting devices to required replacement parts.

### Neo4j Graph Database Integration
For visual traversal and graph queries, tables synchronize to **Neo4j** over the Bolt binary protocol (`bolt://10.13.1.7:7687` via `neo4j_connector.py`).

Example Cypher query to identify high-risk devices and their required replacement parts during peak seasons:

```cypher
MATCH (loc:Location)<-[:LOCATED_IN]-(dev:Device)-[:REQUIRES]->(part:Part)
WHERE dev.base_failure_risk > 0.15 AND loc.is_metropolitan = 1
RETURN loc.name AS Location, dev.sku_name AS Device, part.part_name AS RequiredPart, part.current_stock AS StockOnHand
ORDER BY dev.base_failure_risk DESC
LIMIT 25;
```

---

## 5. Data Governance and Licensing

* **UCI AI4I 2020 Predictive Maintenance**: Creative Commons Attribution 4.0 International (CC BY 4.0).
* **UCI Online Retail II**: Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Kaggle HVAC Power Consumption**: Open Data Commons Attribution License (ODC-By).
* **Indian Seasonal Appliance Catalog**: Sourced from public market research reports, academic grid studies (IIT Kanpur), and commercial catalogs.
* **Privacy and Anonymization**: All transactional records from Online Retail II are stripped of customer names, IP addresses, and payment tokens, retaining only synthetic IDs and aggregate RFM metrics.
