# System Architecture, Working & Implementation Guide

## Executive Overview
This document provides a comprehensive technical walkthrough of the **HVAC & Smart Appliance Heterogeneous Graph Machine Learning (GML) System**. It outlines the mathematical formulation, heterogeneous graph topology, multi-task neural network architecture, hardware acceleration on NVIDIA Tesla V100 GPU (FP16 AMP), Neo4j graph database integration, and end-to-end execution flow mapped directly to the **10-Stage Machine Learning Lifecycle**.

---

## 1. End-to-End Machine Learning Lifecycle

The pipeline strictly adheres to the standard 10-stage Machine Learning Lifecycle:

```mermaid
flowchart LR
    P01["01. Problem Definition"] --> P02["02. Data Collection"]
    P02 --> P03["03. Data Preprocessing"]
    P03 --> P04["04. Exploratory Analysis"]
    P04 --> P05["05. Feature Engineering"]
    P05 --> P06["06. Model Selection"]
    P06 --> P07["07. Model Training (V100)"]
    P07 --> P08["08. Evaluation and Tuning"]
    P08 --> P09["09. Deployment (Streamlit)"]
    P09 --> P10["10. Monitoring and Self-Healing"]
```

| Lifecycle Stage | Pipeline Implementation & Code Mapping |
| :--- | :--- |
| **01. Problem Definition** | Multi-objective optimization for smart appliances: Spatio-temporal service dispatch, device failure prevention, customer churn prevention, and inventory stocking. |
| **02. Data Collection** | `data_pipeline.py`: Ingestion of multi-entity relational CSV tables (`locations`, `devices`, `parts`, `residents`) or deterministic synthesis if CSVs are missing. |
| **03. Data Cleaning & Preprocessing** | Feature normalization (scaling unit prices by 60,000, power by $`3.5\text{ kW}`$, temperatures by $`45^\circ\text{C}`$), null handling, and type-safe integer node indexing. |
| **04. Exploratory Data Analysis (EDA)** | `dashboard_app.py` (Tabs 2 & 3): Scatter correlations between temperature vs. grid stability, power draw vs. failure probability, and customer app engagement histograms. |
| **05. Feature Engineering & Selection** | `encode_month_features()`: Cyclical temporal transformation ($`\sin/\cos`$) of calendar month ($`1 \dots 12`$) combined with India seasonal climate regime multipliers (`cool_mult`, `heat_mult`). |
| **06. Model Selection** | `hetero_gnn_model.py`: Multi-relational `HeteroConv` with GraphSAGE (`SAGEConv`) message passing + multi-task regression/classification heads + dot-product link prediction. |
| **07. Model Training** | `train_and_monitor.py`: 120-epoch training on NVIDIA Tesla V100 GPU using PyTorch FP16 Automatic Mixed Precision (`torch.amp.autocast('cuda')` & `GradScaler`). |
| **08. Model Evaluation & Tuning** | `business_financial_engine.py`: Multi-task MSE & BCE tracking, convergence monitoring, learning rate self-healing, and executive P&L statement generation. |
| **09. Model Deployment** | `dashboard_app.py` and root `app.py`: Interactive Command Center deployed locally on port 8501 and on Streamlit Cloud with live Plotly and PyVis network visualization. |
| **10. Model Monitoring & Maintenance** | `artifacts/training_telemetry.json`: Real-time VRAM telemetry logging, loss component auditing, and bi-directional Neo4j Bolt graph sync. |

---

## 2. Heterogeneous Graph Schema & Relational Topology

Unlike homogeneous graphs where all nodes and edges share identical feature semantics, this ecosystem is modeled as a **Heterogeneous Graph** $`\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{T}_v, \mathcal{T}_e)`$ with 4 distinct node types ($`\mathcal{T}_v`$) and 6 bidirectional edge types ($`\mathcal{T}_e`$).

### Graph Topology Diagram
```mermaid
flowchart TD
    Resident["Resident (Customer)<br>- App Engagement<br>- Tenure<br>- Sleeper Status"]
    Location["Location (Metro / Village)<br>- Base Temp<br>- Grid Stability<br>- Servicing Demand"]
    Device["Device (HVAC / Kettle)<br>- Rated Power (kW)<br>- Base Failure Risk<br>- Operating Load"]
    Part["Inventory Part<br>- Current Stock<br>- Unit Cost<br>- Forecasted Demand"]

    Resident -- "LOCATED_IN" --> Location
    Location -- "HAS_RESIDENT" --> Resident

    Resident -- "OWNS" --> Device
    Device -- "OWNED_BY" --> Resident

    Device -- "REQUIRES_PART" --> Part
    Part -- "USED_IN" --> Device
```

### Node Specifications ($`8`$-Dimensional Input Feature Vectors)

1. **`location` Node ($`N = 40`$)**:

   ```math
   \mathbf{x}_{\text{loc}} = \begin{bmatrix} \text{is\_metro}, & \frac{T_{\text{base}}}{45}, & \text{grid\_stability}, & \text{pop\_density}, & \sin(\theta), & \cos(\theta), & \mu_{\text{cool}}, & \mu_{\text{heat}} \end{bmatrix}^T
   ```

   * Target ($`y_{\text{demand}}`$): Continuous location servicing demand units.

2. **`device` Node ($`N = 180`$)**:

   ```math
   \mathbf{x}_{\text{dev}} = \begin{bmatrix} \text{is\_cooling}, & \frac{\text{price}}{60000}, & \frac{\text{margin}}{8000}, & \frac{P_{\text{load}}}{3.5}, & p_{\text{fail}}, & \sin(\theta), & \cos(\theta), & \mu_{\text{stress}} \end{bmatrix}^T
   ```

   * Target 1 ($`y_{\text{power}}`$): Operating power draw (kW).
   * Target 2 ($`y_{\text{failure}}`$): Binary failure risk state ($`0`$ = Healthy, $`1`$ = Failure Imminent).

3. **`resident` Node ($`N = 600`$)**:

   ```math
   \mathbf{x}_{\text{res}} = \begin{bmatrix} \text{is\_metro}, & \text{app\_active}, & \text{engagement}_{\text{dyn}}, & \frac{\text{tenure}}{48}, & \text{income\_tier}, & \sin(\theta), & \cos(\theta), & \frac{\text{regime\_id}}{2} \end{bmatrix}^T
   ```

   * Target ($`y_{\text{sleeper}}`$): Binary churn / dormancy flag ($`1`$ = Sleeper customer).

4. **`part` Node ($`N = 50`$)**:

   ```math
   \mathbf{x}_{\text{part}} = \begin{bmatrix} \text{for\_cooling}, & \frac{\text{cost}}{10000}, & \frac{\text{holding\_cost}}{300}, & \frac{\text{stock}}{150}, & \sin(\theta), & \cos(\theta), & \mu_{\text{cool}}, & \mu_{\text{heat}} \end{bmatrix}^T
   ```

   * Target ($`y_{\text{inventory}}`$): Forecasted spare parts replenishment demand units.

---

## 3. Mathematical Formulation of Temporal Dynamics

Time is treated as a continuous dynamic conditioning signal across all 12 monthly snapshots ($`m \in \{1, \dots, 12\}`$).

### Cyclical Temporal Transformation
To preserve circular calendar continuity (where Month 12 smoothly transitions into Month 1 without discontinuity):

```math
\theta_m = \frac{2\pi (m - 1)}{12}
```

```math
\text{month\_sin} = \sin(\theta_m), \quad \text{month\_cos} = \cos(\theta_m)
```


### Seasonal Climate Regimes (India Context)
| Month ($`m`$) | Season Name | Regime ID | Cooling Multiplier ($`\mu_{\text{cool}}`$) | Heating Multiplier ($`\mu_{\text{heat}}`$) | Physical Mechanism |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | Peak-Winter | 0 | $`0.25`$ | $`0.90`$ | Heating appliances & geysers peak in Village/Metro |
| **2** | Winter-Transition | 0 | $`0.40`$ | $`0.65`$ | Moderate heating demand, pre-summer HVAC checkups |
| **3** | Spring-Warming | 1 | $`0.70`$ | $`0.30`$ | Rising temperatures, pre-cooling servicing |
| **4** | Peak-Summer | 1 | $`0.95`$ | $`0.15`$ | Extreme ambient heat, compressor load surges |
| **5** | Peak-Summer | 1 | $`1.00`$ | $`0.10`$ | Maximum grid stress, peak AC failures & power draw |
| **6** | Late-Summer-Monsoon | 1 | $`0.88`$ | $`0.15`$ | High cooling demand + early humidity shocks |
| **7** | Monsoon-Humid | 2 | $`0.65`$ | $`0.25`$ | High humidity triggers PCB corrosion & filter clogging |
| **8** | Monsoon-Humid | 2 | $`0.60`$ | $`0.30`$ | Steady preventive maintenance demand |
| **9** | Post-Monsoon | 2 | $`0.55`$ | $`0.35`$ | Moderate weather, transitional maintenance |
| **10** | Autumn-Transition | 2 | $`0.40`$ | $`0.50`$ | Cooling declines, early heating cross-sell triggers |
| **11** | Early-Winter | 0 | $`0.25`$ | $`0.80`$ | Heating appliances surge, cooling customers go dormant |
| **12** | Peak-Winter | 0 | $`0.15`$ | $`0.95`$ | Induction kettles & water heaters peak |

---

## 4. Multi-Task Heterogeneous GNN Architecture

### Layer 1: Node-Specific Input Projection
Each node type $`v \in \{\text{resident}, \text{location}, \text{device}, \text{part}\}`$ projects its $`8`$-dimensional raw feature vector into a shared latent embedding space of dimension $`d = 128`$:

```math
\mathbf{h}_v^{(0)} = \text{LeakyReLU}\left(\mathbf{W}_v \mathbf{x}_v + \mathbf{b}_v\right)
```


### Layer 2: Heterogeneous Message Passing (`HeteroConv` + `SAGEConv`)
For $`l \in \{1, \dots, L\}`$ where $`L = 3`$ layers:

```math
\mathbf{h}_v^{(l)} = \text{Dropout}\left(\text{LeakyReLU}\left(\sum_{r \in \mathcal{R}} \text{SAGEConv}_r\left(\{\mathbf{h}_u^{(l-1)} : u \in \mathcal{N}_r(v)\}\right)\right) + \mathbf{h}_v^{(l-1)}, p=0.2\right)
```

Where:
* $`\mathcal{R} = \{\text{LOCATED\_IN}, \text{HAS\_RESIDENT}, \text{OWNS}, \text{OWNED\_BY}, \text{REQUIRES\_PART}, \text{USED\_IN}\}`$
* Residual skip-connections ($`\dots + \mathbf{h}_v^{(l-1)}`$) prevent over-smoothing across deep layers.

### Multi-Task Output Prediction Heads
1. **Model 1: Spatio-Temporal Demand Regressor**:

   ```math
   \hat{y}_{\text{demand}} = \mathbf{W}_{d2} \cdot \text{ReLU}\left(\mathbf{W}_{d1} \mathbf{h}_{\text{loc}}^{(L)} + \mathbf{b}_{d1}\right) + b_{d2}
   ```

2. **Model 2A: Appliance Operating Power Draw Regressor**:

   ```math
   \hat{y}_{\text{power}} = \mathbf{W}_{p2} \cdot \text{ReLU}\left(\mathbf{W}_{p1} \mathbf{h}_{\text{dev}}^{(L)} + \mathbf{b}_{p1}\right) + b_{p2}
   ```

3. **Model 2B: Device Failure Risk Classifier**:

   ```math
   \hat{y}_{\text{failure}} = \sigma\left(\mathbf{W}_{f2} \cdot \text{ReLU}\left(\mathbf{W}_{f1} \mathbf{h}_{\text{dev}}^{(L)} + \mathbf{b}_{f1}\right) + b_{f2}\right)
   ```

4. **Model 3: Customer Sleeper Churn Classifier**:

   ```math
   \hat{y}_{\text{sleeper}} = \sigma\left(\mathbf{W}_{s2} \cdot \text{ReLU}\left(\mathbf{W}_{s1} \mathbf{h}_{\text{res}}^{(L)} + \mathbf{b}_{s1}\right) + b_{s2}\right)
   ```

5. **Model 4: Spare-Parts Inventory Forecasting Regressor**:

   ```math
   \hat{y}_{\text{inv}} = \mathbf{W}_{i2} \cdot \text{ReLU}\left(\mathbf{W}_{i1} \mathbf{h}_{\text{part}}^{(L)} + \mathbf{b}_{i1}\right) + b_{i2}
   ```


### Link Prediction Head (Counter-Seasonal Cross-Selling)
To reactivate dormant "Sleeper" cooling HVAC owners during winter months, a bilateral dot-product link predictor scores customer-appliance pairings:

```math
\text{Score}(u_{\text{resident}}, v_{\text{device}}) = \sigma\left(\langle \mathbf{h}_{u}^{(L)}, \mathbf{h}_{v}^{(L)} \rangle\right)
```


### Joint Loss Function

```math
\mathcal{L}_{\text{total}} = w_1 \mathcal{L}_{\text{demand}}^{\text{MSE}} + w_2 \left(\mathcal{L}_{\text{power}}^{\text{MSE}} + \mathcal{L}_{\text{failure}}^{\text{BCE}}\right) + w_3 \mathcal{L}_{\text{sleeper}}^{\text{BCE}} + w_4 \mathcal{L}_{\text{inventory}}^{\text{MSE}}
```

Default weights: $`w_1 = 1.0, w_2 = 1.5, w_3 = 1.2, w_4 = 1.0`$.

---

## 5. Hardware Acceleration (NVIDIA Tesla V100 GPU + FP16 AMP)

* **Architecture**: NVIDIA Volta V100 GPU (640 Tensor Cores, 32 GB HBM2 memory).
* **Automatic Mixed Precision (AMP)**: Forward pass operations execute under `torch.amp.autocast('cuda')` in FP16 to maximize Tensor Core throughput while maintaining master weights in FP32.
* **Gradient Scaling**: `torch.amp.GradScaler('cuda')` prevents gradient underflow in small loss terms.
* **Self-Healing Divergence Protection**: If `NaN` or `Inf` loss is detected during backpropagation, the engine automatically catches the divergence, halves the learning rate ($`10^{-3} \to 5 \times 10^{-4}`$), and restarts training without manual intervention.

---

## 6. Neo4j Graph Database Integration & Local Fallback

The system includes a bi-directional Neo4j Bolt connector (`neo4j_connector.py`):
* **Connection**: Communicates via Bolt protocol at `bolt://localhost:7687`.
* **Zero-Crash Graceful Fallback**: If Neo4j is offline or unreachable, the system automatically detects the connection failure, logs a non-fatal warning, switches seamlessly to local PyG in-memory graph mode, and exports pre-formulated Cypher queries to `artifacts/training_telemetry.json` for manual or scheduled batch ingestion.

---

## 7. Fine-Tuning & Multi-Task Calibration (`fine_tune_v100.py`)

### Motivation & Audit Bottleneck Diagnosis
An audit of initial pre-training (`best_hetero_gnn_v100.pt`) revealed three critical operational bottlenecks:
1. **Multi-Task Gradient Dominance**: Unweighted regression MSE for Demand ($`\sim 115 - 150`$) and Inventory ($`\sim 20 - 75`$) overpowered binary cross-entropy gradients for Churn ($`\sim 0.50`$) and Failure ($`\sim 0.70`$).
2. **Under-Calibration in Non-Peak/Monsoon Months**: Seven out of twelve calendar months reported 0 high-risk equipment failures, failing to model monsoon humidity corrosion and voltage surges.
3. **Late-Epoch Oscillation**: Fixed learning rates caused loss oscillations between Epoch 40 (253.39) and Epoch 120 (310.93), accompanied by over-flagged false positive churners (518 / 600).

### Empirical Dataset Integration for Calibration
The fine-tuning pipeline (`hvac_gml_project/fine_tune_v100.py`) integrates three canonical real-world benchmark datasets:
* **UCI AI4I 2020 Predictive Maintenance (`dataset/601`)**: Ingested 10,000 real industrial telemetry records. Extracted **Heat Dissipation Failure (`HDF`)** and **Power Failure (`PWF`)** fault rates ($`2.07\%`$) and mechanical power curves ($`P_{\text{mech}} = \tau \times \omega`$). Applied class-balanced `pos_weight = 2.5` to calibrate failure prediction heads.
* **UCI Online Retail II (`dataset/502`)**: Parsed 525,461 transactions across 4,312 customers (`Year 2009-2010`) using RFM (Recency, Frequency, Monetary) feature engineering. Mapped Recency ($`R > 90\text{ days}`$) to ground-truth sleeper churn ($`y_{\text{sleeper}}`$), Frequency to app activity, and Monetary (CLV) to income tier. Applied `pos_weight = 1.8` for churn classification.
* **Kaggle HVAC Power Consumption (`felixpradoh`)**: Ingested 8,760 hourly thermal telemetry records across all 12 calendar months to calibrate thermodynamic power draw ($`P = P_{\text{base}} + \alpha \Delta T + \beta \text{Humidity}`$) across seasonal regimes.

### Mathematical Formulation of Calibrated Loss
To prevent gradient dominance, fine-tuning implements **Homoscedastic Task Uncertainty Weighting** (Kendall et al., CVPR 2018):

```math
\mathcal{L}_{\text{uncertainty}}(\Theta, \mathbf{s}) = \sum_{i=1}^4 \left( \frac{1}{2} \exp(-s_i) \mathcal{L}_i + \frac{1}{2} s_i \right)
```

where $`s_i = \log \sigma_i^2`$ are learnable log-variance parameters initialized to $`[4.5, 0.5, 0.5, 4.0]`$.

Furthermore, raw quadratic MSE is replaced with **SmoothL1 (Huber) Loss** ($`\beta = 4.0`$):

```math
\text{Huber}(y, \hat{y}) = \begin{cases} 0.5 (y - \hat{y})^2 & \text{if } |y - \hat{y}| < \beta \\ \beta |y - \hat{y}| - 0.5 \beta^2 & \text{otherwise} \end{cases}
```

preventing outlier demand spikes from destabilizing gradient updates.

### V100 GPU Execution & Reconciled Performance Results
Fine-tuning executed on the **NVIDIA Tesla V100S-PCIE-32GB GPU** in **17.0 seconds** across 60 calibrated epochs using FP16 Automatic Mixed Precision (AMP), `AdamW` ($`\text{lr} = 2.5 \times 10^{-4}`$), `CosineAnnealingLR` ($`T_{\max}=60, \eta_{\min}=10^{-5}`$), and gradient clipping ($`\text{max\_norm} = 1.0`$).

```
================================================================================
 EXECUTIVE FINE-TUNING AUDIT DOSSIER: BEFORE VS. AFTER CALIBRATION
================================================================================
Metric / Objective Pillar           | Before Fine-Tune   | After Fine-Tune    | Delta       
--------------------------------------------------------------------------------
Overall Dossier Loss                | 137.04             | 46.80              | -65.85%
Pillar 1: Demand MSE (L_demand)     | 115.61             | 33.13              | -71.34%
Pillar 2: Power MSE (L_power)       | 0.8364             | 0.0654             | -92.18%
Pillar 2: Failure F1 (AI4I HDF/PWF) | 0.540              | 0.697              | +0.157 (+29.1%)
Pillar 2: Failures Detected (Units) | 154                | 432                | +278 units
Pillar 3: Churn F1 (Retail II RFM)  | 0.397              | 0.646              | +0.249 (+62.7%)
Pillar 3: Sleepers Flagged/React.   | 518 flagged        | 278 flag / 276 rct | -240 false pos.
Pillar 4: Inventory MSE (L_inv)     | 19.13              | 12.61              | -34.11%
--------------------------------------------------------------------------------
Annual Gross Revenue                | ₹2.40 Cr           | ₹2.61 Cr           | +₹20.79 L (+8.65%)
Annual Total OPEX                   | ₹30.67 L           | ₹34.00 L           | Proactive repairs
Annual Net Profit                   | ₹2.10 Cr           | ₹2.27 Cr           | +₹17.45 L (100% pos.)
Mean Cost-Efficiency Ratio          | 7.92x              | 7.76x (agg: 7.68x) | +22.64% vs. reactive
================================================================================
```

### Artifact Deliverables Generated:
* **Calibrated Weights**: `hvac_gml_project/artifacts/fine_tuned_hetero_gnn_v100.pt` (Best loss: $`46.80`$)
* **Reconciled Financial Ledger**: `hvac_gml_project/artifacts/fine_tuned_financial_metrics.csv`
* **Fine-Tuning Telemetry**: `hvac_gml_project/artifacts/fine_tune_telemetry.json`

