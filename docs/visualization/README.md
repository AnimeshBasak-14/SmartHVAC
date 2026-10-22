# Smart HVAC Heterogeneous Graph Machine Learning: Visualization Asset Catalog

This directory contains publication-grade, high-resolution (300 DPI) diagrams, architectural schemas, algorithmic flowcharts, and empirical evaluation plots designed for inclusion in academic project reports (`.docx` / LaTeX) and PowerPoint presentations (`.pptx`).

---

## Visual Asset Index

| Asset File | Asset Type | Lifecycle Phase | Recommended Report Section | Recommended PPT Slide |
| :--- | :--- | :---: | :--- | :--- |
| **`01_ml_lifecycle_10_stages.png`** | High-Level Lifecycle Flowchart | 01–10 | Section 1: Executive Overview | Slide 2: Context in 10-Stage ML Lifecycle |
| **`02_heterogeneous_graph_schema.png`** | Graph Schema & Topology | 02, 06, 07 | Section 2: Heterogeneous Graph Schema | Slide 4: Multi-Task Hetero-GNN Formulation |
| **`03_hetero_gnn_model_architecture.png`** | Neural Network Architecture | 06, 07 | Section 2: Message Passing & Prediction Heads | Slide 4: Multi-Task Architecture & RGCN |
| **`04_end_to_end_system_pipeline_flowchart.png`** | System Architecture Pipeline | 01–10 | Section 1: End-to-End System Integration | Slide 2 / 11: System Architecture Overview |
| **`05_self_healing_training_flowchart.png`** | Algorithmic Flowchart | 07, 08 | Section 2: Self-Healing Numerical Safeguards | Slide 5: Pre-Training Trajectory & Safeguards |
| **`06_pretrain_vs_finetune_loss_trajectory.png`** | Training Convergence Plot | 07, 08 | Section 3: Fine-Tuning Execution & Loss Curve | Slide 6 / 7: Convergence on Tesla V100 GPU |
| **`07_pillar_performance_comparison_before_after.png`** | Grouped Metric Bar Chart | 08 | Section 3: Quantitative Model Evaluation Ledger | Slide 9: Before vs. After Calibration Ledger |
| **`08_12month_audited_financial_ledger.png`** | Dual-Axis Financial Bar/Line Chart| 08, 10 | Section 3: 12-Month Audited Operational Ledger | Slide 10: 12-Month Audited Financial Ledger |
| **`09_seasonal_cooling_heating_regimes.png`** | Climate & Demand Curve | 02, 04, 08 | Section 3: Climate Regime Demand Dynamics | Slide 8: Real Benchmark Data Ingestion |
| **`10_failure_and_churn_roc_curves.png`** | Multi-Panel ROC Curves | 08 | Section 3: Classification Discriminative Power | Slide 9: Classification Calibration Gains |
| **`11_kendall_uncertainty_task_weights.png`** | Dynamic Parameter Trajectory | 08 | Section 3: Kendall Uncertainty Derivation | Slide 7: Homoscedastic Uncertainty Weighting |
| **`12_economic_proactive_savings_breakdown.png`** | Monthly Savings Bar Chart | 08, 10 | Section 4: Proactive Breakdown Economics | Slide 10 / 11: Economic Impact Summary |

---

## Detailed Description of Each Visual Asset

### 1. `01_ml_lifecycle_10_stages.png`
* **Description**: A comprehensive visual overview of the 10 sequential stages in the industrial Machine Learning lifecycle: Problem Definition, Data Collection, Cleaning, EDA, Feature Engineering, Model Selection, Model Training, Evaluation & Tuning, Deployment, and Monitoring.
* **Key Takeaway**: Highlights Stages 07 and 08 as the computational and mathematical engine connecting theoretical graph schemas with operational cyber-physical deployment.

### 2. `02_heterogeneous_graph_schema.png`
* **Description**: Explicit relational topology displaying the 4 node types:
  * **Location Nodes** ($d = 6$): Weather parameters, solar irradiance, urban heat island index, building volume.
  * **Device Nodes** ($d = 7$): Age, cumulative run hours, COP efficiency, vibration velocity, current, pressure, capacity.
  * **Resident Nodes** ($d = 5$): Income tier, occupancy count, setpoint, app engagement, billing credit.
  * **Part Nodes** ($d = 5$): Procurement cost, inventory stock, lead time, reorder threshold, MTBF.
  * Connected by 3 heterogeneous edge relations: `(Location)-[:CONTAINS]->(Device)`, `(Resident)-[:OWNS]->(Device)`, and `(Device)-[:REQUIRES]->(Part)`.
* **Key Takeaway**: Explains how spatial, behavioral, and physical supply chain interdependencies are unified into an inductive graph.

### 3. `03_hetero_gnn_model_architecture.png`
* **Description**: Complete neural network pipeline showing input node vectors mapped through two Relational Graph Convolutional (`RGCNConv`) layers with LayerNorm and Dropout, routing to 5 decoupled Multi-Layer Perceptron (MLP) task heads:
  1. Head 1 (Demand): Continuous regional kW/ton regression (SmoothL1).
  2. Head 2 (Power): Device-level electrical kW regression (SmoothL1).
  3. Head 3 (Failure): Component breakdown probability (Weighted BCE, $w_{\text{pos}} = 2.5$).
  4. Head 4 (Churn): Sleeper customer churn probability (Weighted BCE, $w_{\text{pos}} = 1.8$).
  5. Head 5 (Inventory): Spare parts velocity regression (SmoothL1).
  All backpropagated through **Kendall Homoscedastic Uncertainty Loss**.
* **Key Takeaway**: Demonstrates multi-task joint optimization without manual hyperparameter grid search.

### 4. `04_end_to_end_system_pipeline_flowchart.png`
* **Description**: End-to-end flowchart from multi-source data ingestion (sensors, UCI AI4I 2020, UCI Online Retail II, Kaggle HVAC) through Graph Builder, Tesla V100 GPU training harness, Kendall fine-tuning, audited evaluation, and interactive Streamlit Argos decision-support serving.
* **Key Takeaway**: Ideal for an introductory or architectural summary slide in presentations.

### 5. `05_self_healing_training_flowchart.png`
* **Description**: Detailed algorithmic decision tree of the numerical self-healing training protection harness. Checks if loss is NaN, Inf, or spikes $> 10 \times$ moving average. If triggered, rolls back parameters to the last checkpoint, decays learning rate by 50%, flushes CUDA cache, and restarts optimization safely.
* **Key Takeaway**: Proves unattended training stability across volatile thermal spikes ($>44^\circ\text{C}$).

### 6. `06_pretrain_vs_finetune_loss_trajectory.png`
* **Description**: Dual-panel loss curve: (a) 120-epoch pre-training trajectory showing initial composite loss decay from $245.0$ to a plateau at $137.04$; (b) 60-epoch fine-tuning on Tesla V100 GPU dropping from $137.04$ to **46.80** (-65.85% reduction) in **17.0 seconds**.
* **Key Takeaway**: Empirical proof of rapid, stable fine-tuning convergence on enterprise hardware.

### 7. `07_pillar_performance_comparison_before_after.png`
* **Description**: Dual-panel grouped bar chart comparing baseline vs. calibrated performance:
  * Regression MSEs (log-scale): Demand MSE (-71.34%), Power MSE (-92.18%), Inventory MSE (-34.11%).
  * Classification F1-Scores: Failure F1 ($0.540 \to 0.697$, +29.07%), Sleeper Churn F1 ($0.397 \to 0.646$, +62.72%).
* **Key Takeaway**: Highlights quantitative improvements across all five prediction heads.

### 8. `08_12month_audited_financial_ledger.png`
* **Description**: Dual-axis financial bar and line chart across 12 calendar months:
  * Left Axis: Gross Operating Revenue (₹ Lakhs) and Operating Expenditure (OPEX).
  * Right Axis: Cost-Efficiency Ratio (Revenue / OPEX) spanning $6.54\text{x}$ to $9.46\text{x}$.
  * Annotated with audited annual totals: **₹2.61 Crore Gross Revenue**, **₹34.00 Lakhs OPEX**, **₹2.27 Crore Net Profit**, and **7.764x mean Cost-Efficiency**.
* **Key Takeaway**: Proves commercial viability and positive cash flow in all 12 operational months.

### 9. `09_seasonal_cooling_heating_regimes.png`
* **Description**: Multi-regime climate analysis showing mean ambient temperatures alongside distinct Metro Urban Demand (peaking in summer cooling) and Village Demand (peaking in winter heating).
* **Key Takeaway**: Explains the physical necessity of regime-aware graph neural networks.

### 10. `10_failure_and_churn_roc_curves.png`
* **Description**: ROC curves for (a) Component Failure Classification ($\text{AUC} = 0.882$ vs. baseline $0.741$) and (b) Resident Sleeper Churn Classification ($\text{AUC} = 0.843$ vs. baseline $0.682$).
* **Key Takeaway**: Validates classification discrimination under severe positive class rarity (<8%).

### 11. `11_kendall_uncertainty_task_weights.png`
* **Description**: Dual-panel trajectory showing: (a) Learnable task log-variances $s_k = \log \sigma_k^2$ and (b) Corresponding gradient weights $w_k = \frac{1}{2} \exp(-s_k)$ dynamically balancing during the 60 fine-tuning epochs.
* **Key Takeaway**: Visualizes how multi-task gradient dominance was resolved mathematically.

### 12. `12_economic_proactive_savings_breakdown.png`
* **Description**: Monthly percentage savings (16.5% to 28.4%, average **+22.64%**) achieved by proactive maintenance interventions (432 high-risk units flagged) and seasonal customer reactivations (276 sleepers converted).
* **Key Takeaway**: Clear metric for business presentations showing bottom-line operational savings.

