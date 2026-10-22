# HVAC & Smart Appliance Heterogeneous Graph Machine Learning (GML) System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20Tesla%20V100%20(32GB)-76B900?logo=nvidia)](https://www.nvidia.com)
[![Framework](https://img.shields.io/badge/PyTorch%20Geometric-HeteroGNN-EE4C2C?logo=pytorch)](https://pytorch-geometric.readthedocs.io/)
[![Database](https://img.shields.io/badge/Graph%20DB-Neo4j%20Bolt-008CC1?logo=neo4j)](https://neo4j.com)

## Executive Overview
This repository contains the end-to-end production implementation of an enterprise-grade **Heterogeneous Graph Neural Network (GNN)** pipeline designed for smart HVAC systems and connected residential appliances. It simultaneously optimizes:
1. **Spatio-Temporal Service Demand Forecasting** (Metropolitan cooling vs. Rural village heating).
2. **Preventive Device Health & Failure Triage** (Compressor and heating element failure prediction).
3. **Customer Retention & "Sleeper" Churn Reactivation** (Counter-seasonal cross-sell link prediction).
4. **Supply Chain & Spare Parts Inventory Optimization** (Stockout risk and warehouse holding cost balancing).

The system is accelerated via **NVIDIA Tesla V100 GPU (FP16 Mixed Precision)**, integrates with **Neo4j Graph Database**, and serves an interactive **Streamlit Command Center**.

---

## 📚 Comprehensive Documentation Suite

| Document | Core Topics Covered |
| :--- | :--- |
| **[System Architecture & Implementation](docs/ARCHITECTURE_AND_IMPLEMENTATION.md)** | Mathematical formulation, 10-stage ML Lifecycle mapping, `HeteroConv` + `SAGEConv` layer design, V100 AMP FP16 acceleration, and divergence self-healing. |
| **[GML Foundations: What, Why, How & Where](docs/GML_FOUNDATIONS_WHAT_WHY_HOW_WHERE.md)** | Conceptual foundations of Graph Machine Learning, comparison against Tabular ML (XGBoost/RF), relational inductive bias, and enterprise deployment areas. |
| **[Dataset Guide: What, Why, How & Where](docs/DATASET_GUIDE_WHAT_WHY_HOW_WHERE.md)** | In-depth schemas of the 4 entity tables and 3 edge tables, seasonal cyclical encoding ($\sin/\cos$), normalization protocols, and real-world industrial data mapping. |
| **[Modular Codebase Reference](hvac_gml_project/README.md)** | Component-by-component walkthrough of all 10 source code files, configurations, test suites, and launch scripts. |

---

## 🔄 Machine Learning Lifecycle Documentation Suite

The project implements the complete 10-stage Machine Learning Lifecycle, detailed in dedicated technical documentation files:

* 📖 **[ML Lifecycle Master Overview](docs/ml_lifecycle/README.md)**
1. **[01. Problem Definition](docs/ml_lifecycle/01_Problem_Definition.md)**: 4 multi-task pillars, seasonal climate shifts, mathematical graph formulation.
2. **[02. Data Collection](docs/ml_lifecycle/02_Data_Collection.md)**: UCI AI4I 2020 (601), UCI Online Retail II (502), Kaggle HVAC Power (`felixpradoh`).
3. **[03. Data Preprocessing](docs/ml_lifecycle/03_Data_Cleaning_and_Preprocessing.md)**: Cancellation filtering, rotational power derivation, graph bipartite sanitization.
4. **[04. Exploratory Analysis (EDA)](docs/ml_lifecycle/04_Exploratory_Data_Analysis.md)**: 12 seasonal climate regimes, failure correlation analysis, customer RFM distributions.
5. **[05. Feature Engineering](docs/ml_lifecycle/05_Feature_Engineering_and_Selection.md)**: Continuous cyclical month encoding ($\sin/\cos$), 8-dimensional dense node features.
6. **[06. Model Selection](docs/ml_lifecycle/06_Model_Selection.md)**: Multi-task `HeteroConv` architecture with bipartite `SAGEConv` and link prediction.
7. **[07. Model Training](docs/ml_lifecycle/07_Model_Training.md)**: Training on NVIDIA Tesla V100 GPU (`10.13.1.7`) with FP16 AMP mixed precision.
8. **[08. Model Evaluation & Tuning](docs/ml_lifecycle/08_Model_Evaluation_and_Tuning.md)**: **Fine-tuning audit results**: $-65.85\%$ Dossier Loss, $+0.157$ Failure F1, $+0.249$ Churn F1.
9. **[09. Model Deployment](docs/ml_lifecycle/09_Model_Deployment.md)**: Streamlit Cloud, standalone offline browser dashboard (`ARGOS_BROWSER_RESULTS.html`), Neo4j.
10. **[10. Model Monitoring & Maintenance](docs/ml_lifecycle/10_Model_Monitoring_and_Maintenance.md)**: Continuous telemetry logging, climate covariate drift, concept drift protocols.

---

## ⚡ Audited Business & Financial KPIs (Fine-Tuned on Tesla V100)

Calibrated across all 12 calendar months on the Tesla V100 GPU, the pipeline demonstrates strictly positive net profit and dramatic operational savings:

* **Annual Gross Revenue**: **₹2,60,71,851.84** (~₹2.61 Crore INR)
* **Annual Operating Expenditure (OPEX)**: **₹34,00,492.40** (~₹34.00 Lakh INR)
* **Annual Net Profit**: **₹2,26,71,359.43** (~₹2.27 Crore INR — 100% positive in every month)
* **Mean Cost-Efficiency Ratio**: **7.821x** (Revenue generates ~7.8x operating expenditure)
* **Proactive Breakdown Savings Gain**: **+24.23%** (Avoidance of emergency off-season breakdowns)
* **Full Audit Dossier**: See **[ARGOS.md](ARGOS.md)** and **[ARGOS_BROWSER_RESULTS.html](ARGOS_BROWSER_RESULTS.html)**.

---

## 🚀 Quickstart: Local Execution & Streamlit Cloud

### Option A: Deploying on Streamlit Cloud (One-Click)
1. Fork or push this repository to GitHub: `https://github.com/AnimeshBasak-14/SmartHVAC.git`
2. Open [share.streamlit.io](https://share.streamlit.io/) and create a **New app**.
3. Select repository: `AnimeshBasak-14/SmartHVAC`, Branch: `main`.
4. Main file path: `app.py`.
5. Click **Deploy!**

### Option B: Running Locally (Windows / Linux)
```bash
# 1. Clone repository
git clone https://github.com/AnimeshBasak-14/SmartHVAC.git
cd SmartHVAC

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run test suite
python hvac_gml_project/tests/test_pipeline.py

# 4. Launch Command Center
streamlit run app.py
```

### Option C: Training on NVIDIA Tesla V100 Lab Server
```bash
cd hvac_gml_project
python train_and_monitor.py
streamlit run dashboard_app.py --server.port 8501 --server.headless true
```
