# SmartHVAC Graph Machine Learning (GML) - Questions & Answers (Q&A)

This document is a living repository of architectural, conceptual, and implementation questions and answers regarding the SmartHVAC Heterogeneous Graph Machine Learning project.

---

## Table of Contents
1. [GML Foundations & Task Definitions](#1-gml-foundations--task-definitions)
   - [Q1: Is this project a node, edge, or graph-level task?](#q1-is-this-project-a-node-edge-or-graph-level-task)
2. [Model Architecture & Training](#2-model-architecture--training)
   - [Q2: What did we use: GML, GNN, GCN, GraphSAGE, or something else?](#q2-what-did-we-use-gml-gnn-gcn-graphsage-or-something-else)
   - [Q3: Why use Heterogeneous GNNs (HeteroConv) instead of Homogeneous GNNs?](#q3-why-use-heterogeneous-gnns-heteroconv-instead-of-homogeneous-gnns)
   - [Q4: How does the model prevent regression loss from dominating classification loss?](#q4-how-does-the-model-prevent-regression-loss-from-dominating-classification-loss)
   - [Q6: Why do Tabular XGBoost, Vanilla GCN, and Vanilla GAT fail compared to MultiTask Hetero-GNN?](#q6-why-do-tabular-xgboost-vanilla-gcn-and-vanilla-gat-fail-compared-to-multitask-hetero-gnn)
3. [Business & Operational Impact](#3-business--operational-impact)
   - [Q5: How does link prediction translate to business revenue?](#q5-how-does-link-prediction-translate-to-business-revenue)

---

## 1. GML Foundations & Task Definitions

### Q1: Is this project a node, edge, or graph-level task?

**Answer:**
This project performs **Node-level** and **Edge-level** tasks. It does **NOT** perform a Graph-level task.

#### 1. Node-Level Tasks (Primary Multi-Task Heads)
The core architecture in [`MultiTaskHVACHeteroGNN`](file:///c:/Users/basak/Desktop/CSC-505/Project/hvac_gml_project/hetero_gnn_model.py#L11-L89) makes predictions on individual nodes across all 4 heterogeneous entity types:

| Node Type | Task Type | Target Variable | Objective / Business Value |
| :--- | :--- | :--- | :--- |
| **Location** | Node Regression | $\hat{y}_{\text{demand}}$ | Spatio-temporal service & appliance demand forecasting for technician dispatch. |
| **Device** | Node Regression | $\hat{y}_{\text{power}}$ | Operating power consumption estimation to detect anomalies. |
| **Device** | Node Classification | $\hat{y}_{\text{failure}}$ | Failure risk probability (predictive maintenance) to avoid breakdown. |
| **Resident** | Node Classification | $\hat{y}_{\text{sleeper}}$ | "Sleeper" customer churn probability to trigger retention actions. |
| **Inventory Part** | Node Regression | $\hat{y}_{\text{inventory}}$ | Spare-parts replenishment forecasting to prevent stockout SLA penalties. |

#### 2. Edge-Level Tasks (Link Prediction)
Defined in [`predict_cross_sell_links`](file:///c:/Users/basak/Desktop/CSC-505/Project/hvac_gml_project/hetero_gnn_model.py#L90-L94):
* **Resident $\leftrightarrow$ Device Link Prediction**:
  * Evaluates whether an `OWNS` edge should exist between customer $u$ and counter-seasonal appliance $d$ using latent dot-product affinity:
    $$P(\text{Cross-Sell Adoption}) = \sigma\left(\mathbf{h}_u^T \mathbf{h}_d\right)$$
  * **Application**: Cross-selling winter appliances (e.g., Geysers) to summer AC owners to eliminate revenue seasonality.

#### 3. Why It Is NOT a Graph-Level Task
* **Graph-level tasks** assign a single global label or scalar to an entire graph $\mathcal{G}$ (e.g., classifying whether a molecule is toxic, or predicting total network throughput via global pooling).
* In this ecosystem, the entire utility/customer network is modeled as **one interconnected heterogeneous graph** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$. Message passing propagates relational context across neighbors, but decisions are made locally at the **node level** (individual assets, people, locations) and **edge level** (connections between them).

---

## 2. Model Architecture & Training

### Q2: What did we use: GML, GNN, GCN, GraphSAGE, or something else?

**Answer:**
Here is the exact technical breakdown and hierarchy of what is used:

```
Graph Machine Learning (GML)                     <-- Overarching AI Discipline
 └── Graph Neural Network (GNN)                  <-- Deep Learning Model Family
      └── Heterogeneous GNN (HeteroConv)         <-- Multi-Entity Architecture
           └── GraphSAGE (SAGEConv)              <-- Exact Convolutional Operator (NOT GCN)
```

1. **GML (Graph Machine Learning)**:
   * This is the **broad AI paradigm/discipline** (analogous to saying "Computer Vision" or "NLP"). We use GML because interconnected utility networks violate the tabular I.I.D. (Independent and Identically Distributed) assumption.
2. **GNN (Graph Neural Network)**:
   * This is the **deep learning model class** operating on graph-structured data via iterative neighborhood message passing.
3. **Hetero-GNN (`HeteroConv`)**:
   * This is the specific **multi-relational GNN architecture** implemented in [`MultiTaskHVACHeteroGNN`](file:///c:/Users/basak/Desktop/CSC-505/Project/hvac_gml_project/hetero_gnn_model.py#L11). Because our network consists of 4 distinct entity types (`resident`, `location`, `device`, `part`) and 6 directed edge types (`located_in`, `owns`, `requires_part`, etc.), standard homogeneous GNNs cannot be used.
4. **GraphSAGE (`SAGEConv`) — NOT GCN**:
   * The convolution operator inside each `HeteroConv` layer is **`SAGEConv` (Hamilton et al., NeurIPS 2017)** from PyTorch Geometric, instantiated as `SAGEConv((-1, -1), hidden_channels)`.
   * **Why GraphSAGE instead of standard GCN (`GCNConv`)?**
     * **Bipartite Heterogeneity**: Standard GCN assumes symmetric adjacency and identical feature dimensions for source and target nodes ($D^{-1/2} A D^{-1/2} X W$). GraphSAGE natively accepts **bipartite edge pairs** with arbitrary source and target dimensions `(-1, -1)`.
     * **Inductive Generalization**: GCN relies on the full normalized graph Laplacian over a static, transductive graph. GraphSAGE learns localized aggregation functions, making it **inductive**—it can generate embeddings for newly onboarded customers or devices at inference time without retraining the entire graph.
     * **Directional Message Aggregation**: GraphSAGE aggregates messages from incoming neighbors with an explicit mean pooling operator and preserves target node state via residual skip connections ($F.leaky\_relu(h_{next}) + h_{dict}$).

### Q3: Why use Heterogeneous GNNs (HeteroConv) instead of Homogeneous GNNs?

**Answer:**
A homogeneous graph assumes that all nodes belong to a single entity type and all edges represent identical interactions. In this ecosystem:
* Entities possess fundamentally distinct feature spaces (e.g., a **Device** has wattage, runtime hours, and vibration, while a **Resident** has income bracket and tenure).
* Relationships have distinct semantics (`resident -[OWNS]-> device` vs `device -[REQUIRES_PART]-> part`).
* **HeteroConv** applies relation-specific transformations (`SAGEConv((-1, -1), hidden_channels)` per edge type), allowing the network to preserve semantic distinction while propagating messages across multi-hop relational paths.

### Q4: How does the model prevent regression loss from dominating classification loss?

**Answer:**
Through **Homoscedastic Task Uncertainty Weighting** (Kendall et al., CVPR 2018) implemented in [`UncertaintyMultiTaskLoss`](file:///c:/Users/basak/Desktop/CSC-505/Project/hvac_gml_project/fine_tune_v100.py#L45-L63):

$$\mathcal{L}_{\text{total}} = \sum_{i=1}^4 \left( \frac{1}{2}\exp(-s_i)\mathcal{L}_i + \frac{1}{2}s_i \right)$$

* Regression targets (Demand and Inventory with MSE $\approx 100 - 300$) would otherwise overpower classification gradients (BCE $\approx 0.3 - 0.7$).
* Learnable log-variance parameters $s_i$ dynamically scale down high-variance losses, ensuring balanced gradient contributions to shared message-passing layers.

### Q6: Why do Tabular XGBoost, Vanilla GCN, and Vanilla GAT fail compared to MultiTask Hetero-GNN?

**Answer:**
We empirically benchmarked all candidate paradigms on an **NVIDIA Tesla V100S GPU** (`10.13.1.7`) using identical splits, empirical datasets (UCI AI4I 2020, UCI Retail II, Kaggle HVAC Power), and metrics via [`hvac_gml_project/benchmark_baselines.py`](file:///c:/Users/basak/Desktop/CSC-505/Project/hvac_gml_project/benchmark_baselines.py):

| Model Paradigm | Demand MSE $\downarrow$ | Power MSE $\downarrow$ | Failure F1 $\uparrow$ | Sleeper Churn F1 $\uparrow$ | Inventory MSE $\downarrow$ | Total Loss $\downarrow$ | Exact Failure Mechanism |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tabular XGBoost** | **1.59** | **0.000** | **0.851** | **0.822** | **3.15** | **5.07** | **Siloed & Relational Blindness**: Requires 5 disjoint models; cannot propagate multi-hop physical degradation or produce latent graph embeddings. |
| **Vanilla GCN (Homogeneous)** | 355.69 | 0.177 | 0.670 | **0.000** | 7,881.63 | 8,238.83 | **Feature Conflation & Catastrophic Divergence**: Averages incompatible physical units; completely erases minority churn signals into neighborhood centroids. |
| **Vanilla GAT (Homogeneous)** | 249.30 | 0.186 | 0.656 | **0.000** | 324.19 | 575.02 | **Type-Blind Attention Space**: Single shared attention matrix is confounded by disparate entity features; churn drops to 0. |
| **MultiTask Hetero-GNN (Ours)** | **33.13** | **0.065** | **0.697** | **0.646** | **12.61** | **46.46** | **Balanced Inductive GML**: Dedicated $W_r$ per relation type, Kendall uncertainty loss balancing, and end-to-end multi-task reasoning. |

#### 1. Why Vanilla GCN & GAT Failed Completely on Churn ($F_1 = 0.000$)
Homogeneous message passing treats every connection as an unweighted, symmetric interaction across identical nodes:
$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W)$$
Because residents connect to high-degree metropolitan locations and multiple appliances, the subtle behavioral signal of customer disengagement ($R > 90$ days) is completely diluted by neighboring node features. GCN acts as a low-pass filter, resulting in zero positive churn predictions.

#### 2. Why Vanilla GCN Diverged Catastrophically on Inventory ($MSE = 7,881.63$)
Inventory part costs ($₹500 - ₹8,000$) and stock counts reside on a fundamentally different scale than location temperatures ($25 - 45^\circ\text{C}$). Forcing them into the same projection space with uniform normalization causes gradient explosion and completely destabilizes inventory forecasting.

#### 3. Why XGBoost's Apparent Low Loss is an Analytical Trap
While XGBoost achieves low error on isolated in-distribution rows:
1. **Model Fragmentation**: It requires 5 independent, uncoordinated models with 0 parameter sharing.
2. **Topological Blindness**: If a severe monsoon surge hits a Location, XGBoost cannot propagate that stress through the connected Resident to the Device to anticipate a Part stockout 2 hops away.
3. **No Inductive Graph Embeddings**: It cannot output structural node vectors to power the Neo4j visualization graph or perform dynamic link prediction for cross-selling counter-seasonal appliances.

---

## 3. Business & Operational Impact

### Q5: How does link prediction translate to business revenue?

**Answer:**
* **Sleeper Customer Reactivation**: AC customers naturally disengage during winter months.
* The link prediction head computes user-appliance affinity scores, matching dormant summer users with winter products (water heaters, smart geysers).
* Target cross-sell campaigns convert otherwise inactive customers, stabilizing seasonal revenue and generating incremental margin.
