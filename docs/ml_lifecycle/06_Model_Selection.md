# Step 06: Model Selection

> **ML Lifecycle Phase**: 06 / 10  
> **System**: SmartHVAC & Seasonal Appliance Heterogeneous GML  
> **Chosen architecture**: Multi-Task Heterogeneous Graph Neural Network (`HeteroConv` + `SAGEConv`)  
> **Alternative Baselines Compared**: Tabular Gradient Boosted Trees (XGBoost/LightGBM), Homogeneous GNN (GCN/GAT)

---

## 1. Architectural Comparison & Rationale

Why did we choose a Heterogeneous Graph Neural Network (Hetero-GNN) for seasonal multi-task HVAC operations? The reasons are about structure (typed entities, relations, one shared model), not about accuracy; see [WHY_HETERO_GRAPHSAGE_OVER_XGBOOST.md](../WHY_HETERO_GRAPHSAGE_OVER_XGBOOST.md).

| Model Paradigm | Cross-Entity Relational Awareness | Typed Heterogeneous Nodes & Edges | Spatio-Temporal Propagation | Multi-Task Shared Representation | Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Tabular Models**<br>*(XGBoost, LightGBM, CatBoost)* | **None**<br>Assumes I.I.D. rows | **None**<br>Requires flat denormalization | **Weak**<br>Blind to topological hops | **None**<br>Requires 4 siloed models | **Not chosen**: Cannot model failure propagation or customer-device dependencies. |
| **Homogeneous GNN**<br>*(Vanilla GCN, GAT)* | **Partial**<br>Tolerates graphs | **Failed**<br>Forces identical feature spaces on all nodes | **Moderate**<br>Conflates relationship types | **Limited**<br>Prone to oversmoothing | **Not chosen**: Conflates resident attributes ($`R, F, M`$) with part costs and device power. |
| **Heterogeneous GNN**<br>*(HeteroConv + SAGEConv)* | **Native**<br>Multi-hop relational paths | **Native**<br>Dedicated weights per edge relation $`r \in \mathcal{R}`$ | **Strong**<br>Explicit cyclical features + message passing | **Native**<br>Shared heterogeneous graph embeddings | **SELECTED**: Naturally captures multi-relational physical and business topologies. |

---

### 1.1 Baseline benchmark results (Tesla V100, scored on the training data)

To compare the candidates, all baselines were trained and tested on the identical 12-month graph dataset using the benchmark suite [`hvac_gml_project/benchmark_baselines.py`](file:///c:/Users/basak/Desktop/CSC-505/Project/hvac_gml_project/benchmark_baselines.py) on an **NVIDIA Tesla V100S-PCIE-32GB GPU** (`10.13.1.7`):

| Model Architecture | Demand MSE $\downarrow$ | Power MSE $\downarrow$ | Equipment Failure F1 $\uparrow$ | Sleeper Churn F1 $\uparrow$ | Inventory MSE $\downarrow$ | Total loss $\downarrow$ | Status / Empirical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tabular XGBoost** | **1.59** | **0.000** | **0.851** | **0.822** | **3.15** | **5.07** | **Not chosen (Siloed)**: Requires 5 disjoint models with zero topological awareness. Cannot model multi-hop failure propagation or generate shared graph embeddings. |
| **Vanilla Homogeneous GCN**<br>*(Kipf & Welling, ICLR 2017)* | 355.69 | 0.177 | 0.670 | **0.000** | 7,881.63 | 8,238.83 | **Not chosen (Catastrophic)**: Complete collapse on minority churn ($F_1 = 0$) and extreme divergence on inventory ($MSE = 7,881.6$) due to feature conflation. |
| **Vanilla Homogeneous GAT**<br>*(Veličković et al., ICLR 2018)* | 249.30 | 0.186 | 0.656 | **0.000** | 324.19 | 575.02 | **Not chosen (Type-Blind)**: Shared attention space washes out customer churn ($F_1 = 0$) and misinterprets heterogeneous edge types as uniform interactions. |
| **Multi-Task Hetero-GNN**<br>*(HeteroConv + SAGEConv, Ours)* | **33.13** | **0.065** | **0.697** | **0.646** | **12.61** | **46.46** | **Chosen**: one model for all 4 entity types with separate weights per relation. Not the best score in this benchmark. |

> [!WARNING]
> **Read this first.** Neither our model nor the baselines were scored on held-out data, XGBoost was given no neighbour features, and the labels are generated from node-level rules. So this table shows how well each model fits the data it was trained on, not how well it would generalise. See [why we chose the hetero-GraphSAGE model](../WHY_HETERO_GRAPHSAGE_OVER_XGBOOST.md). On these numbers XGBoost scores higher than our model on every metric.

> [!IMPORTANT]
> **What the results show**:
> 1. **Complete Minority Collapse in Homogeneous GNNs ($F_1 = 0.000$)**:
>    Both Vanilla GCN and Vanilla GAT completely failed to flag sleeper churn customers ($0$ flagged out of $278$ true ground-truth sleepers). Because resident nodes are directly connected to high-degree locations and multiple devices, homogeneous message passing acts as a low-pass filter, oversmoothing the delicate behavioral churn signal into the surrounding neighborhood centroid.
> 2. **Catastrophic Divergence in Vanilla GCN Inventory Forecasting ($MSE = 7,881.63$)**:
>    Homogeneous GCN forces part replacement unit costs ($₹500 - ₹8,000$) into the exact same projection space as ambient temperature ($25 - 45^\circ\text{C}$) and wattage. Unweighted symmetric Laplacian normalization ($\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$) destabilizes gradient propagation to isolated inventory part nodes.
> 3. **XGBoost in this benchmark**:
>    XGBoost fits the data best here, and that result is real but limited. It was trained and scored on the same rows (its power error of exactly 0 shows it memorised them), it only saw each node's own features, and our labels are generated from node-level rules, which suits a tabular model.
>    The reasons we still chose the graph model are structural: it is one model for all five outputs, it handles typed nodes and relations, and it can score new nodes and links. XGBoost would need five separate models and hand-built neighbour features.

---

## 2. Selected Architecture: `MultiTaskHVACHeteroGNN`

The model is structured around a shared inductive heterogeneous message-passing backbone connected to four specialized output task heads and a link prediction decoder:

```mermaid
flowchart TD
    subgraph Inputs["Graph Inputs x_dict and edge_index_dict"]
        In_Loc["Location Features (8)"]
        In_Dev["Device Features (8)"]
        In_Res["Resident Features (8)"]
        In_Part["Part Features (8)"]
    end

    subgraph Backbone["Heterogeneous GNN Backbone"]
        HConv1["Per-type Linear 8 to 128, then HeteroConv layer 1 (SAGEConv, 128 to 128) + LeakyReLU + residual + dropout 0.2"]
        HConv2["HeteroConv layer 2 (SAGEConv, 128 to 128) + LeakyReLU + residual + dropout 0.2"]
        HConv3["HeteroConv layer 3 (SAGEConv, 128 to 128) + LeakyReLU + residual + dropout 0.2"]
    end

    subgraph Heads["Specialized Multi-Task Heads"]
        H_Dem["Head 1: Demand Regressor<br>MLP(128, 64, 1) to y_demand"]
        H_Pwr["Head 2a: Power Regressor<br>MLP(128, 64, 1) to y_power"]
        H_Fail["Head 2b: Failure Classifier<br>MLP(128, 64, 1) to failure_logits"]
        H_Churn["Head 3: Sleeper Churn Classifier<br>MLP(128, 64, 1) to sleeper_logits"]
        H_Inv["Head 4: Inventory Forecaster<br>MLP(128, 64, 1) to y_inventory"]
        H_Link["Link Predictor<br>DotProduct(h_res, h_dev) to Cross-Sell Score"]
    end

    Inputs --> HConv1 --> HConv2 --> HConv3
    HConv3 --> H_Dem
    HConv3 --> H_Pwr
    HConv3 --> H_Fail
    HConv3 --> H_Churn
    HConv3 --> H_Inv
    HConv3 --> H_Link
```

### Mathematical Formulation of Message Passing
For each node type $`v \in \mathcal{V}`$ and each incident relation $`r = (u, \text{rel}, v) \in \mathcal{E}`$, layer $`l`$ applies the `SAGEConv` aggregation:

```math
\mathbf{h}_{v, r}^{(l)} = \mathbf{W}_1^{(r)} \mathbf{h}_v^{(l-1)} + \mathbf{W}_2^{(r)} \cdot \text{MEAN}_{u \in \mathcal{N}_r(v)} \left( \mathbf{h}_u^{(l-1)} \right)
```

Aggregations across all incoming relations are summed:

```math
\mathbf{h}_v^{(l)} = \text{LeakyReLU} \left( \sum_{r \in \mathcal{R}_{\text{in}}(v)} \mathbf{h}_{v, r}^{(l)} \right)
```

The code also adds a residual connection (the previous layer's embedding) and applies dropout 0.2 after each layer.

This means that:
* A `device` embedding receives information from its owning `resident` as well as its required spare `parts`.
* A `location` embedding aggregates the collective stress of all its resident customers.
* An `inventory part` embedding anticipates consumption based on the aggregate seasonal power strain of all connected devices.

---

## 3. Output Task Heads Architecture

1. **Demand Head (`head_demand`)**:

```math
\hat{y}_{\text{demand}} = \mathbf{w}_{\text{dem}}^T \mathbf{h}_{\text{loc}} + b_{\text{dem}}
```

2. **Power & Failure Heads (`head_power`, `head_failure`)**:

```math
\hat{y}_{\text{power}} = \mathbf{w}_{\text{pwr}}^T \mathbf{h}_{\text{dev}} + b_{\text{pwr}}
```

```math
\hat{p}_{\text{failure}} = \sigma(\mathbf{w}_{\text{fail}}^T \mathbf{h}_{\text{dev}} + b_{\text{fail}})
```

3. **Sleeper Churn Head (`head_sleeper`)**:

```math
\hat{p}_{\text{sleeper}} = \sigma(\mathbf{w}_{\text{sleep}}^T \mathbf{h}_{\text{res}} + b_{\text{sleep}})
```

4. **Inventory Replenishment Head (`head_inventory`)**:

```math
\hat{y}_{\text{inventory}} = \mathbf{w}_{\text{inv}}^T \mathbf{h}_{\text{part}} + b_{\text{inv}}
```

5. **Cross-Sell Link Predictor (`predict_cross_sell_links`)**:

```math
\hat{s}_{i, j} = \sigma \left( \langle \mathbf{h}_{\text{res}, i}, \mathbf{h}_{\text{dev}, j} \rangle \right)
```

