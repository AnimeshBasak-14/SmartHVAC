# 05. Future work: where a graph neural network may help

This project uses Ridge and XGBoost. A graph neural network (GNN) is **not** part of it. This page says where our results show a gap,
and which of those gaps a graph model might close. It is a hypothesis, not a result.

## What we already know about GNNs on this problem

An earlier version of this project built a heterogeneous graph (locations, residents, devices and spare parts) and trained a GraphSAGE
model. We tested it on held-out months and nodes with three random seeds (summaries are in
`hvac_ml/outputs/results/earlier_gnn_experiment/`):

| Test (held-out months 3, 6, 9, 12) | GraphSAGE GNN | Ridge with pairwise products |
| :--- | :---: | :---: |
| Demand error (MSE, lower is better) | 29.6 ± 2.9 | 28.5 ± 5.2 |
| Inventory error (MSE) | 24.5 ± 1.7 | 19.2 ± 0.8 |
| Forecast of months 10-12, demand error | 73.5 ± 21.7 | 36.2 ± 6.0 |

The GNN never clearly beat the best simple model. The reason is in the data, not necessarily in the GNN: in that synthetic graph every label
was computed from the node's own features, so neighbours carried no extra information and there was nothing for a graph model to use.
So we have **no evidence yet** that a GNN helps, and we do not claim it does.

## Where our current models fall short

| Gap in our results | Why a GNN might help | Why it might not |
| :--- | :--- | :--- |
| **Pillar 3, churn.** Every model reaches about 0.78 ROC-AUC and they all tie, including plain logistic regression. More model complexity buys nothing on a customer's own purchase history. | Customers who share products or behave alike might carry signal that a single customer's row does not. A customer-product graph is a natural way to add it. | The missing information may simply not exist in purchase data. A strong tabular model with "average behaviour of similar customers" as extra columns may do just as well. |
| **Cross-selling a counter-seasonal appliance** (the secondary goal of Pillar 3). We did not attempt it. | It is link prediction: which product would this customer buy next? Graph models learn this directly from the customer-product graph. | Classic recommenders (item co-occurrence, matrix factorisation) are strong and much simpler. |
| **Demand across several buildings** (Pillar 1). Our data has one building. | Sites that share a power grid or a weather system influence each other, and a graph can represent that. | We have no multi-site data to test it. |
| **Heatwave** (Pillar 1). XGBoost and Random Forest fail: their error is 0.34 to 0.37 kW against 0.08 kW for the linear model, and R² is below zero. | none | A GNN would not fix this. Trees cannot predict beyond the range of temperatures they were trained on. The fix is a model that extrapolates (the linear model on the temperature gap), which is what the product already does. |
| **Pillar 2, failure.** Not a gap: XGBoost and Random Forest reach 0.89 to 0.91 PR-AUC. | none | Machines in this dataset are independent rows with no links between them. |

So the honest list of places where a GNN is worth testing is short: **churn and cross-sell on a customer-product graph, and demand across
linked sites.** Heatwave failures need a different remedy.

## How to test the idea properly

1. Build a customer-product graph from Online Retail II (customers and products as nodes, purchases as links).
2. Tasks: churn as node classification, and "which product next" as link prediction on purchases held out in time.
3. Compare the GNN with **XGBoost given neighbour-aggregated features** (for example the average recency of a customer's co-buyers), because that is
   the strongest simple alternative. Also compare with ridge/logistic regression and an item co-occurrence recommender.
4. Use the same split as this project (unseen customers, later period), at least five seeds, and the same tie rule: the GNN must beat the
   best simple model by more than one standard deviation to be preferred.
5. Only then, say that graph structure helps.

## Lessons from the earlier attempt

* Check that the labels really depend on neighbours. Otherwise a graph model has nothing to learn.
* Do not put the label (or a copy of it) in the inputs. The earlier power label was exactly one of the input columns.
* Always score on held-out data and always include a strong simple baseline.
* Report where the more complicated model does **not** win.
