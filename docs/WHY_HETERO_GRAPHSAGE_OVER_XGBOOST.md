# Why we used a heterogeneous GraphSAGE model instead of XGBoost

This note explains our choice, and is honest about what our own benchmark does and does not prove.
Read section 3 before the viva, because an evaluator may ask about it.

## 1. Short answer

We chose a heterogeneous GraphSAGE network (`HeteroConv` with `SAGEConv`) because our problem is made of
connected entities (locations, residents, devices, spare parts) and we wanted **one model that handles all
the tasks and uses those connections**. We did not choose it because it scored better than XGBoost. On
our benchmark XGBoost scores better, and that benchmark has limits that we explain below.

## 2. Why we picked the graph model (design reasons)

1. **The business is relational.** A resident owns devices, devices need parts, residents live in
   locations. Demand for a part depends on the devices that use it. Churn may depend on the neighbourhood.
   A graph model can read these links directly. XGBoost needs a flat table, so someone has to join the
   tables and decide by hand which neighbour features to add.
2. **One model for all tasks.** We predict location demand, device power, device failure, customer
   inactivity and part demand. The graph model has one shared backbone and five small output heads, trained
   together. With XGBoost we would train five separate models that share nothing.
3. **Different node types and different relations.** `HeteroConv` gives each relation its own weights
   (`located_in`, `owns`, `requires_part` and their reverses). A plain GCN or GAT treats all nodes and
   all edges the same way.
4. **GraphSAGE is inductive.** SAGEConv learns a rule for combining a node with the average of its
   neighbours. A new customer, device or part can be scored without retraining from scratch.
5. **Cross-sell as link prediction.** Suggesting a product to an inactive customer is naturally a
   "which resident-device links are missing" question, and the model has a decoder for it
   (`predict_cross_sell_links`).
6. **Learning goal.** The project is about applying graph machine learning, so we wanted to build and
   fine-tune a real graph model, not only a tabular one.

## 3. What our benchmark shows, and what it does not

The comparison is in `hvac_gml_project/artifacts/baseline_model_comparison.csv` (figure 13).

| Model | Demand MSE | Failure F1 | Churn F1 | Total loss |
| :--- | :---: | :---: | :---: | :---: |
| XGBoost | 1.59 | 0.85 | 0.82 | 5.07 |
| Plain GCN | 355.69 | 0.67 | 0.00 | 8,238.83 |
| Plain GAT | 249.30 | 0.66 | 0.00 | 575.02 |
| Hetero-GraphSAGE (ours) | 33.13 | 0.70 | 0.65 | 46.46 |

Four things limit what these numbers mean:

1. **This benchmark has no held-out data.** In `benchmark_baselines.py` every model is trained on all 12 monthly
   graphs and then scored on those same graphs. The same is true of our fine-tuning results (loss 137.04
   to 46.80, F1 scores). These are training-set scores. They show how well each model fits the data it
   saw, not how well it predicts new data. XGBoost with 100 trees of depth 5 on a few hundred rows can
   memorise them, which is why its power error is exactly 0.000.
2. **XGBoost only saw each node's own 8 features.** It got no information about neighbours. A fairer
   tabular baseline would add neighbour averages (for example the mean failure risk of a part's devices).
   We did not run that baseline.
3. **The labels come from rules.** `data_pipeline.py` generates the demand, failure, churn and inventory
   labels from each node's own features plus random noise. When the answer is a function of the node's own
   features, neighbours add little, and a tabular model has the advantage. A graph model can only show its
   strength if the labels really depend on neighbours.
4. **The link-prediction numbers are not comparable.** The XGBoost link model is fitted and scored on the
   same resident-device pairs (AUC 0.965, in-sample). Our model's AUC is 0.5, which is chance level. So we
   have no evidence that our cross-sell predictor works, and no fair evidence that XGBoost's does.

### 3.1 What the held-out run shows

`heldout_evaluation.py` (see [HELDOUT_EVALUATION.md](HELDOUT_EVALUATION.md) and figure 14) scored the GNN and two XGBoost variants on unseen months and unseen nodes. The result is mixed:

* The GNN is better on demand and inventory when testing on unseen months (inventory MSE 23 vs 54 to 352), and in the strict forecast XGBoost's demand error is even worse than predicting the mean.
* XGBoost is far better on device power, and slightly better on failure F1 and (in the forecast) churn F1.
* Neither failure classifier is clearly above a flag-everything baseline.
* Adding neighbour features helped XGBoost on inventory but not in the node split.

So the in-sample claim that XGBoost is simply better does not hold up, and neither does the opposite claim. These are single-run results on rule-generated labels.

The older explanation that the GCN and GAT fail because of over-smoothing is plausible, but we never ran an
experiment to confirm it. Treat it as a hypothesis.

## 4. What we can say in the report and viva

- "We chose a heterogeneous GraphSAGE model because the data is a typed graph and we wanted one shared
  model for five tasks, inductive scoring of new nodes, and a link-prediction decoder."
- "On our in-sample benchmark XGBoost fits the data better. On held-out months and nodes the picture is mixed: the GNN is
  better on seasonal demand and inventory, XGBoost is better on device power, and neither failure classifier is clearly
  better than flagging everything. The labels are generated from rules, so this does not show which approach would work
  better on real data."
- "Fine-tuning improved our model: total loss fell from 137.04 to 46.80 and the failure and churn F1
  scores rose. These are training-set results."

Do not say that the hetero-GNN is the best model, or that XGBoost "fails" or gives a misleading result.

## 5. What would settle the question

Items 1 and 2 below have been done with `hvac_gml_project/heldout_evaluation.py` (see [HELDOUT_EVALUATION.md](HELDOUT_EVALUATION.md)). Items 3 and 4 still need new data or code, and the held-out run should be repeated with several random seeds.

1. Hold out data: train on 9 months and test on the last 3, or hold out a random 20% of nodes.
2. Add a stronger tabular baseline: XGBoost with neighbour-aggregate features.
3. Use labels that depend on neighbours, or real failure and churn data (for example the AI4I and Online
   Retail II tables directly).
4. Evaluate link prediction on edges that were removed before training.

## 6. If we had to pick for a real deployment

XGBoost models would be simpler and faster to train, and they were clearly better on device power. The GNN was better on
seasonal demand and inventory when tested on unseen months. A reasonable choice is XGBoost for power and failure, and either
model for demand and inventory, with the GNN kept if we need the shared embeddings and link prediction. Repeat the held-out
run with several seeds before deciding.
