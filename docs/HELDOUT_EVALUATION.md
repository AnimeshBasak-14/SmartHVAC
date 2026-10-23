# Held-out evaluation: how to run it and how to report it

Our earlier scores (the -65.85% loss, the F1 scores, the XGBoost comparison) were all measured on the data
the models were trained on. `hvac_gml_project/heldout_evaluation.py` fixes that by scoring every model on data it
never saw.

## What it does

It trains three models from scratch and scores them on held-out rows:

| Model | What it sees |
| :--- | :--- |
| Hetero-GNN | the whole graph (trained with the fine-tuning recipe, 200 epochs, no saved checkpoint is reused) |
| XGBoost (own features) | only each node's own 8 features, as in `benchmark_baselines.py` |
| XGBoost (+ neighbour features) | own features plus the mean features of each node's neighbours |

Two splits are run:

* **months**: train on 8 months and test on 4 (default 3, 6, 9, 12). This asks whether a model trained on some
  seasons predicts the others.
* **nodes**: train on 80% of the locations, devices, residents and parts and test on the other 20%. The test
  nodes stay in the graph, as in the usual node-classification setup. With only 10 locations and 20 parts there
  are very few test rows (2 and 4 nodes), so treat the location and part scores as noisy.

For every model it also prints the **train** score next to the **test** score. A large gap means memorisation.
A "reference" row shows what a model that predicts the training mean (or flags everything) would score.

## How to run (lab server)

```bash
cd hvac_gml_project
python heldout_evaluation.py                                   # both splits
python heldout_evaluation.py --split months --test-months 10,11,12   # strict: forecast the last 3 months
```

Output: a table on screen and `hvac_gml_project/artifacts/heldout_evaluation.csv`. It takes a few minutes on the V100.

The script has been run on the V100 server (results below). The first run used the default splits; the strict forecast
split was run afterwards with `--test-months 10,11,12`.

### Prompt for Antigravity

```
In /mnt/hdd/animesh/csc505 (the repo), activate the existing Python environment, then run:
  cd hvac_gml_project
  python heldout_evaluation.py 2>&1 | tee artifacts/heldout_run.log
If it raises an error, show me the full traceback and propose the smallest fix; do not change any model,
loss or split logic without asking. When it finishes, print artifacts/heldout_evaluation.csv and tell me,
for each split, (a) the test scores of the three models, (b) the gap between train and test scores, and
(c) whether the GNN or an XGBoost variant has the lower test error. Do not invent any numbers.
```

## How to read the result

* If the **test** scores are close to the **train** scores, the earlier numbers were not just memorisation.
* If XGBoost with neighbour features is no better than XGBoost on own features, neighbour information adds
  nothing for these labels (they are generated from each node's own features).
* If the GNN's test scores are much worse than its train scores, say so in the report.
* Compare each model with the reference row. A model that does not beat "predict the mean" has learned nothing.

## What to write in the report

Quote the **test** numbers from `heldout_evaluation.csv`, state the split used, and keep the caveat that the
labels are generated from rules plus noise. Replace the old training-set numbers in the report and slides, or
label them clearly as training-set scores.

## Results (run on the Tesla V100S, seed 42, about 51 s per split)

Test-set scores only. The first two splits come from `artifacts/heldout_evaluation_months3-6-9-12_nodes.csv` and the
forecast split from `artifacts/heldout_evaluation_forecast.csv`. Figure 14 plots them. "Reference" predicts the training
mean (for MSE) or flags every row as positive (for F1).

**Held-out months 3, 6, 9, 12 (trained on the other 8 months)**

| Model | Demand MSE | Power MSE | Inventory MSE | Failure F1 | Churn F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Reference | 586.1 | 0.438 | 797.5 | 0.667 | 0.226 |
| Hetero-GNN | **31.9** | 0.048 | **23.2** | 0.708 | 0.520 |
| XGBoost (own features) | 299.1 | **0.0002** | 352.4 | **0.741** | 0.476 |
| XGBoost (+ neighbour features) | 256.0 | **0.0002** | 53.5 | 0.738 | **0.543** |

**Held-out 20% of nodes (2 locations, 8 devices, 20 residents, 4 parts)**

| Model | Demand MSE | Power MSE | Inventory MSE | Failure F1 | Churn F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Reference | 597.2 | 0.350 | 717.9 | 0.638 | 0.229 |
| Hetero-GNN | 73.8 | 0.029 | **16.0** | 0.647 | **0.517** |
| XGBoost (own features) | 43.8 | **0.0001** | 19.3 | 0.653 | 0.475 |
| XGBoost (+ neighbour features) | **40.8** | **0.0001** | 18.4 | **0.654** | 0.417 |

**Forecast: trained on months 1-9, tested on months 10-12**

| Model | Demand MSE | Power MSE | Inventory MSE | Failure F1 | Churn F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Reference | 156.1 | 0.548 | 814.9 | 0.491 | 0.338 |
| Hetero-GNN | **90.5** | 0.092 | **38.7** | **0.497** | 0.672 |
| XGBoost (own features) | 219.5 | **0.0002** | 428.5 | 0.466 | **0.711** |
| XGBoost (+ neighbour features) | 209.2 | **0.0002** | 125.0 | 0.475 | 0.702 |

## What the results show

1. **The training-set numbers flattered XGBoost most.** On the months split its demand MSE goes from 0.72 (train) to 299 (test)
   and its inventory MSE from 1.6 to 352. The GNN goes from 15.1 to 31.9 and from 8.5 to 23.2.
2. **There is no overall winner.** The GNN is better on demand and inventory in both month-based splits (for example
   inventory 23 vs 54-352 on months 3,6,9,12). XGBoost is far better on power (0.0002 vs 0.03-0.09) and slightly better on
   failure F1 in two of three splits and on churn F1 in the forecast. In the node split XGBoost is better on demand
   (41-44 vs 74) and the GNN on inventory and churn F1.
3. **The failure classifiers show almost no skill.** Their F1 is only 0.01 to 0.07 above "flag everything" (0.667 vs 0.708 to 0.741
   on months; 0.638 vs 0.647 to 0.654 on nodes). In the forecast split both XGBoost variants (0.466 and 0.475) are below the
   flag-everything reference (0.491), and the GNN is level with it (0.497).
4. **XGBoost can be worse than guessing the mean.** In the forecast split its demand MSE (219.5 and 209.2) is higher than the
   training-mean reference (156.1). The GNN (90.5) beats the reference. A possible reason is that trees cannot interpolate
   between the few seasonal values seen in training. We have not tested that.
5. **Neighbour features helped XGBoost on inventory** when testing on unseen months (352 to 54, and 429 to 125), which is the
   best evidence so far that neighbour information can matter. The same features did not help in the node split and lowered
   its churn F1 (0.475 to 0.417).

## Caveats

* **One run, one seed.** Differences such as 0.741 vs 0.708 failure F1 are small and may be noise. Repeat with several seeds
  before claiming a winner.
* **The rows are not independent.** The 160 device rows are 40 devices repeated over 4 months, and the 400 resident rows are
  100 residents repeated, so the effective sample size is much smaller.
* **Tiny node test sets.** The node split has 2 test locations and 4 test parts, so its demand and inventory scores are noisy.
* **The labels are rule-generated.** The labels come from rules plus noise, so these scores say how well each model learns
  the rules across seasons and nodes, not how well it would predict real failures.
* **The GNN here was trained from scratch** with the fine-tuning recipe (200 epochs, no saved checkpoint), so its numbers
  differ from the earlier training-set tables.
