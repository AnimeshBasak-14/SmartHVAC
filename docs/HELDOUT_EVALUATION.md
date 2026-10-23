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

The script was written and its split and scoring logic was tested on stand-in data on a laptop, but the GNN and
XGBoost training parts have **not been run yet** (no torch on the laptop). If it fails, send the error and we fix it.

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
