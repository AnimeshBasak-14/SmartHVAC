# Figures for the report and slides

All figures are made by one script:

```bash
python visualization/generate_all_visualizations.py     # needs numpy, pandas, matplotlib
```

The charts read our saved results (`hvac_gml_project/artifacts/`), so they match what was actually run:
`fine_tune_telemetry.json`, `fine_tuned_financial_metrics.csv`, `baseline_model_comparison.csv`, and the
season table in `config.py`. The diagrams (02 to 05) describe the code in `hvac_gml_project/`.
If you retrain and the artifacts change, run the script again.

## What each figure shows

| File | Type | What it shows | Where it fits |
| :--- | :--- | :--- | :--- |
| `01_ml_lifecycle_10_stages.png` | Diagram | The 10 ML lifecycle stages (copied from `docs/ml_lifecycle/`) | Intro |
| `02_heterogeneous_graph_schema.png` | Diagram | The 4 node types with their 8 features each, and the 3 relations | Method |
| `03_hetero_gnn_model_architecture.png` | Diagram | Input layer, 3 HeteroConv layers (SAGEConv), 5 output heads | Method |
| `04_end_to_end_system_pipeline_flowchart.png` | Diagram | Data, graph building, training, fine-tuning, evaluation, dashboard | Overview |
| `05_training_divergence_handling.png` | Diagram | What the pre-training and fine-tuning scripts do when the loss goes bad | Training |
| `06_pretrain_vs_finetune_loss_trajectory.png` | Chart | Total loss, failure F1 and churn F1 at the logged fine-tuning epochs | Results |
| `07_pillar_performance_comparison_before_after.png` | Chart | Regression MSE and F1 before and after fine-tuning | Results |
| `08_12month_financial_ledger.png` | Chart | Simulated monthly revenue, cost and revenue/cost ratio | Business results |
| `09_seasonal_cooling_heating_regimes.png` | Chart | Metro and village demand by month, with the season multipliers | Data and EDA |
| `10_failure_and_churn_classification_detail.png` | Chart | Classification loss and how many devices and customers are flagged | Results |
| `11_kendall_uncertainty_task_weights.png` | Chart | Final learned log-variances and the task weights they imply | Fine-tuning |
| `12_economic_proactive_savings_breakdown.png` | Chart | Simulated saving of planned maintenance over reactive repair, by month | Business results |
| `13_baseline_comparison.png` | Chart | XGBoost, GCN, GAT and our hetero-GNN on the same data | Results |

## Things to say honestly when presenting

- **All scores are in-sample.** No model in `benchmark_baselines.py` or the fine-tuning script was scored on held-out
  data, so figures 06, 07, 10, 11 and 13 show how well models fit the data they trained on. See
  [docs/WHY_HETERO_GRAPHSAGE_OVER_XGBOOST.md](../docs/WHY_HETERO_GRAPHSAGE_OVER_XGBOOST.md).
- **Figure 13:** XGBoost fits better than our hetero-GNN on most metrics (total loss 5.1 vs 46.5, failure F1 0.85 vs 0.70,
  churn F1 0.82 vs 0.65). Our model does beat the plain GCN and GAT. Do not describe the hetero-GNN as the best
  model.
- **Figure 06:** only the fine-tuning history was saved. The log of the first 120-epoch run was overwritten, so we
  show its final loss (137.04) as a single line.
- **Figures 08 and 12:** revenue, cost and savings are simulated from prices and costs we assumed in
  `business_financial_engine.py`. They are not real company accounts.
- **Figure 10:** failure loss barely changed after fine-tuning (0.718 to 0.680). The F1 gain came mostly from
  flagging more devices (154 to 432).
- **Link prediction:** the AUC of our cross-sell link predictor is 0.5 in `baseline_model_comparison.csv`, which is
  chance level. The XGBoost link AUC (0.965) is in-sample too. We do not show a ROC curve.
- **Labels:** the failure, demand and churn labels are generated from rules in `data_pipeline.py` plus noise, so the
  scores say how well the model learns those rules, not how well it predicts real failures.
