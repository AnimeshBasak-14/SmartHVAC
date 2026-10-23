# HVAC Service Planner

CSC-505 group project. **Machine learning for an HVAC service provider** (for example a franchise such as Aire Serv).

We built a small decision-support dashboard with three tools. Each tool is a model plus a ranked list a manager can act on:

| Tool | Question it answers | Task |
| :--- | :--- | :--- |
| **Demand Planner** | How much power will the HVAC draw tomorrow, and when is the peak? | Forecast hourly demand (kW) from the weather |
| **Failure Radar** | Which machines should we inspect first? | Predict machine failure from sensor readings |
| **Retention List** | Which active customers are about to go quiet? | Predict who buys nothing in the next 90 days |

We compare **Ridge regression** and **XGBoost** with a statistical baseline (linear / logistic regression), a classic machine-learning baseline
(Random Forest) and a rule of thumb a company could use today. Every score is measured on data the models never saw, averaged over five seeds.

## What we found

| Pillar | Best result on unseen data | What a company does today (rule of thumb) |
| :--- | :--- | :--- |
| 1. Demand | XGBoost: average error 0.07 to 0.08 kW. In a heatwave (above 40 °C) the linear model is best: 0.08 kW, against 0.34 kW for XGBoost | Average load for that hour: 0.9 to 1.6 kW error |
| 2. Failure | Random Forest and XGBoost tie (PR-AUC 0.91 and 0.89). Inspecting the top 10% of machines finds **94%** of failures | "Most worn tool first": PR-AUC 0.12, finds 37% |
| 3. Churn | All models tie at ROC-AUC 0.77 to 0.78. Calling the top 10% reaches about **20%** of the customers who lapse, twice as many as random | "Longest since last purchase": ROC-AUC 0.61, reaches 16% |

* **No model is best everywhere.** XGBoost (tied with Random Forest) wins failure prediction, and XGBoost wins normal-weather demand. A plain linear model wins heatwave demand. For churn, complexity buys nothing.
* **XGBoost fails in a heatwave.** Trees cannot predict beyond the temperatures they trained on, so the dashboard switches to the linear model above 40 °C.
* **Churn is hard to predict from purchase history alone.** The best models find about twice as many lapsing customers as random contact.
* **The datasets are stand-ins** (simulated power data, machine-tool failures, an online shop). The scores show the method works, not what a real company would gain.

Full tables are in [docs/04_RESULTS.md](docs/04_RESULTS.md). A graph neural network was tried in an earlier version; it did not beat a simple model on our data, so it now appears only in
[docs/05_FUTURE_WORK.md](docs/05_FUTURE_WORK.md).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                    # Linux or Mac: source .venv/bin/activate
pip install -r requirements.txt

python run_all.py --quick                 # check that everything works (about 2 minutes)
python run_all.py                         # full experiment, 5 seeds, writes tables, figures and models
python -m pytest -q hvac_ml/tests         # 14 automatic checks
streamlit run app.py                      # open the dashboard
```

More detail and troubleshooting: [HELP.md](HELP.md).

## Data (public, 1.7 MB in total)

| Pillar | Dataset | Link |
| :--- | :--- | :--- |
| 1 | Kaggle: HVAC power consumption (hourly, 8,760 rows) | <https://www.kaggle.com/datasets/felixpradoh/hvac-power-consumption> |
| 2 | UCI: AI4I 2020 Predictive Maintenance (10,000 machines) | <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset> |
| 3 | UCI: Online Retail II (5,939 customers, stored as 44,870 invoices) | <https://archive.ics.uci.edu/dataset/502/online+retail+ii> |

## Folder structure

```
run_all.py                one command runs everything
app.py                    the dashboard (Streamlit)
HELP.md                   help file: install, run, use, troubleshoot
hvac_ml/
  config.py               every setting in one place (seeds, splits, model settings)
  data.py                 loads and cleans the three datasets
  pillar1_demand.py       demand forecasting
  pillar2_failure.py      failure prediction
  pillar3_churn.py        churn prediction
  common.py               the models (Ridge, XGBoost, baselines) and the metrics
  plots.py                all figures
  run.py                  runs the three pillars and saves the outputs
  data/                   the three datasets
  tests/                  14 automatic checks
  outputs/                figures, result tables, saved models, log
docs/                     01 product, 02 data, 03 method, 04 results, 05 future work, 06 viva questions, 07 report plan
```

## Documentation

| File | Contents |
| :--- | :--- |
| [docs/01_PRODUCT_AND_PROBLEM.md](docs/01_PRODUCT_AND_PROBLEM.md) | What the product is, who uses it, and the three tasks |
| [docs/02_DATA.md](docs/02_DATA.md) | The datasets, cleaning, and how the churn table is built |
| [docs/03_METHOD.md](docs/03_METHOD.md) | Models, features, tests, metrics, and how we avoid fooling ourselves |
| [docs/04_RESULTS.md](docs/04_RESULTS.md) | All results, with an explanation of every figure |
| [docs/05_FUTURE_WORK.md](docs/05_FUTURE_WORK.md) | Where XGBoost falls short and where a graph neural network might help |
| [docs/06_VIVA_QA.md](docs/06_VIVA_QA.md) | Questions we expect, with short answers |
| [docs/07_REPORT_AND_SLIDES_PLAN.md](docs/07_REPORT_AND_SLIDES_PLAN.md) | Report outline, draft conclusion, slides and the submission checklist |

## Team

Animesh Basak (26911001), Dhruv Aggarwal (26911004), Ankit Kumar (26911002), Saurav Kumar Boro (26535026), Kriti Vajpayee.
