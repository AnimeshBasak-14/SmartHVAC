# HVAC Service Planner

CSC-505 Machine Learning, Group 1, IIT Roorkee. **Machine learning for an HVAC service provider** (for example a franchise such as Aire Serv).

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
| 2. Failure | Random Forest and XGBoost tie (PR-AUC 0.91 and 0.89). Inspecting the top 10% of machines finds **94%** of failures | "Most worn tool first": PR-AUC 0.12, finds 36% |
| 3. Churn | All models tie at ROC-AUC 0.77 to 0.78. Calling the top 10% reaches about **20%** of the customers who lapse, twice as many as random | "Longest since last purchase": ROC-AUC 0.61, reaches 16% |

* **No model is best everywhere.** XGBoost (tied with Random Forest) wins failure prediction, and XGBoost wins normal-weather demand. A plain linear model wins heatwave demand. For churn, complexity buys nothing.
* **XGBoost fails in a heatwave.** Trees cannot predict beyond the temperatures they trained on, so the dashboard switches to the linear model above 40 °C.
* **Churn is hard to predict from purchase history alone.** The best models find about twice as many lapsing customers as random contact.
* **The datasets are stand-ins** (simulated power data, machine-tool failures, an online shop). The scores show the method works, not what a real company would gain.

## Recommendation

Pilot the Failure Radar first and the Demand Planner (always with the heatwave safeguard) second, on the company's own data. Use the Retention List as a low-cost aid and keep its model simple. Run the monitoring checks from the first day.

## Documentation: one document per lifecycle stage

| Stage | Document | Written by |
| :--- | :--- | :--- |
| 1 | [Problem definition](docs/lifecycle/01_Problem_Definition.md) | Kriti Vajpayee |
| 2 | [Data collection](docs/lifecycle/02_Data_Collection.md) | Kriti Vajpayee |
| 3 | [Data cleaning and preprocessing](docs/lifecycle/03_Data_Cleaning_and_Preprocessing.md) | Dhruv Aggarwal |
| 4 | [Exploratory data analysis](docs/lifecycle/04_Exploratory_Data_Analysis.md) | Dhruv Aggarwal |
| 5 | [Feature engineering](docs/lifecycle/05_Feature_Engineering.md) | Ankit Kumar |
| 6 | [Model selection](docs/lifecycle/06_Model_Selection.md) | Ankit Kumar |
| 7 | [Model training](docs/lifecycle/07_Model_Training.md) | Animesh Basak |
| 8 | [Model evaluation and tuning](docs/lifecycle/08_Model_Evaluation_and_Tuning.md) | Animesh Basak |
| 9 | [Model deployment](docs/lifecycle/09_Model_Deployment.md) | Sourav Kumar Boro |
| 10 | [Monitoring and maintenance](docs/lifecycle/10_Monitoring_and_Maintenance.md) | Sourav Kumar Boro |

More: [future work](docs/05_FUTURE_WORK.md) (where a graph neural network might help), [viva questions](docs/06_VIVA_QA.md), [submission checklist](docs/07_REPORT_AND_SLIDES_PLAN.md) and the help file [HELP.md](HELP.md).
The report and the slides are in `report/`.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                    # Linux or Mac: source .venv/bin/activate
pip install -r requirements.txt

python run_all.py --quick                 # check that everything works (about 2 minutes)
python run_all.py                         # full experiment, 5 seeds, writes tables, figures and models (about 11 minutes)
python -m pytest -q hvac_ml/tests         # 17 automatic checks
streamlit run app.py                      # open the dashboard
```

More detail and troubleshooting: [HELP.md](HELP.md).

## Data (public, 1.7 MB in total)

| Pillar | Dataset | Link |
| :--- | :--- | :--- |
| 1 | Kaggle: HVAC power consumption (hourly, 8,760 rows) | <https://www.kaggle.com/datasets/felixpradoh/hvac-power-consumption> |
| 2 | UCI: AI4I 2020 Predictive Maintenance (10,000 machines) | <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset> |
| 3 | UCI: Online Retail II (5,939 customers, stored as 44,870 invoices) | <https://archive.ics.uci.edu/dataset/502/online+retail+ii> |

The Kaggle link was rebuilt from the dataset name we recorded. Check that it opens before you submit.

## Folder structure

```
run_all.py                one command runs everything
app.py                    the dashboard (Streamlit)
HELP.md                   help file: install, run, use, troubleshoot
hvac_ml/
  config.py               every setting in one place (seeds, splits, model settings, monitoring thresholds)
  data.py                 loads and cleans the three datasets
  pillar1_demand.py       demand forecasting
  pillar2_failure.py      failure prediction
  pillar3_churn.py        churn prediction
  common.py               the models (Ridge, XGBoost, baselines) and the metrics
  monitoring.py           drift and error checks used after deployment
  plots.py                the first ten figures
  figs_*.py               the report and slide figures (run report_figures.py to redraw them)
  report_facts.py         numbers quoted in the documents, saved to outputs/results/facts.json
  run.py                  runs the three pillars and saves the outputs
  data/                   the three datasets
  tests/                  17 automatic checks
  outputs/                figures, result tables, saved models, log
docs/
  lifecycle/              the ten stage documents
  05_FUTURE_WORK.md, 06_VIVA_QA.md, 07_REPORT_AND_SLIDES_PLAN.md
report/
  HVAC_Service_Planner_Report.docx, .pdf        the report
  HVAC_Service_Planner_Presentation.pptx, .pdf  the slides
  src/, build_report.py, build_slides.py        the sources and the scripts that build them
```

## Team

| Member | Enrollment | Stages |
| :--- | :--- | :--- |
| Kriti Vajpayee | 25931008 | 1 Problem definition, 2 Data collection |
| Dhruv Aggarwal | 25911019 | 3 Data cleaning and preprocessing, 4 Exploratory data analysis |
| Ankit Kumar | 26911002 | 5 Feature engineering, 6 Model selection |
| Animesh Basak | 26911001 | 7 Model training, 8 Model evaluation and tuning |
| Sourav Kumar Boro | 26535026 | 9 Model deployment, 10 Monitoring and maintenance |
