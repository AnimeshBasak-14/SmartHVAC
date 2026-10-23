# HVAC Service Planner: Help File

How to install, run and use the project, and what to do when something goes wrong.

## 1. What you need

* Python 3.10 or newer (we used 3.14)
* About 300 MB of disk space for the packages. The data is only 1.7 MB.
* No GPU and no internet connection is needed to run the project. The three datasets are already in `hvac_ml/data/`.

## 2. Install (one time)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux or Mac
pip install -r requirements.txt
```

## 3. Run

| Command | What it does | Time |
| :--- | :--- | :--- |
| `python run_all.py --quick` | Whole pipeline with one random seed. Use it to check that the install works. Writes to `hvac_ml/outputs/quick_check/`, so it never overwrites the real results. | about 2 minutes |
| `python run_all.py` | Full experiment (5 seeds): trains and tests all models for the three pillars, writes tables, figures and the saved models | about 11 minutes |
| `python -m pytest -q hvac_ml/tests` | 17 automatic checks (data sizes, no look-ahead leakage, no train/test overlap, metrics, monitoring functions) | about 1 minute |
| `python -m hvac_ml.report_figures` | Redraws the 52 report and slide figures into `hvac_ml/outputs/figures/report/` (needs the full run first) | about 3 minutes |
| `python report/build_report.py`, `python report/build_slides.py` | Rebuild the Word report and the slides from the documents, tables and figures | under a minute |
| `streamlit run app.py` | Opens the dashboard in your browser (<http://localhost:8501>). Run `run_all.py` first. | instant |

Results are written to `hvac_ml/outputs/`:

* `results/RESULTS_SUMMARY.md`: all scores as tables (mean ± standard deviation over seeds)
* `results/summary_mean_std.csv`, `all_runs_by_seed.csv`: the same numbers as CSV files
* `results/model_choice.csv`: which model is chosen for each test, and which models tie
* `figures/`: ten PNG figures from the main run, and `figures/report/` with the figures for the report and slides
* `models/hvac_models.joblib`: the trained models used by the dashboard
* `run_log.txt`: the log of the last run

## 4. Using the dashboard

1. **Demand Planner.** Choose the month, indoor setpoint, humidity and the day's lowest and highest outdoor temperature. The chart shows the
   expected power draw hour by hour, the peak hour and the day's energy. If the temperature is 40 °C or more, the planner automatically switches
   from XGBoost to the Ridge model for those hours and says so.
2. **Failure Radar.** Set a machine's sensor readings to see its failure risk and whether it is above the alert threshold. The table lists the 15
   highest-risk machines in a random sample of 500.
3. **Retention List.** Choose how many customers your team can contact. The table shows the active customers most likely to buy nothing in the next 90 days.
The scores on data the models never saw are not shown in the dashboard. They are in `hvac_ml/outputs/results/RESULTS_SUMMARY.md` and in the report.

## 5. Changing the settings

All settings are in `hvac_ml/config.py`: random seeds, the share of data used for testing, which months are held out, the heatwave
temperature, the churn window and the model settings.

## 6. Where the data comes from

| Pillar | Dataset | Link |
| :--- | :--- | :--- |
| 1. Demand | Kaggle, HVAC power consumption (`felixpradoh`) | <https://www.kaggle.com/datasets/felixpradoh/hvac-power-consumption> |
| 2. Failure | UCI, AI4I 2020 Predictive Maintenance | <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset> |
| 3. Churn | UCI, Online Retail II | <https://archive.ics.uci.edu/dataset/502/online+retail+ii> |

The retail data is stored as one row per invoice (`retail_invoices.csv.gz`, 0.76 MB). To rebuild it from the original 45 MB Excel file,
download that file from the UCI link, save it as `hvac_ml/data/online_retail_II.xlsx`, delete `retail_invoices.csv.gz` and run `python run_all.py`
(needs `openpyxl`; takes a few minutes).
The two other files are downloaded automatically from UCI if missing (AI4I) or must be downloaded by hand from Kaggle (needs a free login).

## 7. Troubleshooting

| Problem | Fix |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'xgboost'` (or sklearn, pandas...) | Activate the virtual environment, then `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'hvac_ml'` | Run the commands from the project folder (the one that contains `run_all.py`) |
| Dashboard says "No trained models found" | Run `python run_all.py` first |
| `No space left on device` while installing | Free some disk space, or install with `pip install --no-cache-dir -r requirements.txt` |
| `FileNotFoundError: ... hvac_power_consumption.csv` | Download the Kaggle file and save it in `hvac_ml/data/` with that name |
| Many `FutureWarning` messages from scikit-learn | Harmless. They come from a setting that changes in a later version |
| Results differ slightly from the report | XGBoost and scikit-learn versions can change the last digit. The seeds are fixed, so two runs on the same machine give the same numbers |

## 8. Limits (please read)

* The three datasets are stand-ins: simulated power data, machine-tool failures, and an online shop's customers. None of them comes from an HVAC service company.
* The scores show how the method performs on these data, not what a real company would see.
* The churn model finds roughly twice as many lapsing customers as random contact, not ten times as many. That is a property of the problem.
