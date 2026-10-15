# SmartHVAC Advisor — Help File

This is the help file for the submission. It explains how to install, run and use the project, and what
to do when something goes wrong.

---

## 1. Requirements
- Python **3.10 – 3.13** (3.14 also works if the packages install)
- About **1 GB** of free disk space (for the Python packages)
- No GPU is needed. Training takes **about 2–6 minutes** on a normal laptop CPU and less on a multi-core server.
- Internet is only needed to install packages. The dataset is already included in `data/`.

## 2. Installation (one time)
```bash
cd Project
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # Linux / Mac
pip install -r requirements.txt
```

## 3. Running the experiment
| Command | What it does | Time |
|---|---|---|
| `python run_all.py --quick` | Smoke test: 3 models, no tuning | ~20 s |
| `python run_all.py` | Full experiment: 7 regressors + 5 classifiers, CV, tuning, test, 11 figures | 2–6 min |
| `python -m pytest -q` | Automatic correctness checks | ~30 s |

The progress is printed on the screen **and** saved to `outputs/run_log.txt`. When it finishes you get:
- `outputs/figures/01 … 11_*.png` – all plots (for the report and PPT)
- `outputs/results/RESULTS_SUMMARY.md` – all numbers in one file
- `outputs/results/*.csv` – the same tables as CSV (open them in Excel)
- `outputs/models/hvac_models.joblib` – the trained models used by the GUI

## 4. Using the GUI
```bash
streamlit run app.py
```
Your browser opens at <http://localhost:8501>.
1. In the **left sidebar**, choose the building shape (1–12), the orientation, the window area and where
   the windows are, and the electricity price.
2. The **Building Advisor** tab shows:
   - predicted **heating and cooling load** (kWh/m²) with a **90% range**
   - the **demand class** (Low / Medium / High) and the classifier's confidence
   - the **estimated annual energy cost** (₹)
   - the **recommended equipment and maintenance plan**
   - the **design optimizer** table: the 10 best window/orientation options and the % saving each one gives
3. **Model Performance** shows how accurate the models are on unseen test data.
4. **Data Insights** shows which building features matter most.

## 5. Using the command line (no GUI)
```bash
python predict.py --list-shapes
python predict.py --shape 1 --orientation 2 --glazing-area 0.25 --glazing-dist 1
python predict.py --shape 12 --glazing-area 0.4 --glazing-dist 5 --price 9.5
```
| Option | Allowed values |
|---|---|
| `--shape` | 1–12 (see `--list-shapes`) |
| `--orientation` | 2 = North, 3 = East, 4 = South, 5 = West |
| `--glazing-area` | 0, 0.1, 0.25, 0.4 (window area as a fraction of floor area) |
| `--glazing-dist` | 0 = no windows, 1 = uniform, 2 = N, 3 = E, 4 = S, 5 = W |
| `--price` | electricity price in ₹/kWh (default 8) |

## 6. Changing settings
All settings are in `src/config.py`: random seed, train/validation/test sizes, number of CV folds, interval
coverage (90%), and electricity price. Models and tuning grids are in `src/models.py`.

## 7. Data source
UCI Machine Learning Repository – Energy Efficiency dataset:
<https://archive.ics.uci.edu/dataset/242/energy+efficiency>.
If `data/energy_efficiency.csv` is deleted, `run_all.py` downloads it again automatically (this needs `openpyxl`).

## 8. Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` (or pandas, streamlit…) | Activate the venv, then `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'src'` | Run the commands from the `Project/` folder, not from inside `src/` |
| GUI says *“hvac_models.joblib not found”* | Run `python run_all.py` first |
| `No space left on device` during pip install | Free disk space, or run `pip cache purge`, or install with `pip install --no-cache-dir -r requirements.txt` |
| `ImportError: Missing optional dependency 'tabulate'` | `pip install tabulate` |
| `ConvergenceWarning` from the MLP | Harmless. The neural network stops early; the result is still valid |
| Streamlit on a remote server does not open | Use `ssh -L 8501:localhost:8501 user@server`, then open <http://localhost:8501> on your laptop |
| Results differ slightly between computers | Small differences come from library versions. The seed is fixed (42), so on the same machine the results are identical |

## 9. Limitations (important to state honestly)
- The buildings are **simulated** (Ecotect software), all in **one climate** (Athens, Greece), with only 12 shapes.
- The cost estimate assumes that the loads are **annual kWh per m²**.
- The equipment and maintenance tiers are **business rules** we designed. They are not manufacturer specifications.
- A technician should confirm the result with a proper load calculation before installing equipment.
