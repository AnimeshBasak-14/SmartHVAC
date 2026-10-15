# SmartHVAC Advisor
### Machine Learning for HVAC Service Providers (e.g. Aire Serv) — CSC-505 Group Project

> **One sentence:** we predict how much **heating** and **cooling** energy a building needs from 8 simple
> building features, then turn that into **business advice** for an HVAC company — which equipment to
> sell, which maintenance plan to offer, and which cheap design changes would cut the customer's energy bill.

---

## Why this is useful (the story)
An HVAC service company (like Aire Serv) visits a customer and must decide **what size of system** to
install and **how often to service it**. The proper engineering calculation ("load calculation") is slow.
Our ML model gives an **instant estimate with an uncertainty range**, so the technician can:

1. **Quote faster** – predicted heating & cooling load in < 1 second.
2. **Avoid over/under-sizing** – the demand class (Low / Medium / High) maps to an equipment tier.
3. **Sell the right maintenance plan** – annual / bi-annual / quarterly.
4. **Upsell energy savings** – the *design optimizer* tries every window/orientation option and shows the % saving.

## What is new / interesting (novelty)
| Feature | Why it matters |
|---|---|
| 7 regression models + 5 classifiers compared fairly (CV + validation + hidden test) | Evidence-based model choice |
| **90% prediction ranges** (conformal method) | Technician sees *how sure* the model is, not just one number |
| **Two ways to classify demand** (direct classifier vs. regress-then-classify) | Shows which approach is better for this problem |
| **Design optimizer** (what-if over 60 window/orientation options) | Turns a prediction into an actionable, money-saving recommendation |
| **Business layer** (₹ cost, equipment tier, maintenance plan) | Connects ML output to real service decisions |
| **GUI** (Streamlit) + command-line tool | Usable by a non-programmer technician |

## Data
**UCI Energy Efficiency dataset** – 768 simulated residential buildings, 8 features, 2 targets.
- Online source: <https://archive.ics.uci.edu/dataset/242/energy+efficiency>
- Paper: A. Tsanas & A. Xifara (2012), *Accurate quantitative estimation of energy performance of
  residential buildings using statistical machine learning tools*, Energy and Buildings, 49, 560-567.
- A clean copy is in `data/energy_efficiency.csv` (the original Excel file is in `data/raw/`).

## Quick start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python run_all.py --quick      # 20-second check that everything works
python run_all.py              # full experiment (train + validate + tune + test + all figures)
streamlit run app.py           # open the GUI in the browser
python predict.py --shape 3 --glazing-area 0.4 --glazing-dist 4   # advice for one building
python -m pytest -q            # automatic tests
```

## Folder structure
```
Project/
├── run_all.py            <- runs EVERYTHING (start here)
├── app.py                <- GUI (Streamlit)
├── predict.py            <- command-line advice for one building
├── requirements.txt
├── HELP.md               <- help file (how to install / use / troubleshoot)
├── src/
│   ├── config.py         <- all settings (paths, seed, split sizes, price)
│   ├── data.py           <- load + check + label + split data
│   ├── models.py         <- list of models + tuning grids
│   ├── train.py          <- train / validate / tune / test
│   ├── plots.py          <- all figures
│   └── recommend.py      <- business advice + design optimizer
├── data/                 <- dataset (CSV + original Excel)
├── tests/                <- automatic checks (pytest)
└── outputs/              <- created by run_all.py
    ├── figures/          <- 11 PNG figures for report & PPT
    ├── results/          <- CSV tables + RESULTS_SUMMARY.md
    ├── models/           <- trained models (hvac_models.joblib)
    └── run_log.txt       <- full log of the run
```

## Team (Group 3)
- [Animesh Basak](https://github.com/AnimeshBasak-14) (26911001)
- [Dhruv Aggarwal](https://github.com/dhruvaggarwal419) (26911004)
- [Ankit Kumar](https://github.com/ankit-kr04) (26911002)
- [Saurav Kumar Boro](https://github.com/SauravBoro) (26535026)
- [Kriti Vajpayee](https://github.com/kriti25122001) (25931008)
