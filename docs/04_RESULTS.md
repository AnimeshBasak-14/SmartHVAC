# 04. Results

Every number is the **mean ± standard deviation over 5 random seeds**, measured on data the models never saw.
The same tables are saved in `hvac_ml/outputs/results/RESULTS_SUMMARY.md` and `summary_mean_std.csv`.
A rule of thumb ("Rule: ...") shows what a company could do without machine learning.

## Pillar 1: demand forecasting (error in kW, lower is better)

The average hourly power draw in the data is 1.92 kW.

**Random 20% of hours (the easy test)**

| Model | MAE (kW) | RMSE (kW) | R² | MAE, busiest 10% of hours (kW) |
| :--- | :---: | :---: | :---: | :---: |
| Rule: training mean | 0.942 ± 0.017 | 1.127 ± 0.014 | -0.000 ± 0.001 | 1.528 ± 0.012 |
| Rule: average by hour of day | 0.934 ± 0.017 | 1.106 ± 0.012 | 0.037 ± 0.006 | 1.307 ± 0.014 |
| Linear regression (statistical baseline) | 0.096 ± 0.001 | 0.126 ± 0.001 | 0.987 ± 0.000 | 0.074 ± 0.002 |
| Random Forest (classic ML baseline) | 0.066 ± 0.001 | 0.082 ± 0.001 | 0.995 ± 0.000 | 0.071 ± 0.004 |
| **Ridge**, basic features | 0.182 ± 0.002 | 0.243 ± 0.003 | 0.953 ± 0.001 | 0.208 ± 0.008 |
| **Ridge**, engineered features | 0.096 ± 0.001 | 0.126 ± 0.001 | 0.987 ± 0.000 | 0.074 ± 0.002 |
| **XGBoost**, basic features | 0.065 ± 0.001 | 0.081 ± 0.001 | 0.995 ± 0.000 | 0.072 ± 0.004 |
| **XGBoost**, engineered features | 0.065 ± 0.001 | 0.081 ± 0.001 | 0.995 ± 0.000 | 0.071 ± 0.003 |

**Unseen months (3, 6, 9, 12 held out)**

| Model | MAE (kW) | RMSE (kW) | R² | MAE, busiest 10% of hours (kW) |
| :--- | :---: | :---: | :---: | :---: |
| Rule: training mean | 1.017 ± 0.000 | 1.206 ± 0.000 | -0.000 ± 0.000 | 1.690 ± 0.000 |
| Rule: average by hour of day | 0.995 ± 0.000 | 1.172 ± 0.000 | 0.056 ± 0.000 | 1.516 ± 0.000 |
| Linear regression (statistical baseline) | 0.118 ± 0.000 | 0.160 ± 0.000 | 0.982 ± 0.000 | 0.089 ± 0.000 |
| Random Forest (classic ML baseline) | 0.077 ± 0.000 | 0.099 ± 0.001 | 0.993 ± 0.000 | 0.138 ± 0.001 |
| **Ridge**, basic features | 0.224 ± 0.000 | 0.280 ± 0.000 | 0.946 ± 0.000 | 0.343 ± 0.000 |
| **Ridge**, engineered features | 0.118 ± 0.000 | 0.160 ± 0.000 | 0.982 ± 0.000 | 0.089 ± 0.000 |
| **XGBoost**, basic features | 0.070 ± 0.001 | 0.088 ± 0.002 | 0.995 ± 0.000 | 0.088 ± 0.006 |
| **XGBoost**, engineered features | 0.072 ± 0.002 | 0.091 ± 0.003 | 0.994 ± 0.000 | 0.079 ± 0.002 |

**Forecast (train on months 1-9, predict 10-12)**

| Model | MAE (kW) | RMSE (kW) | R² | MAE, busiest 10% of hours (kW) |
| :--- | :---: | :---: | :---: | :---: |
| Rule: training mean | 1.012 ± 0.000 | 1.310 ± 0.000 | -0.932 ± 0.000 | 0.396 ± 0.000 |
| Rule: average by hour of day | 1.024 ± 0.000 | 1.346 ± 0.000 | -1.040 ± 0.000 | 0.134 ± 0.000 |
| Linear regression (statistical baseline) | 0.101 ± 0.000 | 0.129 ± 0.000 | 0.981 ± 0.000 | 0.080 ± 0.000 |
| Random Forest (classic ML baseline) | 0.083 ± 0.000 | 0.106 ± 0.000 | 0.987 ± 0.000 | 0.155 ± 0.001 |
| **Ridge**, basic features | 0.242 ± 0.000 | 0.312 ± 0.000 | 0.890 ± 0.000 | 0.111 ± 0.000 |
| **Ridge**, engineered features | 0.100 ± 0.000 | 0.128 ± 0.000 | 0.981 ± 0.000 | 0.080 ± 0.000 |
| **XGBoost**, basic features | 0.085 ± 0.002 | 0.107 ± 0.002 | 0.987 ± 0.001 | 0.092 ± 0.005 |
| **XGBoost**, engineered features | 0.076 ± 0.001 | 0.095 ± 0.001 | 0.990 ± 0.000 | 0.079 ± 0.005 |

**Heatwave (train below 40 °C, predict 40 °C and above)**

| Model | MAE (kW) | RMSE (kW) | R² | MAE, busiest 10% of hours (kW) |
| :--- | :---: | :---: | :---: | :---: |
| Rule: training mean | 1.648 ± 0.000 | 1.663 ± 0.000 | -57.297 ± 0.000 | 2.029 ± 0.000 |
| Rule: average by hour of day | 1.574 ± 0.000 | 1.590 ± 0.000 | -52.280 ± 0.000 | 1.956 ± 0.000 |
| Linear regression (statistical baseline) | 0.077 ± 0.000 | 0.097 ± 0.000 | 0.800 ± 0.000 | 0.115 ± 0.000 |
| Random Forest (classic ML baseline) | 0.365 ± 0.001 | 0.426 ± 0.001 | -2.829 ± 0.013 | 0.729 ± 0.002 |
| **Ridge**, basic features | 0.468 ± 0.000 | 0.488 ± 0.000 | -4.027 ± 0.000 | 0.699 ± 0.000 |
| **Ridge**, engineered features | 0.077 ± 0.000 | 0.097 ± 0.000 | 0.800 ± 0.000 | 0.115 ± 0.000 |
| **XGBoost**, basic features | 0.332 ± 0.006 | 0.389 ± 0.007 | -2.194 ± 0.110 | 0.674 ± 0.012 |
| **XGBoost**, engineered features | 0.336 ± 0.009 | 0.394 ± 0.009 | -2.281 ± 0.146 | 0.695 ± 0.012 |

### Figure 4 (`04_p1_model_comparison.png`): error by model and test
* **What is plotted.** Bars: average error of each model in the four tests. Dashed line: the "average by hour of day" rule. Lower is better.
* **What we see.** In the three tests inside the temperature range of the training data, XGBoost is the most accurate:
  0.072 kW on unseen months and 0.076 kW in the forecast. The Ridge and linear models are
  about 0.10 kW, still ten times better than the rule of thumb (about 1.0 kW). Engineered features cut the Ridge error from
  0.24 to 0.10 kW in the forecast. In the heatwave the order reverses:
  Ridge and linear regression reach 0.077 kW, while XGBoost (0.336 kW) and Random Forest (0.365 kW) are about four times worse.
* **What it means.** A single model is not the best choice everywhere. Ridge with engineered features is as good as the linear baseline (the penalty makes no difference with this much data).

### Figure 5 (`05_p1_heatwave_extrapolation.png`): why XGBoost fails in a heatwave
* **What is plotted.** Outdoor temperature (x) against power (y). Grey: the real values. Blue: Ridge. Orange: XGBoost. Left: the forecast months. Right: the heatwave hours at 40 °C and above, which were not in the training data.
* **What we see.** In the forecast months both models follow the data. In the heatwave the real power keeps rising to 4 kW, Ridge follows it, and XGBoost goes flat at about 3.1 kW.
* **What it means.** Trees predict an average of training rows they have seen, so they cannot go beyond the range of the training data. A linear model on the temperature gap
  extends the trend. This is the clearest case where XGBoost fails, and it is exactly the situation (a heatwave) where a service company most needs a good forecast.
  The dashboard therefore uses XGBoost normally and Ridge for hours above 40 °C.

### Figure 1 (`01_hvac_power_vs_temperature.png`): the data
Left: power against outdoor temperature, by mode. Power rises with the distance from the indoor setpoint, in both directions, and the Idle mode uses almost nothing.
The dashed line marks the heatwave test region. Right: the average monthly demand peaks in summer.

## Pillar 2: failure prediction (higher is better)

Only 3.4% of the machines fail, so a random guess has a PR-AUC of 0.034. Accuracy is not used.

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | Failures found in top 10% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Rule: most worn tool first | 0.123 ± 0.013 | 0.655 ± 0.027 | 0.197 ± 0.011 | 0.158 ± 0.013 | 0.262 ± 0.012 | 0.365 ± 0.024 |
| Logistic regression (statistical baseline) | 0.487 ± 0.080 | 0.930 ± 0.021 | 0.480 ± 0.059 | 0.457 ± 0.065 | 0.512 ± 0.075 | 0.744 ± 0.069 |
| Random Forest (classic ML baseline) | 0.912 ± 0.037 | 0.988 ± 0.007 | 0.887 ± 0.041 | 0.927 ± 0.046 | 0.850 ± 0.041 | 0.941 ± 0.040 |
| **Ridge**, basic features | 0.445 ± 0.050 | 0.898 ± 0.022 | 0.470 ± 0.048 | 0.456 ± 0.060 | 0.497 ± 0.085 | 0.662 ± 0.067 |
| **Ridge**, engineered features | 0.492 ± 0.074 | 0.925 ± 0.021 | 0.489 ± 0.059 | 0.514 ± 0.088 | 0.471 ± 0.045 | 0.732 ± 0.069 |
| **XGBoost**, basic features | 0.790 ± 0.088 | 0.980 ± 0.009 | 0.727 ± 0.094 | 0.726 ± 0.138 | 0.738 ± 0.070 | 0.912 ± 0.039 |
| **XGBoost**, engineered features | 0.893 ± 0.042 | 0.987 ± 0.007 | 0.851 ± 0.045 | 0.901 ± 0.039 | 0.806 ± 0.052 | 0.938 ± 0.035 |

### Figure 6 (`06_p2_failure_results.png`): failure prediction
* **What is plotted.** Left: precision-recall curves (seed 42). Precision is the share of alerts that are real failures, recall is the share of all failures that we find; the curve should stay close to the top right.
  Right: PR-AUC of each model (mean ± standard deviation of 5 seeds).
* **What we see.** Random Forest (0.91) and XGBoost with engineered features (0.89) are
  far above the linear and logistic models (0.49) and the wear rule of thumb (0.12).
  The two tree models are within one standard deviation of each other. Engineered features raise XGBoost from 0.79 to 0.89.
* **What it means.** Failures come from combinations of readings crossing thresholds (high torque at low speed, a small temperature gap, a worn tool under load). A straight line cannot represent these, trees can.
  Inspecting the 10% of machines with the highest risk finds 94% of the failures with XGBoost, against 36% with the wear rule.

### Figure 7 (`07_p2_feature_importance.png`): what the models use
Left: the share of XGBoost's gain from each input. Right: the size of each Ridge weight. XGBoost relies mostly on the engineered inputs (temperature gap, wear x torque, mechanical power)
and on speed, while Ridge puts almost all its weight on the raw torque and speed. This agrees with the large gain from engineered features and with the physics of the failure types.

### Figure 2 (`02_ai4i_failure_drivers.png`): the data
The failure rate in ten equal-sized groups of torque, tool wear and mechanical power. Failures cluster at the extremes (high torque, high wear), against an average of 3.4%.

## Pillar 3: churn prediction (higher is better)

Customers who bought in the last 90 days and whether they buy nothing in the next 90 days. About a third of them lapse in the test period.

| Model | ROC-AUC | PR-AUC | F1 | Lapsers reached in top 10% | Lift of top 10% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Rule: recency only | 0.607 ± 0.041 | 0.417 ± 0.019 | 0.478 ± 0.022 | 0.162 ± 0.017 | 1.625 ± 0.170 |
| Logistic regression (statistical baseline) | 0.775 ± 0.033 | 0.575 ± 0.049 | 0.608 ± 0.024 | 0.197 ± 0.021 | 1.967 ± 0.206 |
| Random Forest (classic ML baseline) | 0.772 ± 0.022 | 0.563 ± 0.035 | 0.595 ± 0.017 | 0.201 ± 0.025 | 2.012 ± 0.252 |
| **Ridge**, basic features | 0.750 ± 0.033 | 0.547 ± 0.041 | 0.593 ± 0.032 | 0.187 ± 0.039 | 1.867 ± 0.394 |
| **Ridge**, engineered features | 0.777 ± 0.032 | 0.578 ± 0.048 | 0.609 ± 0.032 | 0.202 ± 0.020 | 2.023 ± 0.205 |
| **XGBoost**, basic features | 0.749 ± 0.031 | 0.529 ± 0.042 | 0.590 ± 0.041 | 0.178 ± 0.037 | 1.777 ± 0.374 |
| **XGBoost**, engineered features | 0.775 ± 0.029 | 0.580 ± 0.050 | 0.609 ± 0.029 | 0.200 ± 0.018 | 2.005 ± 0.176 |

### Figure 8 (`08_p3_churn_results.png`): lapse prediction on unseen customers
* **What is plotted.** Left: a gains curve. Customers are contacted from the highest to the lowest predicted risk; the line shows the share of all lapsing customers reached. The dashed grey diagonal is random contact. Right: ROC-AUC of each model (0.5 means guessing).
* **What we see.** All models with engineered features are almost identical (ROC-AUC 0.77 to 0.78),
  including plain logistic regression. They are clearly better than the recency rule (0.61).
  Basic features alone (recency, frequency, spend) give about 0.75.
* **What it means.** Predicting who will lapse is useful but not precise: contacting the 10% highest-risk customers reaches about 20% of those who lapse,
  roughly 2 times better than random contact and a bit better than the recency rule (1.6 times).
  A more complex model does not help, so the product uses the simplest one.

### Figure 9 (`09_p3_feature_importance.png`): what the models use
How often and how regularly a customer buys (frequency, average gap between orders, orders in the last two 90-day periods) and total spend carry most of the weight. **Recency, the classic rule, is one of the least important inputs** once these are known.

### Figure 3 (`03_churn_base_rates.png`): the data
Left: the share of active customers who lapse differs a lot between cutoff dates (31% to 58%), so the base rate shifts over time; our test period (September to December) has the lowest. Right: the lapse rate rises from about 24% for customers who bought in the last two weeks to about 44% for those who last bought two to three months ago. Recency is informative but far from decisive.

## Where XGBoost helps and where it does not

### Figure 10 (`10_xgboost_vs_ridge.png`)
* **What is plotted.** For each test, how much better (orange) or worse (blue) XGBoost is than Ridge, both using the engineered features.
* **What we see.** XGBoost is better for demand inside the training range (+24% to +39% less error) and for failure (+82% PR-AUC). It is **4 times worse** in the heatwave and equal within noise in churn.
* **What it means.** XGBoost earns its complexity in failure prediction and in normal-weather demand. It adds nothing in churn and it is the wrong tool for a heatwave. See `05_FUTURE_WORK.md` for what to try next.

## Model choice

Best average score; models within one standard deviation of the best are tied and the simplest tied model is chosen.

| Pillar | Test | Metric | Model chosen | Models that tie with the best |
| :--- | :--- | :--- | :--- | :--- |
| demand | random | mae | XGBoost, basic features | XGBoost, basic features; XGBoost, engineered features |
| demand | unseen_months | mae | XGBoost, basic features | XGBoost, basic features |
| demand | forecast | mae | XGBoost, engineered features | XGBoost, engineered features |
| demand | heatwave | mae | Linear regression (statistical baseline) | Linear regression (statistical baseline) |
| failure | random 20% of machines | pr_auc | Random Forest (classic ML baseline) | Random Forest (classic ML baseline); XGBoost, engineered features |
| churn | unseen customers, later period | pr_auc | Logistic regression (statistical baseline) | Logistic regression (statistical baseline); Random Forest (classic ML baseline); Ridge, basic features; Ridge, engineered features; XGBoost, engineered features |

## Recommendation for the product

| Task | Model in the product | Why |
| :--- | :--- | :--- |
| Demand Planner | **XGBoost** with engineered features, and **Ridge** for any hour above 40 °C or outside the training range | XGBoost is the most accurate in normal weather; Ridge is the only model that holds up in a heatwave |
| Failure Radar | **XGBoost** with engineered features (Random Forest is equally good and is a valid alternative) | PR-AUC about 0.9; the top 10% of machines hold about 94% of the failures. Linear models are not good enough here (PR-AUC 0.49) |
| Retention List | **Ridge** (logistic with a penalty) with engineered features | All models tie, so the simplest and easiest to explain is chosen |

In demand and failure the product uses the engineered-feature version of each model. For demand the difference between basic and engineered XGBoost is within noise, so the choice does not matter there; for failure engineered features are clearly better.

## Limits

* The datasets are stand-ins (simulated power data, machine-tool failures, an online shop). The scores show the method works; they do not predict gains at a real HVAC company.
* Five seeds give a rough spread, not a confidence interval. In the unseen-months and forecast tests of Pillar 1 only the models' randomness changes, so the standard deviations there are very small and understate real uncertainty.
* Failure and churn thresholds were chosen on training data. Churn has a lower base rate in the test period than in training, so precision and recall at the chosen threshold shift; the ranking measures (ROC-AUC, PR-AUC, top 10%) are more reliable.
