# 03. Method

## What we compare

For every pillar we train the same families of models on the same data and score them on data they never saw.

| Name in the tables | What it is | Role |
| :--- | :--- | :--- |
| Reference (training mean, average by hour, most worn tool first, recency only) | A rule of thumb a company could use today, no learning | "What you do without our product" |
| Linear regression / Logistic regression (engineered features) | Ordinary least squares, or logistic regression with almost no penalty | Statistical baseline |
| Random Forest (engineered features) | 300 trees, at least 5 rows per leaf | Classic machine-learning baseline |
| **Ridge (basic / engineered features)** | Linear model with an L2 penalty (Ridge regression for numbers, penalised logistic regression for yes/no) | **Our model 1** |
| **XGBoost (basic / engineered features)** | Gradient-boosted trees | **Our model 2** |

"Basic features" are the raw columns. "Engineered features" add a few quantities the physics or the business suggests
(see below). Comparing basic with engineered shows how much of the gain comes from the features and how much from the model.

All settings are fixed in `hvac_ml/config.py`. We did not tune anything on test data. The strength of the Ridge penalty is
chosen by cross-validation on the training data only.

## The three pillars

### Pillar 1: demand (regression, target = power in kW)

* Basic inputs: outdoor and indoor temperature, humidity, hour and month (as sine and cosine, so 23:00 is next to 00:00) and the mode.
* Engineered: the absolute gap between outdoor and indoor temperature, split by mode.
* Four tests, each with data the model never saw:
  1. **random**: 20% of hours chosen at random (the easy case)
  2. **unseen months**: months 3, 6, 9 and 12 held out
  3. **forecast**: train on January to September, predict October to December
  4. **heatwave**: train below 40 °C, predict the hours at 40 °C and above
* Metrics: mean absolute error (MAE) in kW, RMSE, R², and the MAE on the busiest 10% of hours.

### Pillar 2: failure (yes/no, target = machine failure)

* Basic inputs: the five sensor values and the machine type.
* Engineered: temperature gap (process minus air), mechanical power (torque x speed), and wear x torque. These are the physical
  quantities behind how machines fail. We did **not** add the failure thresholds themselves.
* Test: stratified 80/20 split (the 3.4% failure share is the same in both), repeated with 5 seeds.
* Only 3.4% of rows are failures, so **accuracy is meaningless** (always saying "no failure" scores 96.6%). We report:
  * **PR-AUC** (average precision): the main metric for rare events; a random guess scores 0.034
  * ROC-AUC, F1, precision and recall at a threshold chosen on the training data only
  * **top-10% capture**: the share of all failures found if we inspect only the 10% of machines with the highest risk
* The "most worn tool first" reference is the rule of thumb a workshop could use today.

### Pillar 3: churn (yes/no, target = no purchase in the next 90 days)

* Basic inputs: recency (days since last purchase), frequency (number of invoices), monetary (total spend).
* Engineered: tenure, average order value, average gap between orders, orders in the last 90 days and in the 90 days before that,
  the trend between the two, return share, UK flag, and log versions of the money and count columns.
* Test: unseen customers in a later period (see `02_DATA.md`). Each of the 5 seeds picks a different 20% of customers.
* Metrics: ROC-AUC, PR-AUC, F1 and the **top-10% capture and lift**: how many of the customers who really lapse are in the 10% we
  would contact first, compared with contacting at random.
* The "recency only" reference is the classic rule "the longer since the last purchase, the more likely to lapse".

## How we avoid fooling ourselves

| Risk | What we did | Where it is tested |
| :--- | :--- | :--- |
| Using future information (churn) | Features come only from before the cutoff; target only from after it | `test_churn_features_use_only_the_past`, `test_churn_label_comes_from_the_future` |
| Same customer in training and test | Customers are split first, then rows are built | `test_churn_test_customers_never_appear_in_training` |
| Test period overlaps training period | The last training window ends before the test cutoff | `test_train_cutoffs_end_before_test_cutoff` |
| Target hidden in the inputs (failure) | The failure-type columns are never used as inputs | `test_failure_features` |
| Tuning on the test set | Nothing is tuned on test data; the F1 threshold comes from out-of-fold training scores | by design |
| Lucky split | Every number is the mean and standard deviation over 5 seeds | `config.SEEDS` |
| Misleading metric | Accuracy is not used for the rare-event pillars | by design |

## Choosing the model for the product

For each test we pick the model with the best average score. Any model within **one standard deviation** of the best counts as a tie,
and the **simplest tied model wins**. The order, simplest first: linear / logistic regression, Ridge, Random Forest, XGBoost.
This stops us from choosing a complicated model for a difference that is just noise. The result is saved in
`outputs/results/model_choice.csv`.

## Running it

```bash
python run_all.py            # about 5 to 8 minutes: all three pillars, 5 seeds, all figures and the saved models
python run_all.py --quick    # one seed, about 2 minutes; writes to outputs/quick_check/, not over the real results
python -m pytest -q hvac_ml/tests
streamlit run app.py
```
