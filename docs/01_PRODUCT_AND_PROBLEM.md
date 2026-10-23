# 01. The product and the problem

## What we are building: HVAC Service Planner

A decision-support dashboard for an HVAC service provider (for example a franchise such as Aire Serv). It does not replace the
technicians or the engineering calculation. It answers three questions a manager has every week and turns each answer into a short
list to act on.

| Tool | The manager's question | What the tool shows | Who uses it |
| :--- | :--- | :--- | :--- |
| **Demand Planner** (Pillar 1) | How much power will the HVAC systems draw tomorrow, and when is the peak? | An hourly demand curve for a weather forecast, the peak hour, the day's energy | Dispatch and stock planning |
| **Failure Radar** (Pillar 2) | Which machines should we inspect first? | A risk score per machine and a ranked inspection list | Maintenance lead |
| **Retention List** (Pillar 3) | Which active customers are about to go quiet? | A ranked call list with each customer's risk | Customer-success and marketing |

The product is the dashboard (`app.py`) together with the models behind it (`hvac_ml/`).

## Why each tool is worth building

The test of a product is whether it beats what people do without it. For each pillar we therefore measure the model against a
**rule of thumb** a company could use today, and against a **statistical** and a **classic ML** baseline.

| Pillar | Business value | "What you do today" (reference) |
| :--- | :--- | :--- |
| 1. Demand | Staff and stock for a peak, and avoid grid overload in a heatwave or a supply shortage in a cold snap | Use the average load for that hour of the day |
| 2. Failure | Move work from emergency repair to planned inspection, by inspecting the machines most likely to fail | Inspect the most worn tools first |
| 3. Retention | Spend the call budget on customers who are really at risk, and keep their lifetime value | Contact the customers who have not bought for the longest time (recency rule) |

## The three pillars

**Pillar 1: demand forecasting.** Predict the hourly HVAC power draw (a positive number, in kW) from the weather, the indoor setpoint and the
calendar. A manager uses it to prevent a power shortfall in a heatwave and a heating shortage in a cold snap.

**Pillar 2: failure prediction.** Predict whether a machine will fail (yes or no) from its sensor readings, so inspections go first to the
machines most likely to fail. Failures in the data are 3.4% of machines, so the task is about finding a few needles, not about overall accuracy.

**Pillar 3: churn prediction.** Predict whether an active customer will enter a dormant "sleeper" state, meaning no purchase for the next
90 days. The list lets the team call these customers before they leave.

## What the product does not do (yet)

* **Cross-selling a counter-seasonal appliance** (for example a heater to a customer who only bought air conditioning). This needs
  product categories and a link between customers and products. It is listed in `docs/05_FUTURE_WORK.md`.
* **Several locations**: our demand data has one building, so we cannot compare sites or see how a shared grid couples them.
* **Real HVAC service data**: all three datasets are public stand-ins (see `02_DATA.md`). The numbers show the method works on these data;
  they are not a promise for a real company.

## Models

We compare **Ridge regression** and **XGBoost** (our two models) with a statistical baseline (ordinary linear / logistic regression), a classic
machine-learning baseline (Random Forest) and the rule of thumb for each pillar. Details are in `03_METHOD.md`.
A graph neural network was tried in an earlier version of this project and is now only discussed as future work.
