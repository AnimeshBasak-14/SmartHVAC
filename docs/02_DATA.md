# 02. Data

We use three public datasets, one per pillar. Nothing else is needed to run the project. Together they take
**1.7 MB** in `hvac_ml/data/`.

| Pillar | File in `hvac_ml/data/` | Size | Original source |
| :--- | :--- | :--- | :--- |
| 1. Demand | `hvac_power_consumption.csv` | 8,760 hourly rows | Kaggle, `felixpradoh/hvac-power-consumption`: <https://www.kaggle.com/datasets/felixpradoh/hvac-power-consumption> |
| 2. Failure | `ai4i2020.csv` | 10,000 machines | UCI #601, AI4I 2020 Predictive Maintenance: <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset> |
| 3. Churn | `retail_invoices.csv.gz` | 44,870 invoices, 5,939 customers | UCI #502, Online Retail II: <https://archive.ics.uci.edu/dataset/502/online+retail+ii> |

The Kaggle link is rebuilt from the dataset name we recorded, so check it opens before you submit.

## Pillar 1: hourly HVAC power (Kaggle)

One building, one year (2025), one row per hour: month, hour, outdoor temperature, indoor temperature, humidity, HVAC mode
(Cooling, Heating or Idle) and the power draw in kW.

* Power ranges from 0.10 to 4.01 kW (mean 1.92 kW).
* The mode follows the outdoor temperature: Heating below 19 °C, Idle from 19 to 27 °C, Cooling above 27 °C.
* Power grows with the gap between outdoor and indoor temperature (correlation 0.85 with the absolute gap).
* Outdoor temperature reaches 50 °C. Only 824 hours are at or above 40 °C, which is what makes the "heatwave" test possible.
* The data is very regular, so it is probably simulated. We could not confirm this. Treat it as a clean test bed.

## Pillar 2: machine failures (AI4I 2020)

10,000 machines with air temperature, process temperature, rotational speed, torque, tool wear, a product type (L, M, H) and a
yes/no "machine failure" label. **339 machines (3.39%) fail**, so the data is heavily unbalanced.

* The UCI page describes this dataset as synthetic data built to resemble real industrial maintenance data.
* It describes machine tools, not HVAC units. It is a stand-in for "sensor readings that predict a breakdown".
* We predict the combined failure label. The five failure-type columns (TWF, HDF, PWF, OSF, RNF) are **not** used as inputs, because
  they are the answer in disguise (a test checks this).

## Pillar 3: customer purchases (Online Retail II)

Real transactions from a UK online gift shop, December 2009 to December 2011. It is a stand-in for an HVAC company's
customer history (service visits and parts).

The original Excel file has about 1 million lines (45 MB). We keep one row per invoice instead:
customer, invoice, date, value, number of items, number of products, country, and whether the invoice is a return (cancellation).
This is all the churn task needs, and it is 0.76 MB. The Excel file is only needed to rebuild this table.

Cleaning applied when the table was built:

* rows without a customer id are dropped (we cannot follow an anonymous buyer): 22.8% of the 1,067,371 original lines
* rows with a price of zero or less are dropped (0.6% of lines; almost all of them were already in the group above)
* invoices starting with "C" are returns, kept separately and used only for a "return share" feature
* returns and negative quantities never count as purchases

### How the churn table is built (important)

The original file has no "churn" column. Earlier versions of this project used a flag `Recency > 90` as the target, which is circular
when recency is also an input. We build the target properly:

1. choose a **cutoff date**
2. **features** = what we knew about each customer from invoices *before* the cutoff
3. **target** = 1 if the customer buys **nothing in the 90 days after** the cutoff
4. only customers who bought in the 90 days before the cutoff are included (they are active now)

So the model answers: "of the customers who are active today, who will go quiet?". A test deletes all invoices after the
cutoff and checks that no feature changes.

Training rows come from five cutoffs (June 2010 to June 2011); test rows from one later cutoff (9 September 2011, window to 8 December 2011).
Training and test **customers are different people** (80/20 split).

## What we removed

The first version of this project used a larger set of files: synthetic graph tables (locations, devices, parts, residents and their
links), a derived "RFM" file whose `is_sleeper` column was exactly `Recency > 90`, and zip files of the originals. None of them is needed for the three
pillars, so they were deleted. A full backup of the earlier version is kept outside the project folder (`gnn_backup`).

## Limits to state in the report

* All three datasets are stand-ins (simulated, machine tools, a retail shop). None comes from an HVAC service company.
* Pillar 1 has one building, so it cannot show differences between locations.
* Pillar 3 measures "no purchase for 90 days" in a shop. A service company would define churn from contract renewals or visits.
