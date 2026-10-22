# Step 04: Exploratory Data Analysis (EDA)

> **ML Lifecycle Phase**: 04 / 10  
> **System**: SmartHVAC & Seasonal Appliance Heterogeneous GML  
> **Analysis Scope**: Seasonal Climate Regimes, Failure Correlations, Customer RFM Distributions

---

## 1. Seasonal Climate Regimes & Demand Dynamics

Analysis of regional climate data across the 12 calendar snapshots demonstrates that HVAC demand is non-stationary and multimodal:

| Month | Climate Classification | Cooling Mult. | Heating Mult. | Dominant Appliance Category | Regional Vulnerability |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **01 (Jan)** | Peak Winter | 0.25x | **0.90x** | Space Heaters, Geysers | Village Hubs (Cold Waves) |
| **02 (Feb)** | Winter-Transition | 0.40x | 0.65x | Mixed (Transition) | Moderate Load |
| **03 (Mar)** | Spring Warming | 0.70x | 0.30x | Pre-Cooling ACs | Metro Urban Heat Islands |
| **04 (Apr)** | Summer Inception | **0.95x** | 0.15x | Heavy Inverter ACs | Metropolitan Centers |
| **05 (May)** | Peak Summer | **1.00x** | 0.10x | Industrial Chillers, Split ACs | Grid Thermal Strain ($> 40^\circ\text{C}$) |
| **06 (Jun)** | Late Summer | 0.88x | 0.15x | Cooling + Dehumidification | High Ambient Stress |
| **07 (Jul)** | Monsoon-Humid | 0.65x | 0.25x | Dehumidifiers, AC Dry Mode | High Moisture / PCB Shorts |
| **08 (Aug)** | Monsoon-Humid | 0.60x | 0.30x | Dehumidifiers, AC Dry Mode | PCB Corrosion, Lightning Surges |
| **09 (Sep)** | Post-Monsoon | 0.55x | 0.35x | Moderate AC & Water Heating | Sleeper Churn Inception |
| **10 (Oct)** | Autumn Transition | 0.40x | 0.50x | Low Utilization | Dormant Customer Peak |
| **11 (Nov)** | Early Winter | 0.25x | **0.80x** | Water Heaters, Induction Kettles | Village Demand Escalation |
| **12 (Dec)** | Peak Winter | 0.15x | **0.95x** | Room Heaters, Electric Blankets | Village Grid Instability |

### Key Observations:
* **The Metropolitan Summer Peak**: In May, cooling demand peaks at $1.00\text{x}$ with ambient temperatures exceeding $40^\circ\text{C}$, elevating metropolitan demand to $> 620\text{ units/mo}$.
* **The Rural Winter Peak**: In December, village heating demand surges to $> 720\text{ units/mo}$ due to widespread heating appliance adoption, while metropolitan demand drops to $460\text{ units/mo}$.

---

## 2. Predictive Maintenance & Failure Distribution (UCI AI4I 2020)

Statistical inspection of the 10,000 AI4I predictive maintenance logs revealed strong physical correlations between operational parameters and failure events:

```
Total Observations: 10,000
Combined Critical Faults (HDF | PWF): 207 (2.07% Event Rate)
  ├── Heat Dissipation Failure (HDF): 115 occurrences (1.15%)
  └── Power Failure (PWF):             95 occurrences (0.95%)

Mechanical Power Distribution (Torque * RPM):
  ├── Mean:  6.28 kW
  ├── Std:   1.07 kW
  ├── Min:   1.15 kW
  └── Max:  10.47 kW
```

### Critical Cross-Feature Findings:
1. **Heat Dissipation Failure (HDF)** triggers when the difference between Process Temperature and Air Temperature drops below $8.6\text{ K}$ while rotational speed is below $1,380\text{ rpm}$. This directly mirrors HVAC condenser coil clogging and summer fan stalls.
2. **Power Failure (PWF)** occurs when the product of torque and angular velocity falls outside safe power limits ($P_{\text{mech}} < 3,500\text{ W}$ or $P_{\text{mech}} > 9,000\text{ W}$), reflecting electrical compressor lockups and grid undervoltage.

---

## 3. Customer RFM & Sleeper Churn Distribution (UCI Online Retail II)

Analysis of the 4,312 customers in UCI Online Retail II yielded the behavioral baseline:

```
Customer RFM Summary Statistics (N = 4,312):
========================================================================
Dimension          Mean          Std          Min          50%          Max
------------------------------------------------------------------------
Recency (days)     91.17        96.86        1.00        53.00       374.00
Frequency          4.46         8.17        1.00         2.00       205.00
Monetary (£)    2,048.24     8,914.48        2.95       706.02   349,164.35
Sleeper Rate (R > 90 days): 33.28% (1,435 dormant customers)
========================================================================
```

### Key Insights:
* Exactly **$33.28\%$ of customers exceed 90 days of inactivity**, providing the empirical basis for the sleeper churn label $y_{\text{sleeper}}$.
* Monetary expenditure exhibits extreme right-skewness ($\max = £349,164$), necessitating **logarithmic scaling** ($\log(1 + M)$) to prevent high-net-worth outliers from destabilizing resident node embeddings.
