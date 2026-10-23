# SmartHVAC: graph machine learning for an HVAC service company

CSC-505 group project. We model an HVAC and home-appliance service business as a graph
(locations, residents, devices and spare parts) and train one heterogeneous graph neural
network (GNN) to handle four tasks at once:

1. Service demand per location for each month (metro areas need more cooling, villages more heating).
2. Power draw and failure risk per device, so technicians can visit before a breakdown.
3. Customers who have stopped using the service ("sleepers"), and which off-season product to offer them.
4. How many of each spare part to keep in stock.

The model is built with PyTorch Geometric (`HeteroConv` with `SAGEConv` layers). We trained it on
the lab server's Tesla V100 GPU. A Streamlit dashboard shows the results, and a Neo4j
connector is included but optional.

## Documentation

| Document | What it covers |
| :--- | :--- |
| [Architecture and implementation](docs/ARCHITECTURE_AND_IMPLEMENTATION.md) | Problem formulation, model layers, mixed-precision training, how we handle a diverging loss |
| [Graph ML basics](docs/GML_FOUNDATIONS_WHAT_WHY_HOW_WHERE.md) | What graph ML is and when it beats tabular models such as XGBoost or random forests |
| [Dataset guide](docs/DATASET_GUIDE_WHAT_WHY_HOW_WHERE.md) | The 4 node tables and 3 edge tables, month encoding ($`\sin/\cos`$), normalisation |
| [Code reference](hvac_gml_project/README.md) | What each source file does |
| [Q&A](docs/q&a.md) | Questions we expect in the viva |

## ML lifecycle notes

One file per stage, in [docs/ml_lifecycle/](docs/ml_lifecycle/README.md):

1. [Problem definition](docs/ml_lifecycle/01_Problem_Definition.md): the four tasks and how seasons change them.
2. [Data collection](docs/ml_lifecycle/02_Data_Collection.md): UCI AI4I 2020 (id 601), UCI Online Retail II (id 502), Kaggle HVAC power consumption (felixpradoh).
3. [Cleaning and preprocessing](docs/ml_lifecycle/03_Data_Cleaning_and_Preprocessing.md): removing cancelled orders, deriving power from torque and speed, checking the graph edges.
4. [EDA](docs/ml_lifecycle/04_Exploratory_Data_Analysis.md): the 12 monthly climate settings, failure correlations, customer RFM distributions.
5. [Feature engineering](docs/ml_lifecycle/05_Feature_Engineering_and_Selection.md): sin/cos month encoding and 8 features per node.
6. [Model selection](docs/ml_lifecycle/06_Model_Selection.md): why a heterogeneous GNN, compared with XGBoost, GCN and GAT baselines.
7. [Training](docs/ml_lifecycle/07_Model_Training.md): V100 server (`10.13.1.7`), FP16 mixed precision.
8. [Evaluation and tuning](docs/ml_lifecycle/08_Model_Evaluation_and_Tuning.md): fine-tuning cut the total loss by 65.85% and raised failure F1 by 0.157 and churn F1 by 0.249.
9. [Deployment](docs/ml_lifecycle/09_Model_Deployment.md): Streamlit Cloud, Neo4j.
10. [Monitoring](docs/ml_lifecycle/10_Model_Monitoring_and_Maintenance.md): training logs, and what to watch when the climate or customer behaviour changes.

## Business results (after fine-tuning)

These numbers come from `hvac_gml_project/artifacts/fine_tuned_financial_metrics.csv`. They are a
simulation: the model's predictions are combined with the prices and costs we assumed in
`business_financial_engine.py`, so they show what the model could be worth, not real company accounts.

| Over 12 months | Value |
| :--- | :--- |
| Revenue | ₹2,61,01,851.84 (about ₹2.61 crore) |
| Operating cost | ₹34,00,492.40 (about ₹34.0 lakh) |
| Net profit | ₹2,27,01,359.44 (about ₹2.27 crore); profit is positive in every month |
| Revenue / cost ratio | 7.76 (mean of the monthly ratios), 7.68 (total revenue / total cost) |
| Cost saved by preventive instead of reactive repairs | 22.6% on average, 28.4% in the best month |

## Running it

### Streamlit Cloud
1. Push the repository to GitHub (ours is `https://github.com/AnimeshBasak-14/SmartHVAC.git`).
2. On [share.streamlit.io](https://share.streamlit.io/), create a new app from that repository, branch `main`.
3. Set the main file to `app.py` and deploy.

### Locally (Windows or Linux)
```bash
git clone https://github.com/AnimeshBasak-14/SmartHVAC.git
cd SmartHVAC
pip install -r requirements.txt
python hvac_gml_project/tests/test_pipeline.py
streamlit run app.py
```

### Training on the lab server (Tesla V100)
```bash
cd hvac_gml_project
python train_and_monitor.py      # first training run
python fine_tune_v100.py         # fine-tuning
python benchmark_baselines.py    # XGBoost / GCN / GAT comparison
streamlit run dashboard_app.py --server.port 8501 --server.headless true
```

## Team

Animesh Basak (26911001), Dhruv Aggarwal (26911004), Ankit Kumar (26911002),
Saurav Kumar Boro (26535026), Kriti Vajpayee.
