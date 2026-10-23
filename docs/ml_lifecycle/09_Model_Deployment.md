# Step 09: Model Deployment

> **ML Lifecycle Phase**: 09 / 10  
> **System**: SmartHVAC & Seasonal Appliance Heterogeneous GML  
> **Deployment Targets**: Streamlit Cloud, Local dashboard (`dashboard_app.py` / `app.py`), Neo4j Graph DB

---

## 1. Multi-Tiered Deployment Architecture

To suit different users (reviewers, managers, field engineers), the system implements a **two-tier deployment strategy**:

```mermaid
flowchart TD
    subgraph Core_Artifacts["Validated ML Artifacts"]
        M1["fine_tuned_hetero_gnn_v100.pt<br>(Calibrated PyG Weights)"]
        M2["fine_tuned_financial_metrics.csv<br>(12-Month Simulated Ledger)"]
        M3["fine_tune_telemetry.json<br>(Hardware and Training Telemetry)"]
    end

    subgraph Tier1["Tier 1: Cloud Web Application"]
        S1["Streamlit Cloud Deployment<br>Entrypoint: app.py to dashboard_app.py<br>Repo: github.com/AnimeshBasak-14/SmartHVAC"]
    end

    subgraph Tier2["Tier 2: Neo4j graph (optional)"]
        S3["Neo4j Graph Database Server<br>bolt://10.13.1.7:7687<br>Topological Cypher Querying"]
    end

    Core_Artifacts --> Tier1
    Core_Artifacts --> Tier2
```

---

## 2. Deployment Tiers & Specifications

### Tier 1: Streamlit dashboard

> **Note**: the dashboard does not load the trained model. Its per-month demand, failure risk, engagement and stock numbers are computed from simple formulas inside the app, and the revenue and cost table is read from the saved CSV. It shows the ideas and results of the project, but it is not live inference from the GNN.
* **Files**: `app.py` (root entrypoint) and `hvac_gml_project/dashboard_app.py`
* **Features**:
  * **Interactive Company Business POV Controls**: Dynamic sidebar inputs for unit energy tariffs (₹/kWh), emergency repair penalty multipliers ($`1.5\text{x} - 4.0\text{x}`$), discount campaign costs, and monthly inventory holding charges.
  * **In-Memory Failsafe PyVis Topology**: Replaced filesystem write dependencies with in-memory HTML generation (`net.generate_html()`) to eliminate Streamlit Cloud disk permission crashes and blank screen issues.
  * **12-Month Operational Sliders**: Interactive monthly regime selector ($`m = 1 \dots 12`$) dynamically rendering geographic demand maps, high-risk flagged equipment lists, and sleeper customer tables.
  * **Cloud Deployment URL**: `https://smarthvac-advisor.streamlit.app/`

### Tier 2: Neo4j graph export (optional)
* **Connection**: Bolt binary driver (`bolt://localhost:7687` or `bolt://10.13.1.7:7687`).
* **Visual Graph Queries**: Pre-configured Cypher scripts enabling interactive network exploration of resident-device ownership clusters, high-risk failure subgraphs, and parts supply chains.

---

## 3. Deployment Run Commands

### Running Local Streamlit App:
```powershell
# Using Python 3.11 environment
py -3.11 -m streamlit run app.py
```

### Git & Remote Deployment:
* Pushed to GitHub repository `main` branch: `https://github.com/AnimeshBasak-14/SmartHVAC.git`
* Linked directly to Streamlit Community Cloud for automated continuous deployment (CD).

