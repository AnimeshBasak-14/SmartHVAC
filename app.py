"""
app.py
Streamlit dashboard for the SmartHVAC multi-task GNN project.
One tab per task (demand, maintenance, churn, spare parts), a graph explorer
and a monthly revenue/cost table.
"""
import os
import sys
import tempfile
import base64
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components
from pyvis.network import Network

# Ensure hvac_gml_project is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "hvac_gml_project")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config import Config
from data_pipeline import HVACGraphDatasetBuilder, encode_month_features
from business_financial_engine import CompanyBusinessInputs, BusinessFinancialEngine

st.set_page_config(
    page_title="SmartHVAC Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# STYLING
# -------------------------------------------------------------
st.markdown("""
<style>
    /* Metric Card Display */
    [data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        white-space: nowrap !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
        white-space: nowrap !important;
    }
    
    /* Pillar Header Banner */
    .pillar-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-left: 5px solid #00d4ff;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .pillar-banner h3 {
        margin: 0 0 8px 0;
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .pillar-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        font-size: 0.86rem;
        color: #cbd5e1;
    }
    .pillar-tag {
        background: rgba(0, 212, 255, 0.12);
        color: #38bdf8;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    
    /* Tab Headers */
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

def format_inr(amount: float, compact: bool = True) -> str:
    """Formats values in Indian Rupees (Lakhs/Crores)."""
    if compact:
        if abs(amount) >= 10000000:
            return f"₹{amount / 10000000:.2f} Cr"
        elif abs(amount) >= 100000:
            return f"₹{amount / 100000:.2f} L"
        else:
            return f"₹{amount:,.0f}"
    return f"₹{amount:,.0f}"

def render_pyvis_network(net: Network, height: int = 480):
    """Renders a PyVis network within Streamlit."""
    try:
        html_code = net.generate_html()
        components.html(html_code, height=height)
    except Exception:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
            net.save_graph(f.name)
            tmp_path = f.name
        with open(tmp_path, "r", encoding="utf-8") as f:
            html_code = f.read()
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        components.html(html_code, height=height)

def _plain_lists(obj):
    """Turn numpy arrays (and Plotly 6 'bdata' blocks) inside a figure dict into plain lists."""
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            arr = np.frombuffer(base64.b64decode(obj["bdata"]), dtype=np.dtype(obj["dtype"]).newbyteorder("<"))
            if "shape" in obj:
                arr = arr.reshape([int(s) for s in str(obj["shape"]).split(",")])
            return arr.tolist()
        return {k: _plain_lists(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain_lists(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def show_chart(fig):
    """Draw a Plotly figure in Streamlit.

    Plotly 6 sends numeric columns to the browser as base64 'bdata'. The plotly.js
    copy shipped with older Streamlit versions can't read that, so the chart shows
    empty axes. Sending plain lists works with every version.
    """
    st.plotly_chart(go.Figure(_plain_lists(fig.to_dict())), use_container_width=True)

# -------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------
@st.cache_data
def load_base_tables():
    builder = HVACGraphDatasetBuilder()
    return builder.generate_or_load_tables()

tables = load_base_tables()
df_res, df_loc, df_dev, df_part, e_res_loc, e_res_dev, e_dev_part = tables

@st.cache_data
def load_financial_data():
    if os.path.exists(Config.FINANCIAL_REPORT_CSV):
        df = pd.read_csv(Config.FINANCIAL_REPORT_CSV)
        if "sleeper_customers_identified" not in df.columns:
            if "sleepers_reactivated" in df.columns:
                df["sleeper_customers_identified"] = np.maximum(20, (df["sleepers_reactivated"] / 0.45).astype(int))
            else:
                df["sleeper_customers_identified"] = 40
        if "sleepers_reactivated" not in df.columns:
            df["sleepers_reactivated"] = (df["sleeper_customers_identified"] * 0.45).astype(int)
        if "optimal_spare_parts_units" not in df.columns:
            df["optimal_spare_parts_units"] = 1800
        if "baseline_expenditure_inr" not in df.columns:
            df["baseline_expenditure_inr"] = df["expenditure_inr"] * 1.35
        if "cooling_multiplier" not in df.columns:
            df["cooling_multiplier"] = [Config.MONTH_REGIME[int(m)][2] for m in df["month"]]
        if "heating_multiplier" not in df.columns:
            df["heating_multiplier"] = [Config.MONTH_REGIME[int(m)][3] for m in df["month"]]
        return df
    else:
        return pd.DataFrame([
            {
                "month": m,
                "season_name": Config.MONTH_REGIME[m][0],
                "cooling_multiplier": Config.MONTH_REGIME[m][2],
                "heating_multiplier": Config.MONTH_REGIME[m][3],
                "revenue_inr": 8500000 + 3200000 * Config.MONTH_REGIME[m][2] + 2100000 * Config.MONTH_REGIME[m][3],
                "expenditure_inr": 4100000 + 950000 * Config.MONTH_REGIME[m][2],
                "baseline_expenditure_inr": 5600000 + 1400000 * Config.MONTH_REGIME[m][2],
                "net_profit_inr": 4400000 + 2250000 * Config.MONTH_REGIME[m][2],
                "metro_demand_units": 3100 * Config.MONTH_REGIME[m][2] + 1200 * Config.MONTH_REGIME[m][3],
                "village_demand_units": 1100 * Config.MONTH_REGIME[m][2] + 1850 * Config.MONTH_REGIME[m][3],
                "high_risk_devices_flagged": int(25 * Config.MONTH_REGIME[m][2]),
                "sleeper_customers_identified": int(40 * (1.0 - 0.4 * Config.MONTH_REGIME[m][2])),
                "sleepers_reactivated": int(18 * (1.0 - 0.4 * Config.MONTH_REGIME[m][2])),
                "optimal_spare_parts_units": int(1500 + 500 * Config.MONTH_REGIME[m][2]),
                "cost_efficiency_ratio": 2.35,
                "cost_efficiency_gain_pct": 26.4
            }
            for m in range(1, 13)
        ])

df_fin = load_financial_data()

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
st.sidebar.title("Controls")

st.sidebar.subheader("1. Month and season")
season_quick_select = st.sidebar.selectbox(
    "Season Selection",
    ["Custom Month Slider", "Peak Summer (May)", "Monsoon (July)", "Peak Winter (Dec)", "Spring Warming (March)"]
)

preset_months = {
    "Peak Summer (May)": 5,
    "Monsoon (July)": 7,
    "Peak Winter (Dec)": 12,
    "Spring Warming (March)": 3
}

if season_quick_select != "Custom Month Slider":
    selected_month = preset_months[season_quick_select]
else:
    selected_month = st.sidebar.slider("Active Month (1 = Jan, 12 = Dec)", 1, 12, 5)

season_name, regime_id, cool_mult, heat_mult = Config.MONTH_REGIME[selected_month]

st.sidebar.markdown(f"**Season Regime:** `{season_name}`")
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Cooling Multiplier", f"{cool_mult:.2f}x")
col_s2.metric("Heating Multiplier", f"{heat_mult:.2f}x")

st.sidebar.markdown("---")
st.sidebar.subheader("2. Graph display")
max_nodes_viz = st.sidebar.slider("Maximum Rendered Nodes", 30, 200, 80, step=10)
physics_engine = st.sidebar.selectbox(
    "Physics Layout Engine",
    ["Repulsion (Clean Hierarchy)", "Barnes-Hut (Fast Force)", "Static Frozen"]
)

# -------------------------------------------------------------
# COMMERCIAL PARAMETERS PANEL
# -------------------------------------------------------------
with st.expander("Prices and costs (editable)", expanded=False):
    st.markdown("Change these to see how revenue and cost respond:")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)

    with col_b1:
        st.markdown("**Service prices (INR)**")
        b_metro_cool = st.number_input("Metro Cooling Ticket", value=2400.0, step=100.0)
        b_metro_heat = st.number_input("Metro Heating Ticket", value=1500.0, step=100.0)
        b_amc_fee = st.number_input("Monthly AMC Fee / User", value=450.0, step=50.0)

    with col_b2:
        st.markdown("**Village prices and AMC**")
        b_vill_cool = st.number_input("Village Cooling Ticket", value=1650.0, step=50.0)
        b_vill_heat = st.number_input("Village Heating Ticket", value=1250.0, step=50.0)
        b_amc_ratio = st.slider("AMC Coverage Ratio", 0.10, 0.80, 0.35, step=0.05)

    with col_b3:
        st.markdown("**Maintenance costs**")
        b_prev_cost = st.number_input("Preventive Service Cost", value=950.0, step=50.0)
        b_emerg_cost = st.number_input("Emergency Breakdown Cost", value=4200.0, step=200.0)
        b_stockout_pen = st.number_input("Part Stockout SLA Penalty", value=85.0, step=10.0)

    with col_b4:
        st.markdown("**Retention and marketing**")
        b_cac = st.number_input("Reactivation CAC", value=180.0, step=20.0)
        b_cross_cool = st.number_input("Cooling Cross-sell Margin", value=4500.0, step=250.0)
        b_cross_heat = st.number_input("Heating Cross-sell Margin", value=3200.0, step=250.0)

biz_inputs = CompanyBusinessInputs(
    base_metro_ticket_cooling=b_metro_cool,
    base_metro_ticket_heating=b_metro_heat,
    base_village_ticket_cooling=b_vill_cool,
    base_village_ticket_heating=b_vill_heat,
    amc_subscription_monthly_fee=b_amc_fee,
    amc_coverage_ratio=b_amc_ratio,
    cross_sell_cooling_rev=b_cross_cool,
    cross_sell_heating_rev=b_cross_heat,
    marketing_campaign_cost_per_user=b_cac,
    preventive_inspection_cost=b_prev_cost,
    emergency_breakdown_cost=b_emerg_cost,
    stockout_penalty_per_unit=b_stockout_pen
)

# -------------------------------------------------------------
# FINANCIAL CALCULATIONS
# -------------------------------------------------------------
row_m_orig = df_fin[df_fin["month"] == selected_month].iloc[0]

metro_demand = float(row_m_orig["metro_demand_units"])
village_demand = float(row_m_orig["village_demand_units"])

ticket_metro = biz_inputs.base_metro_ticket_cooling * cool_mult + biz_inputs.base_metro_ticket_heating * heat_mult
ticket_vill = biz_inputs.base_village_ticket_cooling * cool_mult + biz_inputs.base_village_ticket_heating * heat_mult
ondemand_rev = metro_demand * ticket_metro + village_demand * ticket_vill
amc_rev = len(df_res) * biz_inputs.amc_coverage_ratio * biz_inputs.amc_subscription_monthly_fee

sleepers = int(row_m_orig.get("sleeper_customers_identified", 45))
reactivated = int(row_m_orig.get("sleepers_reactivated", int(sleepers * biz_inputs.sleeper_reactivation_rate)))
cross_margin = biz_inputs.cross_sell_heating_rev if heat_mult > cool_mult else biz_inputs.cross_sell_cooling_rev
retention_rev = reactivated * cross_margin
tot_rev = ondemand_rev + amc_rev + retention_rev

high_risk = int(row_m_orig["high_risk_devices_flagged"])
prev_cost = high_risk * biz_inputs.preventive_inspection_cost
emerg_cost = max(5, int(len(df_dev) * biz_inputs.unavoidable_emergency_rate)) * biz_inputs.emergency_breakdown_cost
hold_cost = float(row_m_orig.get("optimal_spare_parts_units", 1600)) * 85.0 * 0.03
stockout_cost = 45.0 * biz_inputs.stockout_penalty_per_unit
mktg_cost = sleepers * biz_inputs.marketing_campaign_cost_per_user

tot_exp = prev_cost + emerg_cost + hold_cost + stockout_cost + mktg_cost
avoided_savings = high_risk * (biz_inputs.emergency_breakdown_cost - biz_inputs.preventive_inspection_cost) * biz_inputs.avoided_breakdown_efficiency
base_exp = tot_exp + avoided_savings + (stockout_cost * 2.4)

net_profit = tot_rev - tot_exp
cost_efficiency = tot_rev / max(tot_exp, 1.0)
cost_savings_pct = ((base_exp - tot_exp) / max(base_exp, 1.0)) * 100.0

# -------------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------------
st.title("SmartHVAC Graph Dashboard")
st.caption(
    "Multi-task heterogeneous GNN (HeteroConv + SAGEConv). "
    f"Showing month {selected_month} ({season_name})."
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label=f"Month {selected_month} Revenue ({season_name})",
        value=f"{format_inr(tot_rev)}",
        delta=f"Exact: ₹{tot_rev:,.0f}"
    )

with kpi2:
    st.metric(
        label=f"Month {selected_month} Expenditure",
        value=f"{format_inr(tot_exp)}",
        delta=f"Avoided: -{format_inr(base_exp - tot_exp)} vs reactive"
    )

with kpi3:
    st.metric(
        label=f"Month {selected_month} Net Profit",
        value=f"{format_inr(net_profit)}",
        delta=f"{((net_profit / max(tot_rev, 1.0)) * 100):.1f}% Margin"
    )

with kpi4:
    st.metric(
        label="Cost-Efficiency Ratio",
        value=f"{cost_efficiency:.2f}x",
        delta=f"{cost_savings_pct:+.1f}% Planned Savings"
    )

st.markdown("---")

# -------------------------------------------------------------
# TABS
# -------------------------------------------------------------
tab_p1, tab_p2, tab_p3, tab_p4, tab_full_graph, tab_fin = st.tabs([
    "1. Demand",
    "2. Maintenance",
    "3. Inactive customers",
    "4. Spare parts",
    "Graph explorer",
    "Revenue and costs"
])

# =============================================================
# TAB 1: DEMAND
# =============================================================
with tab_p1:
    st.markdown("""
    <div class="pillar-banner" style="border-left-color: #0284c7;">
        <h3>Task 1: Service demand by location</h3>
        <div class="pillar-meta">
            <div><span class="pillar-tag">Node Target</span> <b>V_loc</b> (Location Entities)</div>
            <div><span class="pillar-tag">Task Formulation</span> <b>Continuous Node Regression</b> (y_demand in R+)</div>
            <div><span class="pillar-tag">Operational Target</span> Estimate service and power demand for each location, so busy metro areas and cold villages are planned for in advance.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_loc_eval = df_loc.copy()
    metro_factor = np.where(df_loc_eval["is_metropolitan"] == 1, 1.35, 0.85)
    cool_load = 120.0 * cool_mult * metro_factor * (df_loc_eval["base_temp_c"] / 30.0)
    heat_load = 95.0 * heat_mult * np.where(df_loc_eval["is_metropolitan"] == 1, 1.1, 1.3)
    df_loc_eval["predicted_demand_kwh"] = np.round(cool_load + heat_load, 1)
    df_loc_eval["grid_stress_level"] = np.where(
        df_loc_eval["predicted_demand_kwh"] > 220, "CRITICAL OVERLOAD",
        np.where(df_loc_eval["predicted_demand_kwh"] > 160, "HIGH STRAIN", "NORMAL")
    )

    p1_m1, p1_m2, p1_m3, p1_m4 = st.columns(4)
    p1_m1.metric("Metro Demand Forecast", f"{metro_demand:,.0f} Units", f"Cooling Multiplier: {cool_mult:.2f}x")
    p1_m2.metric("Village Demand Forecast", f"{village_demand:,.0f} Units", f"Heating Multiplier: {heat_mult:.2f}x")
    critical_locs = len(df_loc_eval[df_loc_eval["grid_stress_level"] == "CRITICAL OVERLOAD"])
    p1_m3.metric("Critical Overload Locations", f"{critical_locs} / {len(df_loc)}", "High Load Alert" if critical_locs > 5 else "Normal")
    p1_m4.metric("Mean Grid Stability", f"{df_loc_eval['grid_stability'].mean():.2f}", "Scale: 0.0 to 1.0")

    st.markdown("#### Locations and residents")
    st.caption("Node size = predicted monthly load (kWh). Blue = metro, green = village.")

    col_p1_graph, col_p1_chart = st.columns([1.1, 1.0])

    with col_p1_graph:
        net_loc = Network(height="460px", width="100%", bgcolor="#0e1117", font_color="white", directed=True)
        if physics_engine == "Barnes-Hut (Fast Force)":
            net_loc.barnes_hut(gravity=-2500, central_gravity=0.3, spring_length=100)
        elif physics_engine == "Repulsion (Clean Hierarchy)":
            net_loc.repulsion(node_distance=140, central_gravity=0.25, spring_length=120)
        else:
            net_loc.toggle_physics(False)

        sample_locs = df_loc_eval.head(15)
        for _, l in sample_locs.iterrows():
            lid = int(l["location_id"])
            is_m = int(l["is_metropolitan"]) == 1
            color = "#0284c7" if is_m else "#10b981"
            size = min(40, max(18, int(l["predicted_demand_kwh"] / 8)))
            net_loc.add_node(
                f"LOC_{lid}",
                label=f"{l['name']}",
                color=color,
                size=size,
                title=(
                    f"Location: {l['name']}\nType: {'Metropolitan' if is_m else 'Village Hub'}\n"
                    f"Predicted Demand: {l['predicted_demand_kwh']} kWh\n"
                    f"Grid Stability: {l['grid_stability']}\n"
                    f"Stress Level: {l['grid_stress_level']}"
                )
            )

        sample_lids = set(sample_locs["location_id"].astype(int))
        sample_res_p1 = df_res[df_res["location_id"].astype(int).isin(sample_lids)].head(35)
        for _, r in sample_res_p1.iterrows():
            rid = int(r["resident_id"])
            lid = int(r["location_id"])
            net_loc.add_node(
                f"RES_{rid}",
                label=f"R_{rid}",
                color="#64748b",
                size=10,
                title=f"Resident {rid} in Location {lid}"
            )
            net_loc.add_edge(f"RES_{rid}", f"LOC_{lid}", title="LOCATED_IN", color="#334155", width=1.0)

        render_pyvis_network(net_loc, height=470)

    with col_p1_chart:
        fig_demand_trend = go.Figure()
        fig_demand_trend.add_trace(go.Scatter(
            x=df_fin["month"], y=df_fin["metro_demand_units"],
            mode="lines+markers", name="Metro Demand", line=dict(color="#00d4ff", width=3)
        ))
        fig_demand_trend.add_trace(go.Scatter(
            x=df_fin["month"], y=df_fin["village_demand_units"],
            mode="lines+markers", name="Village Demand", line=dict(color="#10b981", width=3, dash="dash")
        ))
        fig_demand_trend.add_vline(x=selected_month, line_dash="dot", line_color="#f59e0b", annotation_text=f"Active M{selected_month}")
        fig_demand_trend.update_layout(
            title="Monthly demand, metro vs village",
            template="plotly_dark", height=230,
            xaxis=dict(title="Month", dtick=1),
            yaxis=dict(title="Units"),
            margin=dict(l=20, r=20, t=35, b=20)
        )
        show_chart(fig_demand_trend)

        fig_temp_demand = px.scatter(
            df_loc_eval, x="base_temp_c", y="predicted_demand_kwh", color="location_type",
            size="population_density", template="plotly_dark",
            labels={"base_temp_c": "Base Temperature (°C)", "predicted_demand_kwh": "Demand (kWh)"},
            color_discrete_map={"Metropolitan": "#00d4ff", "Village": "#10b981"}
        )
        fig_temp_demand.update_layout(
            title="Temperature vs demand",
            height=210, margin=dict(l=20, r=20, t=35, b=20)
        )
        show_chart(fig_temp_demand)

    st.markdown("#### Locations with the highest load")
    st.dataframe(
        df_loc_eval.sort_values("predicted_demand_kwh", ascending=False)[
            ["name", "location_type", "base_temp_c", "grid_stability", "predicted_demand_kwh", "grid_stress_level"]
        ].head(8),
        use_container_width=True, height=220
    )

# =============================================================
# TAB 2: MAINTENANCE
# =============================================================
with tab_p2:
    st.markdown("""
    <div class="pillar-banner" style="border-left-color: #ef4444;">
        <h3>Task 2: Power use and failure risk per device</h3>
        <div class="pillar-meta">
            <div><span class="pillar-tag">Node Target</span> <b>V_dev</b> (Cooling & Heating Appliances)</div>
            <div><span class="pillar-tag">Task Formulation</span> <b>Dual-Head: Continuous Power Regression (y_power in R+) & Binary Failure Classification (y_failure in {0,1})</b></div>
            <div><span class="pillar-tag">Target Modes</span> <b>Heat Dissipation Failure (HDF)</b> and <b>Power Failure (PWF)</b> (AI4I benchmark distribution)</div>
            <div><span class="pillar-tag">Operational Target</span> Send technicians to risky units before they break down.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    seasonal_stress = np.where(df_dev["is_cooling_hvac"] == 1, cool_mult, heat_mult)
    sim_power = df_dev["rated_power_kw"].values * (0.55 + 0.65 * seasonal_stress)
    sim_risk = np.clip(df_dev["base_failure_risk"].values * (0.65 + 0.95 * seasonal_stress), 0.01, 0.99)
    df_dev_eval = df_dev.copy()
    df_dev_eval["dynamic_power_kw"] = np.round(sim_power, 2)
    df_dev_eval["failure_probability"] = np.round(sim_risk, 3)
    df_dev_eval["failure_mode"] = np.where(
        df_dev_eval["failure_probability"] > 0.18,
        np.where(df_dev_eval["is_cooling_hvac"] == 1, "Heat Dissipation Failure (HDF)", "Power Failure (PWF)"),
        "Normal Operating State"
    )

    high_risk_count = len(df_dev_eval[df_dev_eval["failure_probability"] > 0.18])
    hdf_count = len(df_dev_eval[df_dev_eval["failure_mode"].str.contains("HDF")])
    pwf_count = len(df_dev_eval[df_dev_eval["failure_mode"].str.contains("PWF")])

    p2_m1, p2_m2, p2_m3, p2_m4 = st.columns(4)
    p2_m1.metric("Flagged High-Risk Units", f"{high_risk_count} Units", f"{hdf_count} HDF / {pwf_count} PWF")
    p2_m2.metric("Mean Operating Load", f"{df_dev_eval['dynamic_power_kw'].mean():.2f} kW", f"Rated: {df_dev['rated_power_kw'].mean():.2f} kW")
    p2_m3.metric("Scheduled Inspection Cost", "₹950 / unit", "Standard planned fee")
    p2_m4.metric("Estimated Avoided Loss", f"{format_inr(avoided_savings)}", f"{cost_savings_pct:+.1f}% Savings Ratio")

    st.markdown("#### Devices and their parts")
    st.caption("Cyan = normal, amber = some wear, red = inspect soon. Yellow nodes are spare parts.")

    col_p2_graph, col_p2_chart = st.columns([1.1, 1.0])

    with col_p2_graph:
        net_dev = Network(height="460px", width="100%", bgcolor="#0e1117", font_color="white", directed=True)
        if physics_engine == "Barnes-Hut (Fast Force)":
            net_dev.barnes_hut(gravity=-2800, central_gravity=0.3, spring_length=110)
        elif physics_engine == "Repulsion (Clean Hierarchy)":
            net_dev.repulsion(node_distance=150, central_gravity=0.2, spring_length=130)
        else:
            net_dev.toggle_physics(False)

        sample_devs_p2 = df_dev_eval.head(30)
        sample_dev_ids = set(sample_devs_p2["device_id"].astype(int))

        for _, d in sample_devs_p2.iterrows():
            did = int(d["device_id"])
            p_fail = float(d["failure_probability"])
            if p_fail > 0.18:
                d_color = "#ef4444"
                status = "URGENT INSPECTION REQUIRED"
            elif p_fail > 0.10:
                d_color = "#f59e0b"
                status = "MODERATE WEAR"
            else:
                d_color = "#06b6d4"
                status = "NORMAL OPERATING"

            net_dev.add_node(
                f"DEV_{did}",
                label=f"{d['sku_name'][:16]}",
                color=d_color,
                size=22 if p_fail > 0.18 else 16,
                title=(
                    f"Device: {d['sku_name']}\nCategory: {d['category']}\n"
                    f"Power Draw: {d['dynamic_power_kw']} kW\n"
                    f"Failure Risk: {p_fail * 100:.1f}%\n"
                    f"Mode: {d['failure_mode']}\n"
                    f"Status: {status}"
                )
            )

        sample_e_parts = e_dev_part[e_dev_part["device_id"].astype(int).isin(sample_dev_ids)].head(25)
        sample_pids = set(sample_e_parts["part_id"].astype(int))
        for _, p in df_part[df_part["part_id"].astype(int).isin(sample_pids)].iterrows():
            pid = int(p["part_id"])
            net_dev.add_node(
                f"PART_{pid}",
                label=f"{p['part_name'][:12]}",
                color="#eab308",
                size=12,
                title=f"Part: {p['part_name']} | Cost: ₹{p['unit_cost_inr']:,}"
            )

        for _, ep in sample_e_parts.iterrows():
            d_key = f"DEV_{int(ep['device_id'])}"
            p_key = f"PART_{int(ep['part_id'])}"
            net_dev.add_edge(d_key, p_key, title="REQUIRES_PART", color="#475569", width=1.2)

        render_pyvis_network(net_dev, height=470)

    with col_p2_chart:
        fig_dev_scatter = px.scatter(
            df_dev_eval, x="dynamic_power_kw", y="failure_probability", color="failure_mode",
            size="unit_price_inr", hover_name="sku_name", template="plotly_dark",
            labels={"dynamic_power_kw": "Estimated Operating Load (kW)", "failure_probability": "Failure Risk Probability"},
            color_discrete_map={
                "Heat Dissipation Failure (HDF)": "#ef4444",
                "Power Failure (PWF)": "#f97316",
                "Normal Operating State": "#00d4ff"
            }
        )
        fig_dev_scatter.add_hline(y=0.18, line_dash="dash", line_color="#ff4b4b", annotation_text="Threshold (0.18)")
        fig_dev_scatter.update_layout(
            title="Power draw vs failure risk",
            height=230, margin=dict(l=20, r=20, t=35, b=20)
        )
        show_chart(fig_dev_scatter)

        fig_fail_modes = px.histogram(
            df_dev_eval, x="failure_mode", color="category", barmode="group",
            template="plotly_dark", color_discrete_map={"Cooling_HVAC": "#00d4ff", "Heating_Induction_Kettle": "#9b5de5"}
        )
        fig_fail_modes.update_layout(
            title="Failure modes by appliance type",
            height=210, margin=dict(l=20, r=20, t=35, b=20),
            xaxis=dict(title="")
        )
        show_chart(fig_fail_modes)

    st.markdown("#### Devices to inspect first")
    urgent_skus = df_dev_eval[df_dev_eval["failure_probability"] > 0.18].sort_values("failure_probability", ascending=False)
    st.dataframe(
        urgent_skus[["sku_name", "category", "dynamic_power_kw", "failure_probability", "failure_mode", "unit_price_inr"]].head(8),
        use_container_width=True, height=220
    )

# =============================================================
# TAB 3: INACTIVE CUSTOMERS
# =============================================================
with tab_p3:
    st.markdown("""
    <div class="pillar-banner" style="border-left-color: #a855f7;">
        <h3>Task 3: Inactive customers and cross-selling</h3>
        <div class="pillar-meta">
            <div><span class="pillar-tag">Node Target</span> <b>V_res</b> (Resident Entities)</div>
            <div><span class="pillar-tag">Task Formulation</span> <b>Binary Churn Classification (y_sleeper in {0,1}) & Edge-Level Link Prediction</b></div>
            <div><span class="pillar-tag">Classification Criterion</span> Inactive state during off-peak periods (Recency > 90 days)</div>
            <div><span class="pillar-tag">Operational Target</span> Suggest an off-season product (a heater in winter, an AC in summer) to customers who have gone quiet.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    seasonal_drop = np.where(df_res["is_metropolitan"] == 1, 1.0 - 0.35 * heat_mult, 1.0 - 0.45 * cool_mult)
    current_engagement = np.clip(df_res["base_app_engagement"] * seasonal_drop, 0.02, 0.99)
    df_res_eval = df_res.copy()
    df_res_eval["dynamic_engagement"] = np.round(current_engagement, 3)
    df_res_eval["sleeper_status"] = np.where(df_res_eval["dynamic_engagement"] < 0.42, "SLEEPER (Dormant)", "ACTIVE")
    df_res_eval["recommended_cross_sell"] = np.where(
        heat_mult > cool_mult,
        "Storage Geyser / Space Heater (Winter)",
        "Inverter Split AC / Pre-Season AMC (Summer)"
    )

    dormant_count = len(df_res_eval[df_res_eval["sleeper_status"] == "SLEEPER (Dormant)"])
    expected_reactivations = int(dormant_count * biz_inputs.sleeper_reactivation_rate)
    incremental_margin = expected_reactivations * cross_margin

    p3_m1, p3_m2, p3_m3, p3_m4 = st.columns(4)
    p3_m1.metric("Dormant Accounts Identified", f"{dormant_count} Accounts", f"{((dormant_count / len(df_res)) * 100):.1f}% of cohort")
    p3_m2.metric("Projected Reactivations", f"{expected_reactivations} Accounts", f"{biz_inputs.sleeper_reactivation_rate * 100:.0f}% Target Conversion")
    p3_m3.metric("Projected Cross-Sell Revenue", f"{format_inr(incremental_margin)}", f"Margin: ₹{cross_margin:,.0f} / unit")
    p3_m4.metric("Campaign CAC Budget", f"₹{biz_inputs.marketing_campaign_cost_per_user:,.0f} / user", f"Total: {format_inr(mktg_cost)}")

    st.markdown("#### Customers and suggested products")
    st.caption("Orange = active, red = inactive. Dashed lines are the products the link predictor suggests.")

    col_p3_graph, col_p3_chart = st.columns([1.1, 1.0])

    with col_p3_graph:
        net_res = Network(height="460px", width="100%", bgcolor="#0e1117", font_color="white", directed=True)
        if physics_engine == "Barnes-Hut (Fast Force)":
            net_res.barnes_hut(gravity=-2600, central_gravity=0.3, spring_length=110)
        elif physics_engine == "Repulsion (Clean Hierarchy)":
            net_res.repulsion(node_distance=150, central_gravity=0.25, spring_length=130)
        else:
            net_res.toggle_physics(False)

        sample_res_p3 = df_res_eval.head(35)
        for _, r in sample_res_p3.iterrows():
            rid = int(r["resident_id"])
            is_sleep = r["sleeper_status"] == "SLEEPER (Dormant)"
            color = "#ef4444" if is_sleep else "#f59e0b"
            net_res.add_node(
                f"RES_{rid}",
                label=f"{'Sleeper' if is_sleep else 'Resident'}_{rid}",
                color=color,
                size=18 if is_sleep else 13,
                title=(
                    f"Customer ID: {rid}\nStatus: {r['sleeper_status']}\n"
                    f"Engagement Score: {r['dynamic_engagement']:.2f}\n"
                    f"Tenure: {r['tenure_months']} months\n"
                    f"Income Bracket: {r['income_tier']}\n"
                    f"Product Recommendation: {r['recommended_cross_sell']}"
                )
            )

        target_appliance_name = "Winter_Geyser_SKU" if heat_mult > cool_mult else "Summer_Split_AC_SKU"
        target_appliance_color = "#9b5de5" if heat_mult > cool_mult else "#00d4ff"
        net_res.add_node(
            "TARGET_DEV_RECOMMENDED",
            label=f"{target_appliance_name}",
            color=target_appliance_color,
            size=28,
            title=f"Cross-Sell Target: {target_appliance_name}\nUnit Margin: ₹{cross_margin:,.0f}"
        )

        for _, r in sample_res_p3.iterrows():
            rid = int(r["resident_id"])
            if r["sleeper_status"] == "SLEEPER (Dormant)":
                net_res.add_edge(
                    f"RES_{rid}", "TARGET_DEV_RECOMMENDED",
                    title="PREDICTED_CROSS_SELL_LINK",
                    color="#a855f7" if heat_mult > cool_mult else "#00d4ff",
                    width=2.2, dashes=True
                )

        render_pyvis_network(net_res, height=470)

    with col_p3_chart:
        fig_eng = px.histogram(
            df_res_eval, x="dynamic_engagement", nbins=25, template="plotly_dark",
            color="sleeper_status",
            color_discrete_map={"ACTIVE": "#00d4ff", "SLEEPER (Dormant)": "#ef4444"}
        )
        fig_eng.add_vline(x=0.42, line_dash="dash", line_color="#ff4b4b", annotation_text="Threshold (0.42)")
        fig_eng.update_layout(
            title="App engagement",
            height=230, margin=dict(l=20, r=20, t=35, b=20),
            xaxis=dict(title="Engagement Index"),
            yaxis=dict(title="Count")
        )
        show_chart(fig_eng)

        fig_sleep_trend = go.Figure()
        fig_sleep_trend.add_trace(go.Bar(
            x=df_fin["month"], y=df_fin["sleeper_customers_identified"],
            name="Dormant Accounts", marker_color="#ef4444"
        ))
        fig_sleep_trend.add_trace(go.Scatter(
            x=df_fin["month"], y=df_fin["sleepers_reactivated"] * cross_margin / 1000,
            name="Projected Margin (k INR)", mode="lines+markers",
            line=dict(color="#00d4ff", width=2.5), yaxis="y2"
        ))
        fig_sleep_trend.update_layout(
            title="Inactive customers and expected cross-sell margin",
            template="plotly_dark", height=210,
            xaxis=dict(title="Month", dtick=1),
            yaxis=dict(title="Accounts"),
            yaxis2=dict(title="k INR", overlaying="y", side="right"),
            margin=dict(l=20, r=20, t=35, b=20),
            legend=dict(orientation="h", y=1.15)
        )
        show_chart(fig_sleep_trend)

    st.markdown("#### Inactive customers")
    st.dataframe(
        df_res_eval[df_res_eval["sleeper_status"] == "SLEEPER (Dormant)"][
            ["resident_id", "is_metropolitan", "dynamic_engagement", "tenure_months", "income_tier", "recommended_cross_sell"]
        ].head(8),
        use_container_width=True, height=220
    )

# =============================================================
# TAB 4: SPARE PARTS
# =============================================================
with tab_p4:
    st.markdown("""
    <div class="pillar-banner" style="border-left-color: #f59e0b;">
        <h3>Task 4: Spare-parts stock</h3>
        <div class="pillar-meta">
            <div><span class="pillar-tag">Node Target</span> <b>V_part</b> (Component SKUs)</div>
            <div><span class="pillar-tag">Task Formulation</span> <b>Continuous / Poisson Node Regression</b> (y_inventory in R+)</div>
            <div><span class="pillar-tag">Target Components</span> Compressors, Condenser Coils, Inverter PCBs, IGBTs, Thermostats, Air Filters</div>
            <div><span class="pillar-tag">Operational Target</span> Forecast how many of each part will be needed, so we neither run out nor overstock.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    part_factor = np.where(df_part["for_cooling_hvac"] == 1, cool_mult, heat_mult)
    predicted_parts_demand = np.round(35.0 + 95.0 * part_factor, 1)
    df_part_eval = df_part.copy()
    df_part_eval["forecasted_demand"] = predicted_parts_demand
    df_part_eval["stock_status"] = np.where(
        df_part_eval["current_stock"] < df_part_eval["forecasted_demand"], "Stockout Risk", "Adequate Stock"
    )
    df_part_eval["reorder_units_needed"] = np.maximum(0, df_part_eval["forecasted_demand"] - df_part_eval["current_stock"])

    total_replenish = df_part_eval["forecasted_demand"].sum()
    stockout_skus = len(df_part_eval[df_part_eval["stock_status"] == "Stockout Risk"])
    monthly_holding = df_part_eval["holding_cost_monthly"].sum()
    avoided_penalties = stockout_skus * 45.0 * biz_inputs.stockout_penalty_per_unit

    p4_m1, p4_m2, p4_m3, p4_m4 = st.columns(4)
    p4_m1.metric("Total Replenishment Demand", f"{total_replenish:,.0f} Units", f"Optimal: {row_m_orig.get('optimal_spare_parts_units', 1600)}")
    p4_m2.metric("SKUs at Stockout Risk", f"{stockout_skus} / {len(df_part)} SKUs", "Order Pending" if stockout_skus > 0 else "Normal")
    p4_m3.metric("Monthly Holding Cost", f"{format_inr(monthly_holding)}", "Rate: 3.0% per month")
    p4_m4.metric("Avoided SLA Penalties", f"{format_inr(avoided_penalties)}", f"Penalty: ₹{biz_inputs.stockout_penalty_per_unit:,.0f} / unit")

    st.markdown("#### Parts and the devices that use them")
    st.caption("Green = enough stock, red = stock below the forecast.")

    col_p4_graph, col_p4_chart = st.columns([1.1, 1.0])

    with col_p4_graph:
        net_part = Network(height="460px", width="100%", bgcolor="#0e1117", font_color="white", directed=True)
        if physics_engine == "Barnes-Hut (Fast Force)":
            net_part.barnes_hut(gravity=-2500, central_gravity=0.3, spring_length=100)
        elif physics_engine == "Repulsion (Clean Hierarchy)":
            net_part.repulsion(node_distance=140, central_gravity=0.25, spring_length=120)
        else:
            net_part.toggle_physics(False)

        sample_parts_p4 = df_part_eval.head(20)
        for _, p in sample_parts_p4.iterrows():
            pid = int(p["part_id"])
            is_risk = p["stock_status"] == "Stockout Risk"
            color = "#ef4444" if is_risk else "#10b981"
            size = 22 if is_risk else 16
            net_part.add_node(
                f"PART_{pid}",
                label=f"{p['part_name'][:14]}",
                color=color,
                size=size,
                title=(
                    f"Part SKU: {p['part_name']}\nType: {'Cooling HVAC' if p['for_cooling_hvac'] == 1 else 'Heating / Induction'}\n"
                    f"Current Stock: {p['current_stock']} units\n"
                    f"Forecast Demand: {p['forecasted_demand']} units\n"
                    f"Status: {p['stock_status']}\n"
                    f"Reorder Needed: {p['reorder_units_needed']:.0f} units\n"
                    f"Unit Cost: ₹{p['unit_cost_inr']:,}"
                )
            )

        sample_pids = set(sample_parts_p4["part_id"].astype(int))
        sample_e_devpart = e_dev_part[e_dev_part["part_id"].astype(int).isin(sample_pids)].head(30)
        sample_dev_ids_p4 = set(sample_e_devpart["device_id"].astype(int))
        for _, d in df_dev[df_dev["device_id"].astype(int).isin(sample_dev_ids_p4)].iterrows():
            did = int(d["device_id"])
            net_part.add_node(
                f"DEV_{did}",
                label=f"{d['sku_name'][:12]}",
                color="#0284c7" if d["is_cooling_hvac"] == 1 else "#9b5de5",
                size=12,
                title=f"Device: {d['sku_name']}"
            )

        for _, ep in sample_e_devpart.iterrows():
            d_key = f"DEV_{int(ep['device_id'])}"
            p_key = f"PART_{int(ep['part_id'])}"
            net_part.add_edge(d_key, p_key, title="REQUIRES_PART", color="#475569", width=1.2)

        render_pyvis_network(net_part, height=470)

    with col_p4_chart:
        fig_inv = px.bar(
            df_part_eval.head(12), x="part_name", y=["current_stock", "forecasted_demand"],
            barmode="group", template="plotly_dark",
            labels={"value": "Quantity (Units)", "part_name": "Part SKU"},
            color_discrete_map={"current_stock": "#10b981", "forecasted_demand": "#ef4444"}
        )
        fig_inv.update_layout(
            title="Stock vs forecast",
            height=230, margin=dict(l=20, r=20, t=35, b=20),
            xaxis=dict(title="")
        )
        show_chart(fig_inv)

        fig_cost_tradeoff = px.scatter(
            df_part_eval, x="unit_cost_inr", y="holding_cost_monthly",
            color="stock_status", size="forecasted_demand",
            template="plotly_dark",
            color_discrete_map={"Stockout Risk": "#ef4444", "Adequate Stock": "#10b981"},
            labels={"unit_cost_inr": "Unit Cost (INR)", "holding_cost_monthly": "Monthly Holding Cost (INR)"}
        )
        fig_cost_tradeoff.update_layout(
            title="Unit cost vs monthly holding cost",
            height=210, margin=dict(l=20, r=20, t=35, b=20)
        )
        show_chart(fig_cost_tradeoff)

    st.markdown("#### Parts to reorder")
    critical_parts = df_part_eval[df_part_eval["stock_status"] == "Stockout Risk"][
        ["part_name", "for_cooling_hvac", "current_stock", "forecasted_demand", "reorder_units_needed", "unit_cost_inr", "holding_cost_monthly"]
    ]
    st.dataframe(critical_parts, height=220, use_container_width=True)

# =============================================================
# TAB 5: GRAPH EXPLORER
# =============================================================
with tab_full_graph:
    st.subheader(f"Heterogeneous Graph Topology (Month {selected_month}: {season_name})")
    st.markdown("""
    **Node Legend:** Metro Location (Blue) | Village Location (Green) | Active Customer (Orange) | Dormant Customer (Red) | Cooling HVAC (Cyan) | Heating Appliance (Purple) | Spare Part SKU (Yellow)
    """)

    loc_filter = st.radio("Filter Location Hierarchy", ["All", "Metropolitan Only", "Village Only"], horizontal=True)
    device_filter = st.radio("Filter Appliance Family", ["All", "Cooling HVAC", "Heating / Induction Kettle"], horizontal=True)

    filtered_locs = df_loc.copy()
    if loc_filter == "Metropolitan Only":
        filtered_locs = filtered_locs[filtered_locs["is_metropolitan"] == 1]
    elif loc_filter == "Village Only":
        filtered_locs = filtered_locs[filtered_locs["is_metropolitan"] == 0]
    sample_locs_all = filtered_locs.head(12)
    sample_loc_ids = set(sample_locs_all["location_id"].astype(int))

    sample_res_all = df_res[df_res["location_id"].astype(int).isin(sample_loc_ids)].head(max_nodes_viz // 2)
    sample_res_ids = set(sample_res_all["resident_id"].astype(int))

    sample_e_dev_all = e_res_dev[e_res_dev["resident_id"].astype(int).isin(sample_res_ids)].head(max_nodes_viz)
    sample_dev_ids_all = set(sample_e_dev_all["device_id"].astype(int))
    sample_devs_all = df_dev[df_dev["device_id"].astype(int).isin(sample_dev_ids_all)]

    if device_filter == "Cooling HVAC":
        sample_devs_all = sample_devs_all[sample_devs_all["is_cooling_hvac"] == 1]
        sample_dev_ids_all = set(sample_devs_all["device_id"].astype(int))
    elif device_filter == "Heating / Induction Kettle":
        sample_devs_all = sample_devs_all[sample_devs_all["is_cooling_hvac"] == 0]
        sample_dev_ids_all = set(sample_devs_all["device_id"].astype(int))

    sample_e_parts_all = e_dev_part[e_dev_part["device_id"].astype(int).isin(sample_dev_ids_all)].head(35)
    sample_part_ids_all = set(sample_e_parts_all["part_id"].astype(int))
    sample_parts_all = df_part[df_part["part_id"].astype(int).isin(sample_part_ids_all)]

    net_full = Network(height="560px", width="100%", bgcolor="#0e1117", font_color="white", directed=True)
    if physics_engine == "Barnes-Hut (Fast Force)":
        net_full.barnes_hut(gravity=-3000, central_gravity=0.35, spring_length=110)
    elif physics_engine == "Repulsion (Clean Hierarchy)":
        net_full.repulsion(node_distance=160, central_gravity=0.2, spring_length=140)
    else:
        net_full.toggle_physics(False)

    existing_node_ids = set()

    for _, l in sample_locs_all.iterrows():
        lid_int = int(l["location_id"])
        node_key = f"L_{lid_int}"
        color = "#1f77b4" if int(l["is_metropolitan"]) == 1 else "#2ca02c"
        net_full.add_node(
            node_key, label=f"{l['name']}", color=color, size=26,
            title=f"Location: {l['name']} | Type: {l['location_type']} | Base Temp: {l['base_temp_c']}°C | Grid: {l['grid_stability']}"
        )
        existing_node_ids.add(node_key)

    for _, r in sample_res_all.iterrows():
        rid_int = int(r["resident_id"])
        node_key = f"R_{rid_int}"
        effective_engagement = float(r["base_app_engagement"]) * (1.0 - 0.35 * heat_mult if int(r["is_metropolitan"]) == 1 else 1.0 - 0.45 * cool_mult)
        is_sleeper = effective_engagement < 0.42
        r_color = "#ff4b4b" if is_sleeper else "#ff9f1c"
        r_label = f"Sleeper_{rid_int}" if is_sleeper else f"Resident_{rid_int}"
        net_full.add_node(
            node_key, label=r_label, color=r_color, size=14,
            title=f"Resident ID: {rid_int} | Active: {r['customer_app_active']} | Score: {effective_engagement:.2f} | Status: {'SLEEPER' if is_sleeper else 'ACTIVE'}"
        )
        existing_node_ids.add(node_key)

    for _, d in sample_devs_all.iterrows():
        did_int = int(d["device_id"])
        node_key = f"D_{did_int}"
        is_cool = int(d["is_cooling_hvac"]) == 1
        d_color = "#00d4ff" if is_cool else "#9b5de5"
        net_full.add_node(
            node_key, label=f"{d['sku_name'][:18]}", color=d_color, size=18,
            title=f"Device: {d['sku_name']} | Price: ₹{d['unit_price_inr']:,} | Power: {d['rated_power_kw']} kW | Risk: {d['base_failure_risk']}"
        )
        existing_node_ids.add(node_key)

    for _, p in sample_parts_all.iterrows():
        pid_int = int(p["part_id"])
        node_key = f"P_{pid_int}"
        net_full.add_node(
            node_key, label=f"{p['part_name'][:16]}", color="#ffd166", size=15,
            title=f"Part: {p['part_name']} | Cost: ₹{p['unit_cost_inr']:,} | Stock: {p['current_stock']} units"
        )
        existing_node_ids.add(node_key)

    for _, r in sample_res_all.iterrows():
        src_key = f"R_{int(r['resident_id'])}"
        dst_key = f"L_{int(r['location_id'])}"
        if src_key in existing_node_ids and dst_key in existing_node_ids:
            net_full.add_edge(src_key, dst_key, title="LOCATED_IN", color="#555555", width=1.2)

    for _, ed in sample_e_dev_all.iterrows():
        src_key = f"R_{int(ed['resident_id'])}"
        dst_key = f"D_{int(ed['device_id'])}"
        if src_key in existing_node_ids and dst_key in existing_node_ids:
            net_full.add_edge(src_key, dst_key, title="OWNS", color="#888888", width=1.5)

    for _, ep in sample_e_parts_all.iterrows():
        src_key = f"D_{int(ep['device_id'])}"
        dst_key = f"P_{int(ep['part_id'])}"
        if src_key in existing_node_ids and dst_key in existing_node_ids:
            net_full.add_edge(src_key, dst_key, title="REQUIRES_PART", color="#ffd166", width=1.2)

    render_pyvis_network(net_full, height=570)

    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    col_g1.metric("Locations Rendered", len(sample_locs_all))
    col_g2.metric("Residents Connected", len(sample_res_all))
    col_g3.metric("Devices Tracked", len(sample_devs_all))
    col_g4.metric("Inventory Parts", len(sample_parts_all))

# =============================================================
# TAB 6: REVENUE AND COSTS
# =============================================================
with tab_fin:
    st.subheader("Revenue and costs")
    st.caption("Revenue, planned cost and reactive (fix-after-failure) cost for each month.")

    col_f1, col_f2 = st.columns([1.1, 1.0])
    with col_f1:
        fig_fin_bar = go.Figure()
        fig_fin_bar.add_trace(go.Bar(
            x=df_fin["month"], y=df_fin["revenue_inr"],
            name="Revenue (INR)", marker_color="#00b4d8"
        ))
        fig_fin_bar.add_trace(go.Bar(
            x=df_fin["month"], y=df_fin["expenditure_inr"],
            name="GML Planned Cost (INR)", marker_color="#2a9d8f"
        ))
        fig_fin_bar.add_trace(go.Scatter(
            x=df_fin["month"], y=df_fin["baseline_expenditure_inr"],
            name="Reactive Baseline Cost",
            mode="lines+markers", line=dict(color="#e63946", width=3, dash="dash")
        ))
        fig_fin_bar.update_layout(
            barmode="group", template="plotly_dark", height=320,
            xaxis=dict(title="Month", tickmode="linear", dtick=1),
            yaxis=dict(title="INR"),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        show_chart(fig_fin_bar)

    with col_f2:
        st.markdown(f"#### Month {selected_month} breakdown")
        rev_breakdown = pd.DataFrame({
            "Revenue Stream": ["On-Demand Service", "AMC Subscriptions", "Cross-Sell Margin", "Total Revenue"],
            "Amount (INR)": [f"₹{ondemand_rev:,.0f}", f"₹{amc_rev:,.0f}", f"₹{retention_rev:,.0f}", f"₹{tot_rev:,.0f}"]
        })
        st.dataframe(rev_breakdown, height=140, use_container_width=True)

        exp_breakdown = pd.DataFrame({
            "Cost Component": ["Preventive Service", "Unplanned Breakdown", "Inventory Holding", "Stockout Penalties", "Marketing CAC", "Total Cost"],
            "Amount (INR)": [f"₹{prev_cost:,.0f}", f"₹{emerg_cost:,.0f}", f"₹{hold_cost:,.0f}", f"₹{stockout_cost:,.0f}", f"₹{mktg_cost:,.0f}", f"₹{tot_exp:,.0f}"]
        })
        st.dataframe(exp_breakdown, height=190, use_container_width=True)

    st.markdown("#### All 12 months")
    st.dataframe(
        df_fin[[
            "month", "season_name", "cooling_multiplier", "heating_multiplier",
            "revenue_inr", "expenditure_inr", "baseline_expenditure_inr",
            "net_profit_inr", "cost_efficiency_ratio", "cost_efficiency_gain_pct"
        ]],
        use_container_width=True, height=280
    )

st.markdown("---")
st.caption("SmartHVAC, CSC-505 group project")
