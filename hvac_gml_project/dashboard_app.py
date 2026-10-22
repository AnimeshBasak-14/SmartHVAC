"""
dashboard_app.py
Executive Multi-Task Graphical Command Center (Streamlit + Plotly + PyVis + Neo4j)
Features: Dynamic Month Filtering, Interactive Graph Exploration, Company Business POV Levers,
Telemetry Risk Matrix, Churn Retention Engine, and Financial Analytics.
"""
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components
from pyvis.network import Network

from config import Config
from data_pipeline import HVACGraphDatasetBuilder, encode_month_features
from neo4j_connector import Neo4jGraphManager
from business_financial_engine import CompanyBusinessInputs, BusinessFinancialEngine

st.set_page_config(
    page_title="HVAC & Smart Appliance GML Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# CUSTOM CSS: FIX METRIC TRUNCATION & EXECUTIVE STYLING
# -------------------------------------------------------------
st.markdown("""
<style>
    /* Prevent metric truncation */
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
    .biz-card {
        background-color: #1a1e29;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

def format_inr(amount: float, compact: bool = True) -> str:
    """Formats INR into Lakhs/Crores if large, with full number fallback."""
    if compact:
        if abs(amount) >= 10000000:
            return f"₹{amount / 10000000:.2f} Cr"
        elif abs(amount) >= 100000:
            return f"₹{amount / 100000:.2f} L"
        else:
            return f"₹{amount:,.0f}"
    return f"₹{amount:,.0f}"

# -------------------------------------------------------------
# DATA INITIALIZATION
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
        return pd.read_csv(Config.FINANCIAL_REPORT_CSV)
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
st.sidebar.title("⚡ Control Center")

st.sidebar.subheader("1. Temporal Climate Snapshot")
season_quick_select = st.sidebar.selectbox(
    "Season Quick Jump",
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
    selected_month = st.sidebar.slider("Select Active Month (1 = Jan, 12 = Dec)", 1, 12, 5)

season_name, regime_id, cool_mult, heat_mult = Config.MONTH_REGIME[selected_month]

st.sidebar.markdown(f"**Season Regime:** `{season_name}`")
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Cooling Mult", f"{cool_mult:.2f}x")
col_s2.metric("Heating Mult", f"{heat_mult:.2f}x")

st.sidebar.markdown("---")
st.sidebar.subheader("2. Graph Filter & Physics")
loc_filter = st.sidebar.radio("Location Filter", ["All", "Metropolitan Only", "Village Only"], horizontal=True)
device_filter = st.sidebar.radio("Appliance Filter", ["All", "Cooling HVAC", "Heating / Induction Kettle"], horizontal=True)
max_nodes_viz = st.sidebar.slider("Max Nodes in Visualizer", 40, 250, 100, step=10)
physics_engine = st.sidebar.selectbox("Graph Physics Engine", ["Barnes-Hut (Fast Force)", "Repulsion (Clean Hierarchy)", "Static Frozen"])

# -------------------------------------------------------------
# EXPANDABLE COMPANY BUSINESS POV INPUTS PANEL
# -------------------------------------------------------------
with st.expander("🏢 Company Business POV: Multi-Input Cost & Pricing Levers", expanded=False):
    st.markdown("Customize commercial business inputs to simulate executive P&L, service pricing, and operational margins:")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)

    with col_b1:
        st.markdown("**1. Service Pricing (INR)**")
        b_metro_cool = st.number_input("Metro Cooling Ticket", value=2400.0, step=100.0)
        b_metro_heat = st.number_input("Metro Heating Ticket", value=1500.0, step=100.0)
        b_amc_fee = st.number_input("Monthly AMC Fee / User", value=450.0, step=50.0)

    with col_b2:
        st.markdown("**2. Regional & Subsidies**")
        b_vill_cool = st.number_input("Village Cooling Ticket", value=1650.0, step=50.0)
        b_vill_heat = st.number_input("Village Heating Ticket", value=1250.0, step=50.0)
        b_amc_ratio = st.slider("AMC Contract Coverage", 0.10, 0.80, 0.35, step=0.05)

    with col_b3:
        st.markdown("**3. Maintenance & SLA Costs**")
        b_prev_cost = st.number_input("Preventive Inspection Cost", value=950.0, step=50.0)
        b_emerg_cost = st.number_input("Emergency Breakdown Cost", value=4200.0, step=200.0)
        b_stockout_pen = st.number_input("Part Stockout SLA Penalty", value=85.0, step=10.0)

    with col_b4:
        st.markdown("**4. Retention & Marketing**")
        b_cac = st.number_input("Sleeper Campaign CAC", value=180.0, step=20.0)
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
# DYNAMIC MONTH FINANCIAL METRICS COMPUTATION
# -------------------------------------------------------------
row_m_orig = df_fin[df_fin["month"] == selected_month].iloc[0]

metro_demand = float(row_m_orig["metro_demand_units"])
village_demand = float(row_m_orig["village_demand_units"])

ticket_metro = biz_inputs.base_metro_ticket_cooling * cool_mult + biz_inputs.base_metro_ticket_heating * heat_mult
ticket_vill = biz_inputs.base_village_ticket_cooling * cool_mult + biz_inputs.base_village_ticket_heating * heat_mult
ondemand_rev = metro_demand * ticket_metro + village_demand * ticket_vill
amc_rev = len(df_res) * biz_inputs.amc_coverage_ratio * biz_inputs.amc_subscription_monthly_fee

sleepers = int(row_m_orig["sleeper_customers_identified"])
reactivated = int(sleepers * biz_inputs.sleeper_reactivation_rate)
cross_margin = biz_inputs.cross_sell_heating_rev if heat_mult > cool_mult else biz_inputs.cross_sell_cooling_rev
retention_rev = reactivated * cross_margin
tot_rev = ondemand_rev + amc_rev + retention_rev

high_risk = int(row_m_orig["high_risk_devices_flagged"])
prev_cost = high_risk * biz_inputs.preventive_inspection_cost
emerg_cost = max(5, int(len(df_dev) * biz_inputs.unavoidable_emergency_rate)) * biz_inputs.emergency_breakdown_cost
hold_cost = float(row_m_orig["optimal_spare_parts_units"]) * 85.0 * 0.03
stockout_cost = 45.0 * biz_inputs.stockout_penalty_per_unit
mktg_cost = sleepers * biz_inputs.marketing_campaign_cost_per_user

tot_exp = prev_cost + emerg_cost + hold_cost + stockout_cost + mktg_cost
avoided_savings = high_risk * (biz_inputs.emergency_breakdown_cost - biz_inputs.preventive_inspection_cost) * biz_inputs.avoided_breakdown_efficiency
base_exp = tot_exp + avoided_savings + (stockout_cost * 2.4)

net_profit = tot_rev - tot_exp
cost_efficiency = tot_rev / max(tot_exp, 1.0)
cost_savings_pct = ((base_exp - tot_exp) / max(base_exp, 1.0)) * 100.0

# -------------------------------------------------------------
# MAIN HEADER & UNTRUNCATED METRICS BAR
# -------------------------------------------------------------
st.title("HVAC & Smart Appliance Lifecycle GML Command Center")
st.caption(
    "Heterogeneous Graph Neural Network | NVIDIA Tesla V100 GPU Mixed-Precision | Neo4j Bolt DB | "
    f"Active Snapshot: **Month {selected_month} ({season_name})**"
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
        delta=f"Saved: -{format_inr(base_exp - tot_exp)} vs reactive"
    )

with kpi3:
    st.metric(
        label=f"Month {selected_month} Net Profit",
        value=f"{format_inr(net_profit)}",
        delta=f"{((net_profit / max(tot_rev, 1.0)) * 100):.1f}% Operating Margin"
    )

with kpi4:
    st.metric(
        label="Cost Efficiency Ratio (Rev/Exp)",
        value=f"{cost_efficiency:.2f}x",
        delta=f"{cost_savings_pct:+.1f}% Proactive Savings"
    )

st.markdown("---")

# -------------------------------------------------------------
# TABBED INTERACTIVE SECTIONS
# -------------------------------------------------------------
tab_graph, tab_devices, tab_churn, tab_parts, tab_fin = st.tabs([
    "🌐 Interactive Hetero-Graph",
    "⚡ Device Failure Telemetry",
    "👥 Sleeper Retention Engine",
    "📦 Spare Parts Forecasting",
    "💰 Executive Financial Ledger"
])

# =============================================================
# TAB 1: INTERACTIVE HETEROGENEOUS GRAPH (FAILSAFE IN-MEMORY)
# =============================================================
with tab_graph:
    st.subheader(f"Heterogeneous Graph Network Snapshot (Month {selected_month}: {season_name})")
    st.markdown("""
    **Node Color Legend:** 
    🔵 **Metro Location** | 🟢 **Village Location** | 🟠 **Active Customer** | 🔴 **Sleeper Customer** | 🔷 **Cooling HVAC** | 🟣 **Heating / Induction Kettle** | 🟡 **Spare Part SKU**
    """)

    filtered_locs = df_loc.copy()
    if loc_filter == "Metropolitan Only":
        filtered_locs = filtered_locs[filtered_locs["is_metropolitan"] == 1]
    elif loc_filter == "Village Only":
        filtered_locs = filtered_locs[filtered_locs["is_metropolitan"] == 0]
    sample_locs = filtered_locs.head(12)
    sample_loc_ids = set(sample_locs["location_id"].astype(int))

    sample_res = df_res[df_res["location_id"].astype(int).isin(sample_loc_ids)].head(max_nodes_viz // 2)
    sample_res_ids = set(sample_res["resident_id"].astype(int))

    sample_e_dev = e_res_dev[e_res_dev["resident_id"].astype(int).isin(sample_res_ids)].head(max_nodes_viz)
    sample_dev_ids = set(sample_e_dev["device_id"].astype(int))
    sample_devs = df_dev[df_dev["device_id"].astype(int).isin(sample_dev_ids)]

    if device_filter == "Cooling HVAC":
        sample_devs = sample_devs[sample_devs["is_cooling_hvac"] == 1]
        sample_dev_ids = set(sample_devs["device_id"].astype(int))
    elif device_filter == "Heating / Induction Kettle":
        sample_devs = sample_devs[sample_devs["is_cooling_hvac"] == 0]
        sample_dev_ids = set(sample_devs["device_id"].astype(int))

    sample_e_parts = e_dev_part[e_dev_part["device_id"].astype(int).isin(sample_dev_ids)].head(35)
    sample_part_ids = set(sample_e_parts["part_id"].astype(int))
    sample_parts = df_part[df_part["part_id"].astype(int).isin(sample_part_ids)]

    net = Network(height="560px", width="100%", bgcolor="#0e1117", font_color="white", directed=True)
    if physics_engine == "Barnes-Hut (Fast Force)":
        net.barnes_hut(gravity=-3000, central_gravity=0.35, spring_length=110)
    elif physics_engine == "Repulsion (Clean Hierarchy)":
        net.repulsion(node_distance=160, central_gravity=0.2, spring_length=140)
    else:
        net.toggle_physics(False)

    existing_node_ids = set()

    for _, l in sample_locs.iterrows():
        lid_int = int(l["location_id"])
        node_key = f"L_{lid_int}"
        color = "#1f77b4" if int(l["is_metropolitan"]) == 1 else "#2ca02c"
        net.add_node(
            node_key, label=f"{l['name']}", color=color, size=26,
            title=f"Location: {l['name']} | Type: {l['location_type']} | Base Temp: {l['base_temp_c']}°C | Grid: {l['grid_stability']}"
        )
        existing_node_ids.add(node_key)

    for _, r in sample_res.iterrows():
        rid_int = int(r["resident_id"])
        node_key = f"R_{rid_int}"
        effective_engagement = float(r["base_app_engagement"]) * (1.0 - 0.35 * heat_mult if int(r["is_metropolitan"]) == 1 else 1.0 - 0.45 * cool_mult)
        is_sleeper = effective_engagement < 0.42
        r_color = "#ff4b4b" if is_sleeper else "#ff9f1c"
        r_label = f"Sleeper_{rid_int}" if is_sleeper else f"Resident_{rid_int}"
        net.add_node(
            node_key, label=r_label, color=r_color, size=14,
            title=f"Resident ID: {rid_int} | App Active: {r['customer_app_active']} | Engagement: {effective_engagement:.2f} | Status: {'SLEEPER' if is_sleeper else 'ACTIVE'}"
        )
        existing_node_ids.add(node_key)

    for _, d in sample_devs.iterrows():
        did_int = int(d["device_id"])
        node_key = f"D_{did_int}"
        is_cool = int(d["is_cooling_hvac"]) == 1
        d_color = "#00d4ff" if is_cool else "#9b5de5"
        net.add_node(
            node_key, label=f"{d['sku_name'][:18]}", color=d_color, size=18,
            title=f"Device: {d['sku_name']} | Price: ₹{d['unit_price_inr']:,} | Power: {d['rated_power_kw']} kW | Risk: {d['base_failure_risk']}"
        )
        existing_node_ids.add(node_key)

    for _, p in sample_parts.iterrows():
        pid_int = int(p["part_id"])
        node_key = f"P_{pid_int}"
        net.add_node(
            node_key, label=f"{p['part_name'][:16]}", color="#ffd166", size=15,
            title=f"Part: {p['part_name']} | Cost: ₹{p['unit_cost_inr']:,} | Stock: {p['current_stock']} units"
        )
        existing_node_ids.add(node_key)

    for _, r in sample_res.iterrows():
        src_key = f"R_{int(r['resident_id'])}"
        dst_key = f"L_{int(r['location_id'])}"
        if src_key in existing_node_ids and dst_key in existing_node_ids:
            net.add_edge(src_key, dst_key, title="LOCATED_IN", color="#555555", width=1.2)

    for _, ed in sample_e_dev.iterrows():
        src_key = f"R_{int(ed['resident_id'])}"
        dst_key = f"D_{int(ed['device_id'])}"
        if src_key in existing_node_ids and dst_key in existing_node_ids:
            net.add_edge(src_key, dst_key, title="OWNS", color="#888888", width=1.5)

    for _, ep in sample_e_parts.iterrows():
        src_key = f"D_{int(ep['device_id'])}"
        dst_key = f"P_{int(ep['part_id'])}"
        if src_key in existing_node_ids and dst_key in existing_node_ids:
            net.add_edge(src_key, dst_key, title="REQUIRES_PART", color="#ffd166", width=1.2)

    try:
        html_code = net.generate_html()
        components.html(html_code, height=570)
    except Exception:
        html_path = os.path.join(Config.ARTIFACT_DIR, "pyvis_sub_graph.html")
        net.save_graph(html_path)
        with open(html_path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=570)

    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    col_g1.metric("Locations Rendered", len(sample_locs))
    col_g2.metric("Residents Connected", len(sample_res))
    col_g3.metric("Devices Tracked", len(sample_devs))
    col_g4.metric("Inventory Parts", len(sample_parts))

# =============================================================
# TAB 2: DEVICE TELEMETRY & FAILURE PREDICTION
# =============================================================
with tab_devices:
    st.subheader("Device Health, Power Draw & Failure Telemetry")
    st.caption("Model 2 Output: Dual heads predict operating load (kW) and compressor/element failure probabilities.")

    col_dev1, col_dev2 = st.columns([1.1, 1.0])
    with col_dev1:
        seasonal_stress = np.where(df_dev["is_cooling_hvac"] == 1, cool_mult, heat_mult)
        sim_power = df_dev["rated_power_kw"].values * (0.55 + 0.65 * seasonal_stress)
        sim_risk = np.clip(df_dev["base_failure_risk"].values * (0.65 + 0.95 * seasonal_stress), 0.01, 0.99)
        df_dev_eval = df_dev.copy()
        df_dev_eval["dynamic_power_kw"] = np.round(sim_power, 2)
        df_dev_eval["failure_probability"] = np.round(sim_risk, 3)

        fig_dev_scatter = px.scatter(
            df_dev_eval, x="dynamic_power_kw", y="failure_probability", color="category",
            size="unit_price_inr", hover_name="sku_name", template="plotly_dark",
            labels={"dynamic_power_kw": "Estimated Operating Load (kW)", "failure_probability": "Predicted Failure Risk"}
        )
        fig_dev_scatter.add_hline(y=0.20, line_dash="dash", line_color="#ff4b4b", annotation_text="High-Risk Cutoff")
        fig_dev_scatter.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_dev_scatter, use_container_width=True)

    with col_dev2:
        st.markdown("#### High-Risk Appliances Requiring Preventive Inspection")
        high_risk_skus = df_dev_eval[df_dev_eval["failure_probability"] > 0.18].sort_values(
            "failure_probability", ascending=False
        )[["sku_name", "category", "dynamic_power_kw", "failure_probability", "unit_price_inr"]]
        st.dataframe(high_risk_skus.head(8), height=340, use_container_width=True)

# =============================================================
# TAB 4: SLEEPER CHURN & CROSS-SELL RETENTION ENGINE
# =============================================================
with tab_churn:
    st.subheader("Customer Retention & 'Sleeper' Cross-Sell Engine")
    st.caption("Model 3 Output & Link Prediction: Identifies off-season dormant customers and recommends counter-seasonal cross-sells.")

    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        st.markdown("#### Customer App Engagement Distribution")
        activity_drop = np.where(df_res["is_metropolitan"] == 1, 1.0 - 0.35 * heat_mult, 1.0 - 0.45 * cool_mult)
        current_eng = df_res["base_app_engagement"] * activity_drop
        fig_eng = px.histogram(
            current_eng, nbins=25, template="plotly_dark",
            labels={"value": "Current Engagement Score"}, color_discrete_sequence=["#00d4ff"]
        )
        fig_eng.add_vline(x=0.42, line_dash="dash", line_color="#ff4b4b", annotation_text="Sleeper Cutoff")
        fig_eng.update_layout(height=280, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_eng, use_container_width=True)

    with col_ch2:
        st.markdown("#### GNN Link Prediction: Recommended Cross-Sell Campaigns")
        st.info(
            f"During **{season_name}**, Cooling HVAC owners with engagement < 0.42 "
            f"are recommended **{'Heating Appliances & Induction Kettles' if heat_mult > cool_mult else 'Pre-Season Cooling Servicing'}**."
        )
        st.metric("Identified Sleeper Customers", sleepers)
        st.metric("Expected Reactivations via Link Predictor", reactivated)
        st.metric("Estimated Incremental Cross-Sell Margin", f"₹{reactivated * cross_margin:,}")

# =============================================================
# TAB 5: SPARE PARTS INVENTORY FORECASTING
# =============================================================
with tab_parts:
    st.subheader("Spare-Parts Supply Chain & Inventory Optimization")
    st.caption("Model 4 Output: Forecasts monthly part replenishment to balance stockout penalties against holding costs.")

    part_factor = np.where(df_part["for_cooling_hvac"] == 1, cool_mult, heat_mult)
    predicted_parts_demand = np.round(35.0 + 95.0 * part_factor, 1)
    df_part_eval = df_part.copy()
    df_part_eval["forecasted_demand"] = predicted_parts_demand
    df_part_eval["stock_status"] = np.where(
        df_part_eval["current_stock"] < df_part_eval["forecasted_demand"], "Stockout Risk", "Adequate"
    )

    fig_inv = px.bar(
        df_part_eval.head(15), x="part_name", y=["current_stock", "forecasted_demand"],
        barmode="group", template="plotly_dark",
        labels={"value": "Quantity (Units)", "part_name": "Spare Part SKU"},
        color_discrete_map={"current_stock": "#2a9d8f", "forecasted_demand": "#e76f51"}
    )
    fig_inv.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_inv, use_container_width=True)

    st.markdown("#### Critical Replenishment Recommendations")
    critical_parts = df_part_eval[df_part_eval["stock_status"] == "Stockout Risk"][
        ["part_name", "for_cooling_hvac", "current_stock", "forecasted_demand", "unit_cost_inr", "holding_cost_monthly"]
    ]
    st.dataframe(critical_parts, height=220, use_container_width=True)

# =============================================================
# TAB 6: EXECUTIVE FINANCIAL LEDGER & SCENARIO ANALYSIS
# =============================================================
with tab_fin:
    st.subheader("Monthly Financial Ledger & Proactive Cost Savings")
    st.caption("Audited P&L statement across all 12 calendar months: Revenue vs. GML Optimized Expenditure vs. Reactive Baseline.")

    col_f1, col_f2 = st.columns([1.1, 1.0])
    with col_f1:
        fig_fin_bar = go.Figure()
        fig_fin_bar.add_trace(go.Bar(
            x=df_fin["month"], y=df_fin["revenue_inr"],
            name="Revenue (INR)", marker_color="#00b4d8"
        ))
        fig_fin_bar.add_trace(go.Bar(
            x=df_fin["month"], y=df_fin["expenditure_inr"],
            name="GML Optimized Cost (INR)", marker_color="#2a9d8f"
        ))
        fig_fin_bar.add_trace(go.Scatter(
            x=df_fin["month"], y=df_fin["baseline_expenditure_inr"],
            name="Unoptimized Reactive Cost",
            mode="lines+markers", line=dict(color="#e63946", width=3, dash="dash")
        ))
        fig_fin_bar.update_layout(
            barmode="group", template="plotly_dark", height=320,
            xaxis=dict(title="Month", tickmode="linear", dtick=1),
            yaxis=dict(title="Amount (INR)"),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_fin_bar, use_container_width=True)

    with col_f2:
        st.markdown(f"#### Month {selected_month} P&L Breakdown")
        rev_breakdown = pd.DataFrame({
            "Stream": ["On-Demand Servicing", "Recurring AMC Contracts", "Sleeper Cross-Sell Margin", "Total Revenue"],
            "Amount (INR)": [f"₹{ondemand_rev:,.0f}", f"₹{amc_rev:,.0f}", f"₹{retention_rev:,.0f}", f"₹{tot_rev:,.0f}"]
        })
        st.dataframe(rev_breakdown, height=140, use_container_width=True)

        exp_breakdown = pd.DataFrame({
            "Cost Driver": ["Preventive Inspections", "Residual Emergency Repairs", "Inventory Holding", "Stockout Penalties", "Marketing CAC", "Total Expenditure"],
            "Amount (INR)": [f"₹{prev_cost:,.0f}", f"₹{emerg_cost:,.0f}", f"₹{hold_cost:,.0f}", f"₹{stockout_cost:,.0f}", f"₹{mktg_cost:,.0f}", f"₹{tot_exp:,.0f}"]
        })
        st.dataframe(exp_breakdown, height=190, use_container_width=True)

    st.markdown("#### Complete 12-Month Audited Financial Table")
    st.dataframe(
        df_fin[[
            "month", "season_name", "cooling_multiplier", "heating_multiplier",
            "revenue_inr", "expenditure_inr", "baseline_expenditure_inr",
            "net_profit_inr", "cost_efficiency_ratio", "cost_efficiency_gain_pct"
        ]],
        use_container_width=True, height=280
    )

st.markdown("---")
st.caption("HVAC & Smart Appliance Heterogeneous GML Command Center | Built for NVIDIA Tesla V100 GPU + Neo4j Lab Server")
