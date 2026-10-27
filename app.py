"""
app.py - HVAC Service Planner (Streamlit dashboard)

    python run_all.py          # once: trains the models and saves them
    streamlit run app.py

Three tools for an HVAC service provider:
  1. Demand Planner   : how much power will the HVAC draw? (plan technicians and grid load)
  2. Failure Radar    : which machines should be inspected first?
  3. Retention List   : which active customers are about to go quiet?
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from hvac_ml import config, data, pillar1_demand as p1, pillar2_failure as p2, pillar3_churn as p3

st.set_page_config(page_title="HVAC Service Planner", layout="wide")


@st.cache_resource
def load_bundle():
    path = config.MODEL_DIR / "hvac_models.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_tables():
    return data.load_hvac(), data.load_ai4i(), data.load_invoices()


@st.cache_data
def churn_snapshot():
    _, _, invoices = load_tables()
    snaps = p3.build_snapshots(invoices)
    return snaps[config.TEST_CUTOFF]


bundle = load_bundle()
st.title("HVAC Service Planner")
st.caption("Decision support for an HVAC service provider: plan demand, inspect the right machines, call the right customers.")
if bundle is None:
    st.error("No trained models found. Run `python run_all.py` first.")
    st.stop()

hvac, ai4i, invoices = load_tables()
RIDGE, XGB = "Ridge (engineered features)", "XGBoost (engineered features)"
tab1, tab2, tab3 = st.tabs(["Demand Planner", "Failure Radar", "Retention List"])

# 1. demand
with tab1:
    st.subheader("Demand Planner: expected HVAC power draw")
    lo, hi = float(hvac["ambient_c"].min()), float(hvac["ambient_c"].max())
    c1, c2 = st.columns(2)
    with c1:
        month = st.slider("Month", 1, 12, 7)
        indoor = st.slider("Indoor setpoint (°C)", 19.0, 26.0, 22.5, 0.5)
        humidity = st.slider("Humidity (%)", 15, 98, 55)
    with c2:
        t_min = st.slider("Coolest outdoor temperature of the day (°C)", -5.0, 50.0, 24.0, 0.5)
        t_max = st.slider("Hottest outdoor temperature of the day (°C)", -5.0, 55.0, 38.0, 0.5)
        t_max = max(t_max, t_min)

    hours = np.arange(24)
    # simple day profile: coldest at 5 am, hottest at 3 pm
    ambient = t_min + (t_max - t_min) * (0.5 - 0.5 * np.cos(2 * np.pi * (hours - 5) / 24))
    mode = np.where(ambient > 27, "Cooling", np.where(ambient < 19, "Heating", "Idle"))
    day = pd.DataFrame({"month": month, "hour": hours, "ambient_c": ambient, "indoor_c": indoor, "humidity": humidity, "mode": mode})
    X = p1.build_features(day, True)
    pred_xgb = bundle["demand"][XGB].predict(X)
    pred_ridge = bundle["demand"][RIDGE].predict(X)
    outside = (ambient > config.HEATWAVE_TEMP_C) | (ambient < lo) | (ambient > hi)
    # safeguard learned from the heatwave test: trees cannot extrapolate, the linear model can
    pred = np.where(outside, pred_ridge, pred_xgb)
    chart = pd.DataFrame({"Planned demand (kW)": pred, "XGBoost only (kW)": pred_xgb, "Ridge only (kW)": pred_ridge}, index=hours)
    st.line_chart(chart)
    m1, m2, m3 = st.columns(3)
    m1.metric("Peak demand", f"{pred.max():.2f} kW", f"at {int(hours[pred.argmax()]):02d}:00")
    m2.metric("Energy for the day", f"{pred.sum():.1f} kWh")
    m3.metric("Hours using the heatwave safeguard", int(outside.sum()))
    if outside.any():
        st.warning(f"Outdoor temperature above {config.HEATWAVE_TEMP_C:.0f} °C (or outside the range seen in training). XGBoost cannot extrapolate there, "
                   "so the planner uses the Ridge model for those hours. In our heatwave test XGBoost was off by 0.34 kW on average, Ridge by 0.08 kW.")

# 2. failure
with tab2:
    st.subheader("Failure Radar: which machines to inspect first")
    left, right = st.columns([1, 2])
    with left:
        mtype = st.selectbox("Machine type", ["L", "M", "H"])
        air = st.slider("Air temperature (K)", 295.0, 305.0, 300.0, 0.1)
        proc = st.slider("Process temperature (K)", 305.0, 315.0, 310.0, 0.1)
        rpm = st.slider("Rotational speed (rpm)", 1100, 2900, 1500, 10)
        torque = st.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.5)
        wear = st.slider("Tool wear (min)", 0, 255, 100)
    one = pd.DataFrame({"Type": [mtype], "air_temp_k": [air], "process_temp_k": [proc], "rpm": [rpm], "torque_nm": [torque], "tool_wear_min": [wear]})
    model, threshold = bundle["failure"][XGB]
    risk = float(model.predict_proba(p2.build_features(one, True))[:, 1][0])
    all_risk = model.predict_proba(p2.build_features(ai4i, True))[:, 1]
    with left:
        st.metric("Failure risk", f"{risk:.1%}", "INSPECT" if risk >= threshold else "ok")
        st.caption(f"Higher than {(all_risk < risk).mean():.0%} of the 10,000 machines in the dataset. Alert threshold: {threshold:.0%}.")
    with right:
        st.markdown("**Inspection list: the 15 highest-risk machines in a random sample of 500**")
        sample = ai4i.sample(500, random_state=1).copy()
        sample["risk"] = model.predict_proba(p2.build_features(sample, True))[:, 1]
        top = sample.sort_values("risk", ascending=False).head(15)
        top = top.assign(risk=(top["risk"] * 100).round(1), actual_failure=np.where(top["failure"] == 1, "yes", "no"))
        st.dataframe(top[["Product ID", "Type", "torque_nm", "rpm", "tool_wear_min", "risk", "actual_failure"]]
                     .rename(columns={"risk": "risk %"}), hide_index=True)
        found = sample.sort_values("risk", ascending=False).head(50)["failure"].sum()
        st.caption(f"Inspecting the top 10% (50 machines) would catch {found} of the {int(sample['failure'].sum())} failures in this sample. "
                   "(The model was trained on all 10,000 machines, so this list is an illustration; honest test scores are in the report and in RESULTS_SUMMARY.md.)")

# 3. retention
with tab3:
    st.subheader("Retention List: active customers most likely to go quiet in the next 90 days")
    snap = churn_snapshot().copy()
    model, threshold = bundle["churn"][RIDGE]
    snap["lapse_risk"] = model.predict_proba(snap[p3.ENGINEERED])[:, 1]
    n = st.slider("How many customers can the team contact?", 10, 200, 50, 10)
    top = snap.sort_values("lapse_risk", ascending=False).head(n)
    top = top.assign(lapse_risk=(top["lapse_risk"] * 100).round(1), monetary=top["monetary"].round(0))
    st.dataframe(top[["customer", "lapse_risk", "recency_days", "frequency", "monetary", "orders_last_90d", "lapsed"]]
                 .rename(columns={"lapse_risk": "risk %", "lapsed": "actually lapsed"}), hide_index=True)
    reached = snap.sort_values("lapse_risk", ascending=False).head(n)["lapsed"].sum() / max(1, snap["lapsed"].sum())
    st.caption(f"Contacting these {n} of {len(snap)} active customers reaches {reached:.0%} of those who actually lapsed "
               f"(random contact would reach {n / len(snap):.0%}). Expect about 2x better than random, not 10x: churn is hard to predict from purchase history alone.")
