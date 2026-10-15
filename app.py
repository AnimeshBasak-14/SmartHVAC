"""
app.py - SmartHVAC Advisor GUI (Streamlit).

    streamlit run app.py          (after `python run_all.py` has trained the models)

Tabs:  Building Advisor | Model Performance | Data Insights | About
"""
import pandas as pd
import streamlit as st

from src import config
from src.recommend import (BUILDING_SHAPES, annual_energy_cost, building_from_shape, design_optimizer,
                           load_bundle, predict_building, service_plan, shape_label)

st.set_page_config(page_title="SmartHVAC Advisor", page_icon="❄️", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle()


st.title("❄️🔥 SmartHVAC Advisor")
st.caption("Machine-learning assistant for HVAC service providers: predict a building's heating & cooling "
           "load, choose the right equipment and maintenance plan, and find cheaper building designs.")

try:
    bundle = get_bundle()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()

# ------------------------------------------------------------------ sidebar inputs
with st.sidebar:
    st.header("Building details")
    shape = st.selectbox("Building shape", range(len(BUILDING_SHAPES)), format_func=shape_label)
    orientation = st.selectbox("Orientation (main facade faces)", [2, 3, 4, 5],
                               format_func=config.ORIENTATION_NAMES.get)
    glazing_area = st.select_slider("Window (glazing) area, % of floor area", [0.0, 0.10, 0.25, 0.40],
                                    value=0.25, format_func=lambda v: f"{v:.0%}")
    dist_options = [0] if glazing_area == 0 else [1, 2, 3, 4, 5]
    glazing_dist = st.selectbox("Where are the windows?", dist_options, format_func=config.GLAZING_DIST_NAMES.get)
    price = st.number_input("Electricity price (₹ / kWh)", 1.0, 50.0, config.ELECTRICITY_PRICE_INR, 0.5)
    s = BUILDING_SHAPES[shape]
    st.caption(f"Compactness {s[0]} · surface {s[1]} m² · wall {s[2]} m² · roof {s[3]} m² · height {s[4]} m")

building = building_from_shape(shape, orientation, glazing_area, glazing_dist)
pred = predict_building(bundle, building)
cost = annual_energy_cost(building, pred, price)
plan = service_plan(pred["demand_class"])

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Building Advisor", "📊 Model Performance", "📈 Data Insights", "ℹ️ About"])

# ------------------------------------------------------------------ tab 1: advisor
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Heating load", f"{pred['heating_load']:.1f} kWh/m²",
              help=f"90% range: {pred['heating_load_low']:.1f} - {pred['heating_load_high']:.1f}")
    c2.metric("Cooling load", f"{pred['cooling_load']:.1f} kWh/m²",
              help=f"90% range: {pred['cooling_load_low']:.1f} - {pred['cooling_load_high']:.1f}")
    c3.metric("Demand class", pred["demand_class"])
    c4.metric("Est. annual cost", f"₹{cost['annual_cost_inr']:,.0f}", help=f"{cost['annual_kwh']:,.0f} kWh/yr "
              f"for {cost['floor_area_m2']:.0f} m² (assumes loads are annual kWh/m²)")
    st.caption(f"90% prediction ranges - heating {pred['heating_load_low']:.1f}–{pred['heating_load_high']:.1f}, "
               f"cooling {pred['cooling_load_low']:.1f}–{pred['cooling_load_high']:.1f} kWh/m². "
               f"Second opinion from the {bundle['classifier_name']} classifier: **{pred['classifier_class']}** "
               f"({pred['classifier_confidence']:.0%} confident).")

    st.subheader("Recommended service package")
    color = {"Low": st.success, "Medium": st.info, "High": st.warning}[pred["demand_class"]]
    color(f"**Equipment:** {plan['equipment']}  \n**Maintenance:** {plan['maintenance']}  \n"
          f"**Advice:** {plan['advice']}")

    st.subheader("Design optimizer - can windows / orientation reduce the load?")
    opt = design_optimizer(bundle, building)
    best = opt.iloc[0]
    if best["saving_vs_current_%"] > 0.5:
        st.write(f"Best option: **{best.glazing_area:.0%} glazing, {best.glazing_distribution} windows, facing "
                 f"{best.orientation}** -> total load {best.total_load:.1f} kWh/m² "
                 f"(**{best['saving_vs_current_%']:.1f}% lower**, ≈ ₹"
                 f"{cost['annual_cost_inr'] * best['saving_vs_current_%'] / 100:,.0f} saved per year).")
    else:
        st.write("This design is already (close to) the best possible for this building shape.")
    show = opt.head(10).copy()
    show["glazing_area"] = show["glazing_area"].map(lambda v: f"{v:.0%}")
    st.dataframe(show.round(2), hide_index=True)

# ------------------------------------------------------------------ tab 2: performance
with tab2:
    res = config.RESULTS_DIR
    st.write(f"Models trained on {bundle['trained_at']} (scikit-learn {bundle['sklearn_version']}). "
             f"Heating model: **{bundle['regressor_names']['heating_load']}**, cooling model: "
             f"**{bundle['regressor_names']['cooling_load']}**, classifier: **{bundle['classifier_name']}**.")
    if (res / "regression_test_metrics.csv").exists():
        st.markdown("**Final results on the hidden test set**")
        st.dataframe(pd.read_csv(res / "regression_test_metrics.csv").round(3), hide_index=True)
        st.markdown("**All models compared (validation set)**")
        st.dataframe(pd.read_csv(res / "regression_model_comparison.csv").round(3), hide_index=True)
    for name in ["05_model_comparison.png", "06_predicted_vs_actual.png", "09_confusion_matrix.png",
                 "10_learning_curve.png"]:
        if (config.FIG_DIR / name).exists():
            st.image(str(config.FIG_DIR / name))

# ------------------------------------------------------------------ tab 3: insights
with tab3:
    for name in ["02_correlation_heatmap.png", "04_loads_by_glazing_and_height.png",
                 "08_feature_importance.png", "11_partial_dependence.png"]:
        if (config.FIG_DIR / name).exists():
            st.image(str(config.FIG_DIR / name))
    if (config.RESULTS_DIR / "design_optimizer_summary.csv").exists():
        st.markdown("**Design optimizer across all 12 building shapes**")
        st.dataframe(pd.read_csv(config.RESULTS_DIR / "design_optimizer_summary.csv").round(2), hide_index=True)

# ------------------------------------------------------------------ tab 4: about
with tab4:
    st.markdown(f"""
**Problem.** HVAC service providers (for example Aire Serv) must size equipment and plan maintenance for
each customer's building. Doing a full engineering load calculation for every quote is slow.
This tool predicts heating & cooling load instantly from 8 simple building features.

**Data.** UCI Energy Efficiency dataset - 768 simulated buildings (Tsanas & Xifara, 2012). [{config.DATASET_PAGE}]({config.DATASET_PAGE})

**Method.** 7 regression models + 5 classifiers compared with 5-fold cross-validation and a validation set,
best model tuned with grid search, final score on a hidden 15% test set. 90% prediction ranges come from
cross-validated errors (conformal method). The design optimizer re-runs the model for every window /
orientation option.

**Limits.** Simulated buildings from one climate (Athens, Greece); 12 shapes only; cost assumes loads are
annual kWh/m². Predictions are a quick estimate - a technician should confirm before installation.

CSC-505 course project.
""")
