"""Interactive Streamlit dashboard for LightGBM ETA prediction and ops briefs."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ai_copilot import generate_ops_brief
from src.portable_model import PortableLightGBM


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


@st.cache_resource
def load_artifacts() -> tuple[dict, dict, PortableLightGBM]:
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((MODELS_DIR / "shap_summary.json").read_text(encoding="utf-8"))
    model = PortableLightGBM(MODELS_DIR / metrics["model_files"]["lightgbm_portable"])
    return metrics, summary, model


def set_category(row: dict, metrics: dict, category: str, value: str) -> None:
    numeric = set(metrics["numeric_features"])
    for col in metrics["feature_cols"]:
        if col.startswith(f"{category}_") and col not in numeric:
            row[col] = int(col == f"{category}_{value}")



def default_level(metrics: dict, category: str) -> str:
    """Level whose one-hot column is 1 in metrics["defaults"].

    The old code used category_levels[category][0] for every hidden
    category, so the simulator always ran as Errand / Buffet /
    electric_scooter, even though the trained defaults are
    Food / Snack / motorcycle. That silently biased every prediction.
    """
    for level in metrics["category_levels"][category]:
        if metrics["defaults"].get(f"{category}_{level}", 0) == 1:
            return level
    return metrics["category_levels"][category][0]


st.set_page_config(page_title="ML Delivery ETA Predictor & AI Operations Copilot", layout="wide")
st.title("ML Delivery ETA Predictor & AI Operations Copilot")
metrics, summary, model = load_artifacts()

tab1, tab2, tab3 = st.tabs(["Live ETA Simulator", "Model & SHAP Insights", "AI Ops Copilot"])

with tab1:
    st.subheader("Simulate Order ETA")
    c1, c2, c3 = st.columns(3)
    distance = c1.slider("Distance (km)", 0.5, 25.0, 8.0, 0.5)
    hour = c2.slider("Hour of day", 0, 23, 18)
    weekday_label = c3.selectbox("Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=4)
    weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(weekday_label)

    c4, c5, c6 = st.columns(3)
    zone = c4.selectbox("Zone", metrics["category_levels"]["zone"])
    weather = c5.selectbox("Weather", metrics["category_levels"]["weather"])
    traffic = c6.selectbox("Traffic", metrics["category_levels"]["traffic"])

    c7, c8, c9 = st.columns(3)
    courier_age = c7.slider("Courier age", 18, 60, 30)
    courier_rating = c8.slider("Courier rating", 1.0, 5.0, 4.5, 0.1)
    multiple_deliveries = c9.slider("Multiple deliveries", 0, 3, 0)

    c10, c11, c12 = st.columns(3)
    prep_wait = c10.slider("Prep wait (min)", 0.0, 30.0, 10.0, 1.0)
    vehicle_condition = c11.slider("Vehicle condition", 0, 2, 0)
    zone_demand = c12.slider("Orders in zone over 3h", 1, 60, 12)

    c13, c14, c15 = st.columns(3)
    order_category = c13.selectbox("Order category", metrics["category_levels"]["order_category"],
                                   index=metrics["category_levels"]["order_category"].index(default_level(metrics, "order_category")))
    type_of_order = c14.selectbox("Order type", metrics["category_levels"]["type_of_order"],
                                  index=metrics["category_levels"]["type_of_order"].index(default_level(metrics, "type_of_order")))
    vehicle_type = c15.selectbox("Vehicle type", metrics["category_levels"]["vehicle_type"],
                                 index=metrics["category_levels"]["vehicle_type"].index(default_level(metrics, "vehicle_type")))

    zone_orders_today = st.slider("Orders in zone today", 10, 300, int(metrics["defaults"]["zone_orders_that_day"]))

    row = dict(metrics["defaults"])
    row.update(
        {
            "distance_km": float(distance),
            "distance_km_original": float(distance),
            "distance_gap_km": 0.0,
            "hour": int(hour),
            "weekday": int(weekday),
            "is_weekend": int(weekday in [4, 5]),  # Fri=4, Sat=5: Saudi weekend, same as src/features.py
            "is_peak": int(12 <= hour <= 14 or 19 <= hour <= 22),
            "is_late_night": int(hour <= 5),
            "is_morning": int(6 <= hour <= 11),
            "courier_age": float(courier_age),
            "courier_rating": float(courier_rating),
            "multiple_deliveries": int(multiple_deliveries),
            "prep_wait_min": float(prep_wait),
            "vehicle_condition": int(vehicle_condition),
            "zone_demand_3h": int(zone_demand),
            "zone_orders_that_day": int(zone_orders_today),
        }
    )
    for category, value in {
        "zone": zone,
        "weather": weather,
        "traffic": traffic,
        "order_category": order_category,
        "type_of_order": type_of_order,
        "vehicle_type": vehicle_type,
        "city_type": default_level(metrics, "city_type"),
        "festival": default_level(metrics, "festival"),
    }.items():
        set_category(row, metrics, category, value)

    x_input = pd.DataFrame([row])[metrics["feature_cols"]]
    prediction = float(model.predict(x_input)[0])
    threshold = metrics["sla_threshold_min"]
    st.metric(
        "Predicted Delivery ETA",
        f"{prediction:.1f} min",
        delta="High SLA risk" if prediction > threshold else "Within SLA",
    )

with tab2:
    st.subheader("Chronological Test Performance")
    winner_metrics = summary["metrics"]
    baseline = summary["baseline"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("LightGBM MAE (min)", winner_metrics["mae_min"], delta=f"{winner_metrics['mae_min'] - baseline['mae_min']:.2f} vs baseline", delta_color="inverse")
    k2.metric("R² Score", winner_metrics["r2"])
    k3.metric("SLA-call Accuracy", f"{winner_metrics['sla_accuracy'] * 100:.1f}%")
    k4.metric("Missed Late Orders", winner_metrics["missed_late_count"])

    st.subheader("Top SHAP Delay Drivers")
    st.bar_chart(pd.DataFrame(summary["top_shap_drivers"]).set_index("feature"))

    st.subheader("Zone SLA Breakdown")
    st.dataframe(pd.DataFrame(summary["zone_summary"]), use_container_width=True)

    st.subheader("Hour-band SHAP and SLA Summary")
    st.dataframe(pd.DataFrame(summary["hour_band_summary"]), use_container_width=True)

with tab3:
    st.subheader("AI Operations Briefing")
    st.caption("Uses Gemini when GOOGLE_API_KEY or GEMINI_API_KEY is configured; otherwise uses the deterministic fallback.")
    if st.button("Generate Daily Ops Brief"):
        st.markdown(generate_ops_brief(summary))
