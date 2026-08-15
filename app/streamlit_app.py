"""
streamlit_app.py
Interactive demo: pick a flight to disrupt, see the ML-predicted passenger
priority scores, and the OR-Tools-optimized rebooking plan.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.express as px

from network import generate_flight_schedule, find_connecting_flights
from model import load_model, predict_priority, MODEL_PATH, train_model
from pipeline import run_disruption_scenario

st.set_page_config(page_title="Flight Disruption Rebooking Optimizer", layout="wide")

st.title("✈️ Flight Disruption Rebooking Optimizer")
st.caption(
    "ML-predicted passenger priority + OR-Tools optimization for automatic "
    "rebooking after a flight disruption. Built as a portfolio project simulating "
    "airline revenue-management style operations."
)

# ---- load / train model ----
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Training model for the first time (only happens once)..."):
            model, encoders = train_model()
    else:
        from model import load_model as _load
        model, encoders = _load()
    return model, encoders

@st.cache_data
def get_schedule():
    return generate_flight_schedule(num_days=1, seed=42)

model, encoders = get_model()
schedule = get_schedule()

# ---- sidebar controls ----
st.sidebar.header("Simulate a Disruption")

flight_options = schedule.sort_values("dep_time").apply(
    lambda r: f"{r.flight_id}  ({r.origin} → {r.dest}, dep {r.dep_time.strftime('%H:%M')})", axis=1
)
flight_map = dict(zip(flight_options, schedule.flight_id))

selected_label = st.sidebar.selectbox("Select flight to disrupt:", flight_options)
selected_flight_id = flight_map[selected_label]

num_passengers = st.sidebar.slider("Number of passengers on disrupted flight", 50, 220, 180)
seed = st.sidebar.number_input("Random seed (for reproducibility)", value=1, step=1)

run_button = st.sidebar.button("🚨 Simulate Disruption & Optimize Rebooking", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works:**\n"
    "1. A flight is marked disrupted (delayed/cancelled)\n"
    "2. XGBoost predicts each passenger's rebooking priority "
    "(fare class, loyalty tier, connection risk, group size)\n"
    "3. OR-Tools solves the optimal seat assignment across available "
    "onward flights, maximizing priority served under seat constraints"
)

if run_button:
    disrupted_row = schedule[schedule.flight_id == selected_flight_id].iloc[0]
    connecting = find_connecting_flights(schedule, selected_flight_id)

    st.subheader(f"Disrupted Flight: {selected_flight_id}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", f"{disrupted_row.origin} → {disrupted_row.dest}")
    c2.metric("Scheduled Departure", disrupted_row.dep_time.strftime("%H:%M"))
    c3.metric("Passengers", num_passengers)
    c4.metric("Candidate Onward Flights", len(connecting))

    if len(connecting) == 0:
        st.error(
            "No onward connecting flights available in this simulated network "
            "for reaccommodation. Try a different flight (ones landing at DEL, BOM, "
            "or BLR tend to have more onward options)."
        )
    else:
        with st.spinner("Predicting passenger priority and solving optimal rebooking..."):
            result = run_disruption_scenario(
                schedule, selected_flight_id, num_passengers, model, encoders, seed=seed
            )

        assignment = result["assignment"]
        passengers = result["passengers"]

        merged = assignment.merge(
            passengers[["passenger_id", "fare_class", "loyalty_tier", "group_size", "has_connecting_flight"]],
            on="passenger_id"
        )

        rebooked = merged[merged.assigned_flight_id.notna()]
        unassigned = merged[merged.assigned_flight_id.isna()]

        st.success(
            f"Solver status: **{result['solve_status']}** — "
            f"**{len(rebooked)} / {len(merged)}** passengers successfully rebooked."
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Rebooking outcome by fare class")
            outcome_summary = merged.groupby("fare_class").apply(
                lambda g: pd.Series({
                    "Rebooked": g.assigned_flight_id.notna().sum(),
                    "Unassigned": g.assigned_flight_id.isna().sum(),
                })
            ).reset_index()
            fig = px.bar(
                outcome_summary, x="fare_class", y=["Rebooked", "Unassigned"],
                barmode="stack", labels={"value": "Passengers", "fare_class": "Fare Class"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Priority score distribution")
            fig2 = px.histogram(
                merged, x="priority_score", color=merged.assigned_flight_id.notna().map(
                    {True: "Rebooked", False: "Unassigned"}
                ),
                nbins=30, labels={"color": "Outcome", "priority_score": "Predicted Priority"},
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Onward flights used for rebooking")
        seats_df = pd.DataFrame([
            {"flight_id": fid, "seats_offered": seats}
            for fid, seats in result["seats_available"].items()
        ])
        filled_df = merged[merged.assigned_flight_id.notna()].groupby(
            "assigned_flight_id"
        ).size().reset_index(name="seats_filled")
        flight_summary = seats_df.merge(
            filled_df, left_on="flight_id", right_on="assigned_flight_id", how="left"
        ).fillna(0)
        st.dataframe(
            flight_summary[["flight_id", "seats_offered", "seats_filled"]],
            use_container_width=True, hide_index=True
        )

        st.markdown("#### Passenger-level rebooking plan")
        display_cols = merged[[
            "passenger_id", "fare_class", "loyalty_tier", "group_size",
            "priority_score", "assigned_flight_id"
        ]].sort_values("priority_score", ascending=False)
        display_cols["priority_score"] = display_cols["priority_score"].round(3)
        display_cols["assigned_flight_id"] = display_cols["assigned_flight_id"].fillna("— Not rebooked —")
        st.dataframe(display_cols, use_container_width=True, hide_index=True, height=350)

        if len(unassigned) > 0:
            st.warning(
                f"{len(unassigned)} passengers could not be rebooked onto currently "
                "available onward flights (no seats left). In a production system, "
                "these would cascade to the next available flight or receive "
                "compensation per airline policy."
            )
else:
    st.info("👈 Select a flight and click **Simulate Disruption & Optimize Rebooking** to run the pipeline.")

st.markdown("---")
st.caption(
    "Portfolio project · Flight network (networkx) → simulated passenger manifest → "
    "XGBoost priority prediction → OR-Tools constrained optimization. "
    "All data is synthetically generated for demonstration; no real airline or passenger data is used."
)
