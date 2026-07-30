import streamlit as st
import pandas as pd
import joblib

from predict import predict_eta

st.set_page_config(
    page_title="Food Delivery ETA Predictor",
    page_icon="🍕",
    layout="wide"
)

st.title("Food Delivery ETA Prediction")

st.caption("Estimate food delivery time using Machine Learning.")

# Sidebar
st.sidebar.header("Model Information")

model = joblib.load("model.pkl")

st.sidebar.success(type(model).__name__)

try:
    results = pd.read_csv("model_results.csv", index_col=0)

    st.sidebar.write("### Model Comparison")

    st.sidebar.dataframe(results)

except:
    st.sidebar.warning("model_results.csv not found.")

with st.form("prediction"):

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Delivery Person Age",
            18,
            60,
            30
        )

        ratings = st.slider(
            "Delivery Rating",
            1.0,
            5.0,
            4.5
        )

        weather = st.selectbox(
            "Weather",
            [
                "Sunny",
                "Cloudy",
                "Fog",
                "Stormy",
                "Sandstorms",
                "Windy"
            ]
        )

        traffic = st.selectbox(
            "Traffic",
            [
                "Low",
                "Medium",
                "High",
                "Jam"
            ]
        )

        vehicle = st.selectbox(
            "Vehicle",
            [
                "motorcycle",
                "scooter",
                "electric_scooter"
            ]
        )

    with col2:

        city = st.selectbox(
            "City",
            [
                "Urban",
                "Semi-Urban",
                "Metropolitian"
            ]
        )

        festival = st.selectbox(
            "Festival",
            [
                "Yes",
                "No"
            ]
        )

        multiple = st.number_input(
            "Multiple Deliveries",
            0,
            5,
            1
        )

        distance = st.number_input(
            "Distance (km)",
            0.5,
            50.0,
            8.0
        )

        prep = st.number_input(
            "Preparation Time",
            1,
            60,
            15
        )

    submit = st.form_submit_button("Predict ETA")

if submit:

    sample = pd.DataFrame({

        "Delivery_person_Age":[age],
        "Delivery_person_Ratings":[ratings],
        "Weatherconditions":[weather],
        "Road_traffic_density":[traffic],
        "Vehicle_condition":[2],
        "Type_of_order":["Snack"],
        "Type_of_vehicle":[vehicle],
        "multiple_deliveries":[multiple],
        "Festival":[festival],
        "City":[city],
        "City_code":["BANG"],
        "distance":[distance],
        "order_prepare_time":[prep],
        "day":[15],
        "month":[4],
        "quarter":[2],
        "year":[2022],
        "day_of_week":[4],
        "is_month_start":[0],
        "is_month_end":[0],
        "is_quarter_start":[0],
        "is_quarter_end":[0],
        "is_year_start":[0],
        "is_year_end":[0],
        "is_weekend":[0]

    })

    eta = predict_eta(sample)

    st.success(f"Estimated Delivery Time : {eta:.2f} minutes")

    st.write("### Prediction Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Distance", f"{distance} km")
    c2.metric("Rating", ratings)
    c3.metric("ETA", f"{eta:.2f} min")

import matplotlib.pyplot as plt
import numpy as np

if hasattr(model, "feature_importances_"):

    st.write("## Feature Importance")

    importance = model.feature_importances_

    names = joblib.load("feature_names.pkl")

    idx = np.argsort(importance)[::-1][:10]

    fig, ax = plt.subplots(figsize=(8,5))

    ax.barh(
        np.array(names)[idx],
        importance[idx]
    )

    ax.invert_yaxis()

    st.pyplot(fig)