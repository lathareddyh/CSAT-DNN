# =====================================
# ADVANCED CSAT STREAMLIT DASHBOARD
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from tensorflow.keras.models import load_model

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="CSAT Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("clean_processed_data.csv")

model = load_model("csat_dnn_model.keras")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Page Title
# -----------------------------
st.title("📊 Customer Satisfaction (CSAT) Analytics Dashboard")

st.markdown(
"""
This dashboard provides **CSAT predictions and service analytics**  
to help understand customer satisfaction patterns.
"""
)

st.divider()

# =============================
# DATA INSIGHTS SECTION
# =============================

st.subheader("📈 Dataset Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Unique Agents", df["Agent_name"].nunique())

with col3:
    st.metric("Channels", df["channel_name"].nunique())

st.divider()

# =============================
# VISUALIZATION SECTION
# =============================

st.subheader("📊 Service Analytics")

chart1, chart2 = st.columns(2)

# Channel Distribution
with chart1:

    channel_counts = df["channel_name"].value_counts()

    fig1 = px.bar(
        x=channel_counts.index,
        y=channel_counts.values,
        title="Channel Distribution",
        labels={"x":"Channel","y":"Count"}
    )

    st.plotly_chart(fig1, use_container_width=True)

# Agent Shift Distribution
with chart2:

    shift_counts = df["Agent Shift"].value_counts()

    fig2 = px.pie(
        names=shift_counts.index,
        values=shift_counts.values,
        title="Agent Shift Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# =============================
# FEATURE IMPORTANCE SECTION
# =============================

st.subheader("⭐ Feature Importance")

if "response_time" in df.columns:

    corr = df.corr(numeric_only=True)["response_time"].sort_values()

    fig3 = px.bar(
        x=corr.values,
        y=corr.index,
        orientation="h",
        title="Correlation with Response Time"
    )

    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# =============================
# CSAT PREDICTION SECTION
# =============================

st.subheader(" CSAT Prediction")

cat_cols = [
'channel_name','category','Sub-category','Agent_name',
'Supervisor','Manager','Tenure Bucket','Agent Shift'
]

num_cols = [
'response_time','day_number_response_date','weekday_num_response_date'
]

col1, col2 = st.columns(2)

input_data = {}

# Categorical Inputs
with col1:

    st.markdown("### Service Details")

    for col in cat_cols:
        options = sorted(df[col].dropna().unique())
        input_data[col] = st.selectbox(col, options)

# Numerical Inputs
with col2:

    st.markdown("### Time Information")

    input_data["response_time"] = st.slider(
        "Response Time", 0, 500, 100
    )

    input_data["day_number_response_date"] = st.slider(
        "Day Number", 1, 31, 10
    )

    input_data["weekday_num_response_date"] = st.slider(
        "Weekday Number", 1, 7, 3
    )

# Convert to dataframe
input_df = pd.DataFrame([input_data])

# =============================
# PREDICTION BUTTON
# =============================

if st.button(" Predict CSAT Score"):

    
    sample_encoded = input_df.copy()
    for col in encoder.keys():
        try:
            sample_encoded[col] = encoder[col].transform(input_df[col])
        except:
            print(" value not seen during training. Using default encoding.")
            sample_encoded[col] = 0


    # encoded_cat = encoder.transform(input_df[cat_cols])
    scaled_input = scaler.transform(sample_encoded)

    prediction = np.argmax(model.predict(scaled_input), axis=1)
    csat_score = prediction[0]+1

    st.subheader("Prediction Result")
    st.success(csat_score)
  
    if csat_score > 3:
        st.success("😊 High Customer Satisfaction")

    elif csat_score == 3:
        st.warning("😐 Moderate Satisfaction")

    else:
        st.error("😟 Low Customer Satisfaction")

st.divider()

st.caption("Deep Learning CSAT Prediction System")