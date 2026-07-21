"""
Streamlit frontend for the Iris Species Prediction API.

Provides an interactive UI where users can adjust sliders for the four
flower measurements and get a prediction from the FastAPI backend.
"""

import os

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🌸 Iris Species Predictor")
st.markdown(
    "Use the sliders to enter flower measurements. "
    "The backend FastAPI service will classify the iris as "
    "**setosa**, **versicolor**, or **virginica**."
)

# ---------------------------------------------------------------------------
# Sidebar: API status + info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ API Status")
    st.caption(f"Backend: `{API_URL}`")

    try:
        health = requests.get(f"{API_URL}/", timeout=3).json()
        st.success(f"✅ {health.get('message', 'API is up')}")
        st.json(health)
    except Exception as exc:
        st.error("❌ API is not reachable")
        st.caption(str(exc))

    st.divider()
    st.header("📊 About")
    st.markdown(
        "- **Model:** Logistic Regression\n"
        "- **Dataset:** Iris (150 samples)\n"
        "- **Classes:** 3 species\n"
        "- **Framework:** FastAPI + Streamlit\n"
        "- **Tracking:** MLflow"
    )

# ---------------------------------------------------------------------------
# Input sliders
# ---------------------------------------------------------------------------
st.subheader("Flower Measurements")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 0.1, 10.0, 5.1, 0.1)
    petal_length = st.slider("Petal length (cm)", 0.1, 10.0, 1.4, 0.1)
with col2:
    sepal_width = st.slider("Sepal width (cm)", 0.1, 10.0, 3.5, 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 10.0, 0.2, 0.1)

# Example presets
st.caption("Try a preset:")
preset_cols = st.columns(3)
if preset_cols[0].button("🌼 Setosa example"):
    sepal_length, sepal_width, petal_length, petal_width = 5.1, 3.5, 1.4, 0.2
    st.rerun()
if preset_cols[1].button("🌺 Versicolor example"):
    sepal_length, sepal_width, petal_length, petal_width = 6.0, 2.7, 4.2, 1.3
    st.rerun()
if preset_cols[2].button("🌷 Virginica example"):
    sepal_length, sepal_width, petal_length, petal_width = 6.5, 3.0, 5.5, 2.0
    st.rerun()

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
st.divider()

if st.button("🔮 Predict Species", type="primary", use_container_width=True):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    with st.spinner("Calling the API..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    # Display result
    st.success(f"🎯 Predicted species: **{result['species'].upper()}**")

    m1, m2 = st.columns(2)
    m1.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
    m2.metric("Species ID", result["species_id"])

    # Probability breakdown
    st.subheader("Probability Distribution")
    probs_df = (
        pd.DataFrame(
            {"Species": list(result["probabilities"].keys()),
             "Probability": list(result["probabilities"].values())}
        )
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )
    st.bar_chart(probs_df.set_index("Species"))
    st.dataframe(probs_df, use_container_width=True, hide_index=True)

    with st.expander("Raw API response"):
        st.json(result)

# Footer
st.divider()
st.caption("Built with FastAPI + Streamlit + MLflow + Docker · SIS-3 assignment")
