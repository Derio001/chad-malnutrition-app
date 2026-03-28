import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Chad Child Malnutrition Risk Predictor",
    page_icon="🍽️",
    layout="centered"
)

with open('malnutrition_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

st.title("🍽️ Child Malnutrition Risk Predictor")
st.subheader("Chad — DHS 2014 | Gradient Boosting Model")
st.markdown("""
> This tool predicts malnutrition risk in children under five in Chad
> using basic anthropometric and household measurements.
> Built on DHS 2014 survey data (9,826 children).
> **Best used by community health workers and NGO field teams.**
""")
st.divider()

st.markdown("### 👶 Child Information")
col1, col2, col3 = st.columns(3)
with col1:
    HW1 = st.number_input("Child age (months)", min_value=0, max_value=60, value=24)
with col2:
    HW2 = st.number_input("Child weight (kg)", min_value=1.0, max_value=25.0, value=10.0, step=0.1)
with col3:
    HW3 = st.number_input("Child height (cm)", min_value=40.0, max_value=120.0, value=80.0, step=0.5)

st.markdown("### 👩 Mother Information")
col4, col5, col6 = st.columns(3)
with col4:
    V445 = st.number_input("Mother BMI", min_value=10.0, max_value=50.0, value=22.0, step=0.1)
with col5:
    V438 = st.number_input("Mother height (cm)", min_value=130.0, max_value=190.0, value=160.0, step=0.5)
with col6:
    V012 = st.number_input("Mother age (years)", min_value=15, max_value=49, value=28)

st.markdown("### 🏠 Household Information")
col7, col8, col9 = st.columns(3)
with col7:
    V191 = st.slider("Wealth index score", min_value=-200000, max_value=200000, value=0, step=1000)
with col8:
    V136 = st.number_input("Household size", min_value=1, max_value=30, value=6)
with col9:
    V133 = st.number_input("Mother education (years)", min_value=0, max_value=20, value=0)

st.markdown("### 🤱 Birth and Feeding Information")
col10, col11, col12 = st.columns(3)
with col10:
    BORD = st.number_input("Birth order", min_value=1, max_value=15, value=2)
with col11:
    M7 = st.number_input("Breastfeeding duration (months)", min_value=0, max_value=48, value=12)
with col12:
    B11 = st.number_input("Birth interval (months)", min_value=0, max_value=60, value=24)

st.divider()

if st.button("🔍 Predict Malnutrition Risk", type="primary", use_container_width=True):
    input_data = {
        'B1': 6,
        'BORD': BORD,
        'B11': B11,
        'V012': V012,
        'V133': V133,
        'V445': V445,
        'V438': V438,
        'V208': 3,
        'V218': 3,
        'V221': 24,
        'V191': V191,
        'V136': V136,
        'V137': 2,
        'V115': 30,
        'V426': 0,
        'M7': M7,
        'M8': 18,
        'HW1': HW1,
        'HW2': HW2 * 10,
        'HW3': HW3 * 10,
        'S829': 21,
        'S828': 30,
        'S830': 0,
        'S831': 95,
    }

    input_df = pd.DataFrame([input_data])[feature_names]
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.divider()
    st.markdown("## 📊 Prediction Result")
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        if prob >= 0.7:
            st.error(f"🔴 HIGH RISK\n\nMalnutrition probability: **{prob*100:.1f}%**")
        elif prob >= 0.4:
            st.warning(f"🟡 MODERATE RISK\n\nMalnutrition probability: **{prob*100:.1f}%**")
        else:
            st.success(f"🟢 LOW RISK\n\nMalnutrition probability: **{prob*100:.1f}%**")

    with col_r2:
        st.metric("Risk Score", f"{prob*100:.1f}%")
        st.metric("Prediction", "Malnourished" if pred == 1 else "Well Nourished")

    st.markdown("### 📋 Recommended Action")
    if prob >= 0.7:
        st.markdown("""
        **Immediate action recommended:**
        - Refer to nearest nutrition center
        - Conduct MUAC measurement
        - Enroll in supplementary feeding program
        - Schedule follow-up within 2 weeks
        """)
    elif prob >= 0.4:
        st.markdown("""
        **Monitoring recommended:**
        - Monthly weight and height monitoring
        - Nutritional counseling for mother
        - Follow up in 1 month
        """)
    else:
        st.markdown("""
        **Preventive care:**
        - Continue regular growth monitoring
        - Maintain breastfeeding if applicable
        - Routine follow-up in 3 months
        """)

st.divider()
st.markdown("""
<small>Built by <b>Mahamat Hanga Derio</b> | M.Tech Data Science, Christ University, Bangalore |
<a href="https://github.com/Derio001/chad-malnutrition-prediction">GitHub</a><br>
Model: Gradient Boosting | Accuracy: 92% | AUC: 0.979 | Data: DHS Chad 2014 (n=9,826)
</small>
""", unsafe_allow_html=True)
