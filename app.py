import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Chad Child Malnutrition Risk Predictor",
    page_icon="🍽️",
    layout="centered"
)

# ── Load model ──────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('malnutrition_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

model, feature_names = load_model()

# ── Header ──────────────────────────────────────────────
st.title("🍽️ Child Malnutrition Risk Predictor")
st.subheader("Chad — DHS 2014 | Gradient Boosting Model")
st.markdown("""
> This tool predicts malnutrition risk in children under five
> in Chad using basic anthropometric and household measurements.
> Built on DHS 2014 survey data (9,826 children).
> **Best used by community health workers and NGO field teams.**
""")

st.divider()

# ── Input form ──────────────────────────────────────────
st.markdown("### 👶 Child Information")
col1, col2, col3 = st.columns(3)

with col1:
    HW1 = st.number_input(
        "Child age (months)", 
        min_value=0, max_value=60, value=24,
        help="Age in months (0-60)")

with col2:
    HW2 = st.number_input(
        "Child weight (kg)",
        min_value=1.0, max_value=25.0, 
        value=10.0, step=0.1,
        help="Weight in kilograms")

with col3:
    HW3 = st.number_input(
        "Child height (cm)",
        min_value=40.0, max_value=120.0,
        value=80.0, step=0.5,
        help="Height in centimeters")

st.markdown("### 👩 Mother Information")
col4, col5, col6 = st.columns(3)

with col4:
    V445 = st.number_input(
        "Mother BMI",
        min_value=10.0, max_value=50.0,
        value=22.0, step=0.1,
        help="Body Mass Index")

with col5:
    V438 = st.number_input(
        "Mother height (cm)",
        min_value=130.0, max_value=190.0,
        value=160.0, step=0.5)

with col6:
    V012 = st.number_input(
        "Mother age (years)",
        min_value=15, max_value=49,
        value=28)

st.markdown("### 🏠 Household Information")
col7, col8, col9 = st.columns(3)

with col7:
    V191 = st.slider(
        "Wealth index score",
        min_value=-200000, max_value=200000,
        value=0, step=1000,
        help="DHS wealth index (-200k to +200k)")

with col8:
    V136 = st.number_input(
        "Household size",
        min_value=1, max_value=30, value=6)

with col9:
    V133 = st.number_input(
        "Mother education (years)",
        min_value=0, max_value=20, value=0)

st.markdown("### 🤱 Birth & Feeding Information")
col10, col11, col12 = st.columns(3)

with col10:
    BORD = st.number_input(
        "Birth order",
        min_value=1, max_value=15, value=2,
        help="1 = first child")

with col11:
    M7 = st.number_input(
        "Breastfeeding duration (months)",
        min_value=0, max_value=48, value=12)

with col12:
    B11 = st.number_input(
        "Birth interval (months)",
        min_value=0, max_value=60, value=24,
        help="Months since previous birth")

st.divider()

# ── Prediction ──────────────────────────────────────────
if st.button("🔍 Predict Malnutrition Risk", 
             type="primary", use_container_width=True):
    
    # Build input with all 24 features
    # Use median values for features not in UI
    input_data = {
        'B1'   : 6,        # Month of birth — median
        'BORD' : BORD,
        'B11'  : B11,
        'V012' : V012,
        'V133' : V133,
        'V445' : V445,
        'V438' : V438,
        'V208' : 3,        # Births in 5 years — median
        'V218' : 3,        # Living children — median
        'V221' : 24,       # Marriage to first birth — median
        'V191' : V191,
        'V136' : V136,
        'V137' : 2,        # Children under 5 — median
        'V115' : 30,       # Time to water — median
        'V426' : 0,        # Breastfeeding status
        'M7'   : M7,
        'M8'   : 18,       # Weaning age — median
        'HW1'  : HW1,
        'HW2'  : HW2 * 10, # DHS stores in 100g units
        'HW3'  : HW3 * 10, # DHS stores in mm units
        'S829' : 21,       # Toilet — median
        'S828' : 30,       # Water source — median
        'S830' : 0,        # Electricity — median
        'S831' : 95,       # Cooking fuel — median
    }

    # Create dataframe in correct feature order
    input_df = pd.DataFrame([input_data])[feature_names]

    # Predict
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.divider()
    st.markdown("## 📊 Prediction Result")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        if prob >= 0.7:
            st.error(f"🔴 HIGH RISK\n\n"
                     f"Malnutrition probability: **{prob*100:.1f}%**")
        elif prob >= 0.4:
            st.warning(f"🟡 MODERATE RISK\n\n"
                       f"Malnutrition probability: **{prob*100:.1f}%**")
        else:
            st.success(f"🟢 LOW RISK\n\n"
                       f"Malnutrition probability: **{prob*100:.1f}%**")

    with col_r2:
        st.metric("Risk Score", f"{prob*100:.1f}%")
        st.metric("Prediction", 
                  "Malnourished" if pred == 1 
                  else "Well Nourished")

    # Risk interpretation
    st.markdown("### 📋 Interpretation")
    if prob >= 0.7:
        st.markdown("""
        **High risk child — immediate action recommended:**
        - Refer to nearest nutrition center
        - Conduct MUAC measurement
        - Enroll in supplementary feeding program
        - Schedule follow-up within 2 weeks
        """)
    elif prob >= 0.4:
        st.markdown("""
        **Moderate risk — monitoring recommended:**
        - Monthly weight and height monitoring
        - Nutritional counseling for mother
        - Ensure vaccination schedule is up to date
        - Follow up in 1 month
        """)
    else:
        st.markdown("""
        **Low risk — preventive care:**
        - Continue regular growth monitoring
        - Maintain breastfeeding if applicable
        - Ensure dietary diversity
        - Routine follow-up in 3 months
        """)

st.divider()
st.markdown("""
<small>
Built by **Mahamat Hanga Derio** | M.Tech Data Science, 
Christ University, Bangalore | 
[GitHub](https://github.com/Derio001/chad-malnutrition-prediction)

Model: Gradient Boosting | Accuracy: 92% | AUC: 0.979 | 
Data: DHS Chad 2014 (n=9,826)
</small>
""", unsafe_allow_html=True)
```

---

## Create `requirements.txt`
```
streamlit
scikit-learn
pandas
numpy
xgboost
