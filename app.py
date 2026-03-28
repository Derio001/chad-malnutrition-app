import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Chad Child Malnutrition Risk Predictor",
    page_icon="🍽️",
    layout="centered"
)

# ── Language selector ────────────────────────────────────
lang = st.selectbox("🌐 Language / Langue", ["English", "Français"], index=0)
FR = lang == "Français"

# ── Load model ──────────────────────────────────────────
with open('malnutrition_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# ── Header ──────────────────────────────────────────────
st.title("🍽️ Child Malnutrition Risk Predictor" if not FR
         else "🍽️ Outil de Prédiction du Risque de Malnutrition")

st.subheader("Chad — DHS 2014 | Gradient Boosting Model" if not FR
             else "Tchad — DHS 2014 | Modèle Gradient Boosting")

st.markdown("""
> This tool predicts malnutrition risk in children under five in Chad
> using basic anthropometric and household measurements.
> Built on DHS 2014 survey data (9,826 children).
> **Best used by community health workers and NGO field teams.**
""" if not FR else """
> Cet outil prédit le risque de malnutrition chez les enfants de moins
> de cinq ans au Tchad à partir de mesures anthropométriques de base.
> Basé sur les données DHS 2014 (9 826 enfants).
> **Conçu pour les agents de santé communautaires et les équipes ONG.**
""")

st.divider()

# ── Child Information ────────────────────────────────────
st.markdown("### 👶 " + ("Child Information" if not FR else "Informations sur l'Enfant"))
col1, col2, col3 = st.columns(3)
with col1:
    HW1 = st.number_input(
        "Child age (months)" if not FR else "Âge de l'enfant (mois)",
        min_value=0, max_value=60, value=24)
with col2:
    HW2 = st.number_input(
        "Child weight (kg)" if not FR else "Poids de l'enfant (kg)",
        min_value=1.0, max_value=25.0, value=10.0, step=0.1)
with col3:
    HW3 = st.number_input(
        "Child height (cm)" if not FR else "Taille de l'enfant (cm)",
        min_value=40.0, max_value=120.0, value=80.0, step=0.5)

# ── Mother Information ───────────────────────────────────
st.markdown("### 👩 " + ("Mother Information" if not FR else "Informations sur la Mère"))
col4, col5, col6 = st.columns(3)
with col4:
    V445 = st.number_input(
        "Mother BMI" if not FR else "IMC de la mère",
        min_value=10.0, max_value=50.0, value=22.0, step=0.1)
with col5:
    V438 = st.number_input(
        "Mother height (cm)" if not FR else "Taille de la mère (cm)",
        min_value=130.0, max_value=190.0, value=160.0, step=0.5)
with col6:
    V012 = st.number_input(
        "Mother age (years)" if not FR else "Âge de la mère (années)",
        min_value=15, max_value=49, value=28)

# ── Household Information ────────────────────────────────
st.markdown("### 🏠 " + ("Household Information" if not FR else "Informations sur le Ménage"))
col7, col8, col9 = st.columns(3)
with col7:
    V191 = st.slider(
        "Wealth index score" if not FR else "Score d'indice de richesse",
        min_value=-200000, max_value=200000, value=0, step=1000)
with col8:
    V136 = st.number_input(
        "Household size" if not FR else "Taille du ménage",
        min_value=1, max_value=30, value=6)
with col9:
    V133 = st.number_input(
        "Mother education (years)" if not FR else "Éducation de la mère (années)",
        min_value=0, max_value=20, value=0)

# ── Birth and Feeding ────────────────────────────────────
st.markdown("### 🤱 " + ("Birth and Feeding Information" if not FR
                          else "Informations sur la Naissance et l'Allaitement"))
col10, col11, col12 = st.columns(3)
with col10:
    BORD = st.number_input(
        "Birth order" if not FR else "Rang de naissance",
        min_value=1, max_value=15, value=2)
with col11:
    M7 = st.number_input(
        "Breastfeeding duration (months)" if not FR
        else "Durée d'allaitement (mois)",
        min_value=0, max_value=48, value=12)
with col12:
    B11 = st.number_input(
        "Birth interval (months)" if not FR else "Intervalle entre naissances (mois)",
        min_value=0, max_value=60, value=24)

st.divider()

# ── Predict button ───────────────────────────────────────
btn_label = "🔍 Predict Malnutrition Risk" if not FR else "🔍 Prédire le Risque de Malnutrition"

if st.button(btn_label, type="primary", use_container_width=True):
    input_data = {
        'B1'  : 6,
        'BORD': BORD,
        'B11' : B11,
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
        'M7'  : M7,
        'M8'  : 18,
        'HW1' : HW1,
        'HW2' : HW2 * 10,
        'HW3' : HW3 * 10,
        'S829': 21,
        'S828': 30,
        'S830': 0,
        'S831': 95,
    }

    input_df = pd.DataFrame([input_data])[feature_names]
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.divider()
    st.markdown("## 📊 " + ("Prediction Result" if not FR else "Résultat de la Prédiction"))

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if prob >= 0.7:
            st.error(
                f"🔴 {'HIGH RISK' if not FR else 'RISQUE ÉLEVÉ'}\n\n"
                f"{'Malnutrition probability' if not FR else 'Probabilité de malnutrition'}: "
                f"**{prob*100:.1f}%**")
        elif prob >= 0.4:
            st.warning(
                f"🟡 {'MODERATE RISK' if not FR else 'RISQUE MODÉRÉ'}\n\n"
                f"{'Malnutrition probability' if not FR else 'Probabilité de malnutrition'}: "
                f"**{prob*100:.1f}%**")
        else:
            st.success(
                f"🟢 {'LOW RISK' if not FR else 'RISQUE FAIBLE'}\n\n"
                f"{'Malnutrition probability' if not FR else 'Probabilité de malnutrition'}: "
                f"**{prob*100:.1f}%**")

    with col_r2:
        st.metric(
            "Risk Score" if not FR else "Score de Risque",
            f"{prob*100:.1f}%")
        st.metric(
            "Prediction" if not FR else "Prédiction",
            ("Malnourished" if not FR else "Malnutri") if pred == 1
            else ("Well Nourished" if not FR else "Bien Nourri"))

    # ── Recommended actions ──────────────────────────────
    st.markdown("### 📋 " + ("Recommended Action" if not FR else "Action Recommandée"))

    if prob >= 0.7:
        if not FR:
            st.markdown("""
            **Immediate action recommended:**
            - Refer to nearest nutrition center
            - Conduct MUAC measurement
            - Enroll in supplementary feeding program
            - Schedule follow-up within 2 weeks
            """)
        else:
            st.markdown("""
            **Action immédiate recommandée :**
            - Référer au centre de nutrition le plus proche
            - Effectuer une mesure du PB (périmètre brachial)
            - Inscrire dans un programme d'alimentation supplémentaire
            - Planifier un suivi dans les 2 semaines
            """)
    elif prob >= 0.4:
        if not FR:
            st.markdown("""
            **Monitoring recommended:**
            - Monthly weight and height monitoring
            - Nutritional counseling for mother
            - Follow up in 1 month
            """)
        else:
            st.markdown("""
            **Surveillance recommandée :**
            - Surveillance mensuelle du poids et de la taille
            - Conseil nutritionnel pour la mère
            - Suivi dans 1 mois
            """)
    else:
        if not FR:
            st.markdown("""
            **Preventive care:**
            - Continue regular growth monitoring
            - Maintain breastfeeding if applicable
            - Routine follow-up in 3 months
            """)
        else:
            st.markdown("""
            **Soins préventifs :**
            - Continuer la surveillance régulière de la croissance
            - Maintenir l'allaitement si applicable
            - Suivi de routine dans 3 mois
            """)

st.divider()
st.markdown("""
<small>
{} <b>Mahamat Hanga Derio</b> | M.Tech Data Science, Christ University, Bangalore |
<a href="https://github.com/Derio001/chad-malnutrition-prediction">GitHub</a><br>
{}: Gradient Boosting | {}: 92% | AUC: 0.979 | {}: DHS Chad 2014 (n=9,826)
</small>
""".format(
    "Built by" if not FR else "Développé par",
    "Model" if not FR else "Modèle",
    "Accuracy" if not FR else "Précision",
    "Data" if not FR else "Données"
), unsafe_allow_html=True)
