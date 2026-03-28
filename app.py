import streamlit as st
import pickle

st.set_page_config(
    page_title="Chad Malnutrition Risk Predictor",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ Chad Child Malnutrition Risk Predictor")

st.write("Testing model load...")

try:
    with open('malnutrition_model.pkl', 'rb') as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")

try:
    with open('feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    st.success(f"Features loaded: {len(features)} features")
except Exception as e:
    st.error(f"Features loading failed: {str(e)}")

st.write("Debug complete.")
