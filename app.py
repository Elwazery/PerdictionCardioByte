import streamlit as st
import pandas as pd
import joblib
import numpy as np
from preprocess import preprocess_input_data

# Load model and preprocessing objects
model = joblib.load("xgb_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
km_male = joblib.load("km_male.pkl")
km_female = joblib.load("km_female.pkl")
clipping_bounds = joblib.load("clipping_bounds.pkl")

# Page setup
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: white; }
    .stButton>button {
        background-color: #FF0000;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🩺 Multiple Disease Prediction System")
pages = {
    "Diabetes Prediction": "diabetes",
    "Heart Disease Prediction": "heart",
    "Parkinsons Prediction": "parkinsons"
}
choice = st.sidebar.radio("", list(pages.keys()), index=1)

if choice != "Heart Disease Prediction":
    st.sidebar.warning("This version is set for Heart Disease Prediction only.")
    st.stop()

# Main title
st.markdown("<h2 style='text-align: center;'>❤️ Heart Disease Prediction using ML</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Form layout
col1, col2, col3 = st.columns(3)

with col1:
    age_years = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
    age = age_years * 365
    gender = st.selectbox("Gender", options={"1": "Female", "2": "Male"})
    cholesterol = st.selectbox("Cholesterol level", options={"1": "Normal", "2": "Above normal", "3": "High"})

with col2:
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, value=120)
    gluc = st.selectbox("Glucose level", options={"1": "Normal", "2": "Above normal", "3": "High"})

with col3:
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)
    smoke = st.selectbox("Do you smoke?", options={0: "No", 1: "Yes"})

# Final row
col4, col5 = st.columns(2)
with col4:
    active = st.selectbox("Are you physically active?", options={0: "No", 1: "Yes"})

# Prediction
if st.button("🔍 Predict Heart Disease", use_container_width=True):
    input_data = pd.DataFrame([{
        'age': age,
        'gender': int(gender),
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': int(cholesterol),
        'gluc': int(gluc),
        'smoke': int(smoke),
        'active': int(active)
    }])

    processed = preprocess_input_data(input_data, label_encoders, km_male, km_female, clipping_bounds)
    probability = model.predict_proba(processed)[0][1]
    prediction = model.predict(processed)[0]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🩺 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ The person is **likely to have heart disease**.\n\nEstimated risk: **{probability * 100:.2f}%**")
    else:
        st.success(f"✅ The person is **not likely to have heart disease**.\n\nEstimated risk: **{(1 - probability) * 100:.2f}%**")
