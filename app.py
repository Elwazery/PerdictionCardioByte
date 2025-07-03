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

# Set page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# Light modern CSS
st.markdown("""
<style>
    .stApp {
        background-color: #FAFAFA;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4, h5 {
        color: #111111;
        font-weight: 700;
    }
    .stSelectbox > div, .stNumberInput > div {
        background-color: #FFFFFF !important;
        border: 1px solid #DDDDDD;
        border-radius: 6px;
        padding: 6px;
    }
    .stButton > button {
        background-color: #FF4040;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #e53935;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        font-size: 16px;
        margin-top: 20px;
    }
    .low-risk {
        background-color: #DFF5E1;
        color: #155724;
        border: 1px solid #A8D5B3;
    }
    .high-risk {
        background-color: #F8D7DA;
        color: #721C24;
        border: 1px solid #F5C6CB;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("❤️ Heart Disease Prediction using ML")
st.write("Enter your medical details to predict your risk of cardiovascular disease.")

# Inputs
col1, col2, col3 = st.columns(3)
with col1:
    age_years = st.number_input("Age (in years)", min_value=1, max_value=120)
    gender = st.selectbox("Gender", options=["Female", "Male"])
    cholesterol = st.selectbox("Cholesterol level", options=["Normal", "Above normal", "High"])

with col2:
    height = st.number_input("Height (cm)", min_value=100, max_value=250)
    ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250)
    gluc = st.selectbox("Glucose level", options=["Normal", "Above normal", "High"])

with col3:
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150)
    smoke = st.selectbox("Do you smoke?", options=["No", "Yes"])

col4, _ = st.columns([1, 2])
with col4:
    active = st.selectbox("Are you physically active?", options=["No", "Yes"])

# Prediction
if st.button("🔍 Predict Heart Disease"):
    age = age_years * 365

    input_data = pd.DataFrame([{
        'age': age,
        'gender': 1 if gender == "Female" else 2,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': {"Normal": 1, "Above normal": 2, "High": 3}[cholesterol],
        'gluc': {"Normal": 1, "Above normal": 2, "High": 3}[gluc],
        'smoke': 1 if smoke == "Yes" else 0,
        'active': 1 if active == "Yes" else 0
    }])

    processed = preprocess_input_data(input_data, label_encoders, km_male, km_female, clipping_bounds)
    probability = model.predict_proba(processed)[0][1]
    prediction = model.predict(processed)[0]

    st.markdown("---")
    st.subheader("🩺 Prediction Result:")

    if probability >= 0.5:
        st.markdown(f"""
        <div class="prediction-box high-risk">
            ⚠️ The person is <strong>likely to have heart disease</strong>.<br><br>
            Estimated risk: <strong>{probability * 100:.2f}%</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box low-risk">
            ✅ The person is <strong>not likely to have heart disease</strong>.<br><br>
            Estimated risk: <strong>{probability * 100:.2f}%</strong>
        </div>
        """, unsafe_allow_html=True)
