import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessing objects
model = joblib.load("xgb_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
km_male = joblib.load("km_male.pkl")
km_female = joblib.load("km_female.pkl")
clipping_bounds = joblib.load("clipping_bounds.pkl")

# Import preprocessing function
from preprocess import preprocess_input_data

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Custom style
st.markdown("""
    <style>
        .stApp {
            background-color: white;
        }
        .stNumberInput input, .stSelectbox div {
            background-color: #F0F0F0;
            color: black;
            border: 1px solid #CCCCCC;
            border-radius: 4px;
            padding: 5px;
        }
        .stButton>button {
            background-color: #FF0000;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 25px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Multiple Disease Prediction System")
pages = {
    "Diabetes Prediction": "diabetes",
    "Heart Disease Prediction": "heart"
}
choice = st.sidebar.radio("", list(pages.keys()), index=1, format_func=lambda x: f"{'❤️' if x == 'Heart Disease Prediction' else ''} {x}")
if choice != "Heart Disease Prediction":
    st.sidebar.warning("This app is currently set for Heart Disease Prediction.")
    st.stop()

# Title
st.markdown("<h1 style='color:black;'>❤️ Heart Disease Prediction using ML</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:black;'>Enter your medical information to predict your risk of cardiovascular disease.</p>", unsafe_allow_html=True)

# Input form
col1, col2, col3 = st.columns(3)

with col1:
    age_years = st.number_input("Age (in years)", min_value=1, max_value=120, step=1, format="%d")
    gender = st.selectbox("Gender", options={"1": "Female", "2": "Male"})

with col2:
    height = st.number_input("Height (cm)", min_value=100, max_value=250, step=1, format="%d")
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, step=1, format="%d")

with col3:
    ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, step=1, format="%d")
    ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, step=1, format="%d")

col4, col5 = st.columns(2)

with col4:
    cholesterol = st.selectbox("Cholesterol level", options={"1": "Normal", "2": "Above normal", "3": "High"})
    gluc = st.selectbox("Glucose level", options={"1": "Normal", "2": "Above normal", "3": "High"})

with col5:
    smoke = st.selectbox("Do you smoke?", options={0: "No", 1: "Yes"})
    active = st.selectbox("Are you physically active?", options={0: "No", 1: "Yes"})

# Predict
if st.button("🔍 Predict Heart Disease"):

    try:
        age = age_years * 365
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

        # Preprocess
        processed = preprocess_input_data(input_data, label_encoders, km_male, km_female, clipping_bounds)

        # Predict
        probability = model.predict_proba(processed)[0][1]
        probability_percent = probability * 100

        # Display result
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:black; font-weight:bold;'>🩺 Prediction Result:</h3>", unsafe_allow_html=True)

        if probability_percent >= 50:
            st.markdown(f"""
                <div style='background-color: #f8d7da; padding: 15px; border-radius: 10px;'>
                    <p style='color:black; font-size:16px;'>
                        ⚠️ <strong>The person is likely to have heart disease.</strong><br><br>
                        <strong>Estimated risk:</strong> {probability_percent:.2f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='background-color: #d4edda; padding: 15px; border-radius: 10px;'>
                    <p style='color:black; font-size:16px;'>
                        ✅ <strong>The person is not likely to have heart disease.</strong><br><br>
                        <strong>Estimated risk:</strong> {probability_percent:.2f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
