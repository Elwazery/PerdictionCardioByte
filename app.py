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

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered", initial_sidebar_state="collapsed")

# Styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🫀 Heart Disease Risk Prediction")
st.markdown("Enter your medical details to predict your risk of cardiovascular disease.")

# Age input as years, convert to days internally
age_years = st.number_input("Age (in years)", min_value=1, max_value=100, value=30)
age = age_years * 365  # convert to days

# Other user inputs
gender = st.selectbox("Gender", options={"1": "Female", "2": "Male"})
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
ap_hi = st.number_input("Systolic blood pressure (ap_hi)", min_value=80, max_value=250, value=120)
ap_lo = st.number_input("Diastolic blood pressure (ap_lo)", min_value=40, max_value=150, value=80)
cholesterol = st.selectbox("Cholesterol level", options={"1": "Normal", "2": "Above normal", "3": "High"})
gluc = st.selectbox("Glucose level", options={"1": "Normal", "2": "Above normal", "3": "High"})
smoke = st.selectbox("Do you smoke?", options={0: "No", 1: "Yes"})
alco = st.selectbox("Do you consume alcohol?", options={0: "No", 1: "Yes"})  # will be ignored
active = st.selectbox("Are you physically active?", options={0: "No", 1: "Yes"})

# When user clicks predict
if st.button("🔍 Predict"):
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
        'alco': int(alco),  # ignored in preprocessing
        'active': int(active)
    }])

    # Preprocess
    processed = preprocess_input_data(input_data, label_encoders, km_male, km_female, clipping_bounds)

    # Predict
    probability = model.predict_proba(processed)[0][1]
    prediction = model.predict(processed)[0]

    # Output
    st.markdown("---")
    st.subheader("🩺 Prediction Result:")
    st.write(f"🔬 Estimated risk of heart disease: **{probability * 100:.2f}%**")
    if prediction == 1:
        st.error("⚠️ Potential risk of cardiovascular disease. Please consult a medical professional.")
    else:
        st.success("✅ Low predicted risk. Stay healthy!")
